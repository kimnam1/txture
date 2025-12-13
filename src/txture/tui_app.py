from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.events import Key
from textual.containers import Container

from pathlib import Path
from typing import Union, Optional
import numpy as np
import cv2
import subprocess
import platform

from dataclasses import replace

import time
from datetime import datetime

from txture.devices import open_auto_camera
from txture.pipeline import process_frame
from txture.loaders import load_lut
from txture.ascii_render import frame_to_ascii
from txture.config import (
    DEFAULT_BRIGHTNESS_THRESHOLD,
    DEFAULT_FPS,
    DEFAULT_GESTURE_CONFIDENCE,
    DEFAULT_FACE_CONFIDENCE,
    KEY_COOLDOWN_S,
    DEFAULT_STABLE_FRAMES,
)
from txture.control import (
    state as ctrl_state,
    handle_key,
    format_help_line,
    format_conf_line,
)
from txture.effects import EffectStack, EffectLayer, EffectContext
from rich.text import Text


def _build_ctx(app: "TxtureApp") -> EffectContext:
    return EffectContext(
        mode=ctrl_state.mode,
        outline=ctrl_state.outline,
        color=ctrl_state.color,
        face_label=app.face_label,
        face_conf=app.face_conf,
        gesture_label=app.gesture_label,
        gesture_conf=app.gesture_conf,
        flags={"debug_effect": getattr(app, "debug_effect_enabled", False)},
        payload={},
    )


BASE = Path(__file__).resolve().parents[2]
METRIC_DIR = BASE / "data" / "metrics"

ML_BASE = BASE / "src" / "ml_models"
GESTURE_CKPT = ML_BASE / "gestures" / "checkpoints" / "gesture_model.pkl"
FACE_CKPT = ML_BASE / "expressions" / "checkpoints" / "expression_model.pth"


def discover_metric_files():
    paths = list(METRIC_DIR.glob("*.json"))
    mapping = {}
    for p in paths:
        name = p.name
        label = name.split("__")[0]
        mapping.setdefault(label, p)
    return mapping


class TxtureApp(App):
    CSS = """
Screen {
    layout: vertical;
}

#root{
    layout: vertical;
}

/* Main area: left(ASCII + status) + right(face + hand) */
#main_area {
    layout: horizontal;
    height: 1fr;
}

/* Left area: ASCII + status*/
#left-plane {
    layout: vertical;
    width: 3fr;
}

#ascii {
    width: 1fr;
    height: 1fr;
    border: #6BA0E3;
}

#status{
    height: 6;
    border: #CDF233;
}

/* right vertical stack (face, hand) */
#right-plane {
    layout: vertical;
    width: 1fr;
}

/* Upper face area */
#face {
    height: 1fr;
    border: #4533F2;
}

/* Lower hand area */
#hand {
    height: 1fr;
    border: #F2D633;
}

/* Bottom help area */
#help {
    height: 4;
    border: none;
}

"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cap = None
        self.lut = None
        self.sat_gain = 1.0
        self.brightness_threshold = DEFAULT_BRIGHTNESS_THRESHOLD

        self._metric_files: dict[str, Path] | None = None
        self._active_metric_key: str | None = None

        self.face_detector = None
        self.expression_recognizer = None
        self.face_filter = None
        self.face_label = None
        self.face_conf = 0.0

        self.hand_detector = None
        self.gesture_recognizer = None
        self.gesture_filter = None
        self.gesture_label = None
        self.gesture_conf = 0.0
        self._gesture_fired = None
        self._gesture_cooldown_s = KEY_COOLDOWN_S
        self._last_gesture_ts = 0.0

        self.effects = EffectStack()
        self.debug_effect = False

        self._last_ascii_text: str = ""
        self._last_face_text: str = ""
        self._last_hand_text: str = ""

    def compose(self) -> ComposeResult:
        yield Container(
            Container(
                Container(
                    Static("ASCII AREA", id="ascii", markup=False),
                    Static("STATUS AREA", id="status", markup=False),
                    id="left-plane",
                ),
                Container(
                    Static("FACE AREA", id="face", markup=False),
                    Static("HAND AREA", id="hand", markup=False),
                    id="right-plane",
                ),
                id="main_area",
            ),
            Static("HELP", id="help", markup=False),
            id="root",
        )

    async def on_mount(self) -> None:
        self.ascii_view: Static = self.query_one("#ascii", Static)
        self.status_view: Static = self.query_one("#status", Static)
        self.face_view: Static = self.query_one("#face", Static)
        self.hand_view: Static = self.query_one("#hand", Static)
        self.help_view: Static = self.query_one("#help", Static)

        self._metric_files = discover_metric_files()
        metric_key = getattr(ctrl_state, "charset", "ascii_punctuation_only")

        if metric_key not in self._metric_files:
            self.status_view.update(f'metric "{metric_key}" not found')
            return

        self.lut = load_lut(self._metric_files[metric_key])
        self._active_metric_key = metric_key

        self.cap, cam_info = open_auto_camera(
            max_devices=3,
        )

        from ml_models.detection.hand_detector import HandDetector
        from ml_models.detection.face_detector import FaceDetector
        from ml_models.gestures.inference import GestureRecognizer
        from ml_models.expressions.inference import ExpressionRecognizer
        from txture.detection_event import EventFilter

        self.hand_detector = HandDetector(
            max_num_hands=1, detection_confidence=0.5
        )
        self.face_detector = FaceDetector()

        self.gesture_recognizer = GestureRecognizer(
            model_path=str(GESTURE_CKPT)
        )
        self.gesture_filter = EventFilter(
            min_conf=DEFAULT_GESTURE_CONFIDENCE,
            stable_frames=DEFAULT_STABLE_FRAMES,
        )

        self.expression_recognizer = ExpressionRecognizer(
            model_path=str(FACE_CKPT)
        )
        self.face_filter = EventFilter(
            min_conf=DEFAULT_FACE_CONFIDENCE,
            stable_frames=DEFAULT_STABLE_FRAMES,
        )
        self.face_label = None
        self.face_conf = 0.0

        self.tick_timer = self.set_interval(1 / DEFAULT_FPS, self._on_tick)

        self.status_view.update(
            f"Camera index = {cam_info.index} backend = {cam_info.backend} score = {cam_info.score: .2f}"
        )

        def _demo_ascii_layer(
            lines: list[str],
            colors: list[list[tuple[int, int, int]]] | None,
            ctx: EffectContext,
        ) -> tuple[list[str], list[list[tuple[int, int, int]]] | None]:
            if not ctx.flags.get("debug_effect", False):
                return lines, colors
            if not lines:
                return lines, colors
            msg = "[DEBUG EFFECT ON]"
            width = max(len(lines[0]), len(msg))
            new_lines = list(lines)
            row = new_lines[0].ljust(width)
            new_lines[0] = (msg + row[len(msg) :])[:width]
            return new_lines, colors

        self.add_effect(
            EffectLayer(
                name="demo_ascii_layer",
                priority=10,
                enabled=True,
                apply_ascii=_demo_ascii_layer,
            )
        )

        def _outline_layer(
            lines: list[str],
            colors: list[list[tuple[int, int, int]]] | None,
            ctx: EffectContext,
        ) -> tuple[list[str], list[list[tuple[int, int, int]]] | None]:
            if not ctx.outline:
                return lines, colors

            grid = ctx.payload.get("outline_grid")
            if grid is None or not lines:
                return lines, colors

            dir_to_ch = {
                0: "|",
                1: "╱",  # U+2571
                2: "─",  # U+2500 ─ or use ━ U+2501
                3: "╲",  # U+2572
                4: "|",
                5: "╱",  # U+2571
                6: "─",
                7: "╲",  # U+2572
            }

            out_lines: list[str] = []

            for y, line in enumerate(lines):
                width = len(line)
                row = [" "] * width

                if y < grid.shape[0]:
                    max_x = min(width, grid.shape[1])
                    for x in range(max_x):
                        d = int(grid[y, x])
                        if d >= 0:
                            row[x] = dir_to_ch.get(d, "|")

                out_lines.append("".join(row))

            return out_lines, None

        self.add_effect(
            EffectLayer(
                name="outline_layer",
                priority=20,
                enabled=True,
                apply_ascii=_outline_layer,
            )
        )

    async def on_key(self, event: Key) -> None:
        # end key
        if event.key in ("escape", "ctrl+q"):
            await self.action_quit()
            return

        # key code mapping
        keycode: int | None = None

        ch = getattr(event, "character", None)

        if ch:
            keycode = ord(ch)

        elif event.key == "backspace":
            keycode = 8

        elif event.key == "left":
            keycode = 81
        elif event.key == "right":
            keycode = 83

        if keycode is None:
            return

        handle_key(ctrl_state, keycode)

        if not ctrl_state.running:
            await self.action_quit()

    def _get_face_crop(self, frame) -> Optional[np.ndarray]:
        """Get cropped face region if detected"""
        if self.face_detector is None:
            self.face_label = None
            self.face_conf = 0.0

            if self.face_filter is not None:
                self.face_filter.update(None, 0.0)
            return None

        faces = self.face_detector.detect(frame)
        if not faces:
            self.face_label = None
            self.face_conf = 0.0

            if self.face_filter is not None:
                self.face_filter.update(None, 0.0)
            return None

        x1, y1, x2, y2 = faces[0]
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        face_crop = self._crop_square_with_padding(
            frame, x1, y1, x2, y2, pad_ratio=0.2
        )
        if face_crop.size == 0:
            self.face_label = None
            self.face_conf = 0.0
            if self.face_filter is not None:
                self.face_filter.update(None, 0.0)
            return None
        label = None
        conf = 0.0

        if self.expression_recognizer is not None:
            try:
                # Use intelligent judgment method with threshold set to 0.3
                label, conf, _ = (
                    self.expression_recognizer.recognize_all_emotions(
                        face_crop, threshold=0.3
                    )
                )
            except Exception:
                label, conf = None, 0.0

        self.face_label = label
        self.face_conf = conf
        fired = None
        if self.face_filter is not None:
            fired = self.face_filter.update(label, conf)

        if fired is not None:
            self._on_face_event(fired, conf)
        return face_crop if face_crop.size > 0 else None

    def _on_face_event(self, label: str, conf: float) -> None:
        pass

    def _apply_gesture(self, gesture: str) -> None:
        now = time.monotonic()
        if now - self._last_gesture_ts < self._gesture_cooldown_s:
            return
        self._last_gesture_ts = now
        gesture = (gesture or "").strip().lower()

        if not gesture:
            return

        handle_key(ctrl_state, ord(gesture[0]))

    def _crop_square_with_padding(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        pad_ratio: float = 0.2,
    ) -> np.ndarray:
        h, w = frame.shape[:2]

        box_w = max(1, x2 - x1)
        box_h = max(1, y2 - y1)

        side = int(max(box_w, box_h) * (1 + 2 * pad_ratio))

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        sx1 = center_x - side // 2
        sy1 = center_y - side // 2
        sx2 = sx1 + side
        sy2 = sy1 + side

        ix1 = max(0, sx1)
        iy1 = max(0, sy1)
        ix2 = min(w, sx2)
        iy2 = min(h, sy2)

        crop = frame[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            return np.zeros((side, side, 3), dtype=frame.dtype)

        top = max(0, -sy1)
        left = max(0, -sx1)
        bottom = max(0, sy2 - h)
        right = max(0, sx2 - w)

        if top or bottom or left or right:
            crop = cv2.copyMakeBorder(
                crop,
                top,
                bottom,
                left,
                right,
                borderType=cv2.BORDER_CONSTANT,
                value=[0, 0, 0],
            )

        return crop

    def _letterbox_to(
        self, img: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        if target_w <= 0 or target_h <= 0:
            return img

        h, w = img.shape[:2]

        if h <= 0 or w <= 0:
            return np.zeros((target_h, target_w, 3), dtype=img.dtype)

        scale = min(target_w / w, target_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

        out = np.zeros((target_h, target_w, 3), dtype=img.dtype)

        x0 = (target_w - new_w) // 2
        y0 = (target_h - new_h) // 2
        out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
        return out

    def _get_hand_crop(self, frame) -> Optional[np.ndarray]:
        """Get cropped hand region with keypoints drawn"""
        if self.hand_detector is None:
            return None
        hands = self.hand_detector.detect(frame)
        if not hands:
            return None

        hand_points = hands[0]

        vis_frame = frame.copy()

        for x, y, z, vis in hand_points:
            cv2.circle(vis_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        # Calculate bounding box
        xs = [p[0] for p in hand_points]
        ys = [p[1] for p in hand_points]

        padding = 30
        h, w = vis_frame.shape[:2]
        x1 = max(0, int(min(xs)) - padding)
        y1 = max(0, int(min(ys)) - padding)
        x2 = min(w, int(max(xs)) + padding)
        y2 = min(h, int(max(ys)) + padding)

        hand_crop = self._crop_square_with_padding(
            vis_frame, x1, y1, x2, y2, pad_ratio=0.2
        )

        if hand_crop.size == 0:
            return None

        return hand_crop

    def outline_grid(
        self, edge_dir: np.ndarray, rows: int, cols: int
    ) -> np.ndarray:
        h, w = edge_dir.shape
        if rows <= 0 or cols <= 0:
            return np.zeros((0, 0), dtype=np.int8)

        cell_h = max(1, h // rows)
        cell_w = max(1, w // cols)

        out = np.full((rows, cols), -1, dtype=np.int8)

        for r in range(rows):
            for c in range(cols):
                sy = r * cell_h
                sx = c * cell_w
                ey = min(h, (r + 1) * cell_h)
                ex = min(w, (c + 1) * cell_w)

                cell = edge_dir[sy:ey, sx:ex]
                if cell.size == 0:
                    continue

                valid = cell[cell >= 0]
                if valid.size == 0:
                    continue

                hist = np.bincount(valid.astype(np.int32), minlength=8)[:8]
                out[r, c] = int(np.argmax(hist))

        return out

    def _render_ascii_raw(
        self, frame, cols: int, rows: int, *, color, outline
    ):
        if self.lut is None:
            return "No LUT loaded."

        features = process_frame(
            frame,
        )

        ctx = replace(_build_ctx(self), outline=outline)

        processed = self.effects.apply_frame(features.processed, ctx)
        edge_arg = None

        lines, colors = frame_to_ascii(
            processed,
            self.lut,
            cols=cols,
            char_aspect=2.0,
            colorize=color,
            saturation_gain=self.sat_gain,
            brightness_threshold=self.brightness_threshold,
            edge_dir=edge_arg,
        )

        max_rows = max(1, rows)

        lines = lines[:max_rows]

        if colors is not None:
            colors = colors[:max_rows]

        if outline and features.edge_dir is not None:
            rows_eff = len(lines)
            cols_eff = len(lines[0]) if lines else 0
            outlined = self.outline_grid(
                features.edge_dir, rows=rows_eff, cols=cols_eff
            )
            # You can use 'outlined' for further processing if needed
            ctx.payload["outline_grid"] = outlined

        lines, colors = self.effects.apply_ascii(lines, colors, ctx)

        if not color or colors is None:
            return "\n".join(lines)

        text = Text()
        for y, line in enumerate(lines):
            for x, ch in enumerate(line):
                r, g, b = colors[y][x]
                text.append(ch, style=f"rgb({r},{g},{b})")
            text.append("\n")

        return text

    def _render_ascii(
        self, frame, cols: int, rows: int, *, color: bool, outline: bool
    ) -> Union[str, Text]:
        fired = None
        if self.lut is None:
            return "No LUT loaded."

        if (
            self.hand_detector is not None
            and self.gesture_recognizer is not None
        ):
            hands = self.hand_detector.detect(frame)
            label = None
            conf = 0.0

            if hands:
                label, conf = self.gesture_recognizer.recognize(hands[0])

            if self.gesture_filter is not None:
                fired = self.gesture_filter.update(label, conf)
            else:
                _ = None

            self.gesture_label = label
            self.gesture_conf = conf
        else:
            self.gesture_label = None
            self.gesture_conf = 0.0

        self._gesture_fired = fired

        features = process_frame(
            frame,
        )

        ctx = replace(_build_ctx(self), outline=outline)

        processed = self.effects.apply_frame(features.processed, ctx)

        edge_arg = None

        lines, colors = frame_to_ascii(
            processed,
            self.lut,
            cols=cols,
            char_aspect=2.0,
            colorize=color,
            saturation_gain=self.sat_gain,
            brightness_threshold=self.brightness_threshold,
            edge_dir=edge_arg,
        )

        max_rows = max(1, rows)

        lines = lines[:max_rows]

        if colors is not None:
            colors = colors[:max_rows]

        if outline and features.edge_dir is not None:
            rows_eff = len(lines)
            cols_eff = len(lines[0]) if lines else 0
            outlined = self.outline_grid(
                features.edge_dir, rows=rows_eff, cols=cols_eff
            )
            # You can use 'outlined' for further processing if needed
            ctx.payload["outline_grid"] = outlined

        lines, colors = self.effects.apply_ascii(lines, colors, ctx)

        if not color or colors is None:
            return "\n".join(lines)

        text = Text()
        for y, line in enumerate(lines):
            for x, ch in enumerate(line):
                r, g, b = colors[y][x]
                text.append(ch, style=f"rgb({r},{g},{b})")
            text.append("\n")

        return text

    async def on_unmount(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def add_effect(self, layer: EffectLayer) -> None:
        self.effects.add_layer(layer)

    def _render_help(self) -> str:
        general_help_line = "[y] YANK(copy) | [esc] QUIT"
        help_line = format_help_line(ctrl_state)

        return general_help_line + "\n" + help_line

    def _render_mode_line(self) -> str:
        outline_flag = "ON" if ctrl_state.outline else "OFF"
        color_flag = (
            "-" if ctrl_state.outline else "ON" if ctrl_state.color else "OFF"
        )
        charset_key = getattr(ctrl_state, "charset", "ascii_punctuation_only")
        charset_label_map = {
            "ascii_letters_only": "LETTERS",
            "ascii_punctuation_only": "PUNCTUATION",
            "ascii_dots_only": "DOTS",
            "ascii_digits_only": "DIGITS",
            "ascii_all": "ALL",
        }
        charset_label = charset_label_map.get(charset_key, charset_key)

        return f"MODE: {ctrl_state.mode}\nOUTLINE: {outline_flag}\nCOLOR: {color_flag}\nCHARACTER: {charset_label}"

    def _append_line_to_ascii(
        self, ascii_renderable: Union[str, Text], line: str
    ) -> Union[str, Text]:
        if isinstance(ascii_renderable, Text):
            out = ascii_renderable.copy()

            out.append("\n")
            out.append(line)

            return out
        return (ascii_renderable or "") + "\n" + line

    def _calc_ascii_size(self) -> tuple[int, int]:
        screen_width = self.size.width
        width = int(screen_width * 0.75)  # ASCII area is 3/4 of width

        screen_height = self.size.height

        ascii_cols = max(20, width - 2)
        ascii_rows = max(3, screen_height)

        return ascii_cols, ascii_rows

    def _ensure_lut_for_charset(self) -> None:
        if self._metric_files is None:
            self._metric_files = discover_metric_files()

        key = getattr(ctrl_state, "charset", "ascii_punctuation_only")

        if key == self._active_metric_key:
            return

        if self._metric_files is None or key not in self._metric_files:
            return

        self.lut = load_lut(self._metric_files[key])
        self._active_metric_key = key

    async def _on_tick(self) -> None:
        if self.cap is not None:
            ok, frame = self.cap.read()
            frame = cv2.flip(frame, 1)
            if not ok:
                ascii_renderable = "Failed to read from camera."
            else:
                cols, target_rows = self._calc_ascii_size()

                target_rows = max(3, target_rows - 1)

                self._ensure_lut_for_charset()

                ascii_renderable = self._render_ascii(
                    frame=frame,
                    cols=cols,
                    rows=target_rows,
                    color=ctrl_state.color,
                    outline=ctrl_state.outline,
                )

        else:
            ascii_renderable = "No camera connected."

        if self.cap is not None and ok:
            self._ensure_lut_for_charset()
            face_crop = self._get_face_crop(frame)

            if face_crop is not None:
                fw = self.face_view.size.width
                fh = self.face_view.size.height
                if fw > 4 and fh > 4:
                    cols = max(20, fw - 2)
                    rows = max(1, fh - 3)

                    stable_face = self._letterbox_to(
                        face_crop, target_w=320, target_h=320
                    )

                    ascii_face = self._render_ascii_raw(
                        frame=stable_face,
                        cols=cols,
                        rows=rows,
                        color=False,
                        outline=True,
                    )

                    max_chars = max(10, cols)
                    bar_len = max(5, max_chars - 20)

                    face_status = format_conf_line(
                        title="FACE",
                        label=self.face_label,
                        conf=self.face_conf,
                        length=bar_len,
                        thresh=0.7,
                    )
                    self.face_view.update(ascii_face + "\n" + face_status)

                    self._last_face_text = (
                        self._to_plain_text(ascii_face) + "\n" + face_status
                    )

            hand_crop = self._get_hand_crop(frame)
            if hand_crop is not None:
                hw = self.hand_view.size.width
                hh = self.hand_view.size.height
                if hw > 4 and hh > 4:
                    cols = max(20, hw - 2)
                    rows = max(1, hh - 3)

                    stable_hand = self._letterbox_to(
                        hand_crop, target_w=320, target_h=320
                    )

                    ascii_hand = self._render_ascii_raw(
                        frame=stable_hand,
                        cols=cols,
                        rows=rows,
                        color=False,
                        outline=True,
                    )

                    max_chars = max(10, cols)
                    bar_len = max(5, max_chars - 20)

                    gesture_status = format_conf_line(
                        title="GESTURE",
                        label=self.gesture_label,
                        conf=self.gesture_conf,
                        length=bar_len,
                        thresh=0.7,
                    )
                    self.hand_view.update(ascii_hand + "\n" + gesture_status)

                    self._last_hand_text = (
                        self._to_plain_text(ascii_hand) + "\n" + gesture_status
                    )

        if getattr(self, "_gesture_fired", None) is not None:
            self._apply_gesture(getattr(self, "_gesture_fired"))
            self._gesture_fired = None

        self._last_ascii_text = self._to_plain_text(ascii_renderable)

        mode_text = self._render_mode_line()
        help_text = self._render_help()

        self.ascii_view.update(ascii_renderable)
        self.status_view.update(mode_text)
        self.help_view.update(help_text)
        self._handle_copy_request()

    def _to_plain_text(self, renderable: Union[str, Text, None]) -> str:
        if renderable is None:
            return ""
        if isinstance(renderable, str):
            return renderable
        elif isinstance(renderable, Text):
            return renderable.plain
        else:
            return str(renderable)

    def _downloads_dir(self) -> Path:
        # Default user Downloads folder
        return Path.home() / "Downloads"

    def _yank_stamp(self) -> str:
        # YYYYMMDD_HHMMSS
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _save_text_file(self, text: str, out_path: Path) -> bool:
        try:
            out_path.write_text(text or "", encoding="utf-8")
            return True
        except Exception:
            return False

    def _save_png_from_text(self, text: str, out_path: Path) -> bool:
        """Render plain text into a PNG file.

        Prefers Pillow for proper monospace layout. If Pillow is unavailable,
        returns False.
        """
        try:
            from PIL import Image, ImageDraw, ImageFont  # type: ignore
        except Exception:
            return False

        lines = (text or "").splitlines() or [""]
        max_cols = max((len(line) for line in lines), default=0)

        # Try to load a common monospace font. Fall back to Pillow default.
        font = None
        font_size = 14
        for fp in [
            "/System/Library/Fonts/Menlo.ttc",
            "/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
        ]:
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                font = None

        if font is None:
            try:
                font = ImageFont.load_default()
            except Exception:
                return False

        # Measure using a representative glyph.
        try:
            bbox = font.getbbox("M")
            char_w = max(1, bbox[2] - bbox[0])
            line_h = max(1, bbox[3] - bbox[1])
        except Exception:
            # Conservative fallback
            char_w = 8
            line_h = 16

        padding = 12
        img_w = max(1, padding * 2 + max_cols * char_w)
        img_h = max(1, padding * 2 + len(lines) * line_h)

        img = Image.new("RGB", (img_w, img_h), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        y = padding
        for line in lines:
            draw.text((padding, y), line, font=font, fill=(255, 255, 255))
            y += line_h

        try:
            img.save(out_path, format="PNG")
            return True
        except Exception:
            return False

    def _save_yanked_files(self, text: str) -> tuple[Path | None, Path | None]:
        """Save yanked content into ~/Downloads as both .txt and .png.

        Returns (txt_path, png_path). Either can be None if saving fails.
        """
        stamp = self._yank_stamp()
        base = f"yanked_{stamp}"

        downloads = self._downloads_dir()
        try:
            downloads.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If Downloads is not writable/creatable, fail gracefully.
            return None, None

        txt_path = downloads / f"{base}.txt"
        png_path = downloads / f"{base}.png"

        ok_txt = self._save_text_file(text, txt_path)
        ok_png = self._save_png_from_text(text, png_path)

        return (txt_path if ok_txt else None), (png_path if ok_png else None)

    def _copy_to_clipboard(self, text: str) -> bool:
        text = text or ""

        system = platform.system().lower()

        try:
            if system == "darwin":
                p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                p.communicate(input=text.encode("utf-8"))
                return p.returncode == 0
            elif system == "windows":
                p = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
                p.communicate(input=text.encode("utf-8"))
                return p.returncode == 0
            p = subprocess.Popen(
                ["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE
            )
            p.communicate(input=text.encode("utf-8"))
            if p.returncode == 0:
                return True
            p = subprocess.Popen(
                ["xsel", "--clipboard", "--input"], stdin=subprocess.PIPE
            )
            p.communicate(input=text.encode("utf-8"))
            return p.returncode == 0
        except Exception:
            return False

    def _handle_copy_request(self) -> None:
        req = getattr(ctrl_state, "copy_request", None)
        if not req:
            return

        if req == "ascii":
            payload = self._last_ascii_text
        elif req == "face":
            payload = self._last_face_text
        elif req == "hand":
            payload = self._last_hand_text
        else:
            payload = self._last_ascii_text

        _ = self._copy_to_clipboard(payload)

        _ = self._save_yanked_files(payload)

        ctrl_state.copy_request = None


def main():
    app = TxtureApp()
    app.run()


if __name__ == "__main__":
    main()
