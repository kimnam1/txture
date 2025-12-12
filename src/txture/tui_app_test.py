import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['GLOG_minloglevel'] = '3'

os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import warnings
warnings.filterwarnings('ignore')

from textual.app import App, ComposeResult
from textual.widgets import Static
from textual.events import Key
from textual.containers import Container

from pathlib import Path
from typing import Union, Optional
import numpy as np
import cv2

from txture.devices import open_auto_camera
from txture.pipeline import process_frame
from txture.loaders import load_lut
from txture.ascii_render import frame_to_ascii
from txture.config import DEFAULT_BRIGHTNESS_THRESHOLD, DEFAULT_FPS
from txture.control import (
    state as ctrl_state,
    handle_key,
    format_help_line,
    format_conf_line,
)
from rich.text import Text


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

/* Main area: left(ASCII) + right(face + hand) */
#main_area {
    layout:horizontal;
    height: 1fr;
}

/* ASCII area */
#ascii {
    width: 3fr;
    border: heavy green;
}

/* right vertical stack (face, hand) */
#side-pane {
    layout: vertical;
    width: 1fr;
}

/* Upper face area */
#face {
    height: 1fr;
    border: heavy blue;
}

/* Lower hand area */
#hand{
    height: 1fr;
    border: heavy yellow;
}

/* Bottom status area */
#status {
    height: 5;
    border: heavy white;
}
"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cap = None
        self.lut = None
        self.sat_gain = 1.0
        self.brightness_threshold = DEFAULT_BRIGHTNESS_THRESHOLD

        self.face_detector = None
        self.expression_recognizer = None
        self.face_label = None
        self.face_conf = 0.0

        self.hand_detector = None
        self.gesture_recognizer = None
        self.gesture_filter = None
        self.gesture_label = None
        self.gesture_conf = 0.0

    def compose(self) -> ComposeResult:
        yield Container(
            Container(
                Static("ASCII AREA", id="ascii", markup=False),
                Container(
                    Static("FACE AREA", id="face", markup=False),
                    Static("HAND AREA", id="hand", markup=False),
                    id="side-pane",
                ),
                id="main_area",
            ),
            Static("STATS / HELP", id="status", markup=False),
            id="root",
        )

    async def on_mount(self) -> None:
        self.ascii_view: Static = self.query_one("#ascii", Static)
        self.face_view: Static = self.query_one("#face", Static)
        self.hand_view: Static = self.query_one("#hand", Static)
        self.status_view: Static = self.query_one("#status", Static)

        files = discover_metric_files()
        metric_key = "ascii_punctuation_only"

        if metric_key not in files:
            self.status_view.update(f'metric "{metric_key}" not found')
            return

        self.lut = load_lut(files[metric_key])

        self.cap, cam_info = open_auto_camera(max_devices=3)

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
        self.gesture_filter = EventFilter(min_conf=0.8, stable_frames=5)

        # 启用表情识别器
        self.expression_recognizer = ExpressionRecognizer(
            model_path=str(FACE_CKPT)
        )
        self.face_filter = EventFilter(min_conf=0.8, stable_frames=5)
        self.face_label = "FACE"
        self.face_conf = 1.0

        self.tick_timer = self.set_interval(1 / DEFAULT_FPS, self._on_tick)

        self.status_view.update(
            f"Camera index = {cam_info.index} backend = {cam_info.backend} score = {cam_info.score: .2f}"
        )

    async def on_key(self, event: Key) -> None:
        # end key
        if event.key in ("escape", "ctrl+q"):
            await self.action_quit()
            return

        # key code mapping
        keycode: int | None = None

        if len(event.key) == 1:
            keycode = ord(event.key)

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

            self.face_filter.update(None, 0.0)
            return None

        faces = self.face_detector.detect(frame)
        if not faces:
            self.face_label = None
            self.face_conf = 0.0

            self.face_filter.update(None, 0.0)
            return None

        x1, y1, x2, y2 = faces[0]
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            self.face_label = None
            self.face_conf = 0.0

            self.face_filter.update(None, 0.0)
            return None
        label = None
        conf = 0.0

        # Using a New Intelligent Emotion Recognition Method
        if self.expression_recognizer is not None:
            try:
                # Use intelligent judgment method with threshold set to 0.3
                label, conf, all_probs = self.expression_recognizer.recognize_all_emotions(face_crop, threshold=0.3)
            except Exception:
                label, conf = None, 0.0

        self.face_label = label
        self.face_conf = conf
        fired = self.face_filter.update(label, conf)

        if fired is not None:
            self._on_face_event(fired, conf)
        return face_crop if face_crop.size > 0 else None

    def _on_face_event(self, label: str, conf: float) -> None:
        pass

    def _get_hand_crop(self, frame) -> Optional[np.ndarray]:
        """Get cropped hand region with keypoints drawn"""
        if self.hand_detector is None:
            return None
        hands = self.hand_detector.detect(frame)
        if not hands:
            return None

        hand_points = hands[0]

        # Calculate bounding box
        xs = [p[0] for p in hand_points]
        ys = [p[1] for p in hand_points]

        padding = 30
        h, w = frame.shape[:2]
        x1 = max(0, int(min(xs)) - padding)
        y1 = max(0, int(min(ys)) - padding)
        x2 = min(w, int(max(xs)) + padding)
        y2 = min(h, int(max(ys)) + padding)

        hand_crop = frame[y1:y2, x1:x2].copy()
        if hand_crop.size == 0:
            return None

        # Draw keypoints
        for x, y, z, vis in hand_points:
            rel_x, rel_y = int(x) - x1, int(y) - y1
            if (
                0 <= rel_x < hand_crop.shape[1]
                and 0 <= rel_y < hand_crop.shape[0]
            ):
                cv2.circle(hand_crop, (rel_x, rel_y), 3, (255, 0, 0), -1)

        return hand_crop

    def _render_ascii_raw(
        self, frame, cols: int, rows: int, *, color, outline
    ):
        if self.lut is None:
            return "No LUT loaded."

        features = process_frame(
            frame,
            outline_mode=outline,
        )

        edge_arg = features.edge_dir if outline else None

        lines, colors = frame_to_ascii(
            features.processed,
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
                _ = self.gesture_filter.update(label, conf)
            else:
                _ = None

            self.gesture_label = label
            self.gesture_conf = conf
        else:
            self.gesture_label = None
            self.gesture_conf = 0.0

        features = process_frame(
            frame,
            outline_mode=outline,
        )

        edge_arg = features.edge_dir if outline else None

        lines, colors = frame_to_ascii(
            features.processed,
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

    def _render_status(self) -> str:
        outline_flag = "ON" if ctrl_state.outline else "OFF"
        color_flag = "ON" if ctrl_state.color else "OFF"
        top_line = f"MODE: {ctrl_state.mode} | outline: {outline_flag} | color: {color_flag}"
        general_help_line = "(h) -> HELP | (esc) -> QUIT"
        help_line = format_help_line(ctrl_state)

        return top_line + "\n" + general_help_line + "\n" + help_line

    def _calc_ascii_size(self) -> tuple[int, int]:
        screen_width = self.size.width
        width = int(screen_width * 0.75)  # ASCII area is 3/4 of width

        screen_height = self.size.height

        ascii_cols = max(20, width - 2)
        ascii_rows = max(3, screen_height)

        return ascii_cols, ascii_rows

    async def _on_tick(self) -> None:
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                ascii_renderable = "Failed to read from camera."
            else:
                cols, target_rows = self._calc_ascii_size()

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
            face_crop = self._get_face_crop(frame)

            if face_crop is not None:
                fw = self.face_view.size.width
                fh = self.face_view.size.height
                if fw > 4 and fh > 4:
                    cols = fw - 2
                    rows = fh - 2
                    ascii_face = self._render_ascii_raw(
                        frame=face_crop,
                        cols=cols,
                        rows=rows,
                        color=True,
                        outline=False,
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

            hand_crop = self._get_hand_crop(frame)
            if hand_crop is not None:
                hw = self.hand_view.size.width
                hh = self.hand_view.size.height
                if hw > 4 and hh > 4:
                    cols = hw - 2
                    rows = hh - 2
                    ascii_hand = self._render_ascii_raw(
                        frame=hand_crop,
                        cols=cols,
                        rows=rows,
                        color=True,
                        outline=False,
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

        status_text = self._render_status()

        self.ascii_view.update(ascii_renderable)
        self.status_view.update(status_text)


def main():
    app = TxtureApp()
    app.run()


if __name__ == "__main__":
    main()