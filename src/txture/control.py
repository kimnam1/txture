from dataclasses import dataclass
import cv2
import numpy as np

from txture.config import (
    CONTROLLER_WINDOW_SIZE,
    CONTROLLER_WINDOW_POS,
    KEY_HELP_DICT,
    DEFAULT_OUTLINE,
    DEFAULT_COLOR,
    FACE_MAP,
)


@dataclass
class ControllerState:
    charset: str = "ascii_punctuation_only"
    outline: bool = DEFAULT_OUTLINE
    color: bool = DEFAULT_COLOR
    # saturation_gain: float = 1.0
    running: bool = True
    brightness_threshold: int = 200
    # gamma: float = 1.0

    # Modes: NORMAL/OUTLINE/VISUAL/
    mode: str = "NORMAL"

    # Clipboard/copy stubs (no actual clipboard implementation here)
    copy_request: str | None = None  # 'any' | 'ascii' | 'face' | 'gesture'

    # Internal key-sequence state for y/yy/yf/yh
    _y_armed: bool = False

    # For HELP mode return
    _mode_before_help: str = "NORMAL"


state = ControllerState()


def run_controller(state: ControllerState) -> None:
    win = "controller"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.moveWindow(win, *CONTROLLER_WINDOW_POS)
    control_blank = np.zeros((1, 1, 3), dtype=np.uint8)
    cv2.resizeWindow(win, *CONTROLLER_WINDOW_SIZE)
    while state.running:
        cv2.imshow(win, control_blank)
        key = cv2.waitKey(50) & 0xFF
        if key != 255:
            handle_key(state, key)


def handle_key(state: ControllerState, key: int) -> None:
    # ESC always quits
    if key == 27:
        state.running = False
        return

    # Allow q as a universal "back" key (except where noted below)
    is_q = key == ord("q")
    is_backspace = key in (8, 127)

    # --- y prefix (global) ---
    # When armed, y/f/g trigger copy requests. Backspace/q cancels ONLY the prefix.
    if state._y_armed:
        if is_backspace or is_q:
            state._y_armed = False
            return
        if key == ord("y"):
            state.copy_request = "ascii"  # yy
            state._y_armed = False
            return
        if key == ord("f"):
            state.copy_request = "face"  # yf
            state._y_armed = False
            return
        if key == ord("g"):
            state.copy_request = "hand"  # yg (hand/gesture)
            state._y_armed = False
            return
        # For any other key, keep `_y_armed` True and continue handling
        # the key normally (so users can toggle modes while armed).

    # Arm y prefix
    if key == ord("y"):
        state._y_armed = True
        return

    # # --- HELP mode toggle ---
    # if key == ord("h"):
    #     if state.mode != "HELP":
    #         state._mode_before_help = state.mode
    #         state.mode = "HELP"
    #     # if already HELP, keep showing HELP (exit via backspace/q)
    #     return

    # # Exit HELP -> return to previous mode
    # if state.mode == "HELP":
    #     if is_backspace or is_q:
    #         state.mode = state._mode_before_help or "NORMAL"
    #     return

    # --- mode-specific navigation and actions ---

    if state.mode == "NORMAL":
        if key == ord("o"):
            state.outline = True
            state.mode = "OUTLINE"
            return
        if key == ord("v"):
            state.mode = "VISUAL"
            return
        if key == ord("t"):
            state.mode = "TONE"
            return
        if key == ord("c"):
            state.color = not state.color
            return
        return

    if state.mode == "OUTLINE":
        if key == ord("o"):
            state.outline = False
            state.mode = "NORMAL"
            return
        if is_backspace or is_q:
            state.outline = False
            state.mode = "NORMAL"
            return
        # No other edits allowed in OUTLINE mode
        return

    if state.mode == "VISUAL":
        if is_backspace or is_q:
            state.mode = "NORMAL"
            return
        if key == ord("l"):
            state.charset = "ascii_letters_only"
            return
        if key == ord("p"):
            state.charset = "ascii_punctuation_only"
            return
        if key == ord("d"):
            state.charset = "ascii_digits_only"
            return
        if key == ord(".") or key == ord("o"):
            state.charset = "ascii_dots_only"
            return
        return

    # if state.mode == "TONE":
    #     if is_backspace or is_q:
    #         state.mode = "NORMAL"
    #         return
    #     if key == ord("s"):
    #         state.mode = "SATURATION"
    #         return
    #     if key == ord("g"):
    #         state.mode = "GAMMA"
    #         return
    #     if key == ord("b"):
    #         state.mode = "BRIGHTNESS"
    #         return
    #     return

    # if state.mode in ("SATURATION", "GAMMA", "BRIGHTNESS"):
    #     # back to TONE
    #     if is_backspace or is_q:
    #         state.mode = "TONE"
    #         return

    #     # Param adjust mapping per spec:
    #     # '-' or RIGHT => UP
    #     # '+' or LEFT  => DOWN
    #     is_left = key in (81, 2, ord("h"))
    #     is_right = key in (83, 3, ord("l"))
    #     if key == ord("-") or is_right:
    #         adjust_param(state, +0.1)
    #         return
    #     if key == ord("+") or is_left:
    #         adjust_param(state, -0.1)
    #         return
    #     return


# def adjust_param(state: ControllerState, delta: float) -> None:
#     if state.mode == "SATURATION":
#         state.saturation_gain = max(
#             0.0, min(3.0, state.saturation_gain + delta)
#         )
#     elif state.mode == "GAMMA":
#         state.gamma = max(0.1, min(3.0, state.gamma + delta))
#     elif state.mode == "BRIGHTNESS":
#         step = int(delta * 10)
#         state.brightness_threshold = int(
#             max(0, min(255, state.brightness_threshold + step))
#         )


def format_mode_line(state: ControllerState) -> str:
    return f"\x1b[1m{state.mode}\x1b[0m "


def format_help_line(state: ControllerState) -> str:
    base = KEY_HELP_DICT.get(state.mode, "")

    if state._y_armed:
        yank = (
            "\nYANK: (y) ascii | (f) face | (g) gesture | (backspace/q) cancel"
        )
        if base:
            return base + " | " + yank
        return yank
    return base


def format_info_line(state: ControllerState) -> str:
    outline_flag = "ON" if state.outline else "OFF"
    color_line = "ON" if state.color else "OFF"
    # sat = f"{state.saturation_gain:.1f}"
    # gamma = f"{state.gamma:.1f}"
    bright_thresh = f"{state.brightness_threshold}"
    return (
        f"outline: {outline_flag} | color: {color_line} | "
        # f"saturation: {sat} | gamma: {gamma} | "
        f"brightness threshold: {bright_thresh}"
    )


def make_conf_bar(
    conf: float, length: int = 20, thresh: float = 0.8
) -> tuple[str, int]:
    p = max(0.0, min(1.0, conf))
    filled = int(p * length)
    bar = ["#"] * filled + ["."] * (length - filled)

    if 0.0 < thresh < 1.0:
        t_pos = int(thresh * length)
        if 0 <= t_pos < length:
            bar[t_pos] = "|"

    return "".join(bar), int(p * 100)


def format_conf_line(
    title: str,
    label: str | None,
    conf: float,
    length: int = 30,
    thresh: float = 0.8,
) -> str:
    bar, pct = make_conf_bar(conf, length=length, thresh=thresh)

    title_show = title if title else "-"

    label_width = 10
    label_fmt = label.center(label_width) if label else "-".center(label_width)

    if title == "FACE":
        label_emoji = FACE_MAP.get(label, "-") if label else "-"
        label_show = f"{label_emoji} ({label_fmt})" if label else "-"
    else:
        label_show = label if label else "-"

    first_line = f"{title_show}: {label_show}"

    pct_field = f"{pct:3d}%"
    second_line = f"( {pct_field:5s} ) {bar}"

    return first_line + "\n" + second_line
