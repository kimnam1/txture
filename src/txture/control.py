from dataclasses import dataclass
import cv2
import numpy as np

from txture.config import (
    CONTROLLER_WINDOW_SIZE,
    CONTROLLER_WINDOW_POS,
    KEY_HELP_DICT,
)


@dataclass
class ControllerState:
    charset: str = "ascii_punctuation_only"
    outline: bool = True
    color: bool = False
    saturation_gain: float = 1.0
    running: bool = True
    brightness_threshold: int = 200
    gamma: float = 1.0
    mode: str = "LIVE"  # 'LIVE/TONE/VISUAL/SATURATION/GAMMA/BRIGHTNESS'


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
    if key == 27:  # ESC
        state.running = False
        return
    if state.mode == "LIVE":
        handle_live_key(state, key)
    elif state.mode == "TONE":
        handle_tone_key(state, key)
    elif state.mode == "VISUAL":
        handle_visual_key(state, key)
    elif state.mode in ("SATURATION", "GAMMA", "BRIGHTNESS"):
        handle_param_key(state, key)


def handle_live_key(state: ControllerState, key: int) -> None:
    if key == ord("v"):
        state.mode = "VISUAL"
    elif key == ord("t"):
        state.mode = "TONE"


def handle_visual_key(state: ControllerState, key: int) -> None:
    if key == ord("p"):
        state.charset = "ascii_punctuation_only"
    elif key == ord("l"):
        state.charset = "ascii_letters_only"
    elif key == ord("d"):
        state.charset = "ascii_digits_only"
    elif key == ord("a"):
        state.charset = "ascii_all"
    elif key == ord("c"):
        state.color = not state.color
    elif key == ord("o"):
        state.outline = not state.outline
    elif key == ord("."):
        state.charset = "ascii_dots_only"
    elif key in (8, 127):  # backspace
        state.mode = "LIVE"


def handle_tone_key(state: ControllerState, key: int) -> None:
    if key == ord("s"):
        state.mode = "SATURATION"
    elif key == ord("g"):
        state.mode = "GAMMA"
    elif key == ord("v"):
        state.mode = "VALUE"
    elif key == ord("b"):
        state.mode = "BRIGHTNESS"
    elif key in (8, 127):  # backspace
        state.mode = "LIVE"


def handle_param_key(state: ControllerState, key: int) -> None:
    if key in (81, 2, ord("h"), ord("-")):  # left arrow
        adjust_param(state, -0.1)
    elif key in (83, 3, ord("l"), ord("+")):  # right arrow
        adjust_param(state, 0.1)
    elif key in (8, 127):  # backspace
        state.mode = "TONE"


def adjust_param(state: ControllerState, delta: float) -> None:
    if state.mode == "SATURATION":
        state.saturation_gain = max(
            0.0, min(3.0, state.saturation_gain + delta)
        )
    elif state.mode == "GAMMA":
        state.gamma = max(0.1, min(3.0, state.gamma + delta))
    elif state.mode == "BRIGHTNESS":
        step = int(delta * 10)
        state.brightness_threshold = int(
            max(0, min(255, state.brightness_threshold + step))
        )


def format_mode_line(state: ControllerState) -> str:
    return f"\x1b[1m{state.mode}\x1b[0m "


def format_help_line(state: ControllerState) -> str:
    return KEY_HELP_DICT[state.mode if state.mode != "LIVE" else "LIVE"]


def format_info_line(state: ControllerState) -> str:
    outline_flag = "ON" if state.outline else "OFF"
    color_line = "ON" if state.color else "OFF"
    sat = f"{state.saturation_gain:.1f}"
    gamma = f"{state.gamma:.1f}"
    bright_thresh = f"{state.brightness_threshold}"
    return (
        f"outline: {outline_flag} | color: {color_line} | "
        f"saturation: {sat} | gamma: {gamma} | "
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
    title_field = f"{title:<7}"
    label_show = label if label else "-"
    label_field = f"({label_show:^7})"

    return f"{title_field} {label_field} : {bar} {pct:3d}%"
