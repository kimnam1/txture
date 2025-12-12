import cv2
import numpy as np

from txture.config import DEFAULT_BRIGHTNESS_THRESHOLD

blank_char = " "
DIR_CHARS = {-1: " ", 0: "─", 1: "╱", 2: "│", 3: "╲"}


def frame_to_ascii(
    frame_bgr,
    lut: list[str],
    cols: int,
    char_aspect: float = 2.0,
    colorize: bool = False,
    saturation_gain: float = 1.0,
    brightness_threshold: int = DEFAULT_BRIGHTNESS_THRESHOLD,
    edge_dir=None,
    mirror: bool = True,
) -> list[str]:
    if mirror:
        frame_bgr = cv2.flip(frame_bgr, 1)
        if edge_dir is not None:
            edge_dir = cv2.flip(edge_dir, 1)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rows = max(1, int(cols * h / w / char_aspect))

    small_gray = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)

    if edge_dir is not None:
        small_dir = cv2.resize(
            edge_dir, (cols, rows), interpolation=cv2.INTER_NEAREST
        )
        mapped = np.empty_like(small_gray, dtype=object)
        for y in range(rows):
            for x in range(cols):
                d = int(small_dir[y, x])
                mapped[y, x] = DIR_CHARS.get(d, blank_char)
    else:
        mapped = np.vectorize(lambda v: lut[int(v)])(small_gray)

    mask = small_gray < brightness_threshold
    mapped = np.where(mask, blank_char, mapped)

    lines = ["".join(row) for row in mapped]

    if not colorize:
        return lines, None

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    small_hsv = cv2.resize(hsv, (cols, rows), interpolation=cv2.INTER_AREA)
    h_ch, s_ch, v_ch = cv2.split(small_hsv)
    s_scaled = np.clip(
        s_ch.astype(np.float32) * saturation_gain, 0, 255
    ).astype(np.uint8)

    hsv_scaled = cv2.merge([h_ch, s_scaled, v_ch])
    small_bgr = cv2.cvtColor(hsv_scaled, cv2.COLOR_HSV2BGR)
    small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)

    color = small_rgb.tolist()

    return lines, color
