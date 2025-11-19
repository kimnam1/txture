import cv2
import numpy as np


def frame_to_ascii(
    frame_bgr,
    lut: list[str],
    cols: int,
    char_aspect: float = 2.0,
    colorize: bool = False,
    saturation_gain: float = 1.0,
) -> list[str]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    rows = max(1, int(cols * h / w / char_aspect))

    small_gray = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)

    mapped = np.vectorize(lambda v: lut[int(v)])(small_gray)

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
