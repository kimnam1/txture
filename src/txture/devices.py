import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


import cv2
import numpy as np

from .detect_os import backend_candidates, detect_os
from .spinner import Spinner


@dataclass
class CameraInfo:
    index: int
    backend: str
    score: float  # average gray mean value


def open_with_backend(index: int, backend: str) -> cv2.VideoCapture:
    if backend == "cv2":
        return cv2.VideoCapture(index)
    elif backend == "avfoundation":
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    elif backend == "dshow":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif backend == "v4l2":
        return cv2.VideoCapture(index, cv2.CAP_V4L2)
    else:
        return cv2.VideoCapture(index)


def probe_device(
    index: int,
    backends: List[str],
    frames: int = 10,
    wait_sec: float = 0.05,
    min_mean: float = 10.0,
) -> Optional[CameraInfo]:
    best: Optional[CameraInfo] = None
    for backend in backends:
        cap = open_with_backend(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        vals = []
        for _ in range(frames):
            ok, frame = cap.read()
            if not ok:
                time.sleep(wait_sec)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vals.append(float(gray.mean()))
            time.sleep(wait_sec)

        cap.release()
        if not vals:
            continue

        avg = float(np.mean(vals))
        if avg < min_mean:
            continue

        best = CameraInfo(index=index, backend=backend, score=avg)
        break
    return best


def auto_scan_devices(
    max_devices: int = 5,
) -> Tuple[str, List[str], List[CameraInfo]]:
    spinner = Spinner(message="Scanning OS...")
    spinner.start()
    os_name = detect_os()
    spinner.stop()
    backends = backend_candidates(os_name)
    spinner = Spinner(message="Scanning camera devices...")
    spinner.start()

    found: List[CameraInfo] = []

    for idx in range(max_devices):
        info = probe_device(index=idx, backends=backends)
        if info is not None:
            found.append(info)
            break  # stop after first found device
    spinner.stop()
    return os_name, backends, found


def auto_pick_camera(max_devices: int = 5) -> Optional[CameraInfo]:
    os_name, backends, devices = auto_scan_devices(max_devices=max_devices)
    if not devices:
        return None
    return devices[0]  # pick the first detected camera


def open_auto_camera(
    max_devices: int = 5,
) -> Tuple[cv2.VideoCapture, CameraInfo]:
    info = auto_pick_camera(max_devices=max_devices)
    if info is None:
        raise RuntimeError("No camera device found")
    cap = open_with_backend(info.index, info.backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index={info.index} backend={info.backend}"
        )
    return cap, info
