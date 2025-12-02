from dataclasses import dataclass
import numpy as np
import cv2
from txture.config import (
    CANNY_LOW,
    CANNY_HIGH,
    BLUR_KERNEL_SIZE,
    SOBEL_KERNEL_SIZE,
    MORPH_KERNEL_SIZE,
)


from ml_models.detection.hand_detector import HandDetector
from ml_models.detection.face_detector import FaceDetector


@dataclass
class FrameFeatures:
    orig: np.ndarray
    hsv: np.ndarray
    gray: np.ndarray
    det_vis: np.ndarray  # detection visualization
    processed: np.ndarray
    edge_dir: np.ndarray  # edge direction
    hand_mask: np.ndarray | None
    face_mask: np.ndarray | None


_hand_detector = HandDetector(max_num_hands=1, detection_confidence=0.5)
_face_detector = FaceDetector(min_detection_confidence=0.5)


def process_frame(frame, outline_mode: bool) -> FrameFeatures:
    orig = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, w = gray.shape

    hands = _hand_detector.detect(orig)
    faces = _face_detector.detect(orig)
    hand_mask = np.zeros((h, w), dtype=np.uint8)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    det_vis = orig.copy()

    ## add hand detection visualization

    if outline_mode:
        blur = cv2.GaussianBlur(gray, BLUR_KERNEL_SIZE, 0)
        kernel = np.ones(MORPH_KERNEL_SIZE, dtype=np.uint8)
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=2)

        gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
        gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=SOBEL_KERNEL_SIZE)

        angle = np.rad2deg(np.arctan2(gy, gx))
        angle[angle < 0] += 180
        edge_dir = np.full((h, w), -1, dtype=np.int8)

        bins = np.array([0.0, 45.0, 90.0, 135.0], dtype=np.float32)

        mask = edges > 0
        if np.any(mask):
            ang_valid = angle[mask]

            diff = np.abs(ang_valid[:, None] - bins[None, :])

            idx = np.argmin(diff, axis=1).astype(np.int8)

            edge_dir[mask] = idx

        processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else:
        processed = orig.copy()
        edge_dir = np.full(gray.shape, -1, dtype=np.int8)

    return FrameFeatures(
        orig=orig,
        gray=gray,
        det_vis=det_vis,
        processed=processed,
        edge_dir=edge_dir,
        hand_mask=hand_mask,
        face_mask=face_mask,
        hsv=hsv,
    )
