import mediapipe as mp
import cv2

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(min_detection_confidence)

    def detect(self, frame):
        """
        Return:
        - List of face bounding boxes [(x1, y1, x2, y2), ...]
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)

        faces = []
        if result.detections:
            h, w, _ = frame.shape
            for detection in result.detections:
                box = detection.location_data.relative_bounding_box
                x1 = int(box.xmin * w)
                y1 = int(box.ymin * h)
                x2 = int((box.xmin + box.width) * w)
                y2 = int((box.ymin + box.height) * h)
                faces.append((x1, y1, x2, y2))

        return faces
