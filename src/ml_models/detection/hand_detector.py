import mediapipe as mp
import cv2

class HandDetector:
    def __init__(self, max_num_hands=1, detection_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.detector = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence
        )

    def detect(self, frame):
        """
        返回：
        - hand_landmarks: 每个手的 21 个关键点列表
          每个关键点是 (x, y, z, visibility)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb)

        all_hands = []
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                landmarks = []
                h, w, _ = frame.shape

                for lm in hand.landmark:
                    landmarks.append((lm.x * w, lm.y * h, lm.z, lm.visibility))

                all_hands.append(landmarks)

        return all_hands
