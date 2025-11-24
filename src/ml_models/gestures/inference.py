from .model import GestureModel


class GestureRecognizer:
    def __init__(self, model_path):
        self.model = GestureModel()
        self.model.load(model_path)
    
    def recognize(self, hand_landmarks):
        """
        Gesture Recognition
        
        Args:
            hand_landmarks: Keypoints returned by HandDetector
                          [(x, y, z, visibility), ...] Total of 21 points
        
        Returns:
            label: Gesture label (e.g., ‘A’, ‘B’, ‘C’, ...)
            confidence: Confidence score
        """
        # Format conversion: Remove visibility
        landmarks = [(x, y, z) for x, y, z, _ in hand_landmarks]
        return self.model.predict(landmarks)