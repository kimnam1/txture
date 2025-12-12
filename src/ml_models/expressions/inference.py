import torch
import cv2
import numpy as np
from torchvision import transforms
from .model import ExpressionModel

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class ExpressionRecognizer:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.label_map = checkpoint['label_map']
        
        self.model = ExpressionModel(num_classes=checkpoint['num_classes'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Initialize MediaPipe facial landmark detection
        if MEDIAPIPE_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            self.use_alignment = True
        else:
            self.use_alignment = False
    
    def align_face(self, face_image):
        """Face alignment based on MediaPipe"""
        if not self.use_alignment:
            return face_image
        
        results = self.face_mesh.process(face_image)
        
        if not results.multi_face_landmarks:
            return face_image
        
        landmarks = results.multi_face_landmarks[0]
        h, w = face_image.shape[:2]
        
        # Extract eye key points (MediaPipe 468-landmark model)
        left_eye = landmarks.landmark[33]
        right_eye = landmarks.landmark[362]
        
        left_eye_center = np.array([left_eye.x * w, left_eye.y * h])
        right_eye_center = np.array([right_eye.x * w, right_eye.y * h])
        
        # Compute the angle between the eyes
        dy = right_eye_center[1] - left_eye_center[1]
        dx = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Compute the center point between the eyes
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)
        
        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        
        # Apply affine transformation
        aligned = cv2.warpAffine(face_image, M, (w, h))
        
        return aligned
    
    def _preprocess_face(self, face_image):
        """
        Shared preprocessing logic
        """
        # Preprocessing
        face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Face alignment
        face = self.align_face(face)
        
        face = cv2.resize(face, (224, 224))
        face = self.transform(face).unsqueeze(0).to(self.device)
        
        return face
    
    def recognize(self, face_image):
        """
        Face Expression Recognition (original method, kept for compatibility)
        
        Args:
            face_image: Cropped face image (BGR format, numpy array)
        
        Returns:
            label: Expression label
            confidence: Confidence score
        """
        face = self._preprocess_face(face_image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(face)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
        
        return self.label_map[pred.item()], conf.item()
    
    def recognize_all_emotions(self, face_image, threshold=0.3):
        """
        Return probabilities for all expressions and intelligently determine the final result
        
        Args:
            face_image: Cropped face image (BGR format, numpy array)
            threshold: Minimum confidence threshold for non-neutral expressions
        
        Returns:
            tuple: (final_emotion, final_confidence, all_emotions_dict)
            - final_emotion: Final predicted expression label
            - final_confidence: Confidence of the final expression
            - all_emotions_dict: Dictionary of probability distribution for all expressions
        """
        face = self._preprocess_face(face_image)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(face)
            probs = torch.softmax(outputs, dim=1).squeeze()  # [num_classes]
        
        # Build probability dictionary for all expressions
        all_emotions = {}
        non_neutral_emotions = {}
        
        for i, prob in enumerate(probs):
            emotion_name = self.label_map[i]
            emotion_prob = prob.item()
            all_emotions[emotion_name] = emotion_prob
            
            # Exclude neutral and collect other expressions
            if emotion_name.lower() != 'neutral':
                non_neutral_emotions[emotion_name] = emotion_prob
        
        # Intelligent decision logic
        if non_neutral_emotions:
            max_non_neutral = max(non_neutral_emotions.values())
            if max_non_neutral >= threshold:
                # There is a clear expression above the threshold
                best_emotion = max(non_neutral_emotions, key=non_neutral_emotions.get)
                return best_emotion, all_emotions[best_emotion], all_emotions
            else:
                # All non-neutral expressions are too weak, return neutral
                neutral_confidence = all_emotions.get('neutral', 0.0)
                return 'neutral', neutral_confidence, all_emotions
        else:
            # No neutral category detected (unlikely, but handled as a fallback)
            best_emotion = max(all_emotions, key=all_emotions.get)
            return best_emotion, all_emotions[best_emotion], all_emotions
    
    def get_top_k_emotions(self, face_image, k=3):
        """
        Return the top-k most likely expressions
        
        Args:
            face_image: Cropped face image (BGR format, numpy array)
            k: Number of top results to return
        
        Returns:
            list: [(emotion, confidence), ...] sorted by confidence in descending order
        """
        face = self._preprocess_face(face_image)
        
        with torch.no_grad():
            outputs = self.model(face)
            probs = torch.softmax(outputs, dim=1).squeeze()
        
        # Get top-k highest probabilities
        top_probs, top_indices = probs.topk(k)
        
        results = []
        for i in range(k):
            emotion = self.label_map[top_indices[i].item()]
            confidence = top_probs[i].item()
            results.append((emotion, confidence))
        
        return results
