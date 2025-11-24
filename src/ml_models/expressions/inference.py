import torch
import cv2
import numpy as np
from torchvision import transforms
from .model import ExpressionModel


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
    
    def recognize(self, face_image):
        """
        Face Expression Recognition
        
        Args:
            face_image: Cropped face image (BGR format, numpy array)
        
        Returns:
            label: Expression label
            confidence: Confidence score
        """
        # Preprocessing
        face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = self.transform(face).unsqueeze(0).to(self.device)
        
        #  Deduction
        with torch.no_grad():
            outputs = self.model(face)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = probs.max(1)
        
        return self.label_map[pred.item()], conf.item()