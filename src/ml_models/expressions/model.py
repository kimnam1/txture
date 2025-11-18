import torch
import torch.nn as nn
from torchvision import models


class ExpressionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.classifier[1] = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'num_classes': self.model.classifier[1].out_features
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.classifier[1] = nn.Linear(1280, checkpoint['num_classes'])
        self.load_state_dict(checkpoint['state_dict'])