import torch
import torch.nn as nn
import timm


class ExpressionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = timm.create_model('efficientnet_b2', pretrained=True, num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)
    
    def save(self, path):
        torch.save({
            'state_dict': self.state_dict(),
            'num_classes': self.model.num_classes
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.num_classes = checkpoint['num_classes']
        self.load_state_dict(checkpoint['state_dict'])