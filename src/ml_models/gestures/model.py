import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pickle


class GestureModel:
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42,
            early_stopping=True
        )
        self.scaler = StandardScaler()
        self.label_map = {}
    
    def extract_features(self, landmarks):
        """
        Extract features from 21 key points
        landmarks: [(x, y, z), ...] Total of 21 points

        """
        landmarks = np.array(landmarks).reshape(-1, 3)
        
        # 以手腕为原点归一化
        wrist = landmarks[0]
        landmarks = landmarks - wrist
        
        # 按手掌大小缩放
        palm_size = np.linalg.norm(landmarks[9])
        if palm_size > 0:
            landmarks = landmarks / palm_size
        
        features = landmarks.flatten().tolist()
        
        # 添加指尖距离特征
        fingertips = [4, 8, 12, 16, 20]
        for i in range(len(fingertips)):
            for j in range(i+1, len(fingertips)):
                dist = np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[j]])
                features.append(dist)
        
        return np.array(features)
    
    def train(self, X, y, labels):
        """
        训练模型
        X: 特征数组
        y: 标签数组
        labels: 类别名称列表
        """
        self.label_map = {i: label for i, label in enumerate(labels)}
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        return self.model.score(X_scaled, y)
    
    def predict(self, landmarks):
        """
        预测手势
        返回: (标签, 置信度)
        """
        features = self.extract_features(landmarks)
        features_scaled = self.scaler.transform([features])
        
        pred = self.model.predict(features_scaled)[0]
        prob = self.model.predict_proba(features_scaled)[0]
        
        return self.label_map[pred], prob[pred]
    
    def save(self, path):
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_map': self.label_map
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_map = data['label_map']