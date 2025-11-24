import os
import cv2
import numpy as np
import mediapipe as mp
from model import GestureModel


def load_dataset(dataset_path):
    """Load the dataset and extract keypoint features"""
    mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
    
    X, y = [], []
    train_path = os.path.join(dataset_path, 'train')
    classes = sorted(os.listdir(train_path))
    
    print(f"Category: {classes}")
    
    for idx, cls in enumerate(classes):
        cls_path = os.path.join(train_path, cls)
        if not os.path.isdir(cls_path):
            continue
        
        count = 0
        for img_name in os.listdir(cls_path):
            img_path = os.path.join(cls_path, img_name)
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Initial Image
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = mp_hands.process(rgb)
            
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
                X.append(landmarks)
                y.append(idx)
                count += 1
            
            # Horizontal flipping of images (data augmentation)）
            flipped = cv2.flip(image, 1)
            rgb_flipped = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
            result_flipped = mp_hands.process(rgb_flipped)
            
            if result_flipped.multi_hand_landmarks:
                hand = result_flipped.multi_hand_landmarks[0]
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
                X.append(landmarks)
                y.append(idx)
                count += 1
        
        print(f"  {cls}: {count} Samples")
    
    mp_hands.close()
    return np.array(X), np.array(y), classes


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, default='checkpoints/gesture_model.pkl')
    args = parser.parse_args()
    
    # Load data
    print("Loading dataset...")
    X_raw, y, labels = load_dataset(args.dataset)
    
    # Extract features and train
    print(f"\Training model has ({len(X_raw)} samples)...")
    model = GestureModel()
    
    X = np.array([model.extract_features(lm) for lm in X_raw])
    print(f"Feature dimension: {X.shape[1]}")
    
    accuracy = model.train(X, y, labels)
    print(f"Training accuracy: {accuracy:.4f}")
    
    # Simple Validation
    from sklearn.model_selection import cross_val_score
    X_scaled = model.scaler.transform(X)
    cv_scores = cross_val_score(model.model, X_scaled, y, cv=5)
    print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    model.save(args.output)
    print(f"Model saved: {args.output}")


if __name__ == '__main__':
    main()