# Hand Gesture & Facial Expression Recognition API

Simple interface documentation for gesture and expression recognition modules.

## 🚀 Installation

```bash
pip install opencv-python mediapipe torch torchvision scikit-learn numpy
```

---

## 📌 Hand Gesture Recognition

### Interface 1: HandDetector (Detection)

```python
from gesture.hand_detector import HandDetector
import cv2

detector = HandDetector(max_num_hands=1, detection_confidence=0.5)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hands = detector.detect(frame)  # Returns list of hand landmarks
    # hands: [[(x1, y1, z1, vis1), (x2, y2, z2, vis2), ..., (x21, y21, z21, vis21)], ...]
    
    cv2.imshow("Hand Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Method**: `detect(frame)` → List of hand landmarks (21 keypoints per hand)

---

### Interface 2: GestureRecognizer (Classification)

```python
from gesture.inference import GestureRecognizer
from gesture.hand_detector import HandDetector
import cv2

hand_detector = HandDetector()
gesture_recognizer = GestureRecognizer(model_path='checkpoints/gesture_model.pkl')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hands = hand_detector.detect(frame)
    
    for hand_landmarks in hands:
        gesture_label, confidence = gesture_recognizer.recognize(hand_landmarks)
        print(f"Gesture: {gesture_label}, Confidence: {confidence:.4f}")
        
        if gesture_label == 'C':
            print("Switch to Chinese characters")
        elif gesture_label == 'K':
            print("Switch to Korean characters")
    
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Method**: `recognize(hand_landmarks)` → (gesture_label: str, confidence: float)

---

## 👤 Facial Expression Recognition

### Interface 3: FaceDetector (Detection)

```python
from gesture.face_detector import FaceDetector
import cv2

detector = FaceDetector(min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = detector.detect(frame)  # Returns list of face bounding boxes
    # faces: [(x1, y1, x2, y2), (x1, y1, x2, y2), ...]
    
    for (x1, y1, x2, y2) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Method**: `detect(frame)` → List of bounding boxes [(x1, y1, x2, y2), ...]

---

### Interface 4: ExpressionRecognizer (Classification)

```python
from gesture.face_detector import FaceDetector
from expression.inference import ExpressionRecognizer
import cv2

face_detector = FaceDetector()
expression_recognizer = ExpressionRecognizer(model_path='checkpoints/expression_model.pth')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    faces = face_detector.detect(frame)
    
    for (x1, y1, x2, y2) in faces:
        face_crop = frame[y1:y2, x1:x2]
        expression_label, confidence = expression_recognizer.recognize(face_crop)
        print(f"Expression: {expression_label}, Confidence: {confidence:.4f}")
        
        # Use confidence to control ASCII art saturation
        saturation = int(confidence * 255)
        print(f"ASCII saturation level: {saturation}")
    
    cv2.imshow("Expression Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Method**: `recognize(face_image)` → (expression_label: str, confidence: float)

---

## 🔗 Quick Integration Example

```python
from gesture.hand_detector import HandDetector
from gesture.inference import GestureRecognizer
from gesture.face_detector import FaceDetector
from expression.inference import ExpressionRecognizer
import cv2

# Initialize all detectors and recognizers
hand_detector = HandDetector()
gesture_recognizer = GestureRecognizer('checkpoints/gesture_model.pkl')
face_detector = FaceDetector()
expression_recognizer = ExpressionRecognizer('checkpoints/expression_model.pth')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Gesture Recognition
    hands = hand_detector.detect(frame)
    gesture_info = None
    for hand_landmarks in hands:
        gesture_label, confidence = gesture_recognizer.recognize(hand_landmarks)
        gesture_info = (gesture_label, confidence)
    
    # Expression Recognition
    faces = face_detector.detect(frame)
    expression_info = None
    for (x1, y1, x2, y2) in faces:
        face_crop = frame[y1:y2, x1:x2]
        expression_label, confidence = expression_recognizer.recognize(face_crop)
        expression_info = (expression_label, confidence)
    
    # Pass to ASCII art module
    if gesture_info and expression_info:
        gesture_label, gesture_conf = gesture_info
        expression_label, expression_conf = expression_info
        
        # Your ASCII art generation logic here
        # ascii_art_generator.update(frame, gesture_label, expression_conf)
    
    cv2.imshow("Combined Recognition", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📊 Output Summary

| Interface | Input | Output |
|-----------|-------|--------|
| `HandDetector.detect()` | Video frame | List of 21 keypoints per hand |
| `GestureRecognizer.recognize()` | Hand landmarks | Gesture label + confidence |
| `FaceDetector.detect()` | Video frame | List of face bounding boxes |
| `ExpressionRecognizer.recognize()` | Face image crop | Expression label + confidence |

---

## 📁 Model Files

Place pre-trained models in `checkpoints/`:
- `gesture_model.pkl` - Trained gesture classifier
- `expression_model.pth` - Trained expression classifier

To train models, see the respective `train.py` scripts in `gesture/` and `expression/` directories.