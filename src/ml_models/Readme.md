# ML Models

This directory contains all **machine learning models and related tools** used in this project.  
The models are mainly responsible for **facial expression recognition** and **hand gesture recognition**, and are designed to be integrated into a real-time, terminal-based ASCII interaction system.

The overall design follows a **modular and pipeline-oriented structure**, allowing each model to be trained, evaluated, and deployed independently.

---

## 📁 Directory Structure

ml_models/
├── detection/          # Face and hand detection (MediaPipe + OpenCV)
├── expressions/        # Facial expression recognition
│   ├── model.py        # Expression recognition network (EfficientNet)
│   ├── train.py        # Training script
│   ├── inference.py    # Inference interface for runtime usage
│   ├── preprocess_align_enhanced.py  # Face detection and alignment preprocessing
│   └── checkpoints/    # Trained model weights
├── gestures/           # Hand gesture recognition
│   ├── model.py        # Feature extraction + MLP classifier
│   ├── train.py        # Training script
│   ├── inference.py    # Inference interface
│   └── checkpoints/    # Trained model weights
├── requirements.txt    # Python dependencies
└── README.md

---

## 🔍 Module Overview

### 1️⃣ Facial Expression Recognition (expressions/)

- **Task**: Classify facial expressions from detected face regions.
- **Model Architecture**:
  - EfficientNet-B2 backbone (via timm)
  - Implemented with PyTorch
- **Key Features**:
  - Expression classification with confidence scores
  - Full probability distribution over all expression classes
  - Threshold-based intelligent emotion decision to suppress weak expressions
  - Optional face alignment using MediaPipe facial landmarks

Main files:
- model.py: Expression recognition model definition
- train.py: Model training and validation
- inference.py: Unified inference interface for real-time systems
- preprocess_align_enhanced.py: Dataset preprocessing with face detection and alignment

---

### 2️⃣ Hand Gesture Recognition (gestures/)

- **Task**: Static hand gesture classification.
- **Model Architecture**:
  - MediaPipe 21-point hand landmarks
  - Geometric feature engineering (relative coordinates + fingertip distances)
  - MLP classifier implemented with scikit-learn
- **Characteristics**:
  - Lightweight and fast inference
  - Suitable for real-time interaction
  - Easy to extend with new gesture classes

Main files:
- model.py: Feature extraction and MLP-based classifier
- train.py: Training script with simple data augmentation
- inference.py: Runtime gesture recognition interface

---

## 🚀 Quick Start

### Install Dependencies

pip install -r requirements.txt

---

### Train Facial Expression Model

python expressions/train.py \
  --dataset path/to/expression_dataset \
  --output expressions/checkpoints/expression_model.pth

Example dataset structure:

dataset/
├── train/
│   ├── happy/
│   ├── sad/
│   └── neutral/
└── test/
    ├── happy/
    ├── sad/
    └── neutral/

---

### Train Gesture Recognition Model

python gestures/train.py \
  --dataset path/to/gesture_dataset \
  --output gestures/checkpoints/gesture_model.pkl

Example dataset structure:

dataset/
└── train/
    ├── A/
    ├── B/
    └── C/

---

## 🧠 Inference Examples

### Facial Expression Recognition

from expressions.inference import ExpressionRecognizer

recognizer = ExpressionRecognizer(
    "expressions/checkpoints/expression_model.pth"
)

label, confidence, all_probs = recognizer.recognize_all_emotions(face_image)

---

### Hand Gesture Recognition

from gestures.inference import GestureRecognizer

recognizer = GestureRecognizer(
    "gestures/checkpoints/gesture_model.pkl"
)

label, confidence = recognizer.recognize(hand_landmarks)

---

## 🔗 Integration with the Main System

- This directory only handles **perception and recognition**
- It does not include:
  - ASCII rendering logic
  - UI or terminal interaction logic
- All models expose clean inference interfaces and are called by the upper-level system to:
  - Map facial expressions to ASCII rendering effects
  - Use hand gestures as control commands

---

## 📝 Notes

- All models are designed to run **in real time on CPU**
- The implementation prioritizes:
  - Interpretability
  - Modularity
  - Educational and experimental use

---
