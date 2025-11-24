import cv2
import numpy as np
import sys
import os

# 获取脚本所在目录和ml_models目录
script_dir = os.path.dirname(os.path.abspath(__file__))
ml_models_dir = os.path.dirname(script_dir)

sys.path.insert(0, ml_models_dir)

from detection.face_detector import FaceDetector
from detection.hand_detector import HandDetector
from gestures import GestureRecognizer
from expressions import ExpressionRecognizer

# 初始化检测器
face_detector = FaceDetector()
hand_detector = HandDetector()

# 初始化识别器（使用正确的路径）
gesture_model_path = os.path.join(ml_models_dir, 'gestures', 'checkpoints', 'gesture_model.pkl')
expression_model_path = os.path.join(ml_models_dir, 'expressions', 'checkpoints', 'expression_model.pth')

gesture_recognizer = GestureRecognizer(gesture_model_path)
expression_recognizer = ExpressionRecognizer(expression_model_path)

cap = cv2.VideoCapture(0)

# 裁剪图像的显示大小
CROP_SIZE = 200

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # 创建用于显示裁剪结果的画布
    face_crop = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    hand_crop = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    
    face_label = "No Face"
    hand_label = "No Hand"

    # Face Detection and Recognition
    faces = face_detector.detect(frame)
    if len(faces) > 0:
        x1, y1, x2, y2 = faces[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 确保坐标在图像范围内
        x1_c, y1_c = max(0, x1), max(0, y1)
        x2_c, y2_c = min(frame.shape[1], x2), min(frame.shape[0], y2)
        face_roi = frame[y1_c:y2_c, x1_c:x2_c]
        
        if face_roi.size > 0:
            # 裁剪显示
            face_crop = cv2.resize(face_roi, (CROP_SIZE, CROP_SIZE))
            
            # 表情识别
            try:
                expression, confidence = expression_recognizer.recognize(face_roi)
                if confidence > 0.5:
                    face_label = f"{expression} ({confidence:.2f})"
                else:
                    face_label = "Low Conf"
                # 在主画面上显示
                cv2.putText(frame, face_label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                face_label = "Error"

    # Gesture Detection and Recognition
    hands = hand_detector.detect(frame)
    if len(hands) > 0:
        hand_points = hands[0]
        
        # 绘制关键点
        for x, y, z, vis in hand_points:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
        
        # 计算边界框
        xs = [p[0] for p in hand_points]
        ys = [p[1] for p in hand_points]
        
        padding = 30
        x1_h = max(0, int(min(xs)) - padding)
        y1_h = max(0, int(min(ys)) - padding)
        x2_h = min(frame.shape[1], int(max(xs)) + padding)
        y2_h = min(frame.shape[0], int(max(ys)) + padding)
        
        # 绘制手部边界框
        cv2.rectangle(frame, (x1_h, y1_h), (x2_h, y2_h), (255, 0, 0), 2)
        
        hand_roi = frame[y1_h:y2_h, x1_h:x2_h]
        if hand_roi.size > 0:
            # 裁剪显示
            hand_crop = cv2.resize(hand_roi, (CROP_SIZE, CROP_SIZE))
            
            # 手势识别（归一化坐标）
            try:
                h, w = frame.shape[:2]
                normalized = [(x/w, y/h, z, vis) for x, y, z, vis in hand_points]
                gesture, confidence = gesture_recognizer.recognize(normalized)
                if confidence > 0.5:
                    hand_label = f"{gesture} ({confidence:.2f})"
                else:
                    hand_label = "Low Conf"
                # 在主画面上显示
                cv2.putText(frame, hand_label, (x1_h, y1_h - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            except Exception as e:
                hand_label = "Error"

    # 在裁剪图像上添加标签
    cv2.putText(face_crop, face_label, (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(hand_crop, hand_label, (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # 水平拼接两个裁剪画面
    crops_combined = np.hstack([face_crop, hand_crop])
    
    # 显示两个窗口
    cv2.imshow("Detection & Recognition", frame)
    cv2.imshow("Crops", crops_combined)
    
    # 调整窗口位置
    cv2.moveWindow("Detection & Recognition", 0, 100)
    cv2.moveWindow("Crops", frame.shape[1] + 10, 100)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()