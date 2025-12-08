import cv2
import sys
sys.path.append('..')

from face_detector import FaceDetector
from hand_detector import HandDetector
from gestures import GestureRecognizer
from expressions import ExpressionRecognizer


def main():
    face_detector = FaceDetector()
    hand_detector = HandDetector()
    gesture_recognizer = GestureRecognizer('../gestures/checkpoints/gesture_model.pkl')
    expression_recognizer = ExpressionRecognizer('../expressions/checkpoints/expression_model.pth')
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Face Detection and Expression Recognition
        faces = face_detector.detect(frame)
        expression_text = "None"
        
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Crop faces and recognize expressions
            face_img = frame[y1:y2, x1:x2]
            if face_img.size > 0:
                label, conf = expression_recognizer.recognize(face_img)
                if conf > 0.5:
                    expression_text = f"{label} ({conf:.2f})"
        
        # Gesture Detection and Recognition
        hands = hand_detector.detect(frame)
        gesture_text = "None"
        
        for hand in hands:
            for x, y, z, vis in hand:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)
            
            h, w = frame.shape[:2]
            normalized = [(x/w, y/h, z, vis) for x, y, z, vis in hand]
            label, conf = gesture_recognizer.recognize(normalized)
            
            if conf > 0.5:
                gesture_text = f"{label} ({conf:.2f})"
        
        # Display results
        cv2.putText(frame, f"Expression: {expression_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Detection Test", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()