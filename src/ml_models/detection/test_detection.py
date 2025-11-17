import cv2
from face_detector import FaceDetector
from hand_detector import HandDetector

face_detector = FaceDetector()
hand_detector = HandDetector()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 人脸检测
    faces = face_detector.detect(frame)
    for (x1, y1, x2, y2) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 手势检测
    hands = hand_detector.detect(frame)
    for hand in hands:
        for x, y, z, vis in hand:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 0, 0), -1)

    cv2.imshow("Detection Test", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
