from ultralytics import YOLO
import cv2

model = YOLO("best.pt")  # Load your model
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=352, stream=True)  # Lower resolution for faster FPS

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{model.names[cls]} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("YOLOv8n Live", frame)
    if cv2.waitKey(1) == ord('q'):
        break
   

cap.release()
cv2.destroyAllWindows()




