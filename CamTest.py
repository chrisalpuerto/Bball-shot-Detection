from ultralytics import YOLO
import cv2

model = YOLO("hoop_model.pt")

# start webcam at 0 (default camera)
cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # make detections
    results = model(frame, conf=0.25)[0]
    annotated_frame = results.plot()

    # display the frame with detections
    cv2.imshow("Webcam Feed", annotated_frame)

    # exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()