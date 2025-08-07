from ultralytics import YOLO
import cv2

hoop_model = YOLO("MLmodels/hoop_model.pt") # loading the hoop detection model
ball_model = YOLO("MLmodels/best.pt") # loading the ball detection model
hoop_model2 = YOLO("MLmodels/hoop_model_2_YOLO11.pt")
ball_hoop_model = YOLO("MLmodels/BallHoop_Model2.pt") # WORKING BALL + HOOP MODEL 
# start webcam at 0 (default camera)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # make detections
    hoop_results = hoop_model2(frame)[0]
    ball_results = ball_model(frame)[0]
    hoop_ball_results = ball_hoop_model(frame)[0]
    annotated_frame = hoop_ball_results.plot()
    # annotated_frame = ball_results.plot(annotated_frame, conf=False)
    # display the frame with detections
    cv2.imshow("Ball + Hoop Detection", annotated_frame)

    # exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()