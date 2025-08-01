import cv2
from ultralytics import YOLO

# Load your models
ball_model = YOLO("MLmodels/best.pt")
hoop_model = YOLO("MLmodels/HoopDetectionModel.pt")

# Load video
video_path = "videoDataset/danielShooting.MOV"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_idx = 0

# Track ball across frames
ball_path = []  # [(frame_num, (x1, y1, x2, y2))]
rim_box = None

# Shot detection state
shots = []
in_shot = False
ball_prev_y = None
shot_candidate = []

def get_center_y(box):
    return (box[1] + box[3]) / 2

def boxes_overlap(b1, b2):
    x1_max = max(b1[0], b2[0])
    y1_max = max(b1[1], b2[1])
    x2_min = min(b1[2], b2[2])
    y2_min = min(b1[3], b2[3])
    return x1_max < x2_min and y1_max < y2_min

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    print(f"Time: {frame_idx / fps:.2f}s")
    # Run hoop detection once (assume static hoop)
    if rim_box is None:
        rim_result = hoop_model(frame, verbose=False)[0]
        for det in rim_result.boxes:
            rim_box = det.xyxy[0].tolist()
            break  # assume one rim
    if not rim_result.boxes:
        print(f"No hoop detected at frame {frame_idx / fps:.2f}s")

    # Run ball detection
    ball_result = ball_model(frame, verbose=False)[0]
    ball_box = None
    for det in ball_result.boxes:
        ball_box = det.xyxy[0].tolist()
        break  # assume one ball
    if not ball_result.boxes:
        print(f"No ball detected at frame {frame_idx / fps:.2f}s")
    if ball_box and rim_box:
        center_y = get_center_y(ball_box)

        # Track if shot might be starting (ball moving downward)
        if ball_prev_y is not None and center_y > ball_prev_y and not in_shot:
            in_shot = True
            shot_candidate = [(frame_idx, ball_box)]
        elif in_shot:
            shot_candidate.append((frame_idx, ball_box))

            # Ball overlaps hoop = potential made shot
            if boxes_overlap(ball_box, rim_box):
                overlap_frame = frame_idx

            # Check if ball goes below hoop (confirms make)
            if get_center_y(ball_box) > rim_box[3]:
                made = True
                for _, box in shot_candidate:
                    if not boxes_overlap(box, rim_box):
                        made = False
                        break
                shots.append(("MADE", overlap_frame / fps))
                in_shot = False
                shot_candidate = []
            elif len(shot_candidate) > int(fps * 1.5):  # more than ~1.5s no result
                shots.append(("MISSED", frame_idx / fps))
                in_shot = False
                shot_candidate = []

        ball_prev_y = center_y

    frame_idx += 1

cap.release()

# Output results
print("\n✅ Shot Summary:")
for i, (status, timestamp) in enumerate(shots, 1):
    print(f"Shot {i}: {status} at {timestamp:.2f}s")
