from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # 'n' is the nano version (lightweight)

# Define blind spot zones (left & right side of the vehicle)
def define_blind_spots(frame):
    height, width, _ = frame.shape

    # Define blind spot regions (adjust coordinates as per your video)
    left_blind_spot = [(50, height // 2), (width // 3, height)]
    right_blind_spot = [(2 * width // 3, height // 2), (width - 50, height)]

    # Draw zones
    cv2.rectangle(frame, left_blind_spot[0], left_blind_spot[1], (0, 255, 0), 2)
    cv2.rectangle(frame, right_blind_spot[0], right_blind_spot[1], (0, 255, 0), 2)

    return left_blind_spot, right_blind_spot


# Open video feed (Ensure correct path format)
video_path = r"C:\Users\kakum\OneDrive\Documents\murthilab\level_0\BSW\vedio1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file: {video_path}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Detect objects using YOLOv8
    results = model(frame)

    # Define blind spot regions
    left_zone, right_zone = define_blind_spots(frame)

    # Loop through detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class label

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Class {cls} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Check if vehicle is in left or right blind spot
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            if cls in [2, 3, 5, 7]:  # Vehicle classes (car, truck, bus, motorcycle)
                if left_zone[0][0] < center_x < left_zone[1][0] and left_zone[0][1] < center_y < left_zone[1][1]:
                    cv2.putText(frame, "⚠ Left Blind Spot!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                if right_zone[0][0] < center_x < right_zone[1][0] and right_zone[0][1] < center_y < right_zone[1][1]:
                    cv2.putText(frame, "⚠ Right Blind Spot!", (frame.shape[1] - 300, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show output
    cv2.imshow("Blind Spot Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
