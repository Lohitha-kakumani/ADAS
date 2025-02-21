from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # 'n' = lightweight version

# Define the danger zone (e.g., an area in front of the vehicle)
def define_danger_zone(frame):
    height, width, _ = frame.shape
    danger_zone = [(width // 4, height // 2), (3 * width // 4, height)]
    cv2.rectangle(frame, danger_zone[0], danger_zone[1], (0, 0, 255), 2)  # Red rectangle
    return danger_zone

# Open video feed
cap = cv2.VideoCapture("C:\\Users\\kakum\\OneDrive\\Documents\\murthilab\\level_0\\PCW\\pedestrian.mp4")  # Use a pedestrian detection video

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Detect objects using YOLOv8
    results = model(frame)

    # Define pedestrian danger zone
    danger_zone = define_danger_zone(frame)

    # Loop through detections
    for result in results[0].boxes.data:  
        x1, y1, x2, y2, conf, cls = result  
        label = int(cls)  

        if label == 0:  # 0 = Pedestrian (Person class in COCO dataset)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Check if pedestrian is in danger zone
            if danger_zone[0][0] < center_x < danger_zone[1][0] and danger_zone[0][1] < center_y < danger_zone[1][1]:
                cv2.putText(frame, "⚠ Pedestrian Ahead!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Draw bounding box around detected pedestrian
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Show output
    cv2.imshow("Pedestrian Collision Warning", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
