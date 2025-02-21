from ultralytics import YOLO
import cv2

# Load the YOLOv8 model (pre-trained on COCO dataset)
model = YOLO("yolov8n.pt")  # 'n' is the nano version (lightweight)

# Open video feed
video_path = r"C:\Users\kakum\OneDrive\Documents\murthilab\level_0\BSW\vedio1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Define a minimum safe distance (you can tune this)
SAFE_DISTANCE = 100  # Adjust based on your scenario

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    # Detect objects using YOLOv8
    results = model(frame)

    # Loop through detections
    for box in results[0].boxes.data:  # Get bounding boxes
        x1, y1, x2, y2, conf, cls = box  # YOLO outputs
        label = int(cls)  # Class label

        # Only detect vehicles (car, truck, bus, motorcycle)
        if label in [2, 3, 5, 7]:  
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            height = int(y2 - y1)

            # Determine if the vehicle is too close (approximate check using bounding box height)
            if height > SAFE_DISTANCE:
                cv2.putText(frame, "⚠ COLLISION WARNING!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Draw red box
            else:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)  # Draw normal box

    # Show output
    resized_frame = cv2.resize(frame, (1280, 720))  # Adjust to your preferred resolution
    cv2.imshow("Forward Collision Warning", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
