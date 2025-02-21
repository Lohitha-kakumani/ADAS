import cv2
import numpy as np

# Load video
video_path = r"C:\Users\kakum\OneDrive\Documents\murthilab\level_0\BSW\vedio1.mp4"
cap = cv2.VideoCapture(video_path)

# Set video resolution to avoid zoom issues
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def canny_edge_detection(frame):
    """ Apply Canny Edge Detection to highlight lane markings """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)  # Lower and upper thresholds
    return edges

def region_of_interest(edges):
    """ Mask the region of interest (ROI) where lane lines are present """
    height, width = edges.shape
    polygon = np.array([[
        (100, height),  # Bottom-left
        (width // 2, height // 2),  # Mid-top
        (width - 100, height)  # Bottom-right
    ]], np.int32)

    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges

def detect_lane_lines(edges):
    """ Use Hough Transform to detect lane lines """
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    return lines

def draw_lane_lines(frame, lines):
    """ Draw lane lines on the frame """
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Green lines
    return frame

def lane_departure_warning(frame, lines, width):
    """ Detect lane departure and trigger warning """
    if lines is None:
        return frame  # No lanes detected

    left_lane = []
    right_lane = []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1 + 0.0001)  # Avoid division by zero

        if slope < 0:
            left_lane.append((x1, y1, x2, y2))  # Left lane lines
        else:
            right_lane.append((x1, y1, x2, y2))  # Right lane lines

    if not left_lane or not right_lane:
        cv2.putText(frame, "⚠ Lane Departure!", (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop if video ends

    frame = cv2.resize(frame, (1280, 720))  # Ensure proper scaling

    edges = canny_edge_detection(frame)
    roi_edges = region_of_interest(edges)
    lane_lines = detect_lane_lines(roi_edges)
    
    frame = draw_lane_lines(frame, lane_lines)
    frame = lane_departure_warning(frame, lane_lines, frame.shape[1])

    cv2.imshow("Lane Departure Warning", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
