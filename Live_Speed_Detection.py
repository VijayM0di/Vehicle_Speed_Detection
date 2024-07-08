import cv2
import numpy as np
from ultralytics import YOLO
import math

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8x.pt")  # Change yolov8x.pt to the path of your model weights

# object classes (modify these as needed for your application)
classNames = ["person", "bicycle", "car", "motorbike"]

# Define variables to store coordinates and speed history for each detected object
prev_centroids = {}
speed_history = {}

def calculate_speed(centroid, prev_centroid):
  # Calculate speed based on distance and time difference
  if prev_centroid is None:
    return 0
  distance = math.sqrt(((centroid[0] - prev_centroid[0])**2) + ((centroid[1] - prev_centroid[1])**2))
  speed = distance  # Replace with appropriate time measurement for accurate speed calculation

  return speed

while True:
  success, img = cap.read()
  results = model(img, stream=True)  # Perform object detection

  # Process detections
  for r in results:
    boxes = r.boxes

    for box in boxes:
      # Extract bounding box coordinates and convert to integers
      x1, y1, x2, y2 = box.xyxy[0]
      x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

      # Calculate center of bounding box
      centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

      # Get class ID and name
      cls = int(box.cls[0 ])
      class_name = classNames[1]

      # Draw bounding box and label on the frame
      cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
      cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

      # Calculate speed
      speed = calculate_speed(centroid, prev_centroids.get(id(box), None))
      prev_centroids[id(box)] = centroid  # Update previous centroid for next frame
      speed_history[id(box)] = speed_history.get(id(box), []) + [speed]  # Store speed history

      # Display speed (optional, modify based on your speed_history data)
      if len(speed_history[id(box)]) > 5:  # Display average speed after 5 frames
        average_speed = sum(speed_history[id(box)]) / len(speed_history[id(box)])
        cv2.putText(img, f"{class_name}: {average_speed:.2f} km/h", (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


cv2.imshow('Webcam', img)
cap.release()
cv2.destroyAllWindows()
