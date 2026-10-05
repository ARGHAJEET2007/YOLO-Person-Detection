from ultralytics import YOLO
import cv2
import csv
import json
import os
import time
from datetime import datetime

# 1. LOAD YOLO26s

model = YOLO("yolo26s.pt")

# 2. OPEN CAMERA

# 0 = laptop camera
# 1 = external USB camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started.")
print("Detecting PERSON only.")
print("Press Q to stop.")

# 3. FILE NAMES

csv_file = "person_detections.csv"
json_file = "person_detections.json"

# 4. OPEN CSV

csv_exists = os.path.exists(csv_file)

csv_output = open(
    csv_file,
    "a",
    newline="",
    encoding="utf-8"
)

csv_writer = csv.writer(csv_output)

if not csv_exists:
    csv_writer.writerow([
        "Timestamp",
        "Object",
        "Confidence",
        "X1",
        "Y1",
        "X2",
        "Y2"
    ])

# 5. LOAD EXISTING JSON

if os.path.exists(json_file):

    try:
        with open(json_file, "r", encoding="utf-8") as file:
            json_data = json.load(file)

    except:
        json_data = []

else:
    json_data = []

# 6. FPS CONTROL

last_detection_time = 0

# 0.2 seconds = approximately 5 FPS
detection_interval = 0.2

# 7. MAIN LOOP


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break


    current_time = time.time()

    # RUN YOLO EVERY 0.2 SECOND

    if current_time - last_detection_time >= detection_interval:

        last_detection_time = current_time

        results = model(
            frame,
            verbose=False
        )

        # PROCESS DETECTIONS

        for result in results:

            boxes = result.boxes

            for box in boxes:

                # Class ID
                class_id = int(box.cls[0])

                # COCO class 0 = person
                if class_id != 0:
                    continue


                # Confidence
                confidence = float(box.conf[0])

                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].tolist()


                # Timestamp
                timestamp = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                # SAVE CSV
               
                csv_writer.writerow([
                    timestamp,
                    "person",
                    round(confidence, 3),
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                ])

                csv_output.flush()


                # SAVE JSON
              
                detection = {
                    "timestamp": timestamp,
                    "object": "person",
                    "confidence": round(confidence, 3),
                    "bounding_box": {
                        "x1": int(x1),
                        "y1": int(y1),
                        "x2": int(x2),
                        "y2": int(y2)
                    }
                }

                json_data.append(detection)


        # WRITE JSON
  

        with open(
            json_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                json_data,
                file,
                indent=4
            )

    # DISPLAY ONLY PERSON DETECTIONS
  
    annotated_frame = frame.copy()

    for result in results if 'results' in locals() else []:

        for box in result.boxes:

            class_id = int(box.cls[0])

            # Only person
            if class_id != 0:
                continue

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                f"Person {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # SHOW CAMERA
 
    cv2.imshow(
        "YOLO26s - Person Detection",
        annotated_frame
    )

    # PRESS Q TO STOP

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 8. CLOSE EVERYTHING

cap.release()

csv_output.close()

cv2.destroyAllWindows()

print("\nDetection stopped.")
print("CSV:", csv_file)
print("JSON:", json_file)