from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Point, Polygon
from collections import defaultdict
import uuid

app = FastAPI()

# Load YOLOv8 model
model = YOLO('yolov8x.pt')  # Ensure the model is available

# Track objects by ID
object_durations = defaultdict(lambda: {"frame_count": 0, "inside_polygon": False, "position": (0, 0)})

# Polygon for detection area
polygon_points = [(100, 100), (500, 100), (500, 500), (100, 500)]
polygon = Polygon(polygon_points)

# Function to check if a point is inside the polygon
def is_inside_polygon(x, y, polygon):
    point = Point(x, y)
    return polygon.contains(point)

@app.post("/detect_frame/")
async def detect_objects_in_frame(file: UploadFile = File(...), frame_threshold: int = 30):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode frame")

    # Perform object detection
    detection_results = model(frame)  # Get predictions from the model

    # Extract bounding boxes and class IDs
    if detection_results:
        boxes = detection_results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
        confidences = detection_results[0].boxes.conf.cpu().numpy()  # Get confidences
        class_ids = detection_results[0].boxes.cls.cpu().numpy()  # Get class IDs

        bboxes = []
        alerts = []

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            conf = confidences[i]
            cls = class_ids[i]
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2

            # Assign an object ID
            object_id = f"{cls}_{int(center_x)}_{int(center_y)}"

            # Check if this object has been seen before
            matched_id = None
            for obj_id, data in object_durations.items():
                previous_center_x, previous_center_y = data["position"]
                distance = np.sqrt((center_x - previous_center_x) ** 2 + (center_y - previous_center_y) ** 2)
                if distance < 50:  # Threshold distance to match objects
                    matched_id = obj_id
                    break

            if matched_id:
                # Update existing object
                object_id = matched_id
                object_durations[object_id]["position"] = (center_x, center_y)

                if is_inside_polygon(center_x, center_y, polygon):
                    if not object_durations[object_id]["inside_polygon"]:
                        object_durations[object_id]["inside_polygon"] = True
                        object_durations[object_id]["frame_count"] = 1  # Start counting frames
                    else:
                        object_durations[object_id]["frame_count"] += 1  # Increment frame count
                        if object_durations[object_id]["frame_count"] > frame_threshold:
                            alerts.append({"object_id": object_id, "alert": f"Object {object_id} inside polygon for {object_durations[object_id]['frame_count']} frames"})
                            print(f"Alert generated for object {object_id}: inside polygon for {object_durations[object_id]['frame_count']} frames")
                else:
                    object_durations[object_id]["inside_polygon"] = False
                    object_durations[object_id]["frame_count"] = 0  # Reset frame count

            else:
                # Add new object
                object_id = str(uuid.uuid4())
                object_durations[object_id] = {
                    "frame_count": 1 if is_inside_polygon(center_x, center_y, polygon) else 0,
                    "position": (center_x, center_y),
                    "inside_polygon": is_inside_polygon(center_x, center_y, polygon)
                }

            # Append detected bounding box
            bboxes.append({
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "confidence": float(conf),
                "class": int(cls),
                "object_id": object_id
            })

        return {"bboxes": bboxes, "alerts": alerts}

    return {"bboxes": [], "alerts": []}
