import cv2
import requests
from tkinter import Tk, Button, Scale, Label, filedialog, messagebox

# Global variables
polygon_points = []
first_frame = None
polygon_complete = False

def select_video():
    global first_frame, polygon_complete, polygon_points

    # Open file dialog to select a video file
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    if not file_path:
        return

    # Read the first frame of the video
    cap = cv2.VideoCapture(file_path)
    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "Cannot read video file.")
        cap.release()
        return

    # Reset polygon points
    polygon_points = []
    polygon_complete = False

    # Display the first frame and set up mouse callback for polygon drawing
    first_frame = frame.copy()
    cv2.imshow("Draw Polygon", first_frame)
    cv2.setMouseCallback("Draw Polygon", draw_polygon)

    # Wait until the polygon is completed
    while True:
        key = cv2.waitKey(0)
        if key == 13:  # Enter key
            if len(polygon_points) > 2:
                complete_polygon()
                break
            else:
                messagebox.showinfo("Incomplete Action", "Draw at least three points to complete the polygon.")

    cv2.destroyAllWindows()

    if polygon_complete:
        process_video(file_path)
    else:
        messagebox.showinfo("Incomplete Action", "Polygon not completed. Try again.")

def draw_polygon(event, x, y, flags, param):
    global polygon_points, first_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        # Add point to the polygon on left-click
        polygon_points.append((x, y))
        cv2.circle(first_frame, (x, y), 3, (0, 255, 0), -1)
        if len(polygon_points) > 1:
            cv2.line(first_frame, polygon_points[-2], polygon_points[-1], (0, 255, 0), 2)
        cv2.imshow('Draw Polygon', first_frame)

def complete_polygon():
    global polygon_points, first_frame, polygon_complete

    if len(polygon_points) > 2:
        # Draw a line from the last point to the first point to close the polygon
        cv2.line(first_frame, polygon_points[-1], polygon_points[0], (0, 255, 0), 2)
        cv2.imshow('Draw Polygon', first_frame)
        polygon_complete = True
        cv2.waitKey(1)

def process_video(file_path):
    global polygon_points

    if not polygon_complete:
        messagebox.showerror("Error", "Please complete the polygon drawing by pressing Enter.")
        return

    # Open the video for frame-by-frame processing
    cap = cv2.VideoCapture(file_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        response = requests.post(
            "http://localhost:8000/detect_frame/",
            files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
            params={"frame_threshold": frame_threshold.get()}
        )

        if response.status_code != 200:
            messagebox.showerror("API Error", f"Status Code: {response.status_code}\nMessage: {response.text}")
            break

        detection = response.json()
        bboxes = detection.get("bboxes", [])
        alerts = detection.get("alerts", [])

        # Draw bounding boxes and alerts
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            object_id = bbox['object_id']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"ID: {object_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display alerts
        for alert in alerts:
            alert_message = f"Alert: {alert['alert']} for object ID: {alert['object_id']}"
            cv2.putText(frame, alert_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(alert_message)  # Print alert to console
            messagebox.showinfo("Alert", alert_message)  # Show alert as pop-up

        cv2.imshow('Video Processing', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", "Video processing completed.")

# Set up the GUI
root = Tk()
root.title("Video Detection and Alert System")

# Create and place GUI components
upload_button = Button(root, text="Select Video", command=select_video)
upload_button.pack(pady=10)

threshold_label = Label(root, text="Frame Threshold:")
threshold_label.pack(pady=5)

frame_threshold = Scale(root, from_=1, to=300, orient='horizontal')  # Set a reasonable range
frame_threshold.set(30)
frame_threshold.pack(pady=5)

root.mainloop()
