# Video Detection and Alert System

This project implements a real-time object detection and alert system using YOLOv8, FastAPI, and OpenCV. It processes video frames, detects objects, and generates alerts based on object persistence within a user-defined polygon area.

## Features

- Object detection using YOLOv8
- Custom detection area definition with polygon drawing
- Real-time video processing
- Alert generation for objects persisting in the defined area
- User-friendly GUI for video selection and parameter adjustment

## Prerequisites

- Python 3.7+
- FastAPI
- OpenCV (cv2)
- Ultralytics YOLOv8
- Shapely
- NumPy
- Requests
- Tkinter

## Installation

1. Clone the repository:
git clone https://github.com/yourusername/video-detection-alert-system.git
cd video-detection-alert-system

2. Install the required packages:
pip install fastapi opencv-python-headless ultralytics shapely numpy requests

## Usage

1. Start the FastAPI server:
uvicorn main:app --reload

2. Run the GUI application:
python gui.py

3. Use the GUI to:
- Select a video file
- Draw a polygon to define the detection area
- Adjust the frame threshold for alerts
- Start video processing

4. The application will display the video with bounding boxes around detected objects and show alerts when objects persist in the defined area.

## Project Structure

- `main.py`: FastAPI server implementation for object detection
- `gui.py`: Tkinter-based GUI for video selection and processing
- `yolov8x.pt`: YOLOv8 pre-trained model (not included in the repository)

## Customization

- Adjust the `frame_threshold` in the GUI to change the sensitivity of alert generation
- Modify the `polygon_points` in `main.py` to change the default detection area

## License

[MIT License](LICENSE)

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenCV](https://opencv.org/)
