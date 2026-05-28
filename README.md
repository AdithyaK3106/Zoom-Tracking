# Gimbal Tracking with Auto-Zooming

This project implements a **real-time object tracking system with auto-zooming capabilities**, designed for use with a gimbal to keep target objects centered and properly framed.

## 🎯 Project Goal
To provide a seamless tracking experience where the camera (webcam or gimbal-mounted) automatically adjusts its zoom level and centering based on the movement and size of a selected target object.

## 🚀 Key Features
- **Real-time Tracking**: Uses YOLO26 for high-performance object detection and persistent tracking.
- **Auto-Zooming**: Dynamically calculates zoom levels based on the object's distance (bounding box area).
- **Smooth Parallax Effect**: Uses affine transformations to keep the target centered while maintaining smooth transitions.
- **Interactive Selection**: Desktop proof-of-concept allows click-to-track functionality.

## 🏗️ Core Components
- **Detection Engine**: YOLO26 (Ultralytics) - tracks objects using ByteTrack or BoTSort.
- **Zoom Logic**: Adaptive scaling based on normalized bounding box area.
- **Centering (Parallax)**: Smooth coordinate translation to follow the target without jerky movements.

## 📂 File Structure
- `proof-of-concept-PC.py`: Desktop-ready script with click-to-track and auto-zoom logic.
- `Take_1/`: Initial iteration containing `Project_Mr-Sippy.py` and first-pass custom model.
- `Take_2/`: Improved iteration with updated tracking parameters and `best_take2.pt`.
- `requirements.txt`: Project dependencies (ultralytics, opencv-python, numpy).

## 🛠️ Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the proof-of-concept: `python proof-of-concept-PC.py`
3. Click on any object in the webcam feed to start tracking.
4. Press `r` to reset selection or `q` to quit.
