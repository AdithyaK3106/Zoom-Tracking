from ultralytics import YOLO
import cv2

# Load your custom trained model
model = YOLO("best_take2.pt")

# Open webcam (use video path instead if needed)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection + tracking with lower confidence threshold
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
    )

    # Draw results
    annotated_frame = results[0].plot()
    
    # Debug: Print number of detections
    if results[0].boxes is not None:
        num_detections = len(results[0].boxes)
        print(f"Detections: {num_detections}")
    else:
        print("No detections found")

    cv2.imshow("Custom YOLO Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()