from ultralytics import YOLO
import cv2

# Load your custom trained model
model = YOLO(
    "best.pt"
)

# Open webcam (use video path instead if needed)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection + tracking
    results = model.track(
        frame,
        persist=True,          # keeps track IDs across frames
        tracker="botsort.yaml" # or "bytetrack.yaml"
    )

    # Draw results
    annotated_frame = results[0].plot()

    cv2.imshow("Custom YOLO Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
