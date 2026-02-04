from ultralytics import YOLO
import cv2
import numpy as np

class SingleObjectTracker:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)
        self.selected_track_id = None
        self.cap = cv2.VideoCapture(0)
        self.first_frame = True
        self.zoom_level = 1.0
        self.target_zoom = 1.0
        self.parallax_offset_x = 0
        self.parallax_offset_y = 0
        self.box_history = []
        self.zoom_smoothness = 0.15
        self.parallax_smoothness = 0.2

    def mouse_click(self, event, x, y, flags, param):
        """Handle mouse click to select object"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find which detection was clicked
            for box, track_id, class_name in self.current_detections:
                x1, y1, x2, y2 = box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_track_id = track_id
                    self.box_history = []
                    print(f"Selected object: {class_name} (Track ID: {track_id})")
                    return
                
    def calculate_zoom_level(self, box):
        """Calculate zoom level based on box size"""
        if not box:
            return 1.0
        
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        
        # Normalize based on frame dimensions (assume 640x480 typical)
        frame_area = 640 * 480
        area_ratio = box_area / frame_area
        
        # Map area ratio to zoom level (0.01 area ratio = 1.5x zoom, 0.1 = 1.0x zoom)
        zoom = 1.5 - (area_ratio * 5)
        zoom = np.clip(zoom, 1.0, 3.0)
        return zoom
    
    def apply_parallax(self, frame, box, zoom_level):
        """Apply parallax zoom effect - center on object and scale"""
        if box is None:
            return frame
        
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        h, w = frame.shape[:2]
        
        # Calculate target parallax offset to keep object centered
        target_offset_x = (w / 2) - center_x
        target_offset_y = (h / 2) - center_y
        
        # Smooth parallax transition
        self.parallax_offset_x += (target_offset_x - self.parallax_offset_x) * self.parallax_smoothness
        self.parallax_offset_y += (target_offset_y - self.parallax_offset_y) * self.parallax_smoothness
        
        # Apply zoom and parallax transformation
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, zoom_level)
        M[0, 2] += self.parallax_offset_x
        M[1, 2] += self.parallax_offset_y
        
        zoomed_frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return zoomed_frame
    
    def run(self):
        """Run the single object tracker"""
        cv2.namedWindow("Single Object Tracking")
        cv2.setMouseCallback("Single Object Tracking", self.mouse_click)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Run YOLOv8 tracking
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
            
            # Extract detections and track IDs
            self.current_detections = []
            selected_box = None
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    track_id = int(box.id[0]) if box.id is not None else None
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    box_coords = [int(x1), int(y1), int(x2), int(y2)]
                    self.current_detections.append((box_coords, track_id, class_name))
                    
                    if track_id == self.selected_track_id:
                        selected_box = box_coords
            
            # Draw frame
            frame_display = frame.copy()
            
            if self.selected_track_id is None:
                # Reset zoom when no object is selected
                self.target_zoom = 1.0
                self.zoom_level += (self.target_zoom - self.zoom_level) * self.zoom_smoothness
                self.parallax_offset_x = 0
                self.parallax_offset_y = 0
                
                # Draw all detections before selection
                cv2.putText(frame_display, "Click on an object to track it", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                for box, track_id, class_name in self.current_detections:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_display, f"{class_name} (ID: {track_id})", (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # Calculate zoom level based on selected object's size
                self.target_zoom = self.calculate_zoom_level(selected_box) if selected_box else 1.0
                
                # Smoothly transition zoom level
                self.zoom_level += (self.target_zoom - self.zoom_level) * self.zoom_smoothness
                
                # Apply parallax zoom effect
                frame_display = self.apply_parallax(frame_display, selected_box, self.zoom_level)
                
                # Draw only the selected object
                for box, track_id, class_name in self.current_detections:
                    if track_id == self.selected_track_id and box:
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame_display, f"Tracking: {class_name} (ID: {track_id})", (x1, y1 - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Display zoom info
                cv2.putText(frame_display, f"Zoom: {self.zoom_level:.2f}x", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Single Object Tracking", frame_display)
            
            # Press 'q' to quit, 'r' to reset selection
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.selected_track_id = None
                print("Selection reset. Click on an object to track it.")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = SingleObjectTracker()
    tracker.run()
