from ultralytics import YOLO
import cv2
import numpy as np
import time

class SingleObjectTracker:
    def __init__(self, model_path=r"yolo26n.pt", camera_index=0):
        self.model = YOLO(model_path, task='detect')
        self.model.to('cpu')
        self.selected_track_id = None
        self.cap = cv2.VideoCapture(camera_index)
        # Set dimensions
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.current_detections = []
        
        # --- CPU Optimization settings ---
        self.inference_size = 320  # Lower resolution for faster inference (default 640)
        self.conf_threshold = 0.5  # Confidence threshold
        self.frame_skip = 0  # Skip frames for inference (0 = every frame, 1 = every other, etc.)
        self.frame_count = 0

        # --- Auto-zoom state ---
        self.prev_area = None                   # last bbox area
        self.area_ema = None                     # smoothed area
        self.area_deriv_ema = 0.0                # smoothed derivative
        self.prev_centroid = None                # last centroid (x,y)
        self.centroid_ema = None                 # smoothed centroid for cropping center
        self.centroid_speed_ema = 0.0            # smoothed speed
        self.zoom = 1.0                          # current digital zoom scale (1.0 = no zoom)
        self.zoom_target = 1.0
        self.last_seen_time = time.time()

        # --- Tuning parameters (playzy with these) ---
        self.TARGET_AREA_FRACTION = 0.20         # desired fraction of frame area occupied by subject
        self.AREA_EMA_ALPHA = 0.4                # smoothing for area
        self.AREA_DERIV_ALPHA = 0.3              # smoothing for area derivative
        self.CENTROID_EMA_ALPHA = 0.25           # smoothing for centroid (reduces cropping jumps)
        self.SPEED_EMA_ALPHA = 0.25              # smoothing for centroid speed
        self.ZOOM_SMOOTHING = 0.08               # how fast zoom moves to target
        self.MAX_ZOOM = 2.5
        self.MIN_ZOOM = 1.0
        self.AREA_WEIGHT = 0.8                   # how much area trend influences zoom
        self.SPEED_WEIGHT = 0.2                  # how much centroid motion influences zoom
        self.RESET_TIMEOUT = 1.0                 # seconds after losing target to reset zoom gradually

        # --- FPS counter ---
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.fps = 0.0

    def mouse_click(self, event, x, y, flags, param):
        """Handle mouse click to select object"""
        if event == cv2.EVENT_LBUTTONDOWN:
            for box, track_id in self.current_detections:
                x1, y1, x2, y2 = box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_track_id = track_id
                    print(f"Selected object: Track ID {track_id}")
                    # reset zoom-related state for a fresh start
                    self.prev_area = None
                    self.area_ema = None
                    self.area_deriv_ema = 0.0
                    self.prev_centroid = None
                    self.centroid_ema = None
                    self.centroid_speed_ema = 0.0
                    self.zoom = 1.0
                    self.zoom_target = 1.0
                    self.last_seen_time = time.time()
                    return

    def _update_auto_zoom(self, frame, box):
        """
        Update zoom based on the selected object's bounding box.
        box = [x1,y1,x2,y2]
        Returns:
            zoomed_frame: frame after crop & resize
            crop_info: (x1_crop, y1_crop, crop_w, crop_h) in original frame coordinates
            mapped_box: original box mapped to zoomed_frame coordinates (x1_m, y1_m, x2_m, y2_m)
        """
        h, w = frame.shape[:2]
        frame_area = w * h
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = bw * bh
        centroid = (x1 + bw // 2, y1 + bh // 2)

        now = time.time()

        # Initialize EMAs if first time
        if self.area_ema is None:
            self.area_ema = float(area)
            self.prev_area = float(area)
            self.area_deriv_ema = 0.0
        else:
            # Update area EMA and derivative EMA
            self.area_ema = (1 - self.AREA_EMA_ALPHA) * self.area_ema + self.AREA_EMA_ALPHA * float(area)
            raw_deriv = (area - self.prev_area)
            self.area_deriv_ema = (1 - self.AREA_DERIV_ALPHA) * self.area_deriv_ema + self.AREA_DERIV_ALPHA * raw_deriv
            self.prev_area = float(area)

        # Update centroid EMA and speed EMA (motion parallax proxy)
        if self.centroid_ema is None:
            self.centroid_ema = centroid
            self.prev_centroid = centroid
            self.centroid_speed_ema = 0.0
        else:
            cx_prev, cy_prev = self.prev_centroid
            cx, cy = centroid
            speed = np.hypot(cx - cx_prev, cy - cy_prev)  # pixel displacement per frame
            self.centroid_speed_ema = (1 - self.SPEED_EMA_ALPHA) * self.centroid_speed_ema + self.SPEED_EMA_ALPHA * speed
            # smooth centroid used for cropping center (reduces jitter)
            sm_cx = (1 - self.CENTROID_EMA_ALPHA) * self.centroid_ema[0] + self.CENTROID_EMA_ALPHA * cx
            sm_cy = (1 - self.CENTROID_EMA_ALPHA) * self.centroid_ema[1] + self.CENTROID_EMA_ALPHA * cy
            self.centroid_ema = (int(sm_cx), int(sm_cy))
            self.prev_centroid = centroid

        # Estimate desired scale from area (linear scale = sqrt(area_ratio))
        desired_box_area = self.TARGET_AREA_FRACTION * frame_area
        area_for_ratio = max(1.0, self.area_ema)
        desired_scale_area = np.sqrt(desired_box_area / area_for_ratio)

        # Clip sensibly
        desired_scale_area = float(np.clip(desired_scale_area, 1.0 / self.MAX_ZOOM, self.MAX_ZOOM))

        # Use centroid speed as a predictor
        frame_diag = np.hypot(w, h)
        speed_norm = self.centroid_speed_ema / (frame_diag + 1e-6)
        speed_factor = 1.0 + speed_norm * 3.0  # tune factor multiplier

        # Blend area-based and speed-based suggestions
        blended_desired = (self.AREA_WEIGHT * desired_scale_area + self.SPEED_WEIGHT * speed_factor) / (self.AREA_WEIGHT + self.SPEED_WEIGHT)

        # Constrain zoom target
        self.zoom_target = float(np.clip(blended_desired, self.MIN_ZOOM, self.MAX_ZOOM))

        # Smoothly update actual zoom
        self.zoom = (1 - self.ZOOM_SMOOTHING) * self.zoom + self.ZOOM_SMOOTHING * self.zoom_target

        # Build crop centered on smoothed centroid (centroid_ema)
        center_x, center_y = int(self.centroid_ema[0]), int(self.centroid_ema[1])
        # compute crop size (linear)
        crop_w = int(w / self.zoom)
        crop_h = int(h / self.zoom)
        crop_w = max(1, min(crop_w, w))
        crop_h = max(1, min(crop_h, h))

        # Ensure crop fits inside frame: adjust center if needed
        x1_crop = max(0, center_x - crop_w // 2)
        y1_crop = max(0, center_y - crop_h // 2)
        x2_crop = x1_crop + crop_w
        y2_crop = y1_crop + crop_h

        # If crop exceeds boundaries, shift
        if x2_crop > w:
            x2_crop = w
            x1_crop = w - crop_w
        if y2_crop > h:
            y2_crop = h
            y1_crop = h - crop_h
        x1_crop, y1_crop = max(0, x1_crop), max(0, y1_crop)

        # Crop and resize back to original frame size for display (digital zoom)
        cropped = frame[y1_crop:y2_crop, x1_crop:x2_crop]
        if cropped.size == 0:
            # safety fallback
            return frame, (0, 0, w, h), (x1, y1, x2, y2)

        zoomed_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        # Map original bbox to zoomed frame coordinates
        scale_x = w / float(crop_w)
        scale_y = h / float(crop_h)
        try:
            x1_m = int((x1 - x1_crop) * scale_x)
            y1_m = int((y1 - y1_crop) * scale_y)
            x2_m = int((x2 - x1_crop) * scale_x)
            y2_m = int((y2 - y1_crop) * scale_y)
            # clip
            x1_m, y1_m = max(0, x1_m), max(0, y1_m)
            x2_m, y2_m = min(w - 1, x2_m), min(h - 1, y2_m)
            mapped_box = (x1_m, y1_m, x2_m, y2_m)
        except Exception:
            mapped_box = (x1, y1, x2, y2)

        return zoomed_frame, (x1_crop, y1_crop, crop_w, crop_h), mapped_box

    def run(self):
        """Run the single object tracker with auto-zoom and debug overlays"""
        cv2.namedWindow("Single Object Tracking")
        cv2.setMouseCallback("Single Object Tracking", self.mouse_click)

        font = cv2.FONT_HERSHEY_SIMPLEX
        first_frame = True

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Print frame dimensions on first frame
            if first_frame:
                h, w = frame.shape[:2]
                print(f"Frame dimensions: {w}x{h} (width x height)")
                print(f"CPU Optimization: inference_size={self.inference_size}, conf={self.conf_threshold}")
                first_frame = False
            
            self.frame_count += 1
            # Skip frames for faster processing
            if self.frame_count % (self.frame_skip + 1) != 0:
                # Use previous detections on skipped frames
                frame_display = frame.copy()
                if self.selected_track_id is not None:
                    # Show frame while skipping inference
                    h, w = frame.shape[:2]
                    cv2.putText(frame_display, f"Skipping inference (FPS: {self.fps:.1f})", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    fps_text = f"FPS: {self.fps:.1f}"
                    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    fps_x = w - text_size[0] - 10
                    fps_y = 30
                    cv2.putText(frame_display, fps_text, (fps_x, fps_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Single Object Tracking", frame_display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.selected_track_id = None
                    self.zoom = 1.0
                continue

            # Run YOLOv8 tracking on the single frame
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", 
                                      imgsz=self.inference_size, conf=self.conf_threshold, verbose=False)

            # Extract detections and track IDs
            self.current_detections = []
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    track_id = int(box.id[0]) if box.id is not None else None
                    self.current_detections.append(
                        ([int(x1), int(y1), int(x2), int(y2)], track_id)
                    )

            frame_display = frame.copy()

            # Update FPS counter
            self.fps_frame_count += 1
            elapsed = time.time() - self.fps_start_time
            if elapsed >= 1.0:
                self.fps = self.fps_frame_count / elapsed
                self.fps_frame_count = 0
                self.fps_start_time = time.time()

            # Display FPS on frame
            fps_text = f"FPS: {self.fps:.1f}"
            text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            fps_x = frame_display.shape[1] - text_size[0] - 10
            fps_y = frame_display.shape[0] - 10
            cv2.putText(frame_display, fps_text, (fps_x, fps_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if self.selected_track_id is None:
                # Draw all detections before selection
                cv2.putText(frame_display, "Click on an object to track it", (10, 30),
                           font, 1, (0, 255, 0), 2)
                for box, track_id in self.current_detections:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_display, f"ID {track_id}", (x1, y1 - 5),
                               font, 0.6, (0, 255, 0), 2)
                cv2.imshow("Single Object Tracking", frame_display)
            else:
                # Find the selected detection in current frame
                found = False
                sel_box = None
                for box, track_id in self.current_detections:
                    if track_id == self.selected_track_id:
                        found = True
                        sel_box = box
                        break

                if found and sel_box is not None:
                    self.last_seen_time = time.time()
                    zoomed, crop_info, mapped_box = self._update_auto_zoom(frame, sel_box)
                    # Draw mapped original detection box (yellow) on zoomed frame
                    x1_m, y1_m, x2_m, y2_m = mapped_box
                    cv2.rectangle(zoomed, (x1_m, y1_m), (x2_m, y2_m), (0, 255, 255), 2)
                    cv2.putText(zoomed, f"Tracking ID {self.selected_track_id}",
                                (10, 30), font, 0.8, (0, 0, 255), 2)
                    # Display FPS on zoomed frame
                    fps_text = f"FPS: {self.fps:.1f}"
                    text_size = cv2.getTextSize(fps_text, font, 0.7, 2)[0]
                    fps_x = zoomed.shape[1] - text_size[0] - 10
                    fps_y = 30  # Position at top right to avoid overlapping with zoom bar
                    cv2.putText(zoomed, fps_text, (fps_x, fps_y),
                                font, 0.7, (0, 255, 0), 2)

                    # Debug numeric overlays (top-left)
                    overlay_x = 10
                    overlay_y = 60
                    line_h = 22
                    # Compose lines
                    area_raw = int(self.prev_area) if self.prev_area is not None else 0
                    area_ema = float(self.area_ema) if self.area_ema is not None else 0.0
                    area_deriv = float(self.area_deriv_ema)
                    speed = float(self.centroid_speed_ema)
                    z = float(self.zoom)
                    zt = float(self.zoom_target)

                    lines = [
                        f"area_raw: {area_raw}",
                        f"area_ema: {area_ema:.1f}",
                        f"area_deriv: {area_deriv:.1f}",
                        f"centroid_speed: {speed:.2f}",
                        f"zoom: {z:.3f}",
                        f"zoom_target: {zt:.3f}"
                    ]
                    for i, txt in enumerate(lines):
                        cv2.putText(zoomed, txt, (overlay_x, overlay_y + i * line_h),
                                    font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(zoomed, txt, (overlay_x, overlay_y + i * line_h),
                                    font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)  # thin shadow for readability

                    # Draw zoom-level bar (bottom-left)
                    bar_w = 220
                    bar_h = 18
                    bar_x = 10
                    bar_y = zoomed.shape[0] - 30
                    # background
                    cv2.rectangle(zoomed, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                    # filled portion
                    frac = (z - self.MIN_ZOOM) / (self.MAX_ZOOM - self.MIN_ZOOM + 1e-9)
                    frac = float(np.clip(frac, 0.0, 1.0))
                    fill_w = int(bar_w * frac)
                    cv2.rectangle(zoomed, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 200, 0), -1)
                    # border and labels
                    cv2.rectangle(zoomed, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1)
                    cv2.putText(zoomed, f"Zoom {z:.2f}x", (bar_x + bar_w + 8, bar_y + bar_h - 2),
                                font, 0.6, (255, 255, 255), 2)

                    # small legend of mapping: draw small original bbox center mapped
                    try:
                        cx_m = int(( (sel_box[0] + sel_box[2]) / 2.0 - crop_info[0]) * (zoomed.shape[1] / float(crop_info[2])))
                        cy_m = int(( (sel_box[1] + sel_box[3]) / 2.0 - crop_info[1]) * (zoomed.shape[0] / float(crop_info[3])))
                        cv2.circle(zoomed, (cx_m, cy_m), 6, (0, 255, 255), -1)
                    except Exception:
                        pass

                    cv2.imshow("Single Object Tracking", zoomed)
                else:
                    # Selected target not found in this frame
                    time_since_seen = time.time() - self.last_seen_time
                    if time_since_seen > self.RESET_TIMEOUT:
                        # slowly go back to no-zoom
                        self.zoom_target = 1.0
                        self.zoom = (1 - self.ZOOM_SMOOTHING) * self.zoom + self.ZOOM_SMOOTHING * self.zoom_target
                        # build crop centered at frame center as zoom decreases
                        h, w = frame.shape[:2]
                        crop_w = int(w / max(self.zoom, 1.0))
                        crop_h = int(h / max(self.zoom, 1.0))
                        x1_crop = (w - crop_w) // 2
                        y1_crop = (h - crop_h) // 2
                        cropped = frame[y1_crop:y1_crop+crop_h, x1_crop:x1_crop+crop_w]
                        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
                        # Show a simpler overlay indicating loss
                        cv2.putText(zoomed, "Target lost — resetting zoom...", (10, 30), font, 0.8, (0, 0, 255), 2)
                        cv2.imshow("Single Object Tracking", zoomed)
                    else:
                        # show normal frame until reset kicks in
                        cv2.putText(frame_display, "Searching for target...", (10, 30), font, 0.8, (0, 255, 255), 2)
                        cv2.imshow("Single Object Tracking", frame_display)

            # Press 'q' to quit, 'r' to reset selection
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.selected_track_id = None
                print("Selection reset. Click on an object to track it.")
                # reset zoom state
                self.prev_area = None
                self.area_ema = None
                self.area_deriv_ema = 0.0
                self.prev_centroid = None
                self.centroid_ema = None
                self.centroid_speed_ema = 0.0
                self.zoom = 1.0
                self.zoom_target = 1.0

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = SingleObjectTracker()
    tracker.run()