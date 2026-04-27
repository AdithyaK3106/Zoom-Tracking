"""
Module 6 — Feedback Loop Pipeline
===================================
Orchestrates all five modules into a single run-loop.

Two-pass detection (feedback loop)
------------------------------------
The key architectural insight is that YOLO on a 640×480 frame may miss an
object that occupies only 10×10 pixels.  By first zooming in on the tracked
object (via ZoomEngine), we get a higher-pixel-density crop of the same
region, resize it back to 640×480, and run YOLO again.  This second pass
provides much richer detections for the target object.

The zoomed detections are mapped back to original-frame coordinates via
`ZoomEngine.map_bbox_to_original()`, then merged with the full-frame
detections.  A lightweight NMS pass removes duplicates before the merged
array is handed to ByteTrack.

Per-frame data flow
-------------------
    raw frame
      │
      ├──► [Pass 1] DetectionModule.detect(frame)         → dets_full
      │
      ├──[if tracking]──► ZoomEngine.apply_zoom()
      │        │              → zoomed_frame, crop_region, zoom_level
      │        │
      │        └──► [Pass 2] DetectionModule.detect(zoomed_frame) → dets_zoom
      │                  └── map_bbox_to_original(dets_zoom, crop_region)
      │
      ├──► merge(dets_full, dets_zoom) → NMS → detections_arr
      │
      ├──► TrackingModule.update(detections_arr) → tracked_objects
      │
      ├──► UserInteractionModule.process_click() → selected_id
      │
      ├──► check_track_alive() → trigger re-selection if lost
      │
      └──► Visualize + display
               ├── full frame with all track boxes + crop indicator
               └── zoomed view inset (bottom-right corner)

Display layout
--------------
┌─────────────────────────────────────────┐
│                                         │
│   Main frame (full view)                │
│   all tracks drawn, selected highlighted│
│   yellow crop-region rectangle          │
│                              ┌─────────┐│
│                              │ zoomed  ││
│                              │  inset  ││
│                              └─────────┘│
│ [banner: click / track lost]            │
└─────────────────────────────────────────┘
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple

from .video_capture    import VideoCapture
from .detection        import DetectionModule, Detection
from .tracker          import TrackingModule, TrackedObject
from .user_interaction import UserInteractionModule
from .zoom_engine      import ZoomEngine
from logs.logger       import FrameLogger
import time
import os


# Palette for track ID coloring (cycles)
_PALETTE = [
    (255,  87,  34),   # deep orange
    ( 33, 150, 243),   # blue
    ( 76, 175,  80),   # green
    (156,  39, 176),   # purple
    (255, 193,   7),   # amber
    (  0, 188, 212),   # cyan
    (244,  67,  54),   # red
    (  0, 150, 136),   # teal
]

def _track_color(track_id: int) -> Tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


# ---------------------------------------------------------------------------
class ZoomTrackingPipeline:
    """
    End-to-end zoom-tracking pipeline.

    Parameters
    ----------
    source              : webcam index (int) or file/RTSP path (str)
    model_path          : Ultralytics YOLO weights (auto-downloaded if absent)
    conf_threshold      : YOLO confidence cutoff
    enable_zoom_redetect: enable/disable Pass-2 feedback detection
    device              : 'cpu', 'cuda', 'mps'
    window_name         : OpenCV window title
    """

    def __init__(
        self,
        source:               int | str = 0,
        model_path:           str       = "yolov8n.pt",
        model_version:        str       = "yolov8n",
        conf_threshold:       float     = 0.35,
        enable_zoom_redetect: bool      = True,
        device:               str       = "cpu",
        window_name:          str       = "Zoom Tracker",
    ):
        self.window_name          = window_name
        self.enable_zoom_redetect = enable_zoom_redetect

        # ── Instantiate modules ───────────────────────────────────────
        self.capture = VideoCapture(source)
        fw, fh       = self.capture.get_frame_size()

        self.detector    = DetectionModule(
            model_path=model_path,
            conf_threshold=conf_threshold,
            device=device,
        )
        self.tracker     = TrackingModule(
            class_names=self.detector.class_names,
            frame_rate=int(self.capture.fps),
        )
        self.interaction = UserInteractionModule(window_name)
        self.zoom_engine = ZoomEngine(frame_width=fw, frame_height=fh)

        # Attach logger using explicitly passed model_version
        self.logger = FrameLogger(model_version=model_version, enable_csv=True)

        # ── Pipeline state ────────────────────────────────────────────
        self._selected_obj:     Optional[TrackedObject] = None
        self._prev_selected_id: Optional[int]           = None

    # ==================================================================
    def run(self):
        """Main loop.  Blocks until 'Q' is pressed or the stream ends."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.interaction.register_mouse_callback()

        print(
            "\n[ZoomTracker] Controls:\n"
            "  Left-click — select an object to track\n"
            "  R          — reset selection\n"
            "  S          — toggle zoom re-detect feedback loop\n"
            "  Q          — quit\n"
        )

        while True:
            t_start = time.time()
            ret, frame = self.capture.read()
            if not ret:
                print("[ZoomTracker] Stream ended or read failure — exiting.")
                break

            self.logger.start_frame(self.capture.frame_count if hasattr(self.capture, 'frame_count') else int(time.time()*30))

            # ── Step 1: Full-frame detection (Pass 1) ─────────────────
            dets_full = self.detector.detect(frame)

            # ── Step 2: Zoomed re-detection (Pass 2 / feedback loop) ──
            dets_zoom:   List[Detection]  = []
            crop_region: Tuple            = (0, 0, self.capture.width, self.capture.height)
            zoom_level:  float            = 1.0
            zoom_view:   Optional[np.ndarray] = None

            if self._selected_obj is not None:
                zoom_view, crop_region, zoom_level = self.zoom_engine.apply_zoom(
                    frame, self._selected_obj
                )
                if self.enable_zoom_redetect:
                    dets_zoom = self.detector.detect(zoom_view)

            # ── Step 3: Merge + NMS + update tracker ──────────────────
            dets_arr      = self._merge_and_nms(dets_full, dets_zoom, crop_region)
            tracked_objs  = self.tracker.update(dets_arr, frame.shape)

            # ── Step 4: User interaction ───────────────────────────────
            selected_id  = self.interaction.process_click(tracked_objs)
            track_alive  = self.interaction.check_track_alive(tracked_objs)

            # Reset zoom smoothing when selection changes
            if selected_id != self._prev_selected_id:
                self.zoom_engine.reset()
                self._prev_selected_id = selected_id

            # Update selected-object reference for next frame
            self._selected_obj = None
            if selected_id is not None and track_alive:
                for obj in tracked_objs:
                    if obj.track_id == selected_id:
                        self._selected_obj = obj
                        
                        # Convert bbox [x1, y1, x2, y2] -> [x, y, w, h]
                        x1, y1, x2, y2 = obj.bbox.tolist()
                        w, h = x2 - x1, y2 - y1
                        bbox_xywh = [float(x1), float(y1), float(w), float(h)]

                        # Log telemetry for tracked object
                        self.logger.log_detection(bbox=bbox_xywh, confidence=float(obj.confidence), track_id=obj.track_id)
                        
                        obj_size, center_err = FrameLogger.compute_derived_metrics(
                            bbox=bbox_xywh, frame_width=self.capture.width, frame_height=self.capture.height
                        )
                        self.logger.log_metrics(obj_size, center_err)

                        break

            # Always log the current zoom level, even if target momentarily lost
            self.logger.log_zoom(float(zoom_level))

            # Measure latency and save frame logs
            latency_ms = (time.time() - t_start) * 1000.0
            self.logger.end_frame(latency_ms)

            # ── Step 5: Visualize ──────────────────────────────────────
            display = self._build_display(
                frame, tracked_objs, selected_id, crop_region, zoom_level, zoom_view
            )
            self.interaction.draw_ui_hints(display)
            cv2.imshow(self.window_name, display)

            # ── Key handling ───────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.interaction.reset_selection()
                self.zoom_engine.reset()
                self._selected_obj = None
            elif key == ord('s'):
                self.enable_zoom_redetect = not self.enable_zoom_redetect
                state = "ON" if self.enable_zoom_redetect else "OFF"
                print(f"[ZoomTracker] Zoom re-detect feedback: {state}")

        self.logger.flush()
        self.capture.release()
        cv2.destroyAllWindows()

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _merge_and_nms(
        self,
        dets_full: List[Detection],
        dets_zoom: List[Detection],
        crop_region: Tuple,
    ) -> np.ndarray:
        """
        Combine full-frame and zoomed detections, remap zoomed ones to
        original coordinates, then suppress duplicates with NMS.

        Returns (N, 6) array ready for TrackingModule.update().
        """
        arr_full = self.detector.to_bytetrack_input(dets_full)

        if dets_zoom:
            arr_zoom = self.detector.to_bytetrack_input(dets_zoom)
            # Remap each zoomed detection back to original-frame space
            remapped = []
            for row in arr_zoom:
                orig_bbox = self.zoom_engine.map_bbox_to_original(row[:4], crop_region)
                remapped.append(np.concatenate([orig_bbox, row[4:]]))
            arr_zoom_mapped = np.array(remapped, dtype=np.float32)

            combined = (
                np.vstack([arr_full, arr_zoom_mapped])
                if arr_full.shape[0] > 0
                else arr_zoom_mapped
            )
        else:
            combined = arr_full

        if combined.shape[0] <= 1:
            return combined

        return self._nms(combined, iou_threshold=0.50)

    # ------------------------------------------------------------------
    def _nms(self, dets_arr: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
        """
        Apply class-agnostic Non-Maximum Suppression to remove duplicate
        detections that arise from merging full-frame and zoomed passes.

        Uses cv2.dnn.NMSBoxes which expects [x, y, w, h] format.
        """
        boxes  = dets_arr[:, :4]
        scores = dets_arr[:, 4]

        xywh = np.column_stack([
            boxes[:, 0],
            boxes[:, 1],
            boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
        ])

        indices = cv2.dnn.NMSBoxes(
            xywh.tolist(),
            scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold,
        )

        if len(indices) == 0:
            return dets_arr
        return dets_arr[np.array(indices).flatten()]

    # ------------------------------------------------------------------
    def _build_display(
        self,
        frame:        np.ndarray,
        tracked_objs: List[TrackedObject],
        selected_id:  Optional[int],
        crop_region:  Tuple,
        zoom_level:   float,
        zoom_view:    Optional[np.ndarray],
    ) -> np.ndarray:
        display = frame.copy()

        # Draw all tracks
        for obj in tracked_objs:
            self._draw_track(display, obj, obj.track_id == selected_id)

        # Crop-region indicator + zoom label
        if self._selected_obj is not None and zoom_level > 1.01:
            cx1, cy1, cx2, cy2 = crop_region
            cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (0, 230, 230), 2)
            cv2.putText(display, f"ZOOM {zoom_level:.1f}x",
                        (cx1 + 4, cy1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 230), 2, cv2.LINE_AA)

            # Zoom re-detect status badge
            badge_txt   = "REDETECT ON" if self.enable_zoom_redetect else "REDETECT OFF"
            badge_color = (0, 200, 80) if self.enable_zoom_redetect else (60, 60, 60)
            cv2.putText(display, badge_txt, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_color, 2, cv2.LINE_AA)

        # Zoomed inset (bottom-right corner, 1/3 of frame dimensions)
        if zoom_view is not None and self._selected_obj is not None:
            h, w  = display.shape[:2]
            iw, ih = w // 3, h // 3
            inset  = cv2.resize(zoom_view, (iw, ih))

            # Border and label on inset
            cv2.rectangle(inset, (0, 0), (iw - 1, ih - 1), (0, 230, 230), 2)
            cv2.putText(inset, f"{zoom_level:.1f}x",
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 230), 2, cv2.LINE_AA)

            display[h - ih: h, w - iw: w] = inset

        return display

    # ------------------------------------------------------------------
    def _draw_track(
        self,
        frame:      np.ndarray,
        obj:        TrackedObject,
        is_selected: bool,
    ):
        """Draw bbox, label, and (if selected) corner highlights."""
        x1, y1, x2, y2 = obj.bbox.astype(int)
        color     = _track_color(obj.track_id)
        thickness = 3 if is_selected else 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"#{obj.track_id} {obj.class_name} {obj.confidence:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 6), (x1 + lw + 2, y1), color, -1)
        cv2.putText(frame, label, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

        if is_selected:
            _draw_corners(frame, x1, y1, x2, y2, color=(0, 255, 80), sz=16, thickness=3)


# ---------------------------------------------------------------------------
def _draw_corners(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color=(0, 255, 0), sz: int = 16, thickness: int = 3,
):
    """Draw L-shaped corner markers at each corner of a rectangle."""
    corners = [
        ((x1, y1), (x1 + sz, y1), (x1, y1 + sz)),
        ((x2, y1), (x2 - sz, y1), (x2, y1 + sz)),
        ((x1, y2), (x1 + sz, y2), (x1, y2 - sz)),
        ((x2, y2), (x2 - sz, y2), (x2, y2 - sz)),
    ]
    for corner, horiz, vert in corners:
        cv2.line(frame, corner, horiz, color, thickness)
        cv2.line(frame, corner, vert,  color, thickness)
