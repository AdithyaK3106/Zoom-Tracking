"""
Module 4 — User Interaction
============================
Handles all user input: mouse clicks to select a tracked object, keyboard
shortcuts, and on-screen UI hints.

Click-to-track matching strategy
---------------------------------
Priority 1 — inside-bbox click:
    If the click (x, y) falls inside a bounding box, select the closest
    bbox *center* among all boxes that contain the click.  This handles
    overlapping detections (pick the "most centered" one).

Priority 2 — nearest-center fallback:
    If no box contains the click, find the nearest bbox center within
    MAX_SNAP_PX pixels.  This tolerates imprecise clicks on small objects.

Re-selection on track loss
---------------------------
`check_track_alive()` is called every frame by the pipeline.  When the
selected track_id is no longer in the active-tracks list, `_pending_reselect`
is set and the UI banner switches to "Track LOST — click to re-select".
The `selected_track_id` is preserved until the user clicks a new one, so
the pipeline can display the last known state.

Design decisions
----------------
- setMouseCallback fires in the OpenCV event loop; we only store the click
  position here and consume it lazily in `process_click()` during the main
  loop.  This avoids race conditions between the callback thread and the
  pipeline thread.
- The window_name must match the cv2.namedWindow() name used in the pipeline.
"""

import cv2
import numpy as np
from typing import Optional, List
from .tracker import TrackedObject


MAX_SNAP_PX = 80   # max pixels from bbox center for fallback snap


class UserInteractionModule:
    def __init__(self, window_name: str = "Zoom Tracker"):
        self.window_name      = window_name
        self.selected_track_id: Optional[int] = None

        self._click_pos:        Optional[tuple] = None   # (x, y) raw pixel
        self._needs_selection:  bool            = True
        self._pending_reselect: bool            = False

    # ------------------------------------------------------------------
    def register_mouse_callback(self):
        """Attach mouse handler.  Call after cv2.namedWindow() has been created."""
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        # Only care about left-button press; store position for lazy processing
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_pos = (x, y)

    # ------------------------------------------------------------------
    def process_click(self, tracked_objects: List[TrackedObject]) -> Optional[int]:
        """
        Consume any pending click and resolve it to the best matching track.

        Should be called once per frame *after* tracker.update().

        Returns
        -------
        The currently selected track_id (may be unchanged if no click was pending).
        """
        if self._click_pos is None:
            return self.selected_track_id

        cx, cy = self._click_pos
        self._click_pos = None            # consume

        best_id    = None
        best_score = float("inf")

        # Priority 1: click inside a bounding box
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                ocx, ocy = obj.bbox_center
                dist = _dist2(cx, cy, ocx, ocy)
                if dist < best_score:
                    best_score = dist
                    best_id    = obj.track_id

        # Priority 2: nearest center within snap radius
        if best_id is None:
            for obj in tracked_objects:
                ocx, ocy = obj.bbox_center
                dist = _dist2(cx, cy, ocx, ocy) ** 0.5
                if dist < min(best_score, MAX_SNAP_PX):
                    best_score = dist
                    best_id    = obj.track_id

        if best_id is not None:
            if best_id != self.selected_track_id:
                # New selection: caller should reset ZoomEngine smoothing
                self._notify_new_selection = True
            self.selected_track_id  = best_id
            self._needs_selection   = False
            self._pending_reselect  = False

        return self.selected_track_id

    # ------------------------------------------------------------------
    def check_track_alive(self, tracked_objects: List[TrackedObject]) -> bool:
        """
        Returns True if the currently selected track is still active.
        Sets the re-selection flag (and the UI banner) if the track was lost.
        """
        if self.selected_track_id is None:
            return False

        active_ids = {obj.track_id for obj in tracked_objects}
        alive = self.selected_track_id in active_ids

        if not alive and not self._pending_reselect:
            self._pending_reselect = True
            self._needs_selection  = True

        return alive

    # ------------------------------------------------------------------
    def needs_selection(self) -> bool:
        return self._needs_selection

    # ------------------------------------------------------------------
    def draw_ui_hints(self, frame: np.ndarray) -> np.ndarray:
        """Render instruction banner at the bottom of the frame (in-place)."""
        h, w = frame.shape[:2]

        if self._needs_selection:
            msg   = "Track LOST — click to re-select" if self._pending_reselect \
                    else "CLICK on an object to track  |  R = reset  |  Q = quit"
            color = (0, 80, 220) if self._pending_reselect else (20, 20, 20)

            # Semi-transparent bar
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 36), (w, h), color, -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.putText(frame, msg, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1,
                        cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    def reset_selection(self):
        """Externally force re-selection (e.g. via 'R' key)."""
        self.selected_track_id = None
        self._needs_selection  = True
        self._pending_reselect = False
        self._click_pos        = None


# ---------------------------------------------------------------------------
def _dist2(ax, ay, bx, by) -> float:
    """Squared Euclidean distance — avoids sqrt for comparison."""
    return (ax - bx) ** 2 + (ay - by) ** 2
