"""
Module 1 — Video Capture
========================
Wraps OpenCV VideoCapture to provide a unified interface over webcams,
video files, and RTSP streams.  Downstream modules only depend on:
  - read()           → (bool, np.ndarray | None)
  - get_frame_size() → (width, height)
  - fps              → float

Design decisions
----------------
- Hardware-level resize is requested at init to reduce bandwidth before
  frames ever reach Python; the *actual* dimensions are read back because
  the driver may ignore the request (e.g. fixed-resolution cameras).
- context-manager support (with VideoCapture(...) as vc) ensures release()
  is always called even if the pipeline raises.
- frame_area is pre-computed once here so the ZoomEngine can reference it
  without recomputing every frame.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class VideoCapture:
    def __init__(
        self,
        source: int | str = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
    ):
        self.source = source
        self.cap = cv2.VideoCapture(source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")

        # Request hardware-level settings (best-effort)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)

        # Read back what the driver actually gave us
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps    = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_area = self.width * self.height

    # ------------------------------------------------------------------
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Grab next frame.  Returns (success, BGR frame | None)."""
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def get_frame_size(self) -> Tuple[int, int]:
        """Returns (width, height)."""
        return self.width, self.height

    def release(self):
        self.cap.release()

    # context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
