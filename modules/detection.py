"""
Module 2 — Detection
=====================
Wraps Ultralytics YOLO (nano by default) and returns typed Detection objects.
The module is intentionally stateless so it can be called on both the full
frame *and* the zoomed crop without side-effects.

Detection → ByteTrack boundary
-------------------------------
ByteTrack expects detections in shape (N, 6) where columns are:
    [x1, y1, x2, y2, confidence, class_id]
`to_bytetrack_input()` converts the list of Detection objects into that array.
This clean boundary means the TrackingModule never imports from Ultralytics
directly and can be swapped for a different tracker without touching detection.

Design decisions
----------------
- `target_classes`: pass a list of COCO class IDs (e.g. [0] for person only)
  to reduce spurious detections and lower inference cost.
- conf_threshold / iou_threshold are stored as instance attributes so the
  pipeline can adjust them at runtime (e.g. lower threshold on zoomed frames
  where the object is larger and easier to detect).
- verbose=False suppresses Ultralytics' per-frame console spam.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from ultralytics import YOLO


@dataclass
class Detection:
    """Single detection output from YOLO."""
    bbox:       np.ndarray   # [x1, y1, x2, y2] float32, pixel coordinates
    confidence: float
    class_id:   int
    class_name: str


class DetectionModule:
    def __init__(
        self,
        model_path:      str              = "yolov8n.pt",
        conf_threshold:  float            = 0.35,
        iou_threshold:   float            = 0.45,
        target_classes:  Optional[List[int]] = None,
        device:          str              = "cpu",
    ):
        self.model          = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.target_classes = target_classes   # None → detect all classes
        self.device         = device
        self.class_names: Dict[int, str] = self.model.names  # {0: "person", …}

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray  shape (H, W, 3), dtype uint8, BGR

        Returns
        -------
        list of Detection — may be empty if nothing found
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.target_classes,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                detections.append(Detection(
                    bbox=boxes.xyxy[i].cpu().numpy().astype(np.float32),
                    confidence=float(boxes.conf[i].cpu()),
                    class_id=int(boxes.cls[i].cpu()),
                    class_name=self.class_names.get(int(boxes.cls[i].cpu()), "?"),
                ))
        return detections

    # ------------------------------------------------------------------
    def to_bytetrack_input(self, detections: List[Detection]) -> np.ndarray:
        """
        Convert detections to a (N, 6) float32 array:
            [x1, y1, x2, y2, confidence, class_id]
        Returns empty (0, 6) array when there are no detections.
        """
        if not detections:
            return np.empty((0, 6), dtype=np.float32)
        return np.array(
            [[*d.bbox, d.confidence, float(d.class_id)] for d in detections],
            dtype=np.float32,
        )
