"""
Module 3 — Tracking (ByteTrack)
================================
Wraps `supervision.ByteTracker` to provide persistent track IDs across frames.

Why supervision.ByteTracker?
-----------------------------
Ultralytics ships ByteTrack internally but only exposes it through
`model.track()`, which couples detection and tracking.  The `supervision`
library re-exports the same ByteTrack algorithm with a clean, standalone API
that takes a `sv.Detections` object (trivially constructed from our arrays)
and returns an updated `sv.Detections` with `tracker_id` populated.
This keeps detection and tracking as independent, swappable modules.

TrackedObject
-------------
A frozen snapshot of one track at the current frame:
  - track_id   : persistent integer, stable across frames
  - bbox        : [x1, y1, x2, y2] in the coordinate space of the frame
                  that was passed to update()  ← important for the ZoomEngine
  - bbox_area   : cached (x2-x1)*(y2-y1) — used by ZoomEngine for depth proxy
  - bbox_center : (cx, cy) used by ZoomEngine crop centering
  - age         : frames since this track was first seen (staleness guard)

Track-loss detection
--------------------
`_track_ages` maps track_id → age.  When a track_id disappears from the
ByteTracker output (object gone or occluded), its entry is removed.
The pipeline checks this dict to detect loss and trigger re-selection.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import supervision as sv


@dataclass
class TrackedObject:
    track_id:   int
    bbox:       np.ndarray   # [x1, y1, x2, y2] float32
    confidence: float
    class_id:   int
    class_name: str
    age:        int = 0      # frames since first seen

    @property
    def bbox_area(self) -> float:
        w = float(self.bbox[2] - self.bbox[0])
        h = float(self.bbox[3] - self.bbox[1])
        return max(w * h, 1.0)   # guard against zero

    @property
    def bbox_center(self) -> Tuple[float, float]:
        cx = (self.bbox[0] + self.bbox[2]) / 2.0
        cy = (self.bbox[1] + self.bbox[3]) / 2.0
        return float(cx), float(cy)

    @property
    def bbox_wh(self) -> Tuple[float, float]:
        return float(self.bbox[2] - self.bbox[0]), float(self.bbox[3] - self.bbox[1])


class TrackingModule:
    def __init__(
        self,
        track_activation_threshold: float            = 0.25,
        lost_track_buffer:          int              = 30,
        minimum_matching_threshold: float            = 0.8,
        minimum_consecutive_frames: int              = 1,
        frame_rate:                 int              = 30,
        class_names:                Optional[Dict[int, str]] = None,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            frame_rate=frame_rate,
        )
        self.class_names:   Dict[int, str] = class_names or {}
        self._track_ages:   Dict[int, int] = {}

    # ------------------------------------------------------------------
    def update(
        self,
        detections_arr: np.ndarray,   # (N, 6): [x1,y1,x2,y2,score,class_id]
        frame_shape:    tuple,        # (H, W, C) — kept for API symmetry
    ) -> List[TrackedObject]:
        """
        Feed new detections into ByteTrack.

        Parameters
        ----------
        detections_arr : (N, 6) array from DetectionModule.to_bytetrack_input()
                         or the merged output from the pipeline
        frame_shape    : original frame shape (unused internally but kept for
                         future trackers that need it)

        Returns
        -------
        List[TrackedObject] — only *active* tracks this frame
        """
        if detections_arr.shape[0] == 0:
            sv_det = sv.Detections.empty()
        else:
            sv_det = sv.Detections(
                xyxy=detections_arr[:, :4],
                confidence=detections_arr[:, 4],
                class_id=detections_arr[:, 5].astype(int),
            )

        tracked_sv = self.tracker.update_with_detections(sv_det)

        active_ids: set = set()
        tracked_objects: List[TrackedObject] = []

        if tracked_sv.tracker_id is not None:
            for i in range(len(tracked_sv)):
                tid = int(tracked_sv.tracker_id[i])
                active_ids.add(tid)
                self._track_ages[tid] = self._track_ages.get(tid, 0) + 1

                cls_id = int(tracked_sv.class_id[i]) if tracked_sv.class_id is not None else 0
                conf   = float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 0.0

                tracked_objects.append(TrackedObject(
                    track_id=tid,
                    bbox=tracked_sv.xyxy[i].astype(np.float32),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=self.class_names.get(cls_id, str(cls_id)),
                    age=self._track_ages[tid],
                ))

        # Prune age table for tracks that disappeared
        for gone_id in set(self._track_ages) - active_ids:
            del self._track_ages[gone_id]

        return tracked_objects

    # ------------------------------------------------------------------
    def reset(self):
        """Reset tracker state — useful after scene cuts or source changes."""
        try:
            self.tracker.reset()
        except AttributeError:
            # Older supervision versions don't expose reset(); re-instantiate.
            self.tracker = sv.ByteTrack(
                track_activation_threshold=self.tracker.track_activation_threshold,
                lost_track_buffer=self.tracker.lost_track_buffer,
                minimum_matching_threshold=self.tracker.minimum_matching_threshold,
                frame_rate=self.tracker.frame_rate,
            )
        self._track_ages.clear()
