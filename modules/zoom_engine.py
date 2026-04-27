"""
Module 5 — Zoom Engine
=======================
Implements dynamic zoom using bounding box area as a depth proxy.

Depth proxy
-----------
    bbox_ratio = bbox_area / frame_area     (0 < bbox_ratio ≤ 1)

A small bbox_ratio means the object appears small → it is far away → zoom in.
A large bbox_ratio means the object is close → zoom level approaches 1×.

Zoom formula
------------
    raw_zoom = REFERENCE_RATIO / bbox_ratio

REFERENCE_RATIO is the bbox_ratio that produces exactly 1× zoom.  Set it to
the fraction of the frame you expect the object to fill when it is at a
"comfortable" distance (default 0.08 → object fills ~8 % of frame area).

    raw_zoom is then clamped to [MIN_ZOOM, MAX_ZOOM].

Smoothing
---------
Both zoom level and the crop center are passed through an exponential moving
average (EMA) to eliminate jitter from bbox noise:
    smoothed = α * new_value + (1 - α) * previous_value
α ≈ 0.12 gives a roughly 8-frame lag, which feels natural at 30 fps.

Crop → resize pipeline
----------------------
1. Compute crop dimensions: frame_size / zoom  (the inverse relationship that
   gives magnification).  Clamp so the crop is never smaller than a padded
   version of the raw bbox.
2. Center the crop on the smoothed bbox center, clamped to frame edges.
3. cv2.resize the crop back to the original frame dimensions → zoomed_frame.

Coordinate remapping
--------------------
`map_bbox_to_original()` inverts the crop+resize operation so that detections
from a zoomed frame can be fed back to the tracker in original-frame space.

    scale_x = crop_width  / frame_width
    scale_y = crop_height / frame_height
    orig_x  = zoomed_x * scale_x + crop_x1
    orig_y  = zoomed_y * scale_y + crop_y1
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from .tracker import TrackedObject


class ZoomEngine:
    # Class-level defaults — override per instance via constructor
    MIN_ZOOM         = 1.0
    MAX_ZOOM         = 6.0
    REFERENCE_RATIO  = 0.08   # bbox/frame area that maps to 1× zoom
    PADDING_FACTOR   = 2.5    # crop is at least PADDING_FACTOR × bbox dims
    SMOOTH_ALPHA     = 0.12   # EMA weight for new values (lower = smoother)

    def __init__(
        self,
        frame_width:     int,
        frame_height:    int,
        min_zoom:        float = MIN_ZOOM,
        max_zoom:        float = MAX_ZOOM,
        reference_ratio: float = REFERENCE_RATIO,
        padding_factor:  float = PADDING_FACTOR,
        smooth_alpha:    float = SMOOTH_ALPHA,
    ):
        self.fw             = frame_width
        self.fh             = frame_height
        self.frame_area     = float(frame_width * frame_height)
        self.min_zoom       = min_zoom
        self.max_zoom       = max_zoom
        self.reference_ratio = reference_ratio
        self.padding_factor = padding_factor
        self.smooth_alpha   = smooth_alpha

        # Smoothing state — reset when switching tracked object
        self._s_zoom: float           = 1.0
        self._s_cx:   Optional[float] = None
        self._s_cy:   Optional[float] = None

    # ------------------------------------------------------------------
    def compute_zoom(self, obj: TrackedObject) -> float:
        """
        Compute raw (un-smoothed) zoom from current bbox area.
        Small bbox → large zoom.  Result clamped to [min_zoom, max_zoom].
        """
        bbox_ratio = obj.bbox_area / self.frame_area
        raw = self.reference_ratio / max(bbox_ratio, 1e-6)
        return float(np.clip(raw, self.min_zoom, self.max_zoom))

    # ------------------------------------------------------------------
    def apply_zoom(
        self,
        frame: np.ndarray,
        obj:   TrackedObject,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
        """
        Main entry point.  Computes smoothed zoom, crops, and resizes.

        Parameters
        ----------
        frame : original BGR frame (H, W, 3)
        obj   : TrackedObject for the currently selected track

        Returns
        -------
        zoomed_frame : BGR frame same size as input, cropped+resized
        crop_region  : (x1, y1, x2, y2) in *original* frame pixels
        zoom_level   : smoothed zoom scalar applied this frame
        """
        # ── Smooth zoom level ──────────────────────────────────────────
        raw_zoom     = self.compute_zoom(obj)
        self._s_zoom = self._ema(raw_zoom, self._s_zoom)
        zoom         = self._s_zoom

        # ── Smooth crop center ─────────────────────────────────────────
        cx, cy = obj.bbox_center
        if self._s_cx is None:
            self._s_cx, self._s_cy = cx, cy
        else:
            self._s_cx = self._ema(cx, self._s_cx)
            self._s_cy = self._ema(cy, self._s_cy)

        scx, scy = self._s_cx, self._s_cy

        # ── Compute crop dimensions ────────────────────────────────────
        # Inverse zoom: smaller crop → more magnification when resized back
        bw, bh = obj.bbox_wh
        crop_w = max(self.fw / zoom, bw * self.padding_factor)
        crop_h = max(self.fh / zoom, bh * self.padding_factor)

        # Keep crop square-ish by taking the max side, then clamp to frame
        crop_w = min(crop_w, self.fw)
        crop_h = min(crop_h, self.fh)

        # ── Compute crop coordinates, centered on smoothed center ──────
        x1 = int(np.clip(scx - crop_w / 2, 0, self.fw - 1))
        y1 = int(np.clip(scy - crop_h / 2, 0, self.fh - 1))
        x2 = int(np.clip(x1  + crop_w,     1, self.fw))
        y2 = int(np.clip(y1  + crop_h,     1, self.fh))

        # Guard against degenerate crops
        if x2 - x1 < 4: x2 = min(x1 + 4, self.fw)
        if y2 - y1 < 4: y2 = min(y1 + 4, self.fh)

        # ── Crop and resize back to original frame dimensions ──────────
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return frame.copy(), (0, 0, self.fw, self.fh), 1.0

        zoomed = cv2.resize(crop, (self.fw, self.fh), interpolation=cv2.INTER_LINEAR)
        return zoomed, (x1, y1, x2, y2), zoom

    # ------------------------------------------------------------------
    def map_bbox_to_original(
        self,
        bbox_in_zoomed: np.ndarray,
        crop_region:    Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Inverse the crop+resize transform to convert a bounding box detected
        in the zoomed frame back to coordinates in the original frame.

        Derivation:
            For a crop (cx1, cy1, cx2, cy2) resized to (fw, fh):
                scale_x = (cx2 - cx1) / fw
                scale_y = (cy2 - cy1) / fh
            So for any point p_z in zoomed space:
                p_orig = p_z * scale + crop_origin

        Parameters
        ----------
        bbox_in_zoomed : [x1, y1, x2, y2] in zoomed-frame pixel coordinates
        crop_region    : (cx1, cy1, cx2, cy2) that produced the zoomed frame

        Returns
        -------
        np.ndarray [x1, y1, x2, y2] in original-frame coordinates
        """
        cx1, cy1, cx2, cy2 = crop_region
        sx = (cx2 - cx1) / self.fw
        sy = (cy2 - cy1) / self.fh

        ox1 = bbox_in_zoomed[0] * sx + cx1
        oy1 = bbox_in_zoomed[1] * sy + cy1
        ox2 = bbox_in_zoomed[2] * sx + cx1
        oy2 = bbox_in_zoomed[3] * sy + cy1

        return np.array([ox1, oy1, ox2, oy2], dtype=np.float32)

    # ------------------------------------------------------------------
    def reset(self):
        """Reset EMA state.  Call when the selected track changes."""
        self._s_zoom = 1.0
        self._s_cx   = None
        self._s_cy   = None

    # ------------------------------------------------------------------
    def _ema(self, new_val: float, old_val: float) -> float:
        return self.smooth_alpha * new_val + (1.0 - self.smooth_alpha) * old_val
