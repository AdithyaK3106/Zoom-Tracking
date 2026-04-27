import json
import math
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import os

class EvaluationMetrics:
    def __init__(self, iou_threshold: float = 0.5, conf_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        """Computes IoU between two boxes [x, y, w, h]"""
        x1_min, y1_min, w1, h1 = box1
        x2_min, y2_min, w2, h2 = box2
        
        x1_max, y1_max = x1_min + w1, y1_min + h1
        x2_max, y2_max = x2_min + w2, y2_min + h2

        intersect_x_min = max(x1_min, x2_min)
        intersect_y_min = max(y1_min, y2_min)
        intersect_x_max = min(x1_max, x2_max)
        intersect_y_max = min(y1_max, y2_max)

        intersect_w = max(0.0, intersect_x_max - intersect_x_min)
        intersect_h = max(0.0, intersect_y_max - intersect_y_min)
        intersect_area = intersect_w * intersect_h

        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - intersect_area

        return intersect_area / union_area if union_area > 0 else 0.0

    def compute_all_metrics(self, gt_frames: List[Dict[str, Any]], pred_frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes all standard and system-specific metrics defined in the research plan.
        gt_frames and pred_frames should be sorted by frame_id.
        """
        if not gt_frames or not pred_frames:
            return {}

        ious = []
        confidences = []
        zoom_factors = []
        center_errors = []
        latencies = []
        track_ids = []
        object_sizes = []
        
        # Track Continuity & Fragmentation
        is_tracking = False
        fragmentation_count = 0
        current_streak = 0
        max_streak = 0
        
        # False tracking count
        false_tracking_frames = 0
        
        # Create lookup for O(1) frame matching
        pred_map = {p.get("frame_id"): p for p in pred_frames}
        
        for gt in gt_frames:
            frame_id = gt["frame_id"]
            gt_bbox = gt["bbox"]
            is_occluded = gt.get("occlusion_flag", False)
            
            pred = pred_map.get(frame_id, {})
            pred_bbox = pred.get("bbox")
            conf = pred.get("confidence", 0.0)
            tid = pred.get("track_id")
            zf = pred.get("zoom_factor", 1.0)
            ce = pred.get("center_error", 0.0)
            lat = pred.get("latency_ms", 0.0)
            osize = pred.get("object_size", 0.0)
            
            latencies.append(lat)
            zoom_factors.append(zf)
            if ce is not None:
                center_errors.append(ce)
            
            if pred_bbox and gt_bbox:
                iou = self.compute_iou(gt_bbox, pred_bbox)
                ious.append(iou)
                confidences.append(conf)
                track_ids.append(tid)
                object_sizes.append(osize)
                
                # False tracking: locking onto wrong object
                if tid is not None and iou < 0.1 and not is_occluded:
                    false_tracking_frames += 1

                # Continuity / Fragmentation Logic
                if tid is not None and iou >= self.iou_threshold:
                    if not is_tracking:
                        is_tracking = True
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    if is_tracking and not is_occluded:
                        is_tracking = False
                        fragmentation_count += 1
                        current_streak = 0
            else:
                ious.append(0.0)
                if is_tracking and not is_occluded:
                    is_tracking = False
                    fragmentation_count += 1
                    current_streak = 0
                    
        # Array conversions for math
        ious_np = np.array(ious)
        conf_np = np.array(confidences)
        zoom_np = np.array(zoom_factors)
        ce_np = np.array(center_errors)
        lat_np = np.array(latencies)
        
        total_frames = len(gt_frames)
        
        # 1. Detection Consistency
        if len(ious_np) > 0 and len(conf_np) > 0:
            consistent_frames = np.sum((ious_np > self.iou_threshold) & (conf_np > self.conf_threshold))
            consistency = float(consistent_frames / total_frames)
        else:
            consistency = 0.0
            
        # 2. Zoom Stability
        zoom_variance = float(np.var(zoom_np)) if len(zoom_np) > 0 else 0.0
        zoom_mad = float(np.mean(np.abs(np.diff(zoom_np)))) if len(zoom_np) > 1 else 0.0
        
        # 3. Temporal Metrics
        continuity_score = float(max_streak / total_frames) if total_frames > 0 else 0.0
        false_tracking_rate = float(false_tracking_frames / total_frames) if total_frames > 0 else 0.0
        
        # Compile results
        results = {
            "iou_mean": float(np.mean(ious_np)) if len(ious_np) > 0 else 0.0,
            "iou_std": float(np.std(ious_np)) if len(ious_np) > 0 else 0.0,
            "consistency": consistency,
            "fragmentation": fragmentation_count,
            "continuity": continuity_score,
            "zoom_variance": zoom_variance,
            "zoom_mad": zoom_mad,
            "center_error_avg": float(np.mean(ce_np)) if len(ce_np) > 0 else 0.0,
            "false_tracking_rate": false_tracking_rate,
            "latency_ms": float(np.mean(lat_np)) if len(lat_np) > 0 else 0.0
        }
        
        return results

    def compare_models(self, baseline_metrics: Dict[str, Any], adaptive_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculates Zoom Gain between baseline and adaptive models."""
        base_iou = baseline_metrics.get("iou_mean", 0.0)
        adap_iou = adaptive_metrics.get("iou_mean", 0.0)
        
        abs_gain = adap_iou - base_iou
        rel_gain = (abs_gain / base_iou) if base_iou > 0 else 0.0
        
        return {
            "zoom_gain_abs": abs_gain,
            "zoom_gain_rel": rel_gain
        }

if __name__ == "__main__":
    # Test block
    print("Metrics module loaded successfully.")
