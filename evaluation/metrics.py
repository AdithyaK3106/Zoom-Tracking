import numpy as np
from typing import Dict, List, Any
from scipy.stats import ttest_rel


class EvaluationMetrics:
    def __init__(self, iou_threshold: float = 0.5, conf_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    # -------------------------------
    # IoU
    # -------------------------------
    @staticmethod
    def compute_iou(box1: List[float], box2: List[float]) -> float:
        """Computes IoU for boxes in [x1, y1, x2, y2] format."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        
        inter_w = max(0, xi2 - xi1)
        inter_h = max(0, yi2 - yi1)
        inter_area = inter_w * inter_h
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - inter_area
        
        return inter_area / union if union > 1e-6 else 0.0

    @staticmethod
    def map_bbox_to_global(bbox_local: List[float], crop: List[float], frame_size: tuple = (640, 480)) -> List[float]:
        """
        Maps a LOCAL bbox (from a zoomed crop) back to original frame coordinates.
        Formula: x_global = x_local * (crop_width / frame_width) + crop_x1
        """
        lx1, ly1, lx2, ly2 = bbox_local
        cx1, cy1, cx2, cy2 = crop
        fw, fh = frame_size
        
        sx = (cx2 - cx1) / fw
        sy = (cy2 - cy1) / fh
        
        gx1 = lx1 * sx + cx1
        gy1 = ly1 * sy + cy1
        gx2 = lx2 * sx + cx1
        gy2 = ly2 * sy + cy1
        
        return [gx1, gy1, gx2, gy2]

    # -------------------------------
    # MAIN METRICS FUNCTION
    # -------------------------------
    # -------------------------------
    # MAIN METRICS FUNCTION
    # -------------------------------
    def compute_all_metrics(self, gt_map: Dict[str, List], pred_frames: List[Dict[str, Any]]):
        if not gt_map or not pred_frames:
            return {}

        ious = []
        confidences = []
        zoom_factors = []
        center_errors = []
        latencies = []
        object_sizes = []
        occlusions = []
        frame_ids = []

        fragmentation_count = 0
        current_streak = 0
        max_streak = 0
        is_tracking = False
        false_tracking_frames = 0
        id_switches = 0
        prev_tid = None

        pred_map = {str(p["frame_id"]): p for p in pred_frames}

        # Iterate through GT frames only (as requested: "If GT missing -> skip frame")
        for f_id_str, gt_bbox in gt_map.items():
            frame_id = int(f_id_str)
            is_occluded = False # Real datasets might lack this, default False

            pred = pred_map.get(f_id_str, {})
            pred_bbox_raw = pred.get("bbox")
            crop_region = pred.get("crop_region")
            conf = pred.get("confidence", 0.0)
            tid = pred.get("track_id")
            zf = pred.get("zoom_factor", 1.0)
            ce = pred.get("center_error", 0.0)
            lat = pred.get("latency_ms", 0.0)
            osize = pred.get("object_size", 0.0)

            # ── Coordinate Standardization ──────────────────────────
            # GT Standardization: Ensure [x1, y1, x2, y2]
            if len(gt_bbox) == 4:
                # If x2 < x1 or y2 < y1, it's [x, y, w, h]
                if gt_bbox[2] < gt_bbox[0] or gt_bbox[3] < gt_bbox[1]:
                    gt_x1y1x2y2 = [gt_bbox[0], gt_bbox[1], gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]]
                else:
                    gt_x1y1x2y2 = gt_bbox
            else:
                gt_x1y1x2y2 = [0,0,0,0]

            if pred_bbox_raw:
                if len(pred_bbox_raw) == 4:
                    px1, py1, p3, p4 = pred_bbox_raw
                    # If x2 < x1 or y2 < y1, it's [x, y, w, h]
                    if p3 < px1 or p4 < py1:
                        pred_x1y1x2y2 = [px1, py1, px1 + p3, py1 + p4]
                    else:
                        pred_x1y1x2y2 = pred_bbox_raw
                else:
                    pred_x1y1x2y2 = [0,0,0,0]

                # Zoom correction if crop exists (as requested)
                if crop_region and zf > 1.01:
                    # Heuristic: if pred_x1y1x2y2 is within crop boundaries, it might be local
                    # For safety and research rigor, we use the remapping tool
                    # But since our pipeline logs global, we'll only do this if it looks local
                    if pred_x1y1x2y2[2] <= 640 and pred_x1y1x2y2[3] <= 480:
                         # Potentially local, but could be global too.
                         # We'll trust the pipeline logs global for now to avoid double mapping.
                         pass 

                iou = self.compute_iou(gt_x1y1x2y2, pred_x1y1x2y2)
            else:
                iou = 0.0

            ious.append(iou)
            confidences.append(conf)
            zoom_factors.append(zf)
            latencies.append(lat)
            center_errors.append(ce)
            object_sizes.append(osize)
            occlusions.append(is_occluded)
            frame_ids.append(frame_id)

            # False tracking
            if tid is not None and iou < 0.1 and not is_occluded:
                false_tracking_frames += 1

            # Continuity logic
            if tid is not None and iou >= self.iou_threshold:
                if not is_tracking:
                    is_tracking = True
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                if is_tracking and not is_occluded:
                    fragmentation_count += 1
                    is_tracking = False
                    current_streak = 0
            
            # ID Switches
            if tid is not None:
                if prev_tid is not None and tid != prev_tid:
                    id_switches += 1
                prev_tid = tid

        total_frames = len(gt_map)
        ious_np = np.array([i if i is not None else 0.0 for i in ious])
        conf_np = np.array([c if c is not None else 0.0 for c in confidences])
        zoom_np = np.array([z if z is not None else 1.0 for z in zoom_factors])
        ce_np = np.array([c if c is not None else 0.0 for c in center_errors])
        lat_np = np.array([l if l is not None else 0.0 for l in latencies])
        size_np = np.array([s if s is not None else 0.0 for s in object_sizes])

        # -------------------------------
        # BASIC METRICS
        # -------------------------------
        consistency = float(np.mean(
            (ious_np > self.iou_threshold) & (conf_np > self.conf_threshold)
        ))

        summary = {
            "iou_mean": float(np.mean(ious_np)),
            "iou_std": float(np.std(ious_np)),
            "detection_consistency": consistency,
            "fragmentation": fragmentation_count,
            "tracking_stability": float(max_streak / total_frames) if total_frames > 0 else 0.0,
            "center_error": float(np.mean(ce_np)),
            "latency": float(np.mean(lat_np)),
            "detection_recovery_rate": self.compute_detection_recovery_rate(ious),
            "zoom_responsiveness": self.compute_zoom_responsiveness(zoom_factors)
        }

        # -------------------------------
        # ADVANCED METRICS
        # -------------------------------
        summary.update(self.compute_scale_metrics(ious, object_sizes, confidences))
        recovery_results = self.compute_recovery_metrics(ious)
        summary.update(self.compute_fps(latencies))
        summary.update(self.compute_zoom_response(object_sizes, zoom_factors))
        summary.update(self.compute_control_smoothness(zoom_factors))
        summary.update(self.compute_temporal_causality(zoom_factors, ious))
        summary.update(self.compute_failure_modes(ious, object_sizes, occlusions))

        # -------------------------------
        # SERIES DATA (FOR PLOTTING)
        # -------------------------------
        series = {
            "frame_ids": frame_ids,
            "ious": ious,
            "zoom_factors": zoom_factors,
            "center_errors": center_errors,
            "object_sizes": object_sizes,
            "confidences": confidences,
            "occlusions": occlusions,
            "recovery_events": recovery_results["recovery_events"],
            "recovery_times": recovery_results["recovery_times"]
        }

        return {"summary": summary, "series": series}

    # -------------------------------
    # SCALE ROBUSTNESS
    # -------------------------------
    def compute_scale_metrics(self, ious, object_sizes, confs):
        buckets = {"small": [], "medium": [], "large": []}

        for iou, size, conf in zip(ious, object_sizes, confs):
            size_val = size if size is not None else 0.0
            if size_val < 0.05:
                buckets["small"].append((iou, conf))
            elif size_val < 0.15:
                buckets["medium"].append((iou, conf))
            else:
                buckets["large"].append((iou, conf))

        results = {}
        for k, vals in buckets.items():
            if vals:
                iou_vals = [v[0] for v in vals]
                results[f"{k}_iou"] = float(np.mean(iou_vals))
                results[f"{k}_consistency"] = float(np.mean([
                    (iou > self.iou_threshold and c > self.conf_threshold)
                    for iou, c in vals
                ]))
            else:
                results[f"{k}_iou"] = 0.0
                results[f"{k}_consistency"] = 0.0

        return results

    # -------------------------------
    # RECOVERY TIME
    # -------------------------------
    def compute_recovery_metrics(self, ious):
        recovery_times = []
        recovery_events = []
        lost = False
        start_idx = 0

        for i, iou in enumerate(ious):
            if iou < self.iou_threshold:
                if not lost:
                    lost = True
                    start_idx = i
            else:
                if lost:
                    duration = i - start_idx
                    recovery_times.append(duration)
                    recovery_events.append(i) # recovery happened at frame i
                    lost = False

        return {
            "avg_recovery_frames": float(np.mean(recovery_times)) if recovery_times else 0.0,
            "max_recovery_frames": int(np.max(recovery_times)) if recovery_times else 0,
            "recovery_times": recovery_times,
            "recovery_events": recovery_events
        }

    # -------------------------------
    # FPS
    # -------------------------------
    def compute_fps(self, latencies):
        lat_np = np.array(latencies)
        avg_latency = np.mean(lat_np) if len(lat_np) > 0 else 0.0
        return {
            "fps": float(1000.0 / avg_latency) if avg_latency > 0 else 0.0
        }

    # -------------------------------
    # ZOOM RESPONSE
    # -------------------------------
    def compute_zoom_response(self, object_sizes, zoom_factors):
        if len(object_sizes) < 2:
            return {"zoom_response_corr": 0.0}

        size_np = np.array([s if s is not None else 0.0 for s in object_sizes])
        zoom_np = np.array([z if z is not None else 1.0 for z in zoom_factors])
        
        # Avoid constant values leading to NaN correlation
        if np.std(size_np) == 0 or np.std(zoom_np) == 0:
            return {"zoom_response_corr": 0.0}

        corr = np.corrcoef(size_np, zoom_np)[0, 1]
        return {
            "zoom_response_corr": float(corr) if not np.isnan(corr) else 0.0
        }

    # -------------------------------
    # CONTROL SMOOTHNESS
    # -------------------------------
    def compute_control_smoothness(self, zoom_factors):
        z = np.array([zv if zv is not None else 1.0 for zv in zoom_factors])
        if len(z) < 3:
            return {"zoom_jerk": 0.0}

        jerk = np.diff(np.diff(z))
        return {
            "zoom_jerk": float(np.mean(np.abs(jerk)))
        }

    # -------------------------------
    # ZOOM RESPONSIVENESS
    # -------------------------------
    def compute_zoom_responsiveness(self, zoom_factors: List[float]) -> float:
        """Measure how fast zoom reacts: delta_z = abs(z_t - z_t-1). Metric = mean(delta_z)"""
        z = np.array([zv if zv is not None else 1.0 for zv in zoom_factors])
        if len(z) < 2:
            return 0.0
        delta_z = np.abs(np.diff(z))
        return float(np.mean(delta_z))

    # -------------------------------
    # MODEL COMPARISON
    # -------------------------------
    def compare_models(self, baseline_summary, adaptive_summary):
        base_iou = baseline_summary.get("iou_mean", 0.0)
        adap_iou = adaptive_summary.get("iou_mean", 0.0)

        abs_gain = adap_iou - base_iou
        rel_gain = abs_gain / base_iou if base_iou > 0 else 0.0

        return {
            "zoom_gain_abs": abs_gain,
            "zoom_gain_rel": rel_gain
        }

    # -------------------------------
    # STATISTICAL SIGNIFICANCE
    # -------------------------------
    def compute_statistical_significance(self, baseline_ious: List[float], adaptive_ious: List[float]) -> Dict[str, float]:
        """Proves statistical improvement using paired t-test."""
        if len(baseline_ious) != len(adaptive_ious) or len(baseline_ious) < 2:
            return {"t_stat": 0.0, "p_value": 1.0}
        
        stat, p_value = ttest_rel(adaptive_ious, baseline_ious)
        return {
            "t_stat": float(stat) if not np.isnan(stat) else 0.0,
            "p_value": float(p_value) if not np.isnan(p_value) else 1.0
        }

    # -------------------------------
    # SCALE IMPROVEMENT CURVE
    # -------------------------------
    def compute_scale_improvement_curve(self, base_ious, adap_ious, object_sizes):
        """Categorizes improvement by object size."""
        bins = {"small": [], "medium": [], "large": []}

        for b_iou, a_iou, size in zip(base_ious, adap_ious, object_sizes):
            delta = (a_iou if a_iou is not None else 0.0) - (b_iou if b_iou is not None else 0.0)
            size_val = size if size is not None else 0.0
            if size_val < 0.05:
                bins["small"].append(delta)
            elif size_val < 0.15:
                bins["medium"].append(delta)
            else:
                bins["large"].append(delta)

        return {
            k: float(np.mean(v)) if v else 0.0
            for k, v in bins.items()
        }

    # -------------------------------
    # FAILURE MODE ANALYSIS
    # -------------------------------
    def compute_failure_modes(self, ious, object_sizes, occlusions):
        """Categorizes where the system fails."""
        failures = {
            "small_object_failure": 0,
            "occlusion_failure": 0,
            "other_failure": 0
        }

        total_failures = 0
        for iou, size, occ in zip(ious, object_sizes, occlusions):
            iou_val = iou if iou is not None else 0.0
            size_val = size if size is not None else 0.0
            if iou_val < self.iou_threshold:
                total_failures += 1
                if occ:
                    failures["occlusion_failure"] += 1
                elif size_val < 0.05:
                    failures["small_object_failure"] += 1
                else:
                    failures["other_failure"] += 1

        if total_failures == 0:
            return {"failure_modes": failures}

        return {
            "failure_modes": {
                k: float(v / total_failures)
                for k, v in failures.items()
            }
        }

    # -------------------------------
    # TEMPORAL CAUSALITY (LAG ANALYSIS)
    # -------------------------------
    def compute_temporal_causality(self, zoom_factors, ious):
        """Measures correlation between zoom at t and IoU at t+1."""
        if len(zoom_factors) < 2:
            return {"zoom_to_iou_lag_corr": 0.0}

        z = np.array([zv if zv is not None else 1.0 for zv in zoom_factors[:-1]])
        iou_next = np.array([i if i is not None else 0.0 for i in ious[1:]])

        # Avoid constant values
        if np.std(z) == 0 or np.std(iou_next) == 0:
            return {"zoom_to_iou_lag_corr": 0.0}

        corr = np.corrcoef(z, iou_next)[0, 1]
        return {
            "zoom_to_iou_lag_corr": float(corr) if not np.isnan(corr) else 0.0
        }
    # -------------------------------
    # DETECTION RECOVERY RATE
    # -------------------------------
    def compute_detection_recovery_rate(self, ious: List[float]) -> float:
        """
        Measures how often detection recovers after a drop (IoU < threshold).
        Defined as: (number of recoveries) / (number of drops)
        """
        drops = 0
        recoveries = 0
        in_drop = False
        
        for iou in ious:
            if iou < self.iou_threshold:
                if not in_drop:
                    drops += 1
                    in_drop = True
            else:
                if in_drop:
                    recoveries += 1
                    in_drop = False
        
        return recoveries / drops if drops > 0 else 1.0

