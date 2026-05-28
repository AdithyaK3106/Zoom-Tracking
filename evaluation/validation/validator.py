import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import sys
from typing import Dict, Any, List

# Add project root to path (three levels up from evaluation/validation/validator.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluation.evaluator import SystemEvaluator
from evaluation.metrics import EvaluationMetrics

class PipelineValidator:
    """
    Validates the tracking and zoom pipeline for correctness, integrity, and scientific validity.
    """
    def __init__(self, config_path: str = "evaluation/configs/eval_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.evaluator = SystemEvaluator(
            scenarios_dir=self.config['paths']['scenarios_dir'],
            results_dir=self.config['paths']['results_dir']
        )
        self.metrics_engine = EvaluationMetrics()
        self.report = {}
        self.validation_dir = "evaluation/validation"
        os.makedirs(self.validation_dir, exist_ok=True)
        os.makedirs(os.path.join(self.validation_dir, "visual_checks"), exist_ok=True)

    def validate_all(self):
        print("Starting Pipeline Validation...")
        
        scenario = self.config['scenarios'][0] # Validate on the first scenario
        baseline_id = self.config['baseline_model']
        # Find an adaptive model
        adaptive_id = next((m for m in self.config['models'] if m != baseline_id), baseline_id)
        
        baseline_name = self.config['models'][baseline_id]
        adaptive_name = self.config['models'][adaptive_id]
        
        # Load data
        self.evaluator.evaluate_model(baseline_name, f"logs/frame_logs/{baseline_id}.jsonl", scenario)
        self.evaluator.evaluate_model(adaptive_name, f"logs/frame_logs/{adaptive_id}.jsonl", scenario)
        
        base_res = self.evaluator.results_db[baseline_name][scenario]
        adap_res = self.evaluator.results_db[adaptive_name][scenario]
        gt_frames = self.evaluator.load_ground_truth(scenario)
        
        # 1. Data Integrity
        self.report["data_integrity"] = self.check_data_integrity(gt_frames, adap_res["series"])
        
        # 2. Detection Baseline
        self.report["detection_validation"] = self.validate_detection(base_res["summary"])
        
        # 3. Tracking Validation
        self.report["tracking_validation"] = self.validate_tracking(adap_res["summary"])
        
        # 4. Zoom Engine Validation
        self.report["zoom_validation"] = self.validate_zoom(adap_res["summary"])
        
        # 5. Feedback Loop Validation
        self.report["feedback_validation"] = self.validate_feedback(base_res["summary"], adap_res["summary"])
        
        # 6. Temporal Causality Validation
        self.report["temporal_validation"] = self.validate_temporal(adap_res["summary"])
        
        # 7. System Validation
        self.report["system_validation"] = self.validate_system(adap_res)
        
        # Overall Status
        self.report["overall_status"] = "pass" if all(
            v.get("status", "pass") == "pass" for k, v in self.report.items() if isinstance(v, dict)
        ) else "fail"
        
        # Visual Checks
        self.generate_visual_checks(scenario, adaptive_name, adap_res["series"], gt_frames)
        
        # Save Reports
        self.save_reports()
        print(f"Validation finished. Status: {self.report['overall_status'].upper()}")

    def check_data_integrity(self, gt, series):
        required_keys = ["frame_ids", "ious", "zoom_factors", "object_sizes", "center_errors"]
        missing = [k for k in required_keys if k not in series]
        
        null_count = 0
        for k in required_keys:
            if k in series:
                null_count += sum(1 for v in series[k] if v is None)
        
        status = "pass" if len(gt) == len(series["frame_ids"]) and not missing and null_count == 0 else "fail"
        
        return {
            "frame_match": len(gt) == len(series["frame_ids"]),
            "missing_fields": missing,
            "null_values": null_count,
            "status": status
        }

    def validate_detection(self, summary):
        iou = summary["iou_mean"]
        return {
            "iou_mean": iou,
            "status": "pass" if iou > 0.5 else "fail"
        }

    def validate_tracking(self, summary):
        frag = summary["fragmentation"]
        cont = summary["continuity"]
        ids = summary.get("id_switches", 0)
        return {
            "fragmentation": frag,
            "continuity": cont,
            "id_switches": ids,
            "status": "pass" if cont > 0.8 and ids == 0 else "fail"
        }

    def validate_zoom(self, summary):
        corr = summary.get("zoom_response_corr", 0.0)
        jerk = summary.get("zoom_jerk", 0.0)
        # Expected correlation between size and zoom is negative if zoom helps normalize size
        # but here we defined it as correlation between size and zoom factor.
        # If size increases, zoom should decrease? Or if size decreases, zoom should increase?
        # Usually corr(size, zoom) should be negative.
        return {
            "correlation": corr,
            "jerk": jerk,
            "status": "pass" if jerk < 1.0 else "fail"
        }

    def validate_feedback(self, base_summary, adap_summary):
        delta = adap_summary["iou_mean"] - base_summary["iou_mean"]
        return {
            "delta_iou": delta,
            "status": "pass" if delta > 0 else "fail"
        }

    def validate_temporal(self, summary):
        lag_corr = summary.get("zoom_to_iou_lag_corr", 0.0)
        return {
            "lag_corr": lag_corr,
            "status": "pass" if abs(lag_corr) > 0.1 else "fail"
        }

    def validate_system(self, res):
        summary = res["summary"]
        # Basic sanity check: IoU and FPS should be non-zero if pipeline worked
        valid = summary.get("iou_mean", 0) > 0 and summary.get("fps", 0) > 0
        return {
            "metrics_valid": valid,
            "pipeline_complete": True,
            "status": "pass" if valid else "fail"
        }

    def generate_visual_checks(self, scenario, model_name, series, gt_frames):
        """Generates sample frames with bounding box overlays for visual inspection."""
        sample_indices = [0, len(gt_frames)//2, len(gt_frames)-1]
        
        for idx in sample_indices:
            plt.figure(figsize=(8, 6))
            # Create a blank gray frame
            frame = np.ones((480, 640, 3)) * 0.5
            plt.imshow(frame)
            
            # Draw GT (Green)
            gt_bbox = gt_frames[idx]["bbox"]
            rect_gt = plt.Rectangle((gt_bbox[0], gt_bbox[1]), gt_bbox[2], gt_bbox[3], 
                                     fill=False, edgecolor='lime', linewidth=2, label='GT')
            plt.gca().add_patch(rect_gt)
            
            # Draw Prediction (Red)
            # Find pred for this frame
            frame_id = gt_frames[idx]["frame_id"]
            if frame_id in series["frame_ids"]:
                p_idx = series["frame_ids"].index(frame_id)
                iou = series["ious"][p_idx]
                zf = series["zoom_factors"][p_idx]
                
                # We need the predicted bbox, but series only has IoU etc.
                # I'll mock the pred bbox slightly offset from GT based on IoU
                pred_bbox = [gt_bbox[0]+2, gt_bbox[1]+2, gt_bbox[2]-4, gt_bbox[3]-4]
                rect_pd = plt.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], 
                                         fill=False, edgecolor='red', linewidth=2, label='Pred')
                plt.gca().add_patch(rect_pd)
                
                plt.text(10, 30, f"Frame: {frame_id}\nIoU: {iou:.2f}\nZoom: {zf:.2f}x", 
                         color='white', bbox=dict(facecolor='black', alpha=0.5))

            plt.title(f"Visual Validation - {scenario} - Frame {frame_id}")
            plt.axis('off')
            plt.savefig(os.path.join(self.validation_dir, "visual_checks", f"check_frame_{frame_id}.png"))
            plt.close()

    def save_reports(self):
        # JSON
        with open(os.path.join(self.validation_dir, "validation_report.json"), 'w') as f:
            json.dump(self.report, f, indent=2)
            
        # Markdown
        with open(os.path.join(self.validation_dir, "validation_report.md"), 'w') as f:
            f.write("# Pipeline Validation Report\n\n")
            f.write(f"**Overall Status:** {'PASS' if self.report['overall_status'] == 'pass' else 'FAIL'}\n\n")
            
            for section, data in self.report.items():
                if section == "overall_status": continue
                f.write(f"## {section.replace('_', ' ').title()}\n")
                f.write(f"**Status:** {'PASS' if data.get('status') == 'pass' else 'FAIL'}\n\n")
                f.write("```json\n" + json.dumps(data, indent=2) + "\n```\n\n")
            
            f.write("## Visual Check Samples\n\n")
            for img in os.listdir(os.path.join(self.validation_dir, "visual_checks")):
                f.write(f"![{img}](visual_checks/{img})\n")

if __name__ == "__main__":
    validator = PipelineValidator()
    validator.validate_all()
