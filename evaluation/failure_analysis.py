import json
import os
import numpy as np
from typing import List, Dict, Any

class FailureAnalyzer:
    """
    Analyzes tracking logs to identify and classify failure segments.
    """
    def __init__(self, iou_threshold: float = 0.3, conf_threshold: float = 0.3):
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    def analyze_scenario(self, series: Dict[str, List], scenario_name: str) -> List[Dict[str, Any]]:
        """
        Detects and classifies failure segments in a single scenario run.
        """
        frames = series["frame_ids"]
        ious = series["ious"]
        confidences = series["confidences"]
        sizes = series["object_sizes"]
        
        # We need centers to compute velocity for motion_failure
        # If not in series, we'll try to estimate or skip motion classification
        # Actually, let's assume we might need center_errors or something similar
        # For now, we'll use object_size for scale_failure and presence for occlusion
        
        failures = []
        in_failure = False
        current_failure = None
        
        for i in range(len(frames)):
            iou = ious[i] if ious[i] is not None else 0.0
            conf = confidences[i] if confidences[i] is not None else 0.0
            
            # Failure condition: IoU < 0.3 or confidence < threshold
            is_failing = (iou < self.iou_threshold) or (conf < self.conf_threshold)
            
            if is_failing:
                if not in_failure:
                    in_failure = True
                    current_failure = {
                        "start": frames[i],
                        "end": frames[i],
                        "start_idx": i,
                        "type": "unknown"
                    }
                else:
                    current_failure["end"] = frames[i]
            else:
                if in_failure:
                    # End of failure segment
                    # Classify before adding
                    current_failure["type"] = self.classify_failure(series, current_failure["start_idx"], i-1)
                    failures.append(current_failure)
                    in_failure = False
                    current_failure = None
        
        # Handle case where failure lasts until end of video
        if in_failure:
            current_failure["type"] = self.classify_failure(series, current_failure["start_idx"], len(frames)-1)
            failures.append(current_failure)
            
        return failures

    def classify_failure(self, series: Dict[str, List], start_idx: int, end_idx: int) -> str:
        """
        Heuristic classification of failure type.
        """
        # 1. Occlusion -> bbox missing (already handled if we had occlusion flags, 
        # but let's check if confidence is very low and iou is 0)
        # 2. Scale failure -> object size very small
        # 3. Motion failure -> high centroid velocity (if we had it)
        
        relevant_sizes = [series["object_sizes"][i] for i in range(start_idx, end_idx+1) if series["object_sizes"][i] is not None]
        size = np.mean(relevant_sizes) if relevant_sizes else 1.0
        
        # scale_failure -> object_size < 0.02 (very small)
        if size < 0.02:
            return "scale_failure"
            
        # occlusion -> if we have the flag in the logs (occlusions series)
        is_occluded = any(series["occlusions"][i] for i in range(start_idx, end_idx+1))
        if is_occluded:
            return "occlusion"
            
        # motion_failure -> high centroid error or just a fallback for now
        # In a real system, we'd check np.diff(centers)
        return "motion_failure"

def generate_failure_report(results_db: Dict[str, Dict], output_path: str):
    analyzer = FailureAnalyzer()
    report = {}
    
    for model_name, scenarios in results_db.items():
        if model_name not in report:
            report[model_name] = {}
        
        for scenario_name, data in scenarios.items():
            if "series" in data:
                failures = analyzer.analyze_scenario(data["series"], scenario_name)
                report[model_name][scenario_name] = failures
                
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"Failure report saved to {output_path}")

if __name__ == "__main__":
    # This would be called by the evaluator or runner
    pass
