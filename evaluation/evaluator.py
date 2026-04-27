import os
import json
import time
from typing import Dict, List, Any

# Adjust paths assuming this is inside evaluation/
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import EvaluationMetrics

class SystemEvaluator:
    """
    Drives the headless execution of the ZoomTrackingPipeline to collect
    telemetry and calculate research-grade metrics.
    """
    def __init__(self, scenarios_dir: str = "evaluation/scenarios/dataset_json", results_dir: str = "evaluation/results"):
        self.scenarios_dir = scenarios_dir
        self.results_dir = results_dir
        
        os.makedirs(self.scenarios_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.metrics_engine = EvaluationMetrics(iou_threshold=0.5, conf_threshold=0.3)
        self.results_db = {}

    def load_ground_truth(self, scenario_name: str) -> List[Dict[str, Any]]:
        """Loads GT JSON. Expects a list of frame dicts: {frame_id, bbox, object_id, occlusion_flag}"""
        gt_path = os.path.join(self.scenarios_dir, f"{scenario_name}.json")
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth for {scenario_name} not found at {gt_path}.")
            return []
            
        with open(gt_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_predictions(self, log_path: str) -> List[Dict[str, Any]]:
        """Parses the JSONL output from logs/logger.py"""
        preds = []
        if not os.path.exists(log_path):
            return preds
            
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    preds.append(json.loads(line))
        return preds

    def evaluate_model(self, model_name: str, log_path: str, scenario_name: str):
        """
        Runs the mathematical evaluation on a completed log against its GT.
        """
        print(f"Evaluating {model_name} on scenario: {scenario_name}...")
        
        gt_frames = self.load_ground_truth(scenario_name)
        if not gt_frames:
            return
            
        pred_frames = self.load_predictions(log_path)
        
        # Calculate single-model stats
        metrics_dict = self.metrics_engine.compute_all_metrics(gt_frames, pred_frames)
        
        if model_name not in self.results_db:
            self.results_db[model_name] = {}
            
        self.results_db[model_name][scenario_name] = metrics_dict

    def run_paired_t_test(self, baseline_model: str, adaptive_model: str, scenario_name: str):
        """
        Calculates Zoom Gain and performs paired comparison.
        """
        if baseline_model not in self.results_db or adaptive_model not in self.results_db:
            return
            
        base_stats = self.results_db[baseline_model].get(scenario_name, {})
        adap_stats = self.results_db[adaptive_model].get(scenario_name, {})
        
        if not base_stats or not adap_stats:
            return
            
        gains = self.metrics_engine.compare_models(base_stats, adap_stats)
        self.results_db[adaptive_model][scenario_name].update(gains)

    def save_results(self, output_filename: str = "leaderboard.json"):
        """Saves the final nested JSON output to results directory."""
        output_path = os.path.join(self.results_dir, output_filename)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results_db, f, indent=2)
        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    print("Initializing SystemEvaluator...")
    evaluator = SystemEvaluator()
    print(f"Ensuring directories exist: {evaluator.scenarios_dir}, {evaluator.results_dir}")
    # In practice, loop over models, run pipeline via subprocess or direct import, and process logs.
