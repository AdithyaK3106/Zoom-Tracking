import os
import json
import time
from typing import Dict, List, Any

# Adjust paths assuming this is inside evaluation/
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import EvaluationMetrics
from evaluation.visualizer import TrackingVisualizer
from evaluation.failure_analysis import FailureAnalyzer

class SystemEvaluator:
    """
    Drives the headless execution of the ZoomTrackingPipeline to collect
    telemetry and calculate research-grade metrics.
    """
    def __init__(self, scenarios_dir: str = "data/annotations", results_dir: str = "evaluation/results"):
        # Normalize paths relative to project root
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.scenarios_dir = os.path.join(self.project_root, scenarios_dir)
        self.results_dir = os.path.join(self.project_root, results_dir)
        self.plots_dir = os.path.join(self.results_dir, "plots")
        
        os.makedirs(self.scenarios_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        self.metrics_engine = EvaluationMetrics(iou_threshold=0.5, conf_threshold=0.3)
        self.visualizer = TrackingVisualizer(output_dir=self.plots_dir)
        self.failure_analyzer = FailureAnalyzer(iou_threshold=0.3, conf_threshold=0.3)
        self.results_db = {} # model -> scenario -> {summary, series}

    def load_ground_truth(self, scenario_name: str) -> Dict[str, Any]:
        """Loads GT JSON from data/annotations."""
        gt_path = os.path.join(self.scenarios_dir, f"{scenario_name}.json")
        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth for {scenario_name} not found at {gt_path}.")
            return {}
            
        with open(gt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if scenario_name == "person_scale":
            converted = {}
            for k, bbox in data.items():
                if len(bbox) == 4:
                    converted[k] = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                else:
                    converted[k] = bbox
            return converted
            
        return data

    def load_predictions(self, log_path: str) -> List[Dict[str, Any]]:
        """Parses the JSONL output."""
        preds = []
        full_log_path = os.path.join(self.project_root, log_path)
        if not os.path.exists(full_log_path):
            print(f"Warning: Log file not found at {full_log_path}")
            return preds
            
        with open(full_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        preds.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return preds

    def evaluate_model(self, model_name: str, log_path: str, scenario_name: str):
        """
        Runs the mathematical evaluation and generates plots.
        """
        print(f"Evaluating {model_name} on scenario: {scenario_name}...")
        
        gt_frames = self.load_ground_truth(scenario_name)
        if not gt_frames:
            return
            
        pred_frames = self.load_predictions(log_path)
        if not pred_frames:
            return
        
        # Calculate stats
        results = self.metrics_engine.compute_all_metrics(gt_frames, pred_frames)
        
        # Run Failure Analysis
        failures = self.failure_analyzer.analyze_scenario(results["series"], scenario_name)
        results["summary"]["failure_segments"] = failures

        if model_name not in self.results_db:
            self.results_db[model_name] = {}
            
        self.results_db[model_name][scenario_name] = results

        # Generate plots
        self.visualizer.plot_time_series(results["series"], model_name, scenario_name)
        self.visualizer.plot_correlations(results["series"], model_name, scenario_name)
        self.visualizer.plot_recovery_timeline(results["series"], model_name, scenario_name)
        self.visualizer.plot_failure_overlay(results["series"], failures, model_name, scenario_name)

    def run_paired_comparison(self, baseline_model: str, adaptive_model: str, scenario_name: str):
        """
        Calculates Zoom Gain, Statistical Significance, and Scale Improvement.
        """
        if baseline_model not in self.results_db or adaptive_model not in self.results_db:
            return
            
        base_res = self.results_db[baseline_model].get(scenario_name, {})
        adap_res = self.results_db[adaptive_model].get(scenario_name, {})
        
        if not base_res or not adap_res:
            return
            
        # 1. Zoom Gain
        gains = self.metrics_engine.compare_models(base_res["summary"], adap_res["summary"])
        self.results_db[adaptive_model][scenario_name]["summary"].update(gains)

        # 2. Statistical Significance
        base_ious = base_res["series"]["ious"]
        adap_ious = adap_res["series"]["ious"]
        stats = self.metrics_engine.compute_statistical_significance(base_ious, adap_ious)
        self.results_db[adaptive_model][scenario_name]["summary"]["statistical_significance"] = stats

        # 3. Scale Improvement
        object_sizes = adap_res["series"]["object_sizes"]
        scale_curve = self.metrics_engine.compute_scale_improvement_curve(base_ious, adap_ious, object_sizes)
        self.results_db[adaptive_model][scenario_name]["summary"]["scale_improvement"] = scale_curve

    def generate_aggregated_plots(self, scenario_name: str, baseline_model: str):
        """Generates plots that compare multiple models."""
        self.visualizer.plot_model_comparison(self.results_db, scenario_name)
        self.visualizer.plot_advanced_research_components(self.results_db, scenario_name, baseline_model)
        self.visualizer.plot_scenario_performance(self.results_db)

    def save_results(self, summary_filename: str = "results.json"):
        """Saves the final nested JSON output in Research-Paper format."""
        
        # Structure: scenario -> model -> summary
        restructured = {}
        
        # First, find all scenarios
        scenarios = set()
        for model in self.results_db:
            scenarios.update(self.results_db[model].keys())
        
        for scenario in scenarios:
            restructured[scenario] = {}
            for model in self.results_db:
                if scenario in self.results_db[model]:
                    raw_summary = self.results_db[model][scenario]["summary"]
                    
                    # Map and rename keys
                    clean_summary = {
                        "IoU": raw_summary.get("iou_mean", 0.0),
                        "zoom_gain": raw_summary.get("zoom_gain_rel", 0.0),
                        "recovery_rate": raw_summary.get("detection_recovery_rate", 0.0),
                        "scale_small_IoU": raw_summary.get("small_iou", 0.0),
                        "scale_medium_IoU": raw_summary.get("medium_iou", 0.0),
                        "scale_large_IoU": raw_summary.get("large_iou", 0.0),
                        "center_error": raw_summary.get("center_error", 0.0),
                        "latency": raw_summary.get("latency", 0.0),
                        "stability": raw_summary.get("tracking_stability", 0.0),
                        "zoom_responsiveness": raw_summary.get("zoom_responsiveness", 0.0)
                    }
                    restructured[scenario][model] = clean_summary

        lb_path = os.path.join(self.results_dir, summary_filename)
        with open(lb_path, 'w', encoding='utf-8') as f:
            json.dump(restructured, f, indent=2)
            
        print(f"Research Results saved to {lb_path}")

if __name__ == "__main__":
    # Example usage for direct debugging
    evaluator = SystemEvaluator()
    evaluator.evaluate_model("v1", "logs/frame_logs/v1.jsonl", "car2")
    evaluator.save_results()

