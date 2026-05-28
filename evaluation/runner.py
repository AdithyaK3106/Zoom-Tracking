import os
import yaml
import json
import sys
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.pipeline import ZoomTrackingPipeline
from evaluation.evaluator import SystemEvaluator

class ResearchRunner:
    """
    Automated Research Runner for Zoom Tracking System.
    Executes 3 modes (A, B, C) across all scenarios and aggregates results.
    """
    def __init__(self, config_path: str = "evaluation/configs/eval_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.evaluator = SystemEvaluator(
            scenarios_dir=self.config['paths']['scenarios_dir'],
            results_dir=self.config['paths']['results_dir']
        )
        
        self.modes = {
            "baseline": {"mode": "baseline", "zoom": False},
            "tracking": {"mode": "tracking", "zoom": False},
            "adaptive_zoom": {"mode": "adaptive_zoom", "zoom": True}
        }

    def run_full_evaluation(self):
        print("\n=== Starting Research Evaluation Pipeline ===\n")
        
        scenarios = self.config['scenarios']
        
        for scenario in scenarios:
            name = scenario['name']
            path = scenario['path']
            print(f"--- Scenario: {name} ({scenario['difficulty']}) ---")
            
            for mode_id, params in self.modes.items():
                print(f"  Running Mode: {mode_id}...")
                
                # Setup model version for logging (filename)
                model_version = f"{name}_{mode_id}"
                log_path = os.path.join(self.config['paths']['logs_dir'], f"{model_version}.json")
                
                # 1. Run Pipeline
                pipeline = ZoomTrackingPipeline(
                    source=path,
                    model_version=model_version,
                    mode=params['mode'],
                    enable_zoom_redetect=params['zoom']
                )
                pipeline.run()
                
                # 2. Evaluate
                self.evaluator.evaluate_model(mode_id, log_path, name)

            # 3. Pair-wise Comparison (for Zoom Gain)
            self.evaluator.run_paired_comparison("baseline", "adaptive_zoom", name)

            # 4. Generate scenario-specific comparison plots
            self.evaluator.generate_aggregated_plots(name, "baseline")

        # 5. Final Aggregation
        self.evaluator.save_results("results.json")
        print("\n=== Evaluation Finished. Results saved to evaluation/results/results.json ===\n")

if __name__ == "__main__":
    runner = ResearchRunner()
    runner.run_full_evaluation()
