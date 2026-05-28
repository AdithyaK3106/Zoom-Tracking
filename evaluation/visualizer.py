import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List

class TrackingVisualizer:
    """
    Generates research-grade plots for tracking and control evaluation.
    """
    def __init__(self, output_dir: str = "evaluation/results/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Use a clean style
        plt.rcParams.update({
            'font.size': 12,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'figure.autolayout': True
        })

    def plot_time_series(self, series: Dict[str, List], model_name: str, scenario_name: str):
        """Generates line plots for IoU, Zoom, and Error over time."""
        frames = series["frame_ids"]
        
        # Sanitize data
        ious = [i if i is not None else 0.0 for i in series["ious"]]
        zooms = [z if z is not None else 1.0 for z in series["zoom_factors"]]
        errors = [e if e is not None else 0.0 for e in series["center_errors"]]
        sizes = [s if s is not None else 0.0 for s in series["object_sizes"]]

        # 1. IoU vs Time
        plt.figure(figsize=(10, 4))
        plt.plot(frames, ious, label="IoU", color='blue', linewidth=1.5)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label="Threshold (0.5)")
        plt.title(f"IoU over Time - {model_name} ({scenario_name})")
        plt.xlabel("Frame ID")
        plt.ylabel("IoU")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_{scenario_name}_iou_time.png"))
        plt.close()

        # 2. Zoom Factor vs Time
        plt.figure(figsize=(10, 4))
        plt.plot(frames, zooms, label="Zoom Factor", color='green', linewidth=1.5)
        plt.title(f"Zoom Factor vs Time - {model_name} ({scenario_name})")
        plt.xlabel("Frame ID")
        plt.ylabel("Zoom Level")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_{scenario_name}_zoom_time.png"))
        plt.close()

        # 3. Center Error vs Time
        plt.figure(figsize=(10, 4))
        plt.plot(frames, errors, label="Center Error", color='red', linewidth=1.5)
        plt.title(f"Center Error vs Time - {model_name} ({scenario_name})")
        plt.xlabel("Frame ID")
        plt.ylabel("Error (Pixels)")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_{scenario_name}_error_time.png"))
        plt.close()

        # 4. Object Size vs Time
        plt.figure(figsize=(10, 4))
        plt.plot(frames, sizes, label="Object Size", color='darkorange', linewidth=1.5)
        plt.title(f"Object Size vs Time - {model_name} ({scenario_name})")
        plt.xlabel("Frame ID")
        plt.ylabel("Size (Relative to Frame)")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_{scenario_name}_size_time.png"))
        plt.close()

        # 5. Confidence vs Time
        plt.figure(figsize=(10, 4))
        confs = [c if c is not None else 0.0 for c in series["confidences"]]
        plt.plot(frames, confs, label="Confidence", color='indigo', linewidth=1.5)
        plt.fill_between(frames, confs, color='indigo', alpha=0.1)
        plt.title(f"Confidence vs Time - {model_name} ({scenario_name})")
        plt.xlabel("Frame ID")
        plt.ylabel("Confidence")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_{scenario_name}_confidence_time.png"))
        plt.close()

    def plot_failure_overlay(self, series: Dict[str, List], failures: List[Dict[str, Any]], model_name: str, scenario_name: str):
        """Highlights failure regions on the IoU timeline."""
        frames = series["frame_ids"]
        ious = [i if i is not None else 0.0 for i in series["ious"]]
        
        plt.figure(figsize=(12, 4))
        plt.plot(frames, ious, color='blue', label="GT-IoU", linewidth=1.0, alpha=0.8)
        
        # Color mapping for failure types
        color_map = {
            "scale_failure": "orange",
            "motion_failure": "red",
            "occlusion": "gray",
            "unknown": "purple"
        }
        
        added_labels = set()
        for fail in failures:
            label = fail["type"].replace('_', ' ').title()
            plt.axvspan(fail["start"], fail["end"], color=color_map.get(fail["type"], "purple"), alpha=0.3, 
                        label=label if label not in added_labels else "")
            added_labels.add(label)
            
        plt.axhline(y=0.3, color='black', linestyle='--', alpha=0.5, label="Failure Threshold (0.3)")
        plt.title(f"Failure Analysis Overlay - {model_name} ({scenario_name})")
        plt.xlabel("Frame ID")
        plt.ylabel("IoU")
        plt.ylim(0, 1.1)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_{scenario_name}_failure_overlay.png"))
        plt.close()

    def plot_scenario_performance(self, results_db: Dict[str, Dict]):
        """Generates a bar plot showing performance across different scenarios."""
        models = list(results_db.keys())
        if not models: return
        
        # Get all scenario names
        scenarios = set()
        for m in models:
            scenarios.update(results_db[m].keys())
        scenarios = sorted(list(scenarios))
        
        if not scenarios: return
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, m in enumerate(models):
            ious = [results_db[m].get(s, {}).get("summary", {}).get("iou_mean", 0.0) for s in scenarios]
            plt.bar(x + i*width, ious, width, label=m)
            
        plt.xlabel('Scenario')
        plt.ylabel('Mean IoU')
        plt.title('Performance Across Scenarios')
        plt.xticks(x + width, scenarios)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "performance_vs_scenario.png"))
        plt.close()

    def plot_correlations(self, series: Dict[str, List], model_name: str, scenario_name: str):
        """Generates scatter plots for IoU vs Size and Zoom vs Size."""
        
        # 5. IoU vs Object Size
        plt.figure(figsize=(8, 6))
        sizes = [s if s is not None else 0.0 for s in series["object_sizes"]]
        ious = [i if i is not None else 0.0 for i in series["ious"]]
        plt.scatter(sizes, ious, alpha=0.4, color='royalblue', s=20)
        
        # Add trend line if enough points
        if len(sizes) > 5:
            try:
                z = np.polyfit(sizes, ious, 1)
                p = np.poly1d(z)
                plt.plot(sorted(sizes), p(sorted(sizes)), "r--", alpha=0.8, label="Trend")
            except Exception as e:
                print(f"Warning: Could not compute trend line for IoU vs Size: {e}")
        
        plt.title(f"IoU vs Object Size - {model_name}")
        plt.xlabel("Object Size (Relative)")
        plt.ylabel("IoU")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_{scenario_name}_iou_vs_size.png"))
        plt.close()

        # 7. Zoom vs Object Size
        plt.figure(figsize=(8, 6))
        zooms = [z if z is not None else 1.0 for z in series["zoom_factors"]]
        plt.scatter(sizes, zooms, alpha=0.4, color='forestgreen', s=20)
        if len(sizes) > 5:
            try:
                z = np.polyfit(sizes, zooms, 1)
                p = np.poly1d(z)
                plt.plot(sorted(sizes), p(sorted(sizes)), "r--", alpha=0.8, label="Response")
            except Exception as e:
                print(f"Warning: Could not compute trend line for Zoom vs Size: {e}")
        
        plt.title(f"Zoom Responsiveness - {model_name}")
        plt.xlabel("Object Size (Relative)")
        plt.ylabel("Zoom Factor")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_{scenario_name}_zoom_vs_size.png"))
        plt.close()

    def plot_recovery_timeline(self, series: Dict[str, List], model_name: str, scenario_name: str):
        """6. Recovery Events Timeline"""
        frames = series["frame_ids"]
        ious = series["ious"]
        events = series["recovery_events"]
        
        plt.figure(figsize=(10, 3))
        plt.plot(frames, ious, color='gray', alpha=0.3)
        plt.scatter(events, [0.5]*len(events), color='red', marker='x', label="Recovery Point")
        plt.title(f"Recovery Events Timeline - {model_name}")
        plt.xlabel("Frame ID")
        plt.yticks([])
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"{model_name}_{scenario_name}_recovery_timeline.png"))
        plt.close()

    def plot_model_comparison(self, results_db: Dict[str, Dict], scenario_name: str):
        """8 & 9. Comparative plots across models."""
        models = list(results_db.keys())
        if not models: return

        # 8. Latency Comparison
        latencies = [results_db[m][scenario_name]["summary"]["latency"] for m in models if scenario_name in results_db[m]]
        model_labels = [m for m in models if scenario_name in results_db[m]]
        
        if latencies:
            plt.figure(figsize=(8, 5))
            colors = plt.cm.viridis(np.linspace(0, 0.8, len(latencies)))
            plt.bar(model_labels, latencies, color=colors)
            plt.title(f"Latency Comparison ({scenario_name})")
            plt.ylabel("Mean Latency (ms)")
            plt.savefig(os.path.join(self.output_dir, f"cmp_{scenario_name}_latency.png"))
            plt.close()

        # 9. IoU Multi-line Comparison
        plt.figure(figsize=(10, 5))
        for m in models:
            if scenario_name in results_db[m] and "series" in results_db[m][scenario_name]:
                ser = results_db[m][scenario_name]["series"]
                plt.plot(ser["frame_ids"], ser["ious"], label=m, alpha=0.7)
        
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
        plt.title(f"Model Performance Comparison - IoU ({scenario_name})")
        plt.xlabel("Frame ID")
        plt.ylabel("IoU")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"cmp_{scenario_name}_iou_all.png"))
        plt.close()

    def plot_advanced_research_components(self, results_db: Dict[str, Dict], scenario_name: str, baseline_model: str):
        """Generates the four advanced research plots."""
        models = [m for m in results_db.keys() if scenario_name in results_db[m]]
        
        for m in models:
            summary = results_db[m][scenario_name]["summary"]
            series = results_db[m][scenario_name]["series"]

            # 1. Failure Distribution (Pie Chart)
            if "failure_modes" in summary:
                f_modes = summary["failure_modes"]
                plt.figure(figsize=(6, 6))
                labels = [k.replace('_', ' ').title() for k in f_modes.keys()]
                sizes = list(f_modes.values())
                if sum(sizes) > 0:
                    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
                    plt.title(f"Failure Mode Analysis - {m}")
                    plt.savefig(os.path.join(self.output_dir, f"{m}_{scenario_name}_failure_pie.png"))
                plt.close()

            # 2. Zoom_t vs IoU_t+1 (Temporal Lag)
            if "zoom_to_iou_lag_corr" in summary:
                z = np.array(series["zoom_factors"][:-1])
                iou_next = np.array(series["ious"][1:])
                plt.figure(figsize=(8, 6))
                plt.scatter(z, iou_next, alpha=0.5, color='purple')
                if len(z) > 5:
                    poly = np.polyfit(z, iou_next, 1)
                    plt.plot(z, np.poly1d(poly)(z), "r--", alpha=0.8)
                plt.title(f"Temporal Causality (Lag=1) - {m}\nCorr: {summary['zoom_to_iou_lag_corr']:.3f}")
                plt.xlabel("Zoom Factor at time t")
                plt.ylabel("IoU at time t+1")
                plt.savefig(os.path.join(self.output_dir, f"{m}_{scenario_name}_lag_analysis.png"))
                plt.close()

            # 3. Scale Improvement (Bar Plot) - Only for adaptive models
            if m != baseline_model and "scale_improvement" in summary:
                improve = summary["scale_improvement"]
                plt.figure(figsize=(8, 5))
                plt.bar(improve.keys(), [v*100 for v in improve.values()], color='teal')
                plt.title(f"Scale Improvement vs Object Size - {m}")
                plt.ylabel("IoU Gain (%)")
                plt.xlabel("Object Size Category")
                plt.savefig(os.path.join(self.output_dir, f"{m}_{scenario_name}_scale_improvement.png"))
                plt.close()

        # 4. Statistical Comparison (Boxplot)
        if baseline_model in results_db and len(models) > 1:
            plt.figure(figsize=(10, 6))
            data = []
            labels = []
            for m in models:
                data.append(results_db[m][scenario_name]["series"]["ious"])
                labels.append(m)
            
            plt.boxplot(data, labels=labels, patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.5))
            plt.title(f"IoU Distribution Comparison ({scenario_name})")
            plt.ylabel("IoU")
            plt.savefig(os.path.join(self.output_dir, f"cmp_{scenario_name}_iou_boxplot.png"))
            plt.close()
