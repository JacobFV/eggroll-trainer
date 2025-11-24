"""Results analysis and visualization for RL comparisons."""

import json
import os
from typing import Dict, List, Any, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def aggregate_trials(trial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate results across multiple trials.
    
    Args:
        trial_results: List of result dictionaries from each trial
        
    Returns:
        Aggregated results with mean, std, min, max
    """
    if not trial_results:
        return {}
    
    aggregated = {}
    
    # Aggregate learning curves
    if "learning_curve" in trial_results[0]:
        learning_curves = [r["learning_curve"] for r in trial_results]
        max_len = max(len(lc) for lc in learning_curves)
        
        # Pad curves to same length
        padded_curves = []
        for lc in learning_curves:
            if len(lc) < max_len:
                # Repeat last value
                lc = lc + [lc[-1]] * (max_len - len(lc))
            padded_curves.append(lc)
        
        learning_curves_array = np.array(padded_curves)
        aggregated["learning_curve"] = {
            "mean": learning_curves_array.mean(axis=0).tolist(),
            "std": learning_curves_array.std(axis=0).tolist(),
            "min": learning_curves_array.min(axis=0).tolist(),
            "max": learning_curves_array.max(axis=0).tolist(),
        }
    
    # Aggregate final metrics
    if "final_metrics" in trial_results[0]:
        final_metrics_list = [r["final_metrics"] for r in trial_results]
        aggregated["final_metrics"] = {}
        
        for key in final_metrics_list[0].keys():
            values = [fm[key] for fm in final_metrics_list]
            aggregated["final_metrics"][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    
    # Aggregate training time
    if "training_time" in trial_results[0]:
        training_times = [r["training_time"] for r in trial_results]
        aggregated["training_time"] = {
            "mean": float(np.mean(training_times)),
            "std": float(np.std(training_times)),
            "min": float(np.min(training_times)),
            "max": float(np.max(training_times)),
        }
    
    return aggregated


def plot_learning_curves(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    title: str = "Learning Curves",
):
    """
    Plot learning curves for different optimizers.
    
    Args:
        results: Dictionary mapping optimizer names to aggregated results
        save_path: Path to save plot (if None, displays)
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {"sgd": "blue", "es": "green", "eggroll": "red"}
    markers = {"sgd": "o", "es": "s", "eggroll": "^"}
    
    for opt_name, opt_results in results.items():
        if "learning_curve" not in opt_results:
            continue
        
        lc = opt_results["learning_curve"]
        x = range(len(lc["mean"]))
        
        ax.plot(x, lc["mean"], label=opt_name.upper(), color=colors.get(opt_name, "black"),
                marker=markers.get(opt_name, "o"), linewidth=2, markersize=4)
        
        # Add shaded error bars
        ax.fill_between(
            x,
            np.array(lc["mean"]) - np.array(lc["std"]),
            np.array(lc["mean"]) + np.array(lc["std"]),
            alpha=0.2,
            color=colors.get(opt_name, "black"),
        )
    
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison_table(
    results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    metric: str = "mean_reward",
):
    """
    Create comparison table plot.
    
    Args:
        results: Dictionary mapping optimizer names to aggregated results
        save_path: Path to save plot
        metric: Metric to compare
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    optimizers = list(results.keys())
    values = []
    errors = []
    
    for opt_name in optimizers:
        if "final_metrics" in results[opt_name] and metric in results[opt_name]["final_metrics"]:
            values.append(results[opt_name]["final_metrics"][metric]["mean"])
            errors.append(results[opt_name]["final_metrics"][metric]["std"])
        else:
            values.append(0.0)
            errors.append(0.0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x_pos = np.arange(len(optimizers))
    colors = [{"sgd": "blue", "es": "green", "eggroll": "red"}.get(opt, "gray") for opt in optimizers]
    
    bars = ax.bar(x_pos, values, yerr=errors, capsize=5, color=colors, alpha=0.7, edgecolor="black")
    
    ax.set_xlabel("Optimizer", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Final Performance Comparison: {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([opt.upper() for opt in optimizers])
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars, values, errors)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + err,
                f'{val:.2f}±{err:.2f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_results(results: Dict[str, Any], filepath: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def print_summary(results: Dict[str, Dict[str, Any]]):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    for opt_name, opt_results in results.items():
        print(f"\n{opt_name.upper()}:")
        
        if "final_metrics" in opt_results:
            for metric, stats in opt_results["final_metrics"].items():
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                      f"(min: {stats['min']:.4f}, max: {stats['max']:.4f})")
        
        if "training_time" in opt_results:
            time_stats = opt_results["training_time"]
            print(f"  Training Time: {time_stats['mean']:.2f}s ± {time_stats['std']:.2f}s")
    
    print("\n" + "=" * 80)

