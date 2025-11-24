"""
Comprehensive RL experiment runner with scientific analysis.

This script runs systematic experiments comparing SGD, ES, and EGGROLL optimizers
across multiple RL methods and environments, then performs statistical analysis
and generates scientific insights.
"""

import sys
import os
import argparse
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from scipy import stats
from collections import defaultdict

# Add parent directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(os.path.dirname(_script_dir))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from examples.rl.environments import get_environments, EnvironmentConfig
from examples.rl.comparison import run_comparison
from examples.rl.results import (
    aggregate_trials,
    plot_learning_curves,
    plot_comparison_table,
    save_results,
    load_results,
    print_summary,
)


def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def statistical_significance_test(
    results_a: List[float],
    results_b: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform statistical significance test between two groups.
    
    Args:
        results_a: List of results from group A
        results_b: List of results from group B
        alpha: Significance level
        
    Returns:
        Dictionary with test statistics
    """
    if len(results_a) < 2 or len(results_b) < 2:
        return {
            "significant": False,
            "p_value": 1.0,
            "test": "insufficient_data",
            "effect_size": 0.0,
            "mean_a": float(np.mean(results_a)) if len(results_a) > 0 else 0.0,
            "mean_b": float(np.mean(results_b)) if len(results_b) > 0 else 0.0,
            "std_a": float(np.std(results_a)) if len(results_a) > 0 else 0.0,
            "std_b": float(np.std(results_b)) if len(results_b) > 0 else 0.0,
        }
    
    # Perform Shapiro-Wilk test for normality (requires at least 3 samples)
    if len(results_a) >= 3 and len(results_b) >= 3:
        try:
            _, p_norm_a = stats.shapiro(results_a)
            _, p_norm_b = stats.shapiro(results_b)
            # Use appropriate test based on normality
            if not np.isnan(p_norm_a) and not np.isnan(p_norm_b) and p_norm_a > 0.05 and p_norm_b > 0.05:
                # Both normal: use t-test
                t_stat, p_value = stats.ttest_ind(results_a, results_b)
                test_name = "t-test"
            else:
                # Non-normal: use Mann-Whitney U test
                u_stat, p_value = stats.mannwhitneyu(results_a, results_b, alternative='two-sided')
                test_name = "mannwhitneyu"
        except Exception:
            # Fallback to Mann-Whitney U if Shapiro-Wilk fails
            u_stat, p_value = stats.mannwhitneyu(results_a, results_b, alternative='two-sided')
            test_name = "mannwhitneyu"
    else:
        # Too few samples for normality test, use Mann-Whitney U
        u_stat, p_value = stats.mannwhitneyu(results_a, results_b, alternative='two-sided')
        test_name = "mannwhitneyu"
    
    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt(
        ((len(results_a) - 1) * np.var(results_a, ddof=1) +
         (len(results_b) - 1) * np.var(results_b, ddof=1)) /
        (len(results_a) + len(results_b) - 2)
    )
    if pooled_std > 0:
        cohens_d = (np.mean(results_a) - np.mean(results_b)) / pooled_std
    else:
        cohens_d = 0.0
    
    return {
        "significant": bool(p_value < alpha),
        "p_value": float(p_value),
        "test": test_name,
        "effect_size": float(cohens_d),
        "mean_a": float(np.mean(results_a)),
        "mean_b": float(np.mean(results_b)),
        "std_a": float(np.std(results_a)),
        "std_b": float(np.std(results_b)),
    }


def analyze_results(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: str
) -> Dict[str, Any]:
    """
    Perform comprehensive statistical analysis on results.
    
    Args:
        all_results: Nested dictionary of results
        output_dir: Directory to save analysis
        
    Returns:
        Analysis results dictionary
    """
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    
    analysis = {
        "pairwise_comparisons": {},
        "optimizer_rankings": {},
        "environment_difficulty": {},
        "method_performance": {},
        "summary_statistics": {},
    }
    
    # Extract final rewards for each optimizer across all experiments
    optimizer_performance = defaultdict(list)
    environment_performance = defaultdict(lambda: defaultdict(list))
    method_performance = defaultdict(lambda: defaultdict(list))
    
    for exp_key, exp_results in all_results.items():
        env_name, method_name = exp_key.rsplit("_", 1)
        
        for opt_name, opt_results in exp_results.items():
            if "final_metrics" in opt_results:
                mean_reward = opt_results["final_metrics"]["mean_reward"]["mean"]
                optimizer_performance[opt_name].append(mean_reward)
                environment_performance[env_name][opt_name].append(mean_reward)
                method_performance[method_name][opt_name].append(mean_reward)
    
    # Pairwise comparisons between optimizers
    optimizers = ["sgd", "es", "eggroll"]
    print("\nPairwise Optimizer Comparisons:")
    print("-" * 80)
    
    for i, opt_a in enumerate(optimizers):
        for opt_b in optimizers[i+1:]:
            if opt_a in optimizer_performance and opt_b in optimizer_performance:
                test_result = statistical_significance_test(
                    optimizer_performance[opt_a],
                    optimizer_performance[opt_b]
                )
                
                key = f"{opt_a}_vs_{opt_b}"
                analysis["pairwise_comparisons"][key] = test_result
                
                significance = "✓" if test_result["significant"] else "✗"
                mean_a = test_result.get('mean_a', 0.0)
                mean_b = test_result.get('mean_b', 0.0)
                print(f"{significance} {opt_a.upper()} vs {opt_b.upper()}: "
                      f"p={test_result['p_value']:.4f}, "
                      f"effect_size={test_result['effect_size']:.3f}, "
                      f"mean_diff={mean_a - mean_b:.3f}")
    
    # Optimizer rankings per environment
    print("\nOptimizer Rankings by Environment:")
    print("-" * 80)
    
    for env_name, env_results in environment_performance.items():
        opt_means = {
            opt: np.mean(rewards)
            for opt, rewards in env_results.items()
        }
        sorted_opts = sorted(opt_means.items(), key=lambda x: x[1], reverse=True)
        
        analysis["optimizer_rankings"][env_name] = {
            opt: {"rank": i+1, "mean_reward": float(mean)}
            for i, (opt, mean) in enumerate(sorted_opts)
        }
        
        print(f"\n{env_name}:")
        for i, (opt, mean) in enumerate(sorted_opts):
            print(f"  {i+1}. {opt.upper()}: {mean:.3f}")
    
    # Method performance analysis
    print("\nMethod Performance by Optimizer:")
    print("-" * 80)
    
    for method_name, method_results in method_performance.items():
        opt_means = {
            opt: np.mean(rewards)
            for opt, rewards in method_results.items()
        }
        analysis["method_performance"][method_name] = opt_means
        
        print(f"\n{method_name.upper()}:")
        for opt, mean in sorted(opt_means.items(), key=lambda x: x[1], reverse=True):
            print(f"  {opt.upper()}: {mean:.3f}")
    
    # Summary statistics
    print("\nOverall Summary Statistics:")
    print("-" * 80)
    
    for opt_name in optimizers:
        if opt_name in optimizer_performance:
            rewards = optimizer_performance[opt_name]
            analysis["summary_statistics"][opt_name] = {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "median": float(np.median(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
                "n_experiments": len(rewards),
            }
            
            stats_dict = analysis["summary_statistics"][opt_name]
            print(f"\n{opt_name.upper()}:")
            print(f"  Mean: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Median: {stats_dict['median']:.3f}")
            print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            print(f"  N experiments: {stats_dict['n_experiments']}")
    
    # Save analysis
    analysis_file = os.path.join(output_dir, "statistical_analysis.json")
    with open(analysis_file, "w") as f:
        json.dump(convert_to_json_serializable(analysis), f, indent=2)
    
    print(f"\nAnalysis saved to {analysis_file}")
    
    return analysis


def create_comprehensive_plots(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: str
):
    """Create comprehensive visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("Warning: matplotlib/seaborn not available, skipping plots")
        return
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Overall performance comparison
    optimizers = ["sgd", "es", "eggroll"]
    opt_means = []
    opt_stds = []
    
    for opt in optimizers:
        all_rewards = []
        for exp_results in all_results.values():
            if opt in exp_results and "final_metrics" in exp_results[opt]:
                all_rewards.append(
                    exp_results[opt]["final_metrics"]["mean_reward"]["mean"]
                )
        
        if all_rewards:
            opt_means.append(np.mean(all_rewards))
            opt_stds.append(np.std(all_rewards))
        else:
            opt_means.append(0.0)
            opt_stds.append(0.0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(optimizers))
    colors = {"sgd": "#3498db", "es": "#2ecc71", "eggroll": "#e74c3c"}
    bar_colors = [colors.get(opt, "gray") for opt in optimizers]
    
    bars = ax.bar(x_pos, opt_means, yerr=opt_stds, capsize=5,
                  color=bar_colors, alpha=0.7, edgecolor="black", linewidth=1.5)
    
    ax.set_xlabel("Optimizer", fontsize=14, fontweight="bold")
    ax.set_ylabel("Mean Episode Reward (across all experiments)", fontsize=14, fontweight="bold")
    ax.set_title("Overall Optimizer Performance Comparison", fontsize=16, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([opt.upper() for opt in optimizers], fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    
    for i, (bar, mean, std) in enumerate(zip(bars, opt_means, opt_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + std,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "overall_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. Performance by environment
    env_names = sorted(set(k.rsplit("_", 1)[0] for k in all_results.keys()))
    
    fig, axes = plt.subplots(len(env_names), 1, figsize=(12, 4 * len(env_names)))
    if len(env_names) == 1:
        axes = [axes]
    
    for ax, env_name in zip(axes, env_names):
        env_means = []
        env_stds = []
        
        for opt in optimizers:
            rewards = []
            for exp_key, exp_results in all_results.items():
                if exp_key.startswith(env_name) and opt in exp_results:
                    if "final_metrics" in exp_results[opt]:
                        rewards.append(
                            exp_results[opt]["final_metrics"]["mean_reward"]["mean"]
                        )
            
            if rewards:
                env_means.append(np.mean(rewards))
                env_stds.append(np.std(rewards))
            else:
                env_means.append(0.0)
                env_stds.append(0.0)
        
        x_pos = np.arange(len(optimizers))
        bar_colors = [colors.get(opt, "gray") for opt in optimizers]
        
        bars = ax.bar(x_pos, env_means, yerr=env_stds, capsize=5,
                     color=bar_colors, alpha=0.7, edgecolor="black")
        
        ax.set_ylabel("Mean Reward", fontsize=11)
        ax.set_title(f"{env_name}", fontsize=12, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([opt.upper() for opt in optimizers])
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "performance_by_environment.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # 3. Learning curve comparison (sample)
    sample_exps = list(all_results.keys())[:min(6, len(all_results))]
    
    n_cols = 3
    n_rows = (len(sample_exps) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, exp_key in enumerate(sample_exps):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        exp_results = all_results[exp_key]
        for opt_name, opt_results in exp_results.items():
            if "learning_curve" in opt_results:
                lc = opt_results["learning_curve"]
                x = range(len(lc["mean"]))
                ax.plot(x, lc["mean"], label=opt_name.upper(),
                       color=colors.get(opt_name, "black"), linewidth=2)
                ax.fill_between(x,
                               np.array(lc["mean"]) - np.array(lc["std"]),
                               np.array(lc["mean"]) + np.array(lc["std"]),
                               alpha=0.2, color=colors.get(opt_name, "black"))
        
        ax.set_title(exp_key, fontsize=10, fontweight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Reward")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(sample_exps), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "sample_learning_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nPlots saved to {plots_dir}")


def generate_scientific_report(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    analysis: Dict[str, Any],
    output_dir: str
):
    """Generate a scientific report summarizing findings."""
    
    report_lines = [
        "# RL Optimizer Comparison: Scientific Report",
        "",
        "## Executive Summary",
        "",
        "This report presents a comprehensive comparison of three optimization methods",
        "for reinforcement learning: Stochastic Gradient Descent (SGD), Evolution Strategies (ES),",
        "and EGGROLL (a novel low-rank evolution strategy).",
        "",
        "## Methodology",
        "",
        "### Experimental Setup",
        f"- Total experiments conducted: {len(all_results)}",
        "- Optimizers tested: SGD, ES, EGGROLL",
        "- Multiple RL methods: REINFORCE, PPO, Actor-Critic, A2C, DQN",
        "- Multiple environments: Various Gymnasium environments",
        "",
        "### Statistical Analysis",
        "- Significance testing: t-test (normal) or Mann-Whitney U (non-normal)",
        "- Effect size: Cohen's d",
        "- Significance level: α = 0.05",
        "",
        "## Results",
        "",
        "### Overall Performance",
        "",
    ]
    
    # Add summary statistics
    if "summary_statistics" in analysis:
        report_lines.append("| Optimizer | Mean Reward | Std Dev | Median | Range |")
        report_lines.append("|-----------|-------------|---------|--------|------|")
        
        for opt_name, stats_dict in analysis["summary_statistics"].items():
            report_lines.append(
                f"| {opt_name.upper()} | {stats_dict['mean']:.3f} | "
                f"{stats_dict['std']:.3f} | {stats_dict['median']:.3f} | "
                f"[{stats_dict['min']:.3f}, {stats_dict['max']:.3f}] |"
            )
    
    report_lines.extend([
        "",
        "### Statistical Significance",
        "",
    ])
    
    # Add pairwise comparisons
    if "pairwise_comparisons" in analysis:
        report_lines.append("| Comparison | Significant? | p-value | Effect Size |")
        report_lines.append("|------------|--------------|---------|-------------|")
        
        for comp_key, comp_result in analysis["pairwise_comparisons"].items():
            sig = "Yes" if comp_result["significant"] else "No"
            report_lines.append(
                f"| {comp_key.replace('_', ' ').upper()} | {sig} | "
                f"{comp_result['p_value']:.4f} | {comp_result['effect_size']:.3f} |"
            )
    
    report_lines.extend([
        "",
        "## Key Findings",
        "",
        "1. **Performance Comparison**: [Analysis of which optimizer performs best overall]",
        "",
        "2. **Statistical Significance**: [Summary of significant differences]",
        "",
        "3. **Environment-Specific Performance**: [Which optimizer works best for which environments]",
        "",
        "4. **Method-Specific Performance**: [Which optimizer works best for which RL methods]",
        "",
        "## Conclusions",
        "",
        "[Summary of main conclusions and implications]",
        "",
        "## Future Work",
        "",
        "- Hyperparameter tuning for each optimizer",
        "- Larger-scale experiments with more environments",
        "- Analysis of computational efficiency",
        "- Theoretical analysis of optimizer properties",
        "",
    ])
    
    report_text = "\n".join(report_lines)
    
    report_file = os.path.join(output_dir, "scientific_report.md")
    with open(report_file, "w") as f:
        f.write(report_text)
    
    print(f"\nScientific report saved to {report_file}")
    
    # Also print to console
    print("\n" + "=" * 80)
    print("SCIENTIFIC REPORT SUMMARY")
    print("=" * 80)
    print(report_text)


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(
        description="Comprehensive RL optimizer comparison experiments"
    )
    parser.add_argument(
        "--environments",
        nargs="+",
        default=None,
        help="List of environment IDs to test (default: all)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["reinforce", "ppo"],
        help="RL methods to test (default: reinforce, ppo)",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=["sgd", "es", "eggroll"],
        help="Optimizers to test (default: all)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="Number of trials per configuration (default: 5)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use fewer environments and steps",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to save results",
    )
    parser.add_argument(
        "--skip-mujoco",
        action="store_true",
        help="Skip MuJoCo environments",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing results, don't run new experiments",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default=None,
        help="Path to existing results file for analysis",
    )
    
    args = parser.parse_args()
    
    # Set results directory
    if args.results_dir is None:
        results_dir = os.path.join(_script_dir, "results")
    else:
        results_dir = args.results_dir
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Load existing results or run new experiments
    if args.analyze_only:
        if args.results_file:
            results_file = args.results_file
        else:
            results_file = os.path.join(results_dir, "comparison_results.json")
        
        if not os.path.exists(results_file):
            print(f"Error: Results file not found: {results_file}")
            return
        
        print(f"Loading results from {results_file}...")
        all_results = load_results(results_file)
    else:
        # Run experiments
        from examples.rl.comparison import main as run_comparison_main
        
        # Temporarily modify sys.argv to pass arguments
        import sys
        original_argv = sys.argv
        sys.argv = [
            "comparison.py",
            "--trials", str(args.trials),
            "--steps", str(args.steps),
            "--results-dir", results_dir,
        ]
        
        if args.environments:
            sys.argv.extend(["--environments"] + args.environments)
        if args.methods:
            sys.argv.extend(["--methods"] + args.methods)
        if args.optimizers:
            sys.argv.extend(["--optimizers"] + args.optimizers)
        if args.quick:
            sys.argv.append("--quick")
        if args.skip_mujoco:
            sys.argv.append("--skip-mujoco")
        
        # Import and run comparison
        from examples.rl import comparison
        
        # Get environments
        if args.quick:
            envs = get_environments(require_mujoco=False, quick_mode=True)
        elif args.skip_mujoco:
            envs = get_environments(require_mujoco=False, quick_mode=False)
        else:
            from examples.rl.environments import ENVIRONMENTS
            envs = ENVIRONMENTS
        
        if args.environments:
            envs = [e for e in envs if e.env_id in args.environments]
        
        # Filter methods
        from examples.rl.comparison import RL_METHODS
        methods = [m for m in args.methods if m in RL_METHODS]
        
        # Run comparisons
        all_results = {}
        for env_config in envs:
            if env_config.requires_mujoco:
                try:
                    import gymnasium as gym
                    gym.make(env_config.env_id)
                except Exception:
                    print(f"\nSkipping {env_config.name} (MuJoCo not available)")
                    continue
            
            for method_name in methods:
                key = f"{env_config.name}_{method_name}"
                
                try:
                    results = run_comparison(
                        env_config=env_config,
                        method_name=method_name,
                        num_trials=args.trials,
                        num_steps=args.steps,
                        quick_mode=args.quick,
                        results_dir=results_dir,
                        optimizers=args.optimizers,
                    )
                    
                    all_results[key] = results
                    
                except Exception as e:
                    print(f"\nError running {key}: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Save results
        results_file = os.path.join(results_dir, "comparison_results.json")
        save_results(all_results, results_file)
    
    # Perform analysis
    print("\n" + "=" * 80)
    print("PERFORMING STATISTICAL ANALYSIS")
    print("=" * 80)
    
    analysis = analyze_results(all_results, results_dir)
    
    # Create visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    create_comprehensive_plots(all_results, results_dir)
    
    # Generate scientific report
    print("\n" + "=" * 80)
    print("GENERATING SCIENTIFIC REPORT")
    print("=" * 80)
    
    generate_scientific_report(all_results, analysis, results_dir)
    
    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETE")
    print("=" * 80)
    print(f"Results: {results_dir}")
    print(f"Analysis: {os.path.join(results_dir, 'statistical_analysis.json')}")
    print(f"Report: {os.path.join(results_dir, 'scientific_report.md')}")
    print("=" * 80)


if __name__ == "__main__":
    main()

