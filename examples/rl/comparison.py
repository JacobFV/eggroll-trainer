"""Main RL comparison script comparing SGD, ES, and EGGROLL optimizers."""

import sys
import os
import argparse
import time
import torch
import numpy as np
import gymnasium as gym
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(os.path.dirname(_script_dir))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from examples.rl.environments import (
    ENVIRONMENTS,
    get_environments,
    EnvironmentConfig,
)
from examples.rl.models import PolicyNetwork, ValueNetwork, QNetwork
from examples.rl.framework import (
    REINFORCETrainer,
    PPOTrainer,
    ActorCriticTrainer,
    A2CTrainer,
    DQNTrainer,
)
from examples.rl.results import (
    aggregate_trials,
    plot_learning_curves,
    plot_comparison_table,
    save_results,
    print_summary,
)


# RL method registry
RL_METHODS = {
    "reinforce": REINFORCETrainer,
    "ppo": PPOTrainer,
    "actor_critic": ActorCriticTrainer,
    "a2c": A2CTrainer,
    "dqn": DQNTrainer,
}

OPTIMIZERS = ["sgd", "es", "eggroll"]


def create_policy(env_config: EnvironmentConfig, method_name: str, device: torch.device) -> torch.nn.Module:
    """Create policy network for environment."""
    obs_dim = env_config.observation_dim
    action_dim = env_config.action_dim
    
    if method_name == "dqn":
        # DQN always uses QNetwork
        return QNetwork(obs_dim, action_dim, hidden_dims=(64, 64))
    else:
        # Policy network for continuous/discrete
        return PolicyNetwork(
            obs_dim,
            action_dim,
            hidden_dims=(64, 64),
            continuous=env_config.continuous_actions,
        )


def run_trial(
    env_config: EnvironmentConfig,
    method_name: str,
    optimizer_type: str,
    num_steps: int,
    seed: int,
    quick_mode: bool = False,
) -> Dict[str, Any]:
    """
    Run a single trial.
    
    Args:
        env_config: Environment configuration
        method_name: RL method name
        optimizer_type: Optimizer type ('sgd', 'es', 'eggroll')
        num_steps: Number of training steps
        seed: Random seed
        quick_mode: If True, use reduced settings
        
    Returns:
        Dictionary with trial results
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create environment
    env = gym.make(env_config.env_id)
    
    # Create policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = create_policy(env_config, method_name, device)
    
    # Get hyperparameters
    hp = env_config.hyperparameters or {}
    learning_rate = hp.get("learning_rate", 0.01)
    population_size = hp.get("population_size", 32)
    sigma = hp.get("sigma", 0.1)
    episodes_per_generation = hp.get("episodes_per_generation", 10)
    
    if quick_mode:
        num_steps = min(num_steps, 20)
        episodes_per_generation = 3
    
    # Create trainer
    trainer_class = RL_METHODS[method_name]
    
    if method_name == "actor_critic" or method_name == "a2c":
        value_network = ValueNetwork(env_config.observation_dim, hidden_dims=(64, 64))
        trainer = trainer_class(
            env=env,
            policy=policy,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            device=device,
            value_network=value_network,
            population_size=population_size,
            sigma=sigma,
            episodes_per_generation=episodes_per_generation,
        )
    elif method_name == "dqn":
        # DQN needs QNetwork, not PolicyNetwork
        if not isinstance(policy, QNetwork):
            policy = QNetwork(env_config.observation_dim, env_config.action_dim, hidden_dims=(64, 64))
            policy = policy.to(device)
        
        trainer = trainer_class(
            env=env,
            policy=policy,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            device=device,
            population_size=population_size,
            sigma=sigma,
            episodes_per_generation=episodes_per_generation,
        )
    else:
        trainer = trainer_class(
            env=env,
            policy=policy,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            device=device,
            population_size=population_size,
            sigma=sigma,
            episodes_per_generation=episodes_per_generation,
        )
    
    # Training loop
    learning_curve = []
    start_time = time.time()
    
    for step in range(num_steps):
        metrics = trainer.step()
        
        # Evaluate periodically
        if step % max(1, num_steps // 20) == 0 or step == num_steps - 1:
            eval_results = trainer.evaluate(num_episodes=5)
            learning_curve.append(eval_results["mean_reward"])
    
    training_time = time.time() - start_time
    
    # Final evaluation
    final_eval = trainer.evaluate(num_episodes=10)
    
    env.close()
    
    return {
        "learning_curve": learning_curve,
        "final_metrics": final_eval,
        "training_time": training_time,
        "seed": seed,
    }


def run_comparison(
    env_config: EnvironmentConfig,
    method_name: str,
    num_trials: int = 5,
    num_steps: int = 100,
    quick_mode: bool = False,
    results_dir: Optional[str] = None,
    optimizers: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run comparison across all optimizers for a given environment and method.
    
    Args:
        env_config: Environment configuration
        method_name: RL method name
        num_trials: Number of trials per optimizer
        num_steps: Number of training steps
        quick_mode: If True, use reduced settings
        results_dir: Directory to save results
        
    Returns:
        Dictionary mapping optimizer names to aggregated results
    """
    print(f"\n{'='*80}")
    print(f"Environment: {env_config.name} | Method: {method_name.upper()}")
    print(f"{'='*80}")
    
    all_results = {}
    
    optimizers_to_use = optimizers or OPTIMIZERS
    
    for optimizer_type in optimizers_to_use:
        print(f"\nOptimizer: {optimizer_type.upper()}")
        print("-" * 80)
        
        trial_results = []
        
        for trial in range(num_trials):
            seed = 42 + trial * 1000
            print(f"  Trial {trial + 1}/{num_trials} (seed={seed})...", end=" ", flush=True)
            
            try:
                result = run_trial(
                    env_config=env_config,
                    method_name=method_name,
                    optimizer_type=optimizer_type,
                    num_steps=num_steps,
                    seed=seed,
                    quick_mode=quick_mode,
                )
                trial_results.append(result)
                
                final_reward = result["final_metrics"]["mean_reward"]
                print(f"✓ Final reward: {final_reward:.2f}")
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
        
        if trial_results:
            aggregated = aggregate_trials(trial_results)
            all_results[optimizer_type] = aggregated
    
    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RL Optimizer Comparison")
    parser.add_argument(
        "--environments",
        nargs="+",
        default=None,
        help="List of environment IDs to test (default: all)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(RL_METHODS.keys()),
        help=f"RL methods to test (default: all). Options: {list(RL_METHODS.keys())}",
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=OPTIMIZERS,
        help=f"Optimizers to test (default: all). Options: {OPTIMIZERS}",
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
        help="Directory to save results (default: examples/rl/results/)",
    )
    parser.add_argument(
        "--skip-mujoco",
        action="store_true",
        help="Skip MuJoCo environments",
    )
    
    args = parser.parse_args()
    
    # Set results directory
    if args.results_dir is None:
        results_dir = os.path.join(_script_dir, "results")
    else:
        results_dir = args.results_dir
    
    os.makedirs(results_dir, exist_ok=True)
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get environments
    if args.quick:
        envs = get_environments(require_mujoco=False, quick_mode=True)
    elif args.skip_mujoco:
        envs = get_environments(require_mujoco=False, quick_mode=False)
    else:
        envs = ENVIRONMENTS
    
    if args.environments:
        envs = [e for e in envs if e.env_id in args.environments]
    
    # Filter optimizers
    optimizers_to_test = args.optimizers
    
    # Filter methods
    methods = [m for m in args.methods if m in RL_METHODS]
    
    print("=" * 80)
    print("RL OPTIMIZER COMPARISON")
    print("=" * 80)
    print(f"Environments: {len(envs)}")
    print(f"Methods: {methods}")
    print(f"Optimizers: {optimizers_to_test}")
    print(f"Trials per config: {args.trials}")
    print(f"Steps per trial: {args.steps}")
    print(f"Quick mode: {args.quick}")
    print("=" * 80)
    
    # Run comparisons
    all_results = {}
    
    for env_config in envs:
        # Skip MuJoCo environments if not available
        if env_config.requires_mujoco:
            try:
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
                    optimizers=optimizers_to_test,
                )
                
                all_results[key] = results
                
                # Save individual comparison plots
                plot_path = os.path.join(plots_dir, f"{key}_learning_curves.png")
                plot_learning_curves(results, save_path=plot_path, title=f"{env_config.name} - {method_name.upper()}")
                
                plot_path = os.path.join(plots_dir, f"{key}_comparison.png")
                plot_comparison_table(results, save_path=plot_path, metric="mean_reward")
                
                # Print summary
                print_summary(results)
                
            except Exception as e:
                print(f"\nError running {key}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save all results
    results_file = os.path.join(results_dir, "comparison_results.json")
    save_results(all_results, results_file)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

