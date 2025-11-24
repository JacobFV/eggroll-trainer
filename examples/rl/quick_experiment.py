"""
Quick experiment script to demonstrate the experimental framework.

This runs a small-scale experiment to verify everything works and
demonstrates the scientific workflow.
"""

import sys
import os

# Add parent directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(os.path.dirname(_script_dir))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from examples.rl.run_experiments import main

if __name__ == "__main__":
    # Override sys.argv for quick experiment
    import sys
    sys.argv = [
        "run_experiments.py",
        "--environments", "CartPole-v1", "Pendulum-v1",
        "--methods", "reinforce",
        "--optimizers", "sgd", "es", "eggroll",
        "--trials", "3",
        "--steps", "50",
        "--quick",
        "--skip-mujoco",
    ]
    
    print("=" * 80)
    print("QUICK EXPERIMENT: Testing RL Optimizer Comparison")
    print("=" * 80)
    print("This will run a small-scale experiment to verify the framework.")
    print("For full experiments, use: python run_experiments.py --help")
    print("=" * 80)
    print()
    
    main()

