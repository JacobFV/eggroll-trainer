"""Basic test to verify RL framework setup."""

import sys
import os

# Add parent directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(os.path.dirname(_script_dir))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import torch
import gymnasium as gym

from examples.rl.environments import get_environments
from examples.rl.models import PolicyNetwork, QNetwork
from examples.rl.framework import REINFORCETrainer, DQNTrainer
from examples.rl.comparison import create_policy


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    try:
        from examples.rl import models, environments, framework, results, comparison
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_training():
    """Test basic training with one environment and method."""
    print("\nTesting basic training...")
    
    try:
        # Get a simple environment
        envs = get_environments(quick_mode=True)
        if not envs:
            print("✗ No environments available")
            return False
        
        env_config = envs[0]
        print(f"  Using environment: {env_config.name}")
        
        # Create environment
        env = gym.make(env_config.env_id)
        
        # Test REINFORCE with SGD
        print("  Testing REINFORCE with SGD...")
        device = torch.device("cpu")
        policy = create_policy(env_config, "reinforce", device)
        
        trainer = REINFORCETrainer(
            env=env,
            policy=policy,
            optimizer_type="sgd",
            learning_rate=0.01,
            device=device,
        )
        
        # Run a few training steps
        for i in range(3):
            metrics = trainer.step()
            print(f"    Step {i+1}: loss={metrics.get('loss', 0):.4f}")
        
        # Evaluate
        eval_results = trainer.evaluate(num_episodes=2)
        print(f"    Evaluation: mean_reward={eval_results['mean_reward']:.2f}")
        
        env.close()
        print("✓ Basic training test passed")
        return True
        
    except Exception as e:
        print(f"✗ Training test error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("RL Framework Basic Test")
    print("=" * 60)
    
    success = True
    success &= test_imports()
    success &= test_basic_training()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)

