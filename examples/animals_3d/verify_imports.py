"""Verify all imports work correctly."""

import sys
import os

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def test_imports():
    """Test all imports."""
    print("=" * 70)
    print("Import Verification")
    print("=" * 70)
    
    all_passed = True
    
    # Test RL imports
    try:
        from examples.rl.models import PolicyNetwork, ValueNetwork, QNetwork
        print("\n✓ examples.rl.models")
        print("  ✓ PolicyNetwork")
        print("  ✓ ValueNetwork")
        print("  ✓ QNetwork")
    except Exception as e:
        print(f"\n✗ examples.rl.models - Error: {e}")
        all_passed = False
    
    try:
        from examples.rl.framework import PPOTrainer, REINFORCETrainer, ActorCriticTrainer
        print("\n✓ examples.rl.framework")
        print("  ✓ PPOTrainer")
        print("  ✓ REINFORCETrainer")
        print("  ✓ ActorCriticTrainer")
    except Exception as e:
        print(f"\n✗ examples.rl.framework - Error: {e}")
        all_passed = False
    
    # Test animals_3d imports
    try:
        from examples.animals_3d import utils
        from examples.animals_3d.utils import EnvironmentWrapper, RewardShaper, MultiAgentCoordinator
        print("\n✓ examples.animals_3d.utils")
        print("  ✓ EnvironmentWrapper")
        print("  ✓ RewardShaper")
        print("  ✓ MultiAgentCoordinator")
    except Exception as e:
        print(f"\n✗ examples.animals_3d.utils - Error: {e}")
        all_passed = False
    
    try:
        from examples.animals_3d import visualization
        from examples.animals_3d.visualization import MuJoCoViewer, plot_training_curves
        print("\n✓ examples.animals_3d.visualization")
        print("  ✓ MuJoCoViewer")
        print("  ✓ plot_training_curves")
    except Exception as e:
        print(f"\n✗ examples.animals_3d.visualization - Error: {e}")
        all_passed = False
    
    try:
        from examples.animals_3d import mujoco_utils
        from examples.animals_3d.mujoco_utils import load_custom_model, create_cloth_mesh
        print("\n✓ examples.animals_3d.mujoco_utils")
        print("  ✓ load_custom_model")
        print("  ✓ create_cloth_mesh")
    except Exception as e:
        print(f"\n✗ examples.animals_3d.mujoco_utils - Error: {e}")
        all_passed = False
    
    try:
        from examples.animals_3d import world_builder
        from examples.animals_3d.world_builder import ObstacleCourseBuilder, TerrainGenerator
        print("\n✓ examples.animals_3d.world_builder")
        print("  ✓ ObstacleCourseBuilder")
        print("  ✓ TerrainGenerator")
    except Exception as e:
        print(f"\n✗ examples.animals_3d.world_builder - Error: {e}")
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All imports verified successfully!")
    else:
        print("❌ Some imports failed")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
