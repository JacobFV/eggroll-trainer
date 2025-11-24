"""Run all examples to verify they work correctly."""

import sys
import os

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import subprocess
import time
from typing import List, Tuple


def run_example(script_path: str, timeout: int = 30) -> Tuple[bool, str]:
    """
    Run an example script and check if it executes without errors.
    
    Args:
        script_path: Path to the example script
        timeout: Maximum time to wait (seconds)
        
    Returns:
        Tuple of (success, message)
    """
    print(f"\n{'='*60}")
    print(f"Testing: {script_path}")
    print(f"{'='*60}")
    
    try:
        # Run with timeout
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=_project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        if result.returncode == 0:
            return True, "✓ Success"
        else:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            return False, f"✗ Failed: {error_msg}"
            
    except subprocess.TimeoutExpired:
        return False, "✗ Timeout"
    except Exception as e:
        return False, f"✗ Error: {str(e)[:200]}"


def main():
    """Run all examples."""
    print("=" * 70)
    print("Running All Animals 3D Examples")
    print("=" * 70)
    
    # Collect all example files
    examples_dir = os.path.dirname(__file__)
    
    simple_examples = [
        os.path.join(examples_dir, "simple_examples", "ant_walk.py"),
        os.path.join(examples_dir, "simple_examples", "halfcheetah_run.py"),
        os.path.join(examples_dir, "simple_examples", "humanoid_stand.py"),
    ]
    
    multi_agent_examples = [
        os.path.join(examples_dir, "multi_agent", "multi_ant_race.py"),
        os.path.join(examples_dir, "multi_agent", "predator_prey.py"),
        os.path.join(examples_dir, "multi_agent", "flocking.py"),
    ]
    
    advanced_examples = [
        os.path.join(examples_dir, "advanced", "custom_obstacle_course.py"),
        os.path.join(examples_dir, "advanced", "terrain_generation.py"),
        os.path.join(examples_dir, "advanced", "object_manipulation.py"),
        os.path.join(examples_dir, "advanced", "underwater_locomotion.py"),
        os.path.join(examples_dir, "advanced", "aerial_locomotion.py"),
        os.path.join(examples_dir, "advanced", "cloth_walking.py"),
        os.path.join(examples_dir, "advanced", "cloth_manipulation.py"),
        os.path.join(examples_dir, "advanced", "tendon_driven_robot.py"),
        os.path.join(examples_dir, "advanced", "skinned_character.py"),
        os.path.join(examples_dir, "advanced", "morphing_creature.py"),
        os.path.join(examples_dir, "advanced", "soft_rigid_interaction.py"),
        os.path.join(examples_dir, "advanced", "muscle_actuated.py"),
    ]
    
    all_examples = (
        [("Simple Examples", simple_examples)] +
        [("Multi-Agent Examples", multi_agent_examples)] +
        [("Advanced Examples", advanced_examples)]
    )
    
    results = []
    total = 0
    passed = 0
    
    for category, examples in all_examples:
        print(f"\n{'='*70}")
        print(f"{category}")
        print(f"{'='*70}")
        
        for example in examples:
            if os.path.exists(example):
                total += 1
                success, message = run_example(example, timeout=10)  # Short timeout for quick check
                results.append((example, success, message))
                if success:
                    passed += 1
                print(f"{message}")
            else:
                print(f"✗ File not found: {example}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total examples tested: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if passed == total:
        print("\n✅ All examples work correctly!")
    else:
        print("\n⚠️  Some examples had issues (may be due to missing MuJoCo dependencies)")
        print("\nFailed examples:")
        for example, success, message in results:
            if not success:
                print(f"  - {os.path.basename(example)}: {message}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

