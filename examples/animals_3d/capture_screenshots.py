"""Capture screenshots from 3D MuJoCo simulations for documentation."""

import sys
import os

# Add project root to path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import gymnasium as gym
from pathlib import Path

try:
    import mujoco
    import imageio
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    print("Warning: MuJoCo or imageio not available. Install with: pip install 'gymnasium[mujoco]' imageio")


def capture_screenshot(env, output_path: str, camera_id: int = 0, width: int = 800, height: int = 600):
    """
    Capture a screenshot from a MuJoCo environment.
    
    Args:
        env: Gymnasium MuJoCo environment
        output_path: Path to save screenshot
        camera_id: Camera ID to use (0 for default)
        width: Image width (max 800 due to framebuffer limits)
        height: Image height
    """
    if not HAS_MUJOCO:
        print(f"Skipping screenshot capture (MuJoCo not available): {output_path}")
        return False
    
    try:
        # Get MuJoCo model and data
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'model'):
            model = env.unwrapped.model
            data = env.unwrapped.data
        elif hasattr(env, 'model'):
            model = env.model
            data = env.data
        else:
            print(f"Could not access MuJoCo model for {output_path}")
            return False
        
        # Limit width to framebuffer size (typically 480)
        width = min(width, 480)
        height = min(height, 360)
        
        # Create renderer
        renderer = mujoco.Renderer(model, height=height, width=width)
        
        # Update physics to get a good pose
        mujoco.mj_forward(model, data)
        
        # Render with the specified camera
        renderer.update_scene(data)
        pixels = renderer.render()
        
        # Save image
        imageio.imwrite(output_path, pixels)
        print(f"Saved screenshot: {output_path} ({width}x{height})")
        return True
        
    except Exception as e:
        print(f"Error capturing screenshot {output_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def capture_ant_screenshots():
    """Capture screenshots from Ant environment."""
    print("Capturing Ant screenshots...")
    try:
        env = gym.make("Ant-v4", render_mode=None)
        obs, _ = env.reset()
        
        # Run a few steps to get interesting poses
        best_step = 0
        best_reward = -float('inf')
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            if reward > best_reward:
                best_reward = reward
                best_step = step
            if terminated or truncated:
                obs, _ = env.reset()
        
        # Reset and run to best step
        obs, _ = env.reset()
        for _ in range(min(best_step, 50)):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        # Capture screenshot
        output_dir = Path(_project_root) / "docs" / "assets" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        success = capture_screenshot(env, str(output_dir / "ant_walk.png"), width=480, height=360)
        
        env.close()
        return success
    except Exception as e:
        print(f"Error capturing Ant screenshots: {e}")
        import traceback
        traceback.print_exc()
        return False


def capture_halfcheetah_screenshots():
    """Capture screenshots from HalfCheetah environment."""
    print("Capturing HalfCheetah screenshots...")
    try:
        env = gym.make("HalfCheetah-v4", render_mode=None)
        obs, _ = env.reset()
        
        # Run a few steps
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        output_dir = Path(_project_root) / "docs" / "assets" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        success = capture_screenshot(env, str(output_dir / "halfcheetah_run.png"), width=480, height=360)
        
        env.close()
        return success
    except Exception as e:
        print(f"Error capturing HalfCheetah screenshots: {e}")
        import traceback
        traceback.print_exc()
        return False


def capture_humanoid_screenshots():
    """Capture screenshots from Humanoid environment."""
    print("Capturing Humanoid screenshots...")
    try:
        env = gym.make("Humanoid-v4", render_mode=None)
        obs, _ = env.reset()
        
        # Run a few steps
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        output_dir = Path(_project_root) / "docs" / "assets" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        success = capture_screenshot(env, str(output_dir / "humanoid_stand.png"), width=480, height=360)
        
        env.close()
        return success
    except Exception as e:
        print(f"Error capturing Humanoid screenshots: {e}")
        import traceback
        traceback.print_exc()
        return False


def capture_hopper_screenshots():
    """Capture screenshots from Hopper environment."""
    print("Capturing Hopper screenshots...")
    try:
        env = gym.make("Hopper-v4", render_mode=None)
        obs, _ = env.reset()
        
        # Run a few steps
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        output_dir = Path(_project_root) / "docs" / "assets" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        success = capture_screenshot(env, str(output_dir / "hopper.png"), width=480, height=360)
        
        env.close()
        return success
    except Exception as e:
        print(f"Error capturing Hopper screenshots: {e}")
        import traceback
        traceback.print_exc()
        return False


def capture_walker_screenshots():
    """Capture screenshots from Walker2d environment."""
    print("Capturing Walker2d screenshots...")
    try:
        env = gym.make("Walker2d-v4", render_mode=None)
        obs, _ = env.reset()
        
        # Run a few steps
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        output_dir = Path(_project_root) / "docs" / "assets" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        success = capture_screenshot(env, str(output_dir / "walker2d.png"), width=480, height=360)
        
        env.close()
        return success
    except Exception as e:
        print(f"Error capturing Walker2d screenshots: {e}")
        import traceback
        traceback.print_exc()
        return False


def capture_swimmer_screenshots():
    """Capture screenshots from Swimmer environment."""
    print("Capturing Swimmer screenshots...")
    try:
        env = gym.make("Swimmer-v4", render_mode=None)
        obs, _ = env.reset()
        
        # Run a few steps
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        output_dir = Path(_project_root) / "docs" / "assets" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        success = capture_screenshot(env, str(output_dir / "swimmer.png"), width=480, height=360)
        
        env.close()
        return success
    except Exception as e:
        print(f"Error capturing Swimmer screenshots: {e}")
        import traceback
        traceback.print_exc()
        return False


def capture_reacher_screenshots():
    """Capture screenshots from Reacher environment."""
    print("Capturing Reacher screenshots...")
    try:
        env = gym.make("Reacher-v4", render_mode=None)
        obs, _ = env.reset()
        
        # Run a few steps
        for _ in range(50):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        output_dir = Path(_project_root) / "docs" / "assets" / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        success = capture_screenshot(env, str(output_dir / "reacher.png"), width=480, height=360)
        
        env.close()
        return success
    except Exception as e:
        print(f"Error capturing Reacher screenshots: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Capture screenshots from all environments."""
    print("=" * 70)
    print("Capturing 3D Simulation Screenshots")
    print("=" * 70)
    
    if not HAS_MUJOCO:
        print("\nERROR: MuJoCo is not available.")
        print("Please install with: pip install 'gymnasium[mujoco]' imageio")
        return
    
    # Create output directory
    output_dir = Path(_project_root) / "docs" / "assets" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Capture screenshots from different environments
    results = []
    
    results.append(("Ant", capture_ant_screenshots()))
    results.append(("HalfCheetah", capture_halfcheetah_screenshots()))
    results.append(("Humanoid", capture_humanoid_screenshots()))
    results.append(("Hopper", capture_hopper_screenshots()))
    results.append(("Walker2d", capture_walker_screenshots()))
    results.append(("Swimmer", capture_swimmer_screenshots()))
    results.append(("Reacher", capture_reacher_screenshots()))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    successful = sum(1 for _, success in results if success)
    print(f"\nSuccessfully captured {successful}/{len(results)} screenshots")
    print(f"Screenshots saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

