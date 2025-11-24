"""Visualization utilities for 3D animal RL examples."""

import numpy as np
from typing import List, Dict, Optional, Tuple
import gymnasium as gym

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import mujoco
    import mujoco.viewer
    HAS_MUJOCO_VIEWER = True
except ImportError:
    HAS_MUJOCO_VIEWER = False


class MuJoCoViewer:
    """Wrapper for MuJoCo viewer with trajectory recording."""
    
    def __init__(self, env: gym.Env, render: bool = True):
        """
        Initialize MuJoCo viewer.
        
        Args:
            env: Gymnasium MuJoCo environment
            render: Whether to render in real-time
        """
        self.env = env
        self.render = render
        self.viewer = None
        self.trajectory = []
        
        if render and HAS_MUJOCO_VIEWER:
            try:
                # Get MuJoCo model and data from environment
                if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'model'):
                    self.model = env.unwrapped.model
                    self.data = env.unwrapped.data
                elif hasattr(env, 'model'):
                    self.model = env.model
                    self.data = env.data
                else:
                    self.model = None
                    self.data = None
            except Exception:
                self.model = None
                self.data = None
    
    def render_step(self, obs: np.ndarray, action: np.ndarray, reward: float):
        """Render a single step."""
        if self.render and HAS_MUJOCO_VIEWER and self.model is not None:
            try:
                if self.viewer is None:
                    self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                
                # Record trajectory
                self.trajectory.append({
                    'obs': obs.copy(),
                    'action': action.copy(),
                    'reward': reward,
                })
                
                # Update viewer
                mujoco.mj_step(self.model, self.data)
                self.viewer.sync()
            except Exception:
                pass  # Viewer might not be available
    
    def close(self):
        """Close viewer."""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass


def plot_trajectory(
    trajectory: List[Dict],
    save_path: Optional[str] = None,
    title: str = "Trajectory",
):
    """
    Plot trajectory data.
    
    Args:
        trajectory: List of trajectory dictionaries with 'obs', 'action', 'reward'
        save_path: Path to save plot
        title: Plot title
    """
    if not HAS_MATPLOTLIB or not trajectory:
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Extract data
    rewards = [t['reward'] for t in trajectory]
    actions = np.array([t['action'] for t in trajectory])
    obs = np.array([t['obs'] for t in trajectory])
    
    # Plot rewards
    axes[0].plot(rewards, linewidth=2)
    axes[0].set_ylabel('Reward', fontsize=12)
    axes[0].set_title(f'{title} - Rewards', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot actions (first few dimensions)
    num_action_dims = min(actions.shape[1], 5)
    for i in range(num_action_dims):
        axes[1].plot(actions[:, i], label=f'Action {i}', alpha=0.7)
    axes[1].set_ylabel('Action', fontsize=12)
    axes[1].set_title('Actions', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot observations (first few dimensions)
    num_obs_dims = min(obs.shape[1], 5)
    for i in range(num_obs_dims):
        axes[2].plot(obs[:, i], label=f'Obs {i}', alpha=0.7)
    axes[2].set_xlabel('Step', fontsize=12)
    axes[2].set_ylabel('Observation', fontsize=12)
    axes[2].set_title('Observations', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_curves(
    rewards_history: List[float],
    save_path: Optional[str] = None,
    title: str = "Training Progress",
):
    """
    Plot training curves.
    
    Args:
        rewards_history: List of episode rewards
        save_path: Path to save plot
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute moving average
    window = min(10, len(rewards_history) // 4)
    if window > 1:
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        x_avg = np.arange(window-1, len(rewards_history))
        ax.plot(x_avg, moving_avg, label='Moving Average', linewidth=2, color='orange')
    
    ax.plot(rewards_history, alpha=0.3, label='Raw', color='blue')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curve saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_multi_agent(
    trajectories: List[List[Dict]],
    save_path: Optional[str] = None,
    title: str = "Multi-Agent Trajectories",
):
    """
    Visualize multiple agent trajectories.
    
    Args:
        trajectories: List of trajectories, one per agent
        save_path: Path to save plot
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    # Plot rewards for each agent
    for i, traj in enumerate(trajectories):
        rewards = [t['reward'] for t in traj]
        axes[0].plot(rewards, label=f'Agent {i+1}', color=colors[i], alpha=0.7)
    
    axes[0].set_ylabel('Reward', fontsize=12)
    axes[0].set_title(f'{title} - Rewards', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot positions (if available in observations)
    for i, traj in enumerate(trajectories):
        if traj and 'obs' in traj[0]:
            obs = np.array([t['obs'] for t in traj])
            # Try to extract position (usually first few dimensions)
            if obs.shape[1] >= 2:
                axes[1].plot(obs[:, 0], obs[:, 1], label=f'Agent {i+1}', color=colors[i], alpha=0.7)
    
    axes[1].set_xlabel('X Position', fontsize=12)
    axes[1].set_ylabel('Y Position', fontsize=12)
    axes[1].set_title('Agent Positions', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Multi-agent visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_animation(
    states: List[np.ndarray],
    save_path: Optional[str] = None,
    interval: int = 50,
):
    """
    Create animation from state sequence.
    
    Args:
        states: List of state arrays
        save_path: Path to save animation
        interval: Animation interval in milliseconds
    """
    if not HAS_MATPLOTLIB or not states:
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Initialize plot
    if len(states[0].shape) == 1 and states[0].shape[0] >= 2:
        line, = ax.plot([], [], 'o-', linewidth=2, markersize=8)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
    else:
        # Fallback: plot first two dimensions
        line, = ax.plot([], [], 'o-', linewidth=2, markersize=8)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
    
    def animate(frame):
        state = states[frame]
        if len(state.shape) == 1 and state.shape[0] >= 2:
            line.set_data([state[0]], [state[1]])
        return line,
    
    anim = FuncAnimation(fig, animate, frames=len(states), interval=interval, blit=True)
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

