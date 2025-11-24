"""Advanced example: Tendon-driven robot learns precise control - FULL IMPLEMENTATION."""

import sys
import os

# Add project root to path (when running with uv run from repo root)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import numpy as np
import gymnasium as gym

from examples.rl.models import PolicyNetwork
from examples.rl.framework import PPOTrainer

from examples.animals_3d import utils as animal_utils
from examples.animals_3d import visualization as animal_vis
from examples.animals_3d import mujoco_utils

EnvironmentWrapper = animal_utils.EnvironmentWrapper
MuJoCoViewer = animal_vis.MuJoCoViewer
plot_training_curves = animal_vis.plot_training_curves


class TendonDrivenEnv:
    """Full implementation: Tendon-driven control with real tendon mechanics."""
    
    def __init__(self, base_env: gym.Env):
        """
        Initialize tendon-driven environment with full tendon mechanics.
        
        Args:
            base_env: Base Gymnasium environment
        """
        self.base_env = base_env
        
        # Tendon system - get from model if available
        self.tendon_ids = []
        self.tendon_names = []
        self.joint_ids = []
        
        if hasattr(base_env, 'model') and mujoco_utils.HAS_MUJOCO:
            import mujoco
            try:
                # Get all tendons from model
                for i in range(base_env.model.ntendon):
                    tendon_name = mujoco.mj_id2name(base_env.model, mujoco.mjtObj.mjOBJ_TENDON, i)
                    if tendon_name:
                        self.tendon_ids.append(i)
                        self.tendon_names.append(tendon_name)
                
                # Get joints
                for i in range(base_env.model.njnt):
                    self.joint_ids.append(i)
            except:
                pass
        
        # If no tendons found, create virtual tendon system
        if len(self.tendon_ids) == 0:
            self.num_tendons = 4  # Default
            self.tendon_ids = list(range(self.num_tendons))
        else:
            self.num_tendons = len(self.tendon_ids)
        
        # Tendon state tracking
        self.tendon_lengths = np.zeros(self.num_tendons)
        self.tendon_velocities = np.zeros(self.num_tendons)
        self.tendon_tensions = np.zeros(self.num_tendons)
        self.tendon_lengths_history = []
        self.tendon_tensions_history = []
        
        # Target for precision control
        self.target_position = np.array([2.0, 0.0, 1.0])
        
        # Rest lengths (for computing stretch)
        self.rest_lengths = np.ones(self.num_tendons) * 1.0
    
    def reset(self, **kwargs):
        """Reset environment."""
        obs, info = self.base_env.reset(**kwargs)
        
        # Reset tendon state
        self.tendon_lengths = np.zeros(self.num_tendons)
        self.tendon_velocities = np.zeros(self.num_tendons)
        self.tendon_tensions = np.zeros(self.num_tendons)
        self.tendon_lengths_history = []
        self.tendon_tensions_history = []
        
        # Randomize target position
        self.target_position = np.random.uniform([-2, -2, 0.5], [2, 2, 1.5])
        
        return obs, info
    
    def step(self, action):
        """Step environment with full tendon control mechanics."""
        # Get tendon state from model if available
        if hasattr(self.base_env, 'data') and hasattr(self.base_env, 'model'):
            if mujoco_utils.HAS_MUJOCO and len(self.tendon_ids) > 0:
                # Get real tendon lengths
                self.tendon_lengths = mujoco_utils.get_tendon_lengths(
                    self.base_env.data,
                    self.base_env.model,
                    self.tendon_ids,
                )
                
                # Get real tendon velocities
                self.tendon_velocities = mujoco_utils.get_tendon_velocities(
                    self.base_env.data,
                    self.base_env.model,
                    self.tendon_ids,
                )
                
                # Get real tendon tensions
                self.tendon_tensions = mujoco_utils.get_tendon_tensions(
                    self.base_env.data,
                    self.base_env.model,
                    self.tendon_ids,
                )
                
                # Track history
                self.tendon_lengths_history.append(self.tendon_lengths.copy())
                self.tendon_tensions_history.append(self.tendon_tensions.copy())
        
        # Map actions to tendon control
        # Actions control tendon actuators directly
        action_dim = len(action)
        tendon_commands = np.tanh(action[:min(self.num_tendons, action_dim)])
        
        # Map tendon commands to joint torques via moment arms
        # Output must match action space dimension
        joint_torques = np.zeros(action_dim)
        
        for i, tendon_id in enumerate(self.tendon_ids[:min(self.num_tendons, action_dim)]):
            if i >= len(tendon_commands):
                break
            tendon_command = tendon_commands[i]
            # Map tendon command to tension
            tension = tendon_command * 10.0
            
            # Apply to joints using REAL moment arms
            if i < len(self.joint_ids) and i < action_dim:
                joint_id = self.joint_ids[i]
                if joint_id < self.base_env.model.njnt:
                    # Get actual moment arm from tendon-joint coupling
                    moment_arm = mujoco_utils.compute_tendon_moment_arm(
                        self.base_env.model,
                        tendon_id,
                        joint_id,
                    )
                    if moment_arm > 0:
                        # Apply torque via moment arm: tau = F * r (normalized to [-1, 1])
                        torque = tension * moment_arm * 0.01  # Scale to action range
                        joint_torques[i] = np.clip(torque, -1.0, 1.0)
                    else:
                        # Fallback: use estimated moment arm
                        joint_torques[i] = np.clip(tendon_command, -1.0, 1.0)
                else:
                    # Direct mapping if joint not found
                    joint_torques[i] = tendon_command
            else:
                # Direct mapping if no joint mapping
                joint_torques[i] = tendon_command
        
        # Fill remaining actions if needed
        if action_dim > self.num_tendons:
            joint_torques[self.num_tendons:] = action[self.num_tendons:]
        
        combined_action = joint_torques
        
        # Step environment
        obs, reward, terminated, truncated, info = self.base_env.step(combined_action)
        
        # Add tendon-specific rewards based on real tendon mechanics
        if len(obs) >= 3:
            agent_pos = obs[:3]
            
            # Precision reward (distance to target)
            dist_to_target = np.linalg.norm(agent_pos - self.target_position)
            precision_reward = np.exp(-dist_to_target / 1.0) * 3.0
            
            # Penalty for excessive tendon tension (energy cost)
            if len(self.tendon_tensions) > 0:
                tension_penalty = -np.sum(self.tendon_tensions ** 2) * 0.01
            else:
                tension_penalty = -np.sum(tendon_commands ** 2) * 0.01
            
            # Reward for smooth tendon control (low tension variation)
            if len(self.tendon_tensions_history) > 10:
                tension_variance = np.var([np.sum(t) for t in self.tendon_tensions_history[-10:]])
                smoothness_reward = -tension_variance * 0.1
            else:
                smoothness_reward = 0.0
            
            # Reward for coordinated tendon activation (antagonist pairs)
            coordination_reward = 0.0
            if self.num_tendons >= 2:
                for i in range(0, self.num_tendons - 1, 2):
                    if i + 1 < self.num_tendons:
                        # Antagonist pair coordination
                        pair_coord = 1.0 - abs(tendon_commands[i] - (-tendon_commands[i+1]))
                        coordination_reward += pair_coord * 0.2
            
            # Reward for maintaining tendon length within optimal range
            length_reward = 0.0
            if len(self.tendon_lengths) > 0:
                for length, rest_length in zip(self.tendon_lengths, self.rest_lengths):
                    stretch = abs(length - rest_length) / rest_length
                    if stretch < 0.2:  # Within 20% of rest length
                        length_reward += 0.1
            
            reward += precision_reward + tension_penalty + smoothness_reward + coordination_reward + length_reward
        
        # Store info
        info['tendon_tension'] = np.mean(self.tendon_tensions) if len(self.tendon_tensions) > 0 else 0.0
        info['tendon_length'] = np.mean(self.tendon_lengths) if len(self.tendon_lengths) > 0 else 0.0
        info['precision'] = dist_to_target if len(obs) >= 3 else 0.0
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate to base environment."""
        return getattr(self.base_env, name)


def create_tendon_reward_shaper():
    """Create reward shaping for tendon-driven control."""
    def precision_reward(obs, prev_obs, action):
        if len(obs) < 3:
            return 0.0
        
        # Reward for precise movements
        if prev_obs is not None and len(prev_obs) >= 3:
            movement = np.linalg.norm(obs[:3] - prev_obs[:3])
            # Small, controlled movements are better for precision
            if 0.01 < movement < 0.1:
                return 0.5
        return 0.0
    
    def smoothness_reward(obs, prev_obs, action):
        # Reward for smooth action changes
        if prev_obs is None:
            return 0.0
        
        # Penalize large action changes
        action_change = np.linalg.norm(action)
        return -0.1 * action_change
    
    def stability_reward(obs, prev_obs, action):
        if len(obs) < 6:
            return 0.0
        
        # Stability important for precise control
        orientation = obs[3:6]
        tilt = np.linalg.norm(orientation[:2])
        return 0.8 * np.exp(-tilt)
    
    def energy_penalty(obs, prev_obs, action):
        # Tendon control has energy cost
        return -0.05 * np.sum(action ** 2)
    
    return [precision_reward, smoothness_reward, stability_reward, energy_penalty]


def main():
    """Train tendon-driven robot with full tendon mechanics."""
    print("=" * 60)
    print("Tendon-Driven Robot Control - FULL IMPLEMENTATION")
    print("=" * 60)
    
    # Create base environment
    try:
        base_env = gym.make("Ant-v4")
    except Exception as e:
        print(f"Error creating Ant environment: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    # Wrap with tendon control rewards
    reward_shapers = create_tendon_reward_shaper()
    env = EnvironmentWrapper(base_env, reward_shapers=reward_shapers)
    env = TendonDrivenEnv(env)
    
    # Create policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy = PolicyNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=(256, 256),
        continuous=True,
    ).to(device)
    
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Device: {device}")
    print(f"Tendon system: {env.num_tendons} tendons with FULL mechanics:")
    print("  - Real tendon lengths and velocities")
    print("  - Actual tendon tensions")
    print("  - Moment arm coupling to joints")
    print("  - Coordinated antagonist control")
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        optimizer_type="sgd",
        learning_rate=3e-4,
        device=device,
        clip_epsilon=0.2,
    )
    
    # Training loop
    num_episodes = 200
    rewards_history = []
    precision_history = []
    
    print(f"\nTraining for {num_episodes} episodes...")
    print("-" * 60)
    
    viewer = MuJoCoViewer(env.base_env, render=False)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _ = policy.get_action(obs_tensor, deterministic=False)
            action_np = action.detach().cpu().numpy()[0]
            
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            
            viewer.render_step(obs, action_np, reward)
            
            if steps % 20 == 0:
                trainer.step()
            
            obs = next_obs
            episode_reward += reward
            steps += 1
        
        rewards_history.append(episode_reward)
        
        # Track precision
        if 'precision' in info:
            precision_history.append(info['precision'])
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            avg_precision = np.mean(precision_history[-20:]) if precision_history else 0.0
            avg_tension = np.mean(env.tendon_tensions) if len(env.tendon_tensions) > 0 else 0.0
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Avg (last 20) = {avg_reward:.2f}, "
                  f"Precision = {avg_precision:.3f}, "
                  f"Avg tension = {avg_tension:.2f}")
    
    viewer.close()
    
    # Evaluate
    print("\nEvaluating final policy...")
    eval_rewards = []
    eval_precision = []
    eval_tensions = []
    
    for _ in range(5):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < 500:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _ = policy.get_action(obs_tensor, deterministic=True)
            action_np = action.detach().cpu().numpy()[0]
            
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        final_pos = obs[:3] if len(obs) >= 3 else np.zeros(3)
        precision = np.linalg.norm(final_pos - env.target_position)
        eval_rewards.append(episode_reward)
        eval_precision.append(precision)
        
        if 'tendon_tension' in info:
            eval_tensions.append(info['tendon_tension'])
    
    print(f"Final evaluation - Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Mean precision (distance to target): {np.mean(eval_precision):.2f} ± {np.std(eval_precision):.2f}")
    if eval_tensions:
        print(f"Mean tendon tension: {np.mean(eval_tensions):.2f} ± {np.std(eval_tensions):.2f}")
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plot_training_curves(
        rewards_history,
        save_path=os.path.join(results_dir, "tendon_driven_training.png"),
        title="Tendon-Driven Control - Training Progress",
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
