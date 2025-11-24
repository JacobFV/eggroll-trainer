"""Advanced example: Complex locomotion patterns with hierarchical control."""

import sys
import os

# Add project root to path (when running with uv run from repo root)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from examples.rl.models import PolicyNetwork, ValueNetwork
from examples.rl.framework import ActorCriticTrainer

from examples.animals_3d import utils as animal_utils
from examples.animals_3d import visualization as animal_vis

EnvironmentWrapper = animal_utils.EnvironmentWrapper
RewardShaper = animal_utils.RewardShaper
ObservationPreprocessor = animal_utils.ObservationPreprocessor
MuJoCoViewer = animal_vis.MuJoCoViewer
plot_training_curves = animal_vis.plot_training_curves

class HierarchicalPolicy:
    """Hierarchical policy with high-level and low-level controllers."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
    ):
        """
        Initialize hierarchical policy.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            device: Device to run on
        """
        self.device = device
        
        # High-level policy: decides which behavior mode to use
        self.high_level_policy = PolicyNetwork(
            obs_dim=obs_dim,
            action_dim=3,  # 3 behavior modes: walk, run, jump
            hidden_dims=(128, 128),
            continuous=False,  # Discrete mode selection
        ).to(device)
        
        # Low-level policies: one for each behavior mode
        self.low_level_policies = nn.ModuleList([
            PolicyNetwork(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=(256, 256),
                continuous=True,
            ).to(device)
            for _ in range(3)
        ])
        
        self.current_mode = 0
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Get action from hierarchical policy.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic mode selection
            
        Returns:
            Tuple of (action, log_prob)
        """
        # High-level: select mode
        mode_logits = self.high_level_policy(obs)
        
        if deterministic:
            mode = mode_logits.argmax(dim=-1)
        else:
            mode_dist = torch.distributions.Categorical(logits=mode_logits)
            mode = mode_dist.sample()
        
        self.current_mode = mode.item()
        
        # Low-level: get action for selected mode
        low_level_policy = self.low_level_policies[mode.item()]
        action, log_prob = low_level_policy.get_action(obs, deterministic=deterministic)
        
        return action, log_prob
    
    def parameters(self):
        """Get all parameters."""
        params = list(self.high_level_policy.parameters())
        for policy in self.low_level_policies:
            params.extend(list(policy.parameters()))
        return params

def create_complex_reward_shaper():
    """Create reward shaping for complex locomotion."""
    def forward_reward(obs, prev_obs, action):
        if prev_obs is None:
            return 0.0
        return 1.0 * RewardShaper.forward_velocity_reward(obs, prev_obs)
    
    def stability_reward_fn(obs, prev_obs, action):
        return 0.8 * RewardShaper.stability_reward(obs)
    
    def height_reward_fn(obs, prev_obs, action):
        # Reward for maintaining height (jumping behavior)
        return 0.3 * RewardShaper.height_reward(obs, z_pos_idx=2, target_height=1.0)
    
    def energy_penalty_fn(obs, prev_obs, action):
        return 0.05 * RewardShaper.energy_penalty(action)
    
    def smoothness_reward(obs, prev_obs, action):
        """Reward for smooth actions."""
        if prev_obs is None:
            return 0.0
        # Penalize large action changes
        action_change = np.linalg.norm(action)
        return -0.1 * action_change
    
    return [forward_reward, stability_reward_fn, height_reward_fn, energy_penalty_fn, smoothness_reward]

def main():
    """Train complex locomotion with hierarchical control."""
    print("=" * 60)
    print("Complex Locomotion with Hierarchical Control")
    print("=" * 60)
    
    try:
        env = gym.make("Ant-v4")
    except Exception as e:
        print(f"Error creating Ant environment: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    # Wrap with complex reward shaping
    reward_shapers = create_complex_reward_shaper()
    
    # Add observation preprocessing
    def obs_preprocessor(obs, prev_obs):
        return ObservationPreprocessor.add_velocity_info(obs, prev_obs)
    
    env = EnvironmentWrapper(env, reward_shapers=reward_shapers, obs_preprocessor=obs_preprocessor)
    
    # Create hierarchical policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Adjust obs_dim for preprocessing
    obs_dim = obs_dim * 2  # Velocity info added
    
    hierarchical_policy = HierarchicalPolicy(obs_dim, action_dim, device)
    
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Device: {device}")
    print("Using hierarchical policy with 3 behavior modes")
    
    # Create trainer (using Actor-Critic for better learning)
    value_network = ValueNetwork(obs_dim, hidden_dims=(256, 256)).to(device)
    
    # Custom trainer wrapper for hierarchical policy
    class HierarchicalTrainer(ActorCriticTrainer):
        def __init__(self, *args, **kwargs):
            # Replace policy with hierarchical policy
            kwargs['policy'] = hierarchical_policy
            super().__init__(*args, **kwargs)
            self.policy = hierarchical_policy
    
    trainer = HierarchicalTrainer(
        env=env,
        policy=hierarchical_policy,
        optimizer_type="sgd",
        learning_rate=3e-4,
        device=device,
        value_network=value_network,
    )
    
    # Training loop
    num_episodes = 200
    rewards_history = []
    mode_history = []
    
    print(f"\nTraining for {num_episodes} episodes...")
    print("-" * 60)
    
    viewer = MuJoCoViewer(env, render=False)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_modes = []
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _ = hierarchical_policy.get_action(obs_tensor, deterministic=False)
            action_np = action.detach().cpu().numpy()[0]
            
            episode_modes.append(hierarchical_policy.current_mode)
            
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            viewer.render_step(obs, action_np, reward)
            
            if steps % 20 == 0:
                trainer.train_step()
            
            obs = next_obs
            episode_reward += reward
            steps += 1
        
        rewards_history.append(episode_reward)
        mode_history.append(episode_modes)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            mode_counts = {}
            for modes in mode_history[-20:]:
                for mode in modes:
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Avg (last 20) = {avg_reward:.2f}")
            print(f"  Mode usage: {mode_counts}")
    
    viewer.close()
    
    # Evaluate
    print("\nEvaluating final policy...")
    eval_rewards = []
    for _ in range(5):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < 500:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _ = hierarchical_policy.get_action(obs_tensor, deterministic=True)
            action_np = action.detach().cpu().numpy()[0]
            
            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        eval_rewards.append(episode_reward)
    
    print(f"Final evaluation - Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plot_training_curves(
        rewards_history,
        save_path=os.path.join(results_dir, "complex_locomotion_training.png"),
        title="Complex Locomotion - Training Progress",
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    env.close()

if __name__ == "__main__":
    main()

