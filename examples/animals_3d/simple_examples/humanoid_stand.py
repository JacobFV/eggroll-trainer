"""Simple example: Humanoid learning to stand and walk."""

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

EnvironmentWrapper = animal_utils.EnvironmentWrapper
create_locomotion_reward_shaper = animal_utils.create_locomotion_reward_shaper
RewardShaper = animal_utils.RewardShaper
MuJoCoViewer = animal_vis.MuJoCoViewer
plot_training_curves = animal_vis.plot_training_curves


def create_humanoid_reward_shaper():
    """Create reward shaping for humanoid (emphasize stability and height)."""
    def forward_reward(obs, prev_obs, action):
        if prev_obs is None:
            return 0.0
        return 1.0 * RewardShaper.forward_velocity_reward(obs, prev_obs)
    
    def stability_reward_fn(obs, prev_obs, action):
        return 1.0 * RewardShaper.stability_reward(obs)
    
    def height_reward_fn(obs, prev_obs, action):
        return 0.5 * RewardShaper.height_reward(obs, z_pos_idx=2, target_height=1.4)
    
    def energy_penalty_fn(obs, prev_obs, action):
        return 0.1 * RewardShaper.energy_penalty(action)
    
    return [forward_reward, stability_reward_fn, height_reward_fn, energy_penalty_fn]


def main():
    """Train Humanoid to stand and walk."""
    print("=" * 60)
    print("Humanoid Standing/Walking Example")
    print("=" * 60)
    
    # Create environment
    try:
        env = gym.make("Humanoid-v4")
    except Exception as e:
        print(f"Error creating Humanoid environment: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    # Wrap with reward shaping
    reward_shapers = create_humanoid_reward_shaper()
    env = EnvironmentWrapper(env, reward_shapers=reward_shapers)
    
    # Create policy
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy = PolicyNetwork(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=(512, 512),  # Larger network for complex humanoid
        continuous=True,
    ).to(device)
    
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Device: {device}")
    
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
    num_episodes = 200  # More episodes for complex task
    rewards_history = []
    
    print(f"\nTraining for {num_episodes} episodes...")
    print("-" * 60)
    
    viewer = MuJoCoViewer(env, render=False)
    
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
            
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            viewer.render_step(obs, action_np, reward)
            
            if steps % 20 == 0:
                trainer.train_step()
            
            obs = next_obs
            episode_reward += reward
            steps += 1
        
        rewards_history.append(episode_reward)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Avg (last 20) = {avg_reward:.2f}")
    
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
            action, _ = policy.get_action(obs_tensor, deterministic=True)
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
        save_path=os.path.join(results_dir, "humanoid_stand_training.png"),
        title="Humanoid Standing/Walking - Training Progress",
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
