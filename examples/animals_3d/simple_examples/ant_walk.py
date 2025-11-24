"""Simple example: Ant learning to walk."""

import sys
import os

# Add project root to path (when running with uv run from repo root)
# File is at: examples/animals_3d/simple_examples/ant_walk.py
# So go up 3 levels to get to project root
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import numpy as np
import gymnasium as gym

from examples.rl.models import PolicyNetwork
from examples.rl.framework import REINFORCETrainer, PPOTrainer

from examples.animals_3d import utils as animal_utils
from examples.animals_3d import visualization as animal_vis

EnvironmentWrapper = animal_utils.EnvironmentWrapper
create_locomotion_reward_shaper = animal_utils.create_locomotion_reward_shaper
MuJoCoViewer = animal_vis.MuJoCoViewer
plot_training_curves = animal_vis.plot_training_curves
plot_trajectory = animal_vis.plot_trajectory


def main():
    """Train ant to walk."""
    print("=" * 60)
    print("Ant Walking Example")
    print("=" * 60)
    
    # Create environment
    try:
        env = gym.make("Ant-v4")
    except Exception as e:
        print(f"Error creating Ant environment: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    # Wrap with reward shaping
    reward_shapers = create_locomotion_reward_shaper(
        forward_weight=1.0,
        stability_weight=0.5,
        energy_weight=0.1,
    )
    env = EnvironmentWrapper(env, reward_shapers=reward_shapers)
    
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
    
    # Create trainer (using PPO for better stability)
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        optimizer_type="sgd",  # Can use "es" or "eggroll" too
        learning_rate=3e-4,
        device=device,
        clip_epsilon=0.2,
    )
    
    # Training loop
    num_episodes = 100
    rewards_history = []
    trajectory = []
    
    print(f"\nTraining for {num_episodes} episodes...")
    print("-" * 60)
    
    viewer = MuJoCoViewer(env, render=False)  # Set render=True for visualization
    
    for episode in range(num_episodes):
        # Reset environment
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_trajectory = []
        
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            # Get action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _ = policy.get_action(obs_tensor, deterministic=False)
            action_np = action.detach().cpu().numpy()[0]
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            # Record trajectory
            episode_trajectory.append({
                'obs': obs.copy(),
                'action': action_np.copy(),
                'reward': reward,
            })
            
            # Render
            viewer.render_step(obs, action_np, reward)
            
            # Train (every few steps for PPO)
            if steps % 20 == 0:
                trainer.train_step()
            
            obs = next_obs
            episode_reward += reward
            steps += 1
        
        rewards_history.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Avg (last 10) = {avg_reward:.2f}")
        
        if episode == num_episodes - 1:
            trajectory = episode_trajectory
    
    viewer.close()
    
    # Evaluate final policy
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
        save_path=os.path.join(results_dir, "ant_walk_training.png"),
        title="Ant Walking - Training Progress",
    )
    
    if trajectory:
        plot_trajectory(
            trajectory,
            save_path=os.path.join(results_dir, "ant_walk_trajectory.png"),
            title="Ant Walking - Episode Trajectory",
        )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()

