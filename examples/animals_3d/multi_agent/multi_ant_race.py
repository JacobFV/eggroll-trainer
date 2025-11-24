"""Multi-agent example: Multiple ants racing."""

import sys
import os

# Add project root to path (when running with uv run from repo root)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import numpy as np
import gymnasium as gym
from typing import List

from examples.rl.models import PolicyNetwork
from examples.rl.framework import PPOTrainer

from examples.animals_3d import utils as animal_utils
from examples.animals_3d import visualization as animal_vis

EnvironmentWrapper = animal_utils.EnvironmentWrapper
create_locomotion_reward_shaper = animal_utils.create_locomotion_reward_shaper
MuJoCoViewer = animal_vis.MuJoCoViewer
plot_training_curves = animal_vis.plot_training_curves
visualize_multi_agent = animal_vis.visualize_multi_agent

class MultiAntRace:
    """Multi-agent ant racing environment."""
    
    def __init__(self, num_ants: int = 4):
        """
        Initialize multi-ant race.
        
        Args:
            num_ants: Number of ants racing
        """
        self.num_ants = num_ants
        self.envs = []
        self.policies = []
        self.trainers = []
        
        # Create environments and policies
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for i in range(num_ants):
            try:
                env = gym.make("Ant-v4")
                reward_shapers = create_locomotion_reward_shaper(
                    forward_weight=2.0,  # Emphasize speed for racing
                    stability_weight=0.3,
                    energy_weight=0.05,
                )
                env = EnvironmentWrapper(env, reward_shapers=reward_shapers)
                self.envs.append(env)
                
                # Create policy for each ant
                obs_dim = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0]
                
                policy = PolicyNetwork(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dims=(256, 256),
                    continuous=True,
                ).to(device)
                
                self.policies.append(policy)
                
                # Create trainer
                trainer = PPOTrainer(
                    env=env,
                    policy=policy,
                    optimizer_type="sgd",
                    learning_rate=3e-4,
                    device=device,
                    clip_epsilon=0.2,
                )
                self.trainers.append(trainer)
                
            except Exception as e:
                print(f"Error creating ant {i+1}: {e}")
                raise
    
    def train_episode(self) -> List[float]:
        """Train all ants for one episode."""
        episode_rewards = [0.0] * self.num_ants
        observations = []
        actions_list = []
        
        # Reset all environments
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        
        done_flags = [False] * self.num_ants
        steps = 0
        max_steps = 1000
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        while not all(done_flags) and steps < max_steps:
            # Get actions for all ants
            actions = []
            for i, (obs, policy) in enumerate(zip(observations, self.policies)):
                if not done_flags[i]:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action, _ = policy.get_action(obs_tensor, deterministic=False)
                    action_np = action.detach().cpu().numpy()[0]
                    actions.append(action_np)
                else:
                    actions.append(np.zeros(self.envs[0].action_space.shape[0]))
            
            actions_list.append(actions)
            
            # Step all environments
            new_observations = []
            for i, (env, action) in enumerate(zip(self.envs, actions)):
                if not done_flags[i]:
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done_flags[i] = terminated or truncated
                    episode_rewards[i] += reward
                    new_observations.append(obs)
                else:
                    new_observations.append(observations[i])
            
            observations = new_observations
            
            # Train periodically
            if steps % 20 == 0:
                for trainer in self.trainers:
                    trainer.step()
            
            steps += 1
        
        return episode_rewards
    
    def evaluate(self, num_episodes: int = 5) -> List[List[float]]:
        """Evaluate all ants."""
        all_rewards = [[] for _ in range(self.num_ants)]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for _ in range(num_episodes):
            observations = []
            for env in self.envs:
                obs, _ = env.reset()
                observations.append(obs)
            
            done_flags = [False] * self.num_ants
            episode_rewards = [0.0] * self.num_ants
            steps = 0
            max_steps = 500
            
            while not all(done_flags) and steps < max_steps:
                actions = []
                for i, (obs, policy) in enumerate(zip(observations, self.policies)):
                    if not done_flags[i]:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                        action, _ = policy.get_action(obs_tensor, deterministic=True)
                        action_np = action.detach().cpu().numpy()[0]
                        actions.append(action_np)
                    else:
                        actions.append(np.zeros(self.envs[0].action_space.shape[0]))
                
                new_observations = []
                for i, (env, action) in enumerate(zip(self.envs, actions)):
                    if not done_flags[i]:
                        obs, reward, terminated, truncated, _ = env.step(action)
                        done_flags[i] = terminated or truncated
                        episode_rewards[i] += reward
                        new_observations.append(obs)
                    else:
                        new_observations.append(observations[i])
                
                observations = new_observations
                steps += 1
            
            for i, reward in enumerate(episode_rewards):
                all_rewards[i].append(reward)
        
        return all_rewards
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

def main():
    """Train multiple ants to race."""
    print("=" * 60)
    print("Multi-Agent Ant Race")
    print("=" * 60)
    
    num_ants = 4
    num_episodes = 100
    
    try:
        race = MultiAntRace(num_ants=num_ants)
    except Exception as e:
        print(f"Error setting up race: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    print(f"Training {num_ants} ants for {num_episodes} episodes...")
    print("-" * 60)
    
    all_rewards_history = [[] for _ in range(num_ants)]
    
    for episode in range(num_episodes):
        episode_rewards = race.train_episode()
        
        for i, reward in enumerate(episode_rewards):
            all_rewards_history[i].append(reward)
        
        if (episode + 1) % 10 == 0:
            avg_rewards = [np.mean(rewards[-10:]) for rewards in all_rewards_history]
            print(f"Episode {episode + 1}/{num_episodes}:")
            for i, avg_reward in enumerate(avg_rewards):
                print(f"  Ant {i+1}: Avg reward = {avg_reward:.2f}")
    
    # Evaluate
    print("\nEvaluating all ants...")
    eval_results = race.evaluate(num_episodes=5)
    
    print("\nFinal Evaluation Results:")
    for i, rewards in enumerate(eval_results):
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        print(f"  Ant {i+1}: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Find winner
    final_means = [np.mean(rewards) for rewards in eval_results]
    winner = np.argmax(final_means)
    print(f"\n🏆 Winner: Ant {winner + 1} with {final_means[winner]:.2f} average reward!")
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Convert to trajectory format for visualization
    trajectories = []
    for rewards in all_rewards_history:
        traj = [{'obs': np.array([r]), 'action': np.array([0]), 'reward': r} for r in rewards]
        trajectories.append(traj)
    
    visualize_multi_agent(
        trajectories,
        save_path=os.path.join(results_dir, "multi_ant_race.png"),
        title="Multi-Agent Ant Race",
    )
    
    race.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()

