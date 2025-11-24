"""Advanced example: Multi-agent cooperation."""

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
MultiAgentCoordinator = animal_utils.MultiAgentCoordinator
visualize_multi_agent = animal_vis.visualize_multi_agent


class CooperativeSystem:
    """Multi-agent cooperative system."""
    
    def __init__(self, num_agents: int = 4):
        """
        Initialize cooperative system.
        
        Args:
            num_agents: Number of cooperating agents
        """
        self.num_agents = num_agents
        self.envs = []
        self.policies = []
        self.trainers = []
        self.coordinator = MultiAgentCoordinator()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for i in range(num_agents):
            try:
                env = gym.make("Ant-v4")
                reward_shapers = create_locomotion_reward_shaper(
                    forward_weight=1.0,
                    stability_weight=0.5,
                    energy_weight=0.1,
                )
                env = EnvironmentWrapper(env, reward_shapers=reward_shapers)
                self.envs.append(env)
                
                obs_dim = env.observation_space.shape[0]
                action_dim = env.action_space.shape[0]
                
                policy = PolicyNetwork(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    hidden_dims=(256, 256),
                    continuous=True,
                ).to(device)
                
                self.policies.append(policy)
                
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
                print(f"Error creating agent {i+1}: {e}")
                raise
        
        # Shared goal position (agents try to reach together)
        self.goal_position = np.array([10.0, 0.0, 1.0])
    
    def compute_cooperative_rewards(
        self,
        positions: List[np.ndarray],
    ) -> List[float]:
        """
        Compute cooperative rewards.
        
        Args:
            positions: List of agent positions
            
        Returns:
            List of rewards, one per agent
        """
        rewards = []
        
        # Individual goal-reaching reward
        for pos in positions:
            dist_to_goal = np.linalg.norm(pos - self.goal_position)
            goal_reward = np.exp(-dist_to_goal / 5.0)  # Closer = better
            rewards.append(goal_reward)
        
        # Cooperative reward: agents should stay together
        centroid = self.coordinator.compute_centroid(positions)
        distances_to_centroid = [np.linalg.norm(pos - centroid) for pos in positions]
        avg_distance = np.mean(distances_to_centroid)
        
        # Reward for staying close together
        cohesion_reward = np.exp(-avg_distance / 2.0)
        
        # Add cohesion reward to all agents
        for i in range(len(rewards)):
            rewards[i] += 0.5 * cohesion_reward
        
        # Shared success: if all agents are near goal, extra reward
        all_near_goal = all(np.linalg.norm(pos - self.goal_position) < 2.0 for pos in positions)
        if all_near_goal:
            for i in range(len(rewards)):
                rewards[i] += 5.0
        
        return rewards
    
    def train_episode(self) -> List[float]:
        """Train one episode."""
        episode_rewards = [0.0] * self.num_agents
        
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        
        done_flags = [False] * self.num_agents
        steps = 0
        max_steps = 1000
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        while not all(done_flags) and steps < max_steps:
            # Get actions
            actions = []
            for i, (obs, policy) in enumerate(zip(observations, self.policies)):
                if not done_flags[i]:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    action, _ = policy.get_action(obs_tensor, deterministic=False)
                    action_np = action.detach().cpu().numpy()[0]
                    actions.append(action_np)
                else:
                    actions.append(np.zeros(self.envs[0].action_space.shape[0]))
            
            # Step environments
            new_observations = []
            positions = []
            
            for i, (env, action) in enumerate(zip(self.envs, actions)):
                if not done_flags[i]:
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done_flags[i] = terminated or truncated
                    
                    pos = obs[:3] if len(obs) >= 3 else np.array([0.0, 0.0, 0.0])
                    positions.append(pos)
                    
                    episode_rewards[i] += reward
                    new_observations.append(obs)
                else:
                    new_observations.append(observations[i])
                    if len(observations[i]) >= 3:
                        positions.append(observations[i][:3])
                    else:
                        positions.append(np.array([0.0, 0.0, 0.0]))
            
            observations = new_observations
            
            # Add cooperative rewards
            if len(positions) == self.num_agents:
                coop_rewards = self.compute_cooperative_rewards(positions)
                for i in range(self.num_agents):
                    episode_rewards[i] += coop_rewards[i]
            
            # Train periodically
            if steps % 20 == 0:
                for trainer in self.trainers:
                    trainer.train_step()
            
            steps += 1
        
        return episode_rewards
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

def main():
    """Train cooperative system."""
    print("=" * 60)
    print("Multi-Agent Cooperation Example")
    print("=" * 60)
    
    num_agents = 4
    num_episodes = 150
    
    try:
        system = CooperativeSystem(num_agents=num_agents)
    except Exception as e:
        print(f"Error setting up system: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    print(f"Training {num_agents} cooperating agents for {num_episodes} episodes...")
    print("Goal: Agents work together to reach a shared goal position")
    print("-" * 60)
    
    rewards_history = [[] for _ in range(num_agents)]
    
    for episode in range(num_episodes):
        episode_rewards = system.train_episode()
        
        for i, reward in enumerate(episode_rewards):
            rewards_history[i].append(reward)
        
        if (episode + 1) % 15 == 0:
            avg_rewards = [np.mean(rewards[-15:]) for rewards in rewards_history]
            print(f"Episode {episode + 1}/{num_episodes}:")
            for i, avg_reward in enumerate(avg_rewards):
                print(f"  Agent {i+1}: Avg reward = {avg_reward:.2f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    trajectories = []
    for rewards in rewards_history:
        traj = [{'obs': np.array([r]), 'action': np.array([0]), 'reward': r} for r in rewards]
        trajectories.append(traj)
    
    visualize_multi_agent(
        trajectories,
        save_path=os.path.join(results_dir, "multi_agent_cooperation.png"),
        title="Multi-Agent Cooperation",
    )
    
    system.close()

if __name__ == "__main__":
    main()

