"""Multi-agent example: Flocking behavior."""

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


class FlockingSystem:
    """Multi-agent flocking system."""
    
    def __init__(self, num_agents: int = 5):
        """
        Initialize flocking system.
        
        Args:
            num_agents: Number of agents in the flock
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
    
    def compute_flocking_rewards(
        self,
        positions: List[np.ndarray],
        velocities: List[np.ndarray],
    ) -> List[float]:
        """
        Compute flocking rewards (separation, alignment, cohesion).
        
        Args:
            positions: List of agent positions
            velocities: List of agent velocities
            
        Returns:
            List of rewards, one per agent
        """
        # Separation: avoid crowding neighbors
        separation_rewards = self.coordinator.separation_reward(
            positions,
            min_distance=0.5,
            max_distance=2.0,
        )
        
        # Alignment: steer towards average heading
        alignment_reward = self.coordinator.alignment_reward(velocities)
        alignment_rewards = [alignment_reward] * self.num_agents
        
        # Cohesion: steer towards average position
        centroid = self.coordinator.compute_centroid(positions)
        cohesion_rewards = []
        for pos in positions:
            dist_to_centroid = np.linalg.norm(pos - centroid)
            # Reward for being near centroid (but not too close)
            if 1.0 < dist_to_centroid < 3.0:
                cohesion_rewards.append(1.0)
            else:
                cohesion_rewards.append(-abs(dist_to_centroid - 2.0))
        
        # Combine rewards
        total_rewards = []
        for i in range(self.num_agents):
            total = (
                0.5 * separation_rewards[i] +
                0.3 * alignment_rewards[i] +
                0.2 * cohesion_rewards[i]
            )
            total_rewards.append(total)
        
        return total_rewards
    
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
            velocities = []
            
            for i, (env, action) in enumerate(zip(self.envs, actions)):
                if not done_flags[i]:
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done_flags[i] = terminated or truncated
                    
                    # Extract position and velocity
                    pos = obs[:3] if len(obs) >= 3 else np.array([0.0, 0.0, 0.0])
                    vel = obs[3:6] if len(obs) >= 6 else np.array([0.0, 0.0, 0.0])
                    
                    positions.append(pos)
                    velocities.append(vel)
                    
                    episode_rewards[i] += reward
                    new_observations.append(obs)
                else:
                    new_observations.append(observations[i])
                    if len(observations[i]) >= 3:
                        positions.append(observations[i][:3])
                        velocities.append(observations[i][3:6] if len(observations[i]) >= 6 else np.array([0.0, 0.0, 0.0]))
                    else:
                        positions.append(np.array([0.0, 0.0, 0.0]))
                        velocities.append(np.array([0.0, 0.0, 0.0]))
            
            observations = new_observations
            
            # Add flocking rewards
            if len(positions) == self.num_agents and len(velocities) == self.num_agents:
                flocking_rewards = self.compute_flocking_rewards(positions, velocities)
                for i in range(self.num_agents):
                    episode_rewards[i] += flocking_rewards[i]
            
            # Train periodically
            if steps % 20 == 0:
                for trainer in self.trainers:
                    trainer.step()
            
            steps += 1
        
        return episode_rewards
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

def main():
    """Train flocking system."""
    print("=" * 60)
    print("Flocking Multi-Agent Example")
    print("=" * 60)
    
    num_agents = 5
    num_episodes = 100
    
    try:
        system = FlockingSystem(num_agents=num_agents)
    except Exception as e:
        print(f"Error setting up system: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    print(f"Training {num_agents} agents for {num_episodes} episodes...")
    print("-" * 60)
    
    rewards_history = [[] for _ in range(num_agents)]
    
    for episode in range(num_episodes):
        episode_rewards = system.train_episode()
        
        for i, reward in enumerate(episode_rewards):
            rewards_history[i].append(reward)
        
        if (episode + 1) % 10 == 0:
            avg_rewards = [np.mean(rewards[-10:]) for rewards in rewards_history]
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
        save_path=os.path.join(results_dir, "flocking.png"),
        title="Flocking Behavior",
    )
    
    system.close()

if __name__ == "__main__":
    main()

