"""Multi-agent example: Predator-prey dynamics."""

import sys
import os

# Add project root to path (when running with uv run from repo root)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import numpy as np
import gymnasium as gym
from typing import List, Tuple

from examples.rl.models import PolicyNetwork
from examples.rl.framework import PPOTrainer

from examples.animals_3d import utils as animal_utils
from examples.animals_3d import visualization as animal_vis

EnvironmentWrapper = animal_utils.EnvironmentWrapper
create_locomotion_reward_shaper = animal_utils.create_locomotion_reward_shaper
MultiAgentCoordinator = animal_utils.MultiAgentCoordinator
visualize_multi_agent = animal_vis.visualize_multi_agent

class PredatorPrey:
    """Predator-prey multi-agent environment."""
    
    def __init__(self, num_prey: int = 3, num_predators: int = 1):
        """
        Initialize predator-prey scenario.
        
        Args:
            num_prey: Number of prey agents
            num_predators: Number of predator agents
        """
        self.num_prey = num_prey
        self.num_predators = num_predators
        self.total_agents = num_prey + num_predators
        
        self.envs = []
        self.policies = []
        self.trainers = []
        self.is_predator = []
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create prey agents
        for i in range(num_prey):
            try:
                env = gym.make("Ant-v4")
                # Prey: maximize forward velocity and distance from predators
                reward_shapers = create_locomotion_reward_shaper(
                    forward_weight=1.5,
                    stability_weight=0.5,
                    energy_weight=0.1,
                )
                env = EnvironmentWrapper(env, reward_shapers=reward_shapers)
                self.envs.append(env)
                self.is_predator.append(False)
                
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
                print(f"Error creating prey {i+1}: {e}")
                raise
        
        # Create predator agents
        for i in range(num_predators):
            try:
                env = gym.make("Ant-v4")
                # Predator: maximize forward velocity and proximity to prey
                reward_shapers = create_locomotion_reward_shaper(
                    forward_weight=2.0,
                    stability_weight=0.3,
                    energy_weight=0.05,
                )
                env = EnvironmentWrapper(env, reward_shapers=reward_shapers)
                self.envs.append(env)
                self.is_predator.append(True)
                
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
                print(f"Error creating predator {i+1}: {e}")
                raise
    
    def compute_predator_prey_rewards(
        self,
        positions: List[np.ndarray],
    ) -> Tuple[List[float], List[float]]:
        """
        Compute additional rewards based on predator-prey dynamics.
        
        Args:
            positions: List of agent positions
            
        Returns:
            Tuple of (prey_rewards, predator_rewards)
        """
        prey_positions = [positions[i] for i in range(self.num_prey)]
        predator_positions = [positions[i + self.num_prey] for i in range(self.num_predators)]
        
        prey_rewards = [0.0] * self.num_prey
        predator_rewards = [0.0] * self.num_predators
        
        # Compute distances between predators and prey
        for i, prey_pos in enumerate(prey_positions):
            min_dist_to_predator = float('inf')
            for pred_pos in predator_positions:
                dist = np.linalg.norm(prey_pos - pred_pos)
                min_dist_to_predator = min(min_dist_to_predator, dist)
            
            # Prey reward: higher when farther from predators
            if min_dist_to_predator < 2.0:
                prey_rewards[i] -= (2.0 - min_dist_to_predator) * 5.0
            else:
                prey_rewards[i] += 1.0
        
        for i, pred_pos in enumerate(predator_positions):
            min_dist_to_prey = float('inf')
            for prey_pos in prey_positions:
                dist = np.linalg.norm(pred_pos - prey_pos)
                min_dist_to_prey = min(min_dist_to_prey, dist)
            
            # Predator reward: higher when closer to prey
            if min_dist_to_prey < 1.0:
                predator_rewards[i] += 10.0  # Caught prey!
            else:
                predator_rewards[i] += 1.0 / (min_dist_to_prey + 0.1)
        
        return prey_rewards, predator_rewards
    
    def train_episode(self) -> Tuple[List[float], List[float]]:
        """Train one episode."""
        prey_rewards = [0.0] * self.num_prey
        predator_rewards = [0.0] * self.num_predators
        
        observations = []
        for env in self.envs:
            obs, _ = env.reset()
            observations.append(obs)
        
        done_flags = [False] * self.total_agents
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
                    
                    # Extract position (first 3 dimensions typically)
                    pos = obs[:3] if len(obs) >= 3 else np.array([0.0, 0.0, 0.0])
                    positions.append(pos)
                    
                    if self.is_predator[i]:
                        predator_rewards[i - self.num_prey] += reward
                    else:
                        prey_rewards[i] += reward
                    
                    new_observations.append(obs)
                else:
                    new_observations.append(observations[i])
                    if len(observations[i]) >= 3:
                        positions.append(observations[i][:3])
                    else:
                        positions.append(np.array([0.0, 0.0, 0.0]))
            
            observations = new_observations
            
            # Add predator-prey dynamics rewards
            if len(positions) == self.total_agents:
                pp_prey_rewards, pp_pred_rewards = self.compute_predator_prey_rewards(positions)
                for i in range(self.num_prey):
                    prey_rewards[i] += pp_prey_rewards[i]
                for i in range(self.num_predators):
                    predator_rewards[i] += pp_pred_rewards[i]
            
            # Train periodically
            if steps % 20 == 0:
                for trainer in self.trainers:
                    trainer.train_step()
            
            steps += 1
        
        return prey_rewards, predator_rewards
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

def main():
    """Train predator-prey system."""
    print("=" * 60)
    print("Predator-Prey Multi-Agent Example")
    print("=" * 60)
    
    num_prey = 3
    num_predators = 1
    num_episodes = 100
    
    try:
        system = PredatorPrey(num_prey=num_prey, num_predators=num_predators)
    except Exception as e:
        print(f"Error setting up system: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    print(f"Training {num_prey} prey and {num_predators} predator(s) for {num_episodes} episodes...")
    print("-" * 60)
    
    prey_rewards_history = [[] for _ in range(num_prey)]
    predator_rewards_history = [[] for _ in range(num_predators)]
    
    for episode in range(num_episodes):
        prey_rewards, predator_rewards = system.train_episode()
        
        for i, reward in enumerate(prey_rewards):
            prey_rewards_history[i].append(reward)
        for i, reward in enumerate(predator_rewards):
            predator_rewards_history[i].append(reward)
        
        if (episode + 1) % 10 == 0:
            avg_prey = [np.mean(rewards[-10:]) for rewards in prey_rewards_history]
            avg_pred = [np.mean(rewards[-10:]) for rewards in predator_rewards_history]
            print(f"Episode {episode + 1}/{num_episodes}:")
            for i, avg in enumerate(avg_prey):
                print(f"  Prey {i+1}: Avg reward = {avg:.2f}")
            for i, avg in enumerate(avg_pred):
                print(f"  Predator {i+1}: Avg reward = {avg:.2f}")
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Combine all trajectories
    all_trajectories = []
    for rewards in prey_rewards_history:
        traj = [{'obs': np.array([r]), 'action': np.array([0]), 'reward': r} for r in rewards]
        all_trajectories.append(traj)
    for rewards in predator_rewards_history:
        traj = [{'obs': np.array([r]), 'action': np.array([0]), 'reward': r} for r in rewards]
        all_trajectories.append(traj)
    
    visualize_multi_agent(
        all_trajectories,
        save_path=os.path.join(results_dir, "predator_prey.png"),
        title="Predator-Prey Dynamics",
    )
    
    system.close()

if __name__ == "__main__":
    main()

