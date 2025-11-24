"""Advanced example: Agent learns to manipulate objects."""

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
from examples.animals_3d import world_builder
from examples.animals_3d import mujoco_utils

EnvironmentWrapper = animal_utils.EnvironmentWrapper
create_locomotion_reward_shaper = animal_utils.create_locomotion_reward_shaper
MuJoCoViewer = animal_vis.MuJoCoViewer
plot_training_curves = animal_vis.plot_training_curves


class ObjectManipulationEnv:
    """Environment for object manipulation tasks."""
    
    def __init__(self, base_env: gym.Env, object_spawner: world_builder.ObjectSpawner):
        """
        Initialize object manipulation environment.
        
        Args:
            base_env: Base Gymnasium environment
            object_spawner: Object spawner instance
        """
        self.base_env = base_env
        self.object_spawner = object_spawner
        
        # Spawn target object
        self.target_object = object_spawner.spawn_box(
            position=(5.0, 0.0, 0.5),
            size=(0.3, 0.3, 0.3),
            mass=0.5,
            name="target_box",
        )
        
        self.goal_position = np.array([10.0, 0.0, 0.5])
        self.object_position = np.array(self.target_object['pos'])
    
    def reset(self, **kwargs):
        """Reset environment."""
        obs, info = self.base_env.reset(**kwargs)
        
        # Reset object position
        self.object_position = np.array(self.target_object['pos'])
        
        return obs, info
    
    def step(self, action):
        """Step environment with FULL object tracking from physics."""
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Get REAL object position from MuJoCo physics
        if hasattr(self.base_env, 'data') and hasattr(self.base_env, 'model'):
            if mujoco_utils.HAS_MUJOCO:
                import mujoco
                try:
                    # Find object body ID
                    object_body_id = None
                    for i in range(self.base_env.model.nbody):
                        body_name = mujoco.mj_id2name(self.base_env.model, mujoco.mjtObj.mjOBJ_BODY, i)
                        if body_name and ('target_box' in body_name.lower() or 'object' in body_name.lower()):
                            object_body_id = i
                            break
                    
                    if object_body_id is not None:
                        # Get actual object position from physics
                        self.object_position = self.base_env.data.xpos[object_body_id].copy()
                        
                        # Get object velocity
                        object_vel = self.base_env.data.qvel[self.base_env.model.body_jntadr[object_body_id]:self.base_env.model.body_jntadr[object_body_id] + 6][:3]
                        
                        # Get contact forces on object
                        object_contact_force = mujoco_utils.get_contact_forces(
                            self.base_env.data,
                            object_body_id,
                        )
                        
                        info['object_position'] = self.object_position.copy()
                        info['object_velocity'] = object_vel.copy()
                        info['object_contact_force'] = np.linalg.norm(object_contact_force)
                except Exception as e:
                    # Fallback if object not found
                    pass
        
        # Add manipulation rewards based on REAL physics
        if len(obs) >= 3:
            agent_pos = obs[:3]
            
            # Reward for agent being near object (pushing it)
            dist_to_object = np.linalg.norm(agent_pos - self.object_position)
            if dist_to_object < 1.0:
                proximity_reward = (1.0 - dist_to_object) * 2.0
            else:
                proximity_reward = 0.0
            
            # Reward for object being near goal
            dist_to_goal = np.linalg.norm(self.object_position - self.goal_position)
            goal_reward = np.exp(-dist_to_goal / 3.0) * 5.0
            
            # Bonus if object reaches goal
            if dist_to_goal < 0.5:
                goal_reward += 10.0
            
            # Reward for applying force to object (real contact force)
            if 'object_contact_force' in info:
                contact_reward = min(info['object_contact_force'] / 50.0, 1.0) * 1.0
            else:
                contact_reward = 0.0
            
            # Reward for moving object toward goal (real velocity)
            if 'object_velocity' in info:
                object_vel = info['object_velocity']
                goal_direction = (self.goal_position - self.object_position)
                goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)
                velocity_toward_goal = np.dot(object_vel, goal_direction)
                movement_reward = max(0.0, velocity_toward_goal) * 2.0
            else:
                movement_reward = 0.0
            
            reward += proximity_reward + goal_reward + contact_reward + movement_reward
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate to base environment."""
        return getattr(self.base_env, name)


def main():
    """Train agent to manipulate objects."""
    print("=" * 60)
    print("Object Manipulation Learning")
    print("=" * 60)
    
    # Create base environment
    try:
        base_env = gym.make("Ant-v4")
    except Exception as e:
        print(f"Error creating Ant environment: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    # Create object spawner
    spawner = world_builder.ObjectSpawner()
    
    # Wrap environment
    reward_shapers = create_locomotion_reward_shaper(
        forward_weight=1.0,
        stability_weight=0.8,  # Higher stability for manipulation
        energy_weight=0.1,
    )
    env = EnvironmentWrapper(base_env, reward_shapers=reward_shapers)
    env = ObjectManipulationEnv(env, spawner)
    
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
    print("Task: Push object to goal position")
    
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
    success_count = 0
    
    print(f"\nTraining for {num_episodes} episodes...")
    print("-" * 60)
    
    viewer = MuJoCoViewer(env.base_env, render=False)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        max_steps = 2000
        
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
            
            # Check for success
            if len(obs) >= 3:
                agent_pos = obs[:3]
                dist_to_goal = np.linalg.norm(agent_pos - env.goal_position)
                if dist_to_goal < 0.5:
                    success_count += 1
                    break
        
        rewards_history.append(episode_reward)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            success_rate = success_count / (episode + 1) * 100
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Avg (last 20) = {avg_reward:.2f}, "
                  f"Success rate = {success_rate:.1f}%")
    
    viewer.close()
    
    # Evaluate
    print("\nEvaluating final policy...")
    eval_rewards = []
    eval_successes = 0
    
    for _ in range(10):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done and steps < 1000:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _ = policy.get_action(obs_tensor, deterministic=True)
            action_np = action.detach().cpu().numpy()[0]
            
            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Check success
            if len(obs) >= 3:
                agent_pos = obs[:3]
                dist_to_goal = np.linalg.norm(agent_pos - env.goal_position)
                if dist_to_goal < 0.5:
                    eval_successes += 1
                    break
        
        eval_rewards.append(episode_reward)
    
    print(f"Final evaluation - Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Success rate: {eval_successes}/10 ({eval_successes*10}%)")
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plot_training_curves(
        rewards_history,
        save_path=os.path.join(results_dir, "object_manipulation_training.png"),
        title="Object Manipulation - Training Progress",
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()

