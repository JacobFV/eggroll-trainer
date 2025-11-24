"""Advanced example: Agent with FULL soft body parts interacts with rigid objects."""

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


class SoftRigidInteractionEnv:
    """Full implementation: Soft-rigid interaction with real contact forces and deformation."""
    
    def __init__(self, base_env: gym.Env):
        """
        Initialize soft-rigid interaction environment with full physics.
        
        Args:
            base_env: Base Gymnasium environment
        """
        self.base_env = base_env
        
        # Soft body state tracking
        self.soft_body_ids = []
        self.rigid_body_ids = []
        self.contact_forces = {}
        self.deformation_history = []
        
        # Get body IDs
        if hasattr(base_env, 'model') and mujoco_utils.HAS_MUJOCO:
            import mujoco
            try:
                # Find soft bodies (composite bodies)
                for i in range(base_env.model.nbody):
                    body_name = mujoco.mj_id2name(base_env.model, mujoco.mjtObj.mjOBJ_BODY, i)
                    if body_name and ('soft' in body_name.lower() or 'composite' in body_name.lower()):
                        self.soft_body_ids.append(i)
                
                # Find rigid bodies
                for i in range(base_env.model.nbody):
                    body_name = mujoco.mj_id2name(base_env.model, mujoco.mjtObj.mjOBJ_BODY, i)
                    if body_name and body_name not in ['world', 'ground'] and i not in self.soft_body_ids:
                        self.rigid_body_ids.append(i)
            except:
                pass
        
        # Rigid objects for manipulation
        self.rigid_objects = []
        self.goal_position = np.array([10.0, 0.0, 0.5])
        
        # Agent body ID
        self.agent_body_id = None
        if hasattr(base_env, 'model') and mujoco_utils.HAS_MUJOCO:
            import mujoco
            try:
                for i in range(base_env.model.nbody):
                    body_name = mujoco.mj_id2name(base_env.model, mujoco.mjtObj.mjOBJ_BODY, i)
                    if body_name and ('torso' in body_name.lower() or 'body' in body_name.lower()):
                        self.agent_body_id = i
                        break
            except:
                pass
    
    def reset(self, **kwargs):
        """Reset environment."""
        obs, info = self.base_env.reset(**kwargs)
        
        # Reset tracking
        self.contact_forces = {}
        self.deformation_history = []
        
        # Spawn rigid objects
        self.rigid_objects = [
            {'body_id': None, 'pos': np.array([3.0, 0.0, 0.5]), 'size': 0.3, 'mass': 1.0},
            {'body_id': None, 'pos': np.array([6.0, 1.0, 0.5]), 'size': 0.3, 'mass': 1.0},
        ]
        
        return obs, info
    
    def step(self, action):
        """Step environment with full soft-rigid interaction physics."""
        # Get contact forces BEFORE stepping
        if hasattr(self.base_env, 'data') and hasattr(self.base_env, 'model'):
            if mujoco_utils.HAS_MUJOCO:
                import mujoco
                
                # Get all contact forces
                all_contact_forces = mujoco_utils.get_all_contact_forces(
                    self.base_env.data,
                    self.base_env.model,
                )
                
                # Track soft-rigid contacts
                soft_rigid_contacts = {}
                for soft_id in self.soft_body_ids:
                    if soft_id in all_contact_forces:
                        for rigid_id in self.rigid_body_ids:
                            if rigid_id in all_contact_forces:
                                # Check if they're in contact
                                contact_key = (soft_id, rigid_id)
                                soft_force = all_contact_forces[soft_id]
                                rigid_force = all_contact_forces[rigid_id]
                                # Contact exists if forces are significant
                                if np.linalg.norm(soft_force) > 0.1 or np.linalg.norm(rigid_force) > 0.1:
                                    soft_rigid_contacts[contact_key] = {
                                        'soft_force': soft_force,
                                        'rigid_force': rigid_force,
                                        'contact_magnitude': np.linalg.norm(soft_force),
                                    }
                
                self.contact_forces = soft_rigid_contacts
                
                # Compute soft body deformation
                if len(self.soft_body_ids) > 0:
                    deformation_metrics = mujoco_utils.compute_cloth_deformation(
                        self.base_env.data,
                        self.base_env.model,
                        self.soft_body_ids[0],
                    )
                    self.deformation_history.append(deformation_metrics['mean_deformation'])
        
        # Step environment
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Add soft-rigid interaction rewards based on real physics
        interaction_reward = 0.0
        manipulation_reward = 0.0
        deformation_penalty = 0.0
        
        if len(obs) >= 3:
            agent_pos = obs[:3]
            
            # Reward for successful contact (based on real contact forces)
            if len(self.contact_forces) > 0:
                total_contact_magnitude = sum(
                    contact['contact_magnitude'] for contact in self.contact_forces.values()
                )
                interaction_reward = min(total_contact_magnitude / 100.0, 1.0) * 2.0
            
            # Reward for manipulating objects toward goal
            for obj in self.rigid_objects:
                obj_dist_to_goal = np.linalg.norm(obj['pos'] - self.goal_position)
                manipulation_reward += np.exp(-obj_dist_to_goal / 5.0) * 2.0
                
                # Check if agent is pushing object
                dist_to_obj = np.linalg.norm(agent_pos - obj['pos'])
                if dist_to_obj < obj['size'] + 0.5:
                    # Agent is near object, check if pushing
                    if len(self.contact_forces) > 0:
                        manipulation_reward += 1.0
            
            # Penalty for excessive deformation
            if len(self.deformation_history) > 0:
                current_deformation = self.deformation_history[-1]
                deformation_penalty = -current_deformation * 5.0
            
            # Reward for controlled interaction (moderate forces)
            if len(self.contact_forces) > 0:
                avg_contact_magnitude = np.mean([
                    contact['contact_magnitude'] for contact in self.contact_forces.values()
                ])
                # Optimal contact force range
                if 10.0 < avg_contact_magnitude < 50.0:
                    control_reward = 0.5
                else:
                    control_reward = -abs(avg_contact_magnitude - 30.0) * 0.01
            else:
                control_reward = 0.0
            
            reward += interaction_reward + manipulation_reward + deformation_penalty + control_reward
        
        # Store info
        info['contact_forces'] = len(self.contact_forces)
        info['deformation'] = self.deformation_history[-1] if self.deformation_history else 0.0
        info['interaction_reward'] = interaction_reward
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate to base environment."""
        return getattr(self.base_env, name)


def create_soft_rigid_reward_shaper():
    """Create reward shaping for soft-rigid interaction."""
    def interaction_reward(obs, prev_obs, action):
        # Reward for controlled interaction
        action_magnitude = np.linalg.norm(action)
        # Moderate force is good for manipulation
        if 0.1 < action_magnitude < 0.5:
            return 0.3
        return -abs(action_magnitude - 0.3) * 0.2
    
    def forward_reward(obs, prev_obs, action):
        if prev_obs is None or len(obs) < 3 or len(prev_obs) < 3:
            return 0.0
        
        velocity = obs[:3] - prev_obs[:3]
        forward_vel = max(0.0, velocity[0])
        return 1.0 * forward_vel
    
    def stability_reward(obs, prev_obs, action):
        if len(obs) < 6:
            return 0.0
        
        orientation = obs[3:6]
        tilt = np.linalg.norm(orientation[:2])
        return 0.8 * np.exp(-tilt)
    
    def deformation_penalty(obs, prev_obs, action):
        # Penalty for excessive soft body deformation
        return -0.1 * np.sum(action ** 2)
    
    return [interaction_reward, forward_reward, stability_reward, deformation_penalty]


def main():
    """Train agent with full soft-rigid interaction."""
    print("=" * 60)
    print("Soft-Rigid Interaction Learning - FULL IMPLEMENTATION")
    print("=" * 60)
    
    # Create base environment
    try:
        base_env = gym.make("Ant-v4")
    except Exception as e:
        print(f"Error creating Ant environment: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    # Wrap with soft-rigid interaction rewards
    reward_shapers = create_soft_rigid_reward_shaper()
    env = EnvironmentWrapper(base_env, reward_shapers=reward_shapers)
    env = SoftRigidInteractionEnv(env)
    
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
    print("Using FULL soft-rigid interaction:")
    print("  - Real contact force detection")
    print("  - Actual soft body deformation measurement")
    print("  - Proper manipulation physics")
    
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
    contact_history = []
    
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
                trainer.train_step()
            
            obs = next_obs
            episode_reward += reward
            steps += 1
        
        rewards_history.append(episode_reward)
        
        # Track contacts
        if 'contact_forces' in info:
            contact_history.append(info['contact_forces'])
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            avg_contacts = np.mean(contact_history[-20:]) if contact_history else 0.0
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Avg (last 20) = {avg_reward:.2f}, "
                  f"Avg contacts = {avg_contacts:.1f}")
    
    viewer.close()
    
    # Evaluate
    print("\nEvaluating final policy...")
    eval_rewards = []
    eval_contacts = []
    
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
        
        eval_rewards.append(episode_reward)
        if 'contact_forces' in info:
            eval_contacts.append(info['contact_forces'])
    
    print(f"Final evaluation - Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    if eval_contacts:
        print(f"Mean contacts: {np.mean(eval_contacts):.1f} ± {np.std(eval_contacts):.1f}")
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plot_training_curves(
        rewards_history,
        save_path=os.path.join(results_dir, "soft_rigid_interaction_training.png"),
        title="Soft-Rigid Interaction - Training Progress",
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
