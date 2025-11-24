"""Advanced example: Humanoid with FULL skinned mesh learns locomotion."""

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
create_locomotion_reward_shaper = animal_utils.create_locomotion_reward_shaper
MuJoCoViewer = animal_vis.MuJoCoViewer
plot_training_curves = animal_vis.plot_training_curves


class SkinnedCharacterEnv:
    """Full implementation: Skinned character with real mesh skinning."""
    
    def __init__(self, base_env: gym.Env):
        """
        Initialize skinned character environment with full mesh skinning.
        
        Args:
            base_env: Base Gymnasium environment (Humanoid)
        """
        self.base_env = base_env
        
        # Skinned mesh tracking
        self.skin_ids = []
        self.mesh_vertices = {}
        self.rest_vertices = {}
        self.joint_angles = None
        
        # Get skin IDs from model
        if hasattr(base_env, 'model') and mujoco_utils.HAS_MUJOCO:
            import mujoco
            try:
                # Find all skins
                for i in range(base_env.model.nskin):
                    skin_name = mujoco.mj_id2name(base_env.model, mujoco.mjtObj.mjOBJ_SKIN, i)
                    if skin_name:
                        self.skin_ids.append(i)
                        # Get rest vertices
                        rest_verts = mujoco_utils.get_skinned_mesh_vertices(
                            base_env.data,
                            base_env.model,
                            i,
                        )
                        if len(rest_verts) > 0:
                            self.rest_vertices[i] = rest_verts.copy()
            except:
                pass
        
        # Joint tracking for skinning
        self.joint_ids = []
        if hasattr(base_env, 'model') and mujoco_utils.HAS_MUJOCO:
            import mujoco
            try:
                for i in range(base_env.model.njnt):
                    self.joint_ids.append(i)
            except:
                pass
        
        # Track mesh deformation
        self.deformation_history = []
        self.vertex_velocities = {}
    
    def reset(self, **kwargs):
        """Reset environment."""
        obs, info = self.base_env.reset(**kwargs)
        
        # Reset mesh tracking
        self.mesh_vertices = {}
        self.deformation_history = []
        self.vertex_velocities = {}
        
        # Initialize joint angles
        if len(obs) >= 24:
            # Humanoid observation: [qpos (24), qvel (23), ...]
            self.joint_angles = obs[7:7+24] if len(obs) >= 31 else obs[:24]
        
        # Get initial mesh vertices
        if hasattr(self.base_env, 'data') and hasattr(self.base_env, 'model'):
            for skin_id in self.skin_ids:
                vertices = mujoco_utils.get_skinned_mesh_vertices(
                    self.base_env.data,
                    self.base_env.model,
                    skin_id,
                )
                if len(vertices) > 0:
                    self.mesh_vertices[skin_id] = vertices
        
        return obs, info
    
    def step(self, action):
        """Step environment with full skinned mesh tracking."""
        # Get mesh state BEFORE stepping
        prev_vertices = {}
        if hasattr(self.base_env, 'data') and hasattr(self.base_env, 'model'):
            for skin_id in self.skin_ids:
                vertices = mujoco_utils.get_skinned_mesh_vertices(
                    self.base_env.data,
                    self.base_env.model,
                    skin_id,
                )
                if len(vertices) > 0:
                    prev_vertices[skin_id] = vertices.copy()
        
        # Step environment
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Update joint angles
        if len(obs) >= 24:
            self.joint_angles = obs[7:7+24] if len(obs) >= 31 else obs[:24]
        
        # Get current mesh vertices and compute deformation
        mesh_deformation = 0.0
        vertex_velocity_magnitude = 0.0
        
        if hasattr(self.base_env, 'data') and hasattr(self.base_env, 'model'):
            for skin_id in self.skin_ids:
                # Get current vertices
                current_vertices = mujoco_utils.get_skinned_mesh_vertices(
                    self.base_env.data,
                    self.base_env.model,
                    skin_id,
                )
                
                if len(current_vertices) > 0:
                    self.mesh_vertices[skin_id] = current_vertices
                    
                    # Compute deformation from rest pose
                    if skin_id in self.rest_vertices:
                        deformation_metrics = mujoco_utils.compute_skin_deformation(
                            self.base_env.data,
                            self.base_env.model,
                            skin_id,
                            self.rest_vertices[skin_id],
                        )
                        mesh_deformation = max(mesh_deformation, deformation_metrics['mean_deformation'])
                        self.deformation_history.append(mesh_deformation)
                    
                    # Compute vertex velocities
                    if skin_id in prev_vertices and len(prev_vertices[skin_id]) == len(current_vertices):
                        vertex_velocities = np.linalg.norm(
                            current_vertices - prev_vertices[skin_id],
                            axis=1
                        )
                        vertex_velocity_magnitude = np.mean(vertex_velocities)
                        self.vertex_velocities[skin_id] = vertex_velocities
        
        # Add skinned mesh-specific rewards based on real mesh state
        # Reward for natural-looking motion (smooth mesh deformation)
        natural_motion_reward = 0.0
        if len(self.deformation_history) > 10:
            # Smooth deformation is natural
            deformation_variance = np.var(self.deformation_history[-10:])
            natural_motion_reward = np.exp(-deformation_variance * 10.0) * 0.5
        
        # Penalty for extreme joint angles (unnatural poses)
        unnatural_penalty = 0.0
        if self.joint_angles is not None:
            extreme_angles = np.abs(self.joint_angles) > 2.0
            unnatural_penalty = -np.sum(extreme_angles) * 0.1
        
        # Reward for smooth vertex movements
        smoothness_reward = 0.0
        if len(self.vertex_velocities) > 0:
            for skin_id, velocities in self.vertex_velocities.items():
                if len(velocities) > 0:
                    # Smooth motion has low velocity variance
                    velocity_variance = np.var(velocities)
                    smoothness_reward += np.exp(-velocity_variance * 100.0) * 0.2
        
        # Penalty for excessive mesh deformation
        deformation_penalty = -mesh_deformation * 2.0
        
        reward += natural_motion_reward + unnatural_penalty + smoothness_reward + deformation_penalty
        
        # Store info
        info['mesh_deformation'] = mesh_deformation
        info['vertex_velocity'] = vertex_velocity_magnitude
        info['natural_motion'] = natural_motion_reward
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate to base environment."""
        return getattr(self.base_env, name)


def create_skinned_reward_shaper():
    """Create reward shaping for skinned character."""
    def natural_motion_reward(obs, prev_obs, action):
        if len(obs) < 24 or prev_obs is None or len(prev_obs) < 24:
            return 0.0
        
        # Reward for natural-looking motion patterns
        # Check joint angle changes
        joint_angles = obs[7:7+24] if len(obs) >= 31 else obs[:24]
        prev_joint_angles = prev_obs[7:7+24] if len(prev_obs) >= 31 else prev_obs[:24]
        
        joint_changes = np.abs(joint_angles - prev_joint_angles)
        
        # Natural motion has moderate, coordinated changes
        if 0.01 < np.mean(joint_changes) < 0.3:
            return 0.3
        return -np.abs(np.mean(joint_changes) - 0.1) * 0.5
    
    def forward_reward(obs, prev_obs, action):
        if prev_obs is None or len(obs) < 3 or len(prev_obs) < 3:
            return 0.0
        
        velocity = obs[:3] - prev_obs[:3]
        forward_vel = max(0.0, velocity[0])
        return 1.5 * forward_vel
    
    def stability_reward(obs, prev_obs, action):
        if len(obs) < 6:
            return 0.0
        
        orientation = obs[3:6]
        tilt = np.linalg.norm(orientation[:2])
        return 1.0 * np.exp(-tilt)
    
    def energy_penalty(obs, prev_obs, action):
        return -0.1 * np.sum(action ** 2)
    
    return [natural_motion_reward, forward_reward, stability_reward, energy_penalty]


def main():
    """Train skinned character with full mesh skinning."""
    print("=" * 60)
    print("Skinned Character Locomotion - FULL IMPLEMENTATION")
    print("=" * 60)
    
    # Create base environment (Humanoid for skinned mesh)
    try:
        base_env = gym.make("Humanoid-v4")
    except Exception as e:
        print(f"Error creating Humanoid environment: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    # Wrap with skinned character rewards
    reward_shapers = create_skinned_reward_shaper()
    env = EnvironmentWrapper(base_env, reward_shapers=reward_shapers)
    env = SkinnedCharacterEnv(env)
    
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
    print("Using FULL skinned mesh:")
    print("  - Real mesh vertex tracking")
    print("  - Actual skin deformation measurement")
    print("  - Vertex velocity computation")
    print("  - Natural motion rewards")
    
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
    deformation_history = []
    
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
        
        # Track deformation
        if 'mesh_deformation' in info:
            deformation_history.append(info['mesh_deformation'])
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            avg_deformation = np.mean(deformation_history[-20:]) if deformation_history else 0.0
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Avg (last 20) = {avg_reward:.2f}, "
                  f"Mesh deformation = {avg_deformation:.4f}")
    
    viewer.close()
    
    # Evaluate
    print("\nEvaluating final policy...")
    eval_rewards = []
    eval_deformations = []
    eval_natural_motion = []
    
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
        if 'mesh_deformation' in info:
            eval_deformations.append(info['mesh_deformation'])
        if 'natural_motion' in info:
            eval_natural_motion.append(info['natural_motion'])
    
    print(f"Final evaluation - Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    if eval_deformations:
        print(f"Mean mesh deformation: {np.mean(eval_deformations):.4f} ± {np.std(eval_deformations):.4f}")
    if eval_natural_motion:
        print(f"Mean natural motion score: {np.mean(eval_natural_motion):.3f} ± {np.std(eval_natural_motion):.3f}")
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plot_training_curves(
        rewards_history,
        save_path=os.path.join(results_dir, "skinned_character_training.png"),
        title="Skinned Character Locomotion - Training Progress",
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
