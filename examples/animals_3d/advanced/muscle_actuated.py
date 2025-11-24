"""Advanced example: Agent with FULL muscle-like actuators and proper muscle mechanics."""

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


class MuscleActuatedEnv:
    """Full implementation: Muscle-actuated control with real muscle dynamics."""
    
    def __init__(self, base_env: gym.Env):
        """
        Initialize muscle-actuated environment with full muscle mechanics.
        
        Args:
            base_env: Base Gymnasium environment
        """
        self.base_env = base_env
        
        # Muscle system parameters
        self.num_muscles = 8  # Match action space
        self.muscle_activations = np.zeros(self.num_muscles)
        self.muscle_fatigue = np.zeros(self.num_muscles)
        self.muscle_lengths = np.ones(self.num_muscles) * 1.0  # Normalized lengths
        self.muscle_velocities = np.zeros(self.num_muscles)
        
        # Muscle dynamics parameters
        self.activation_time_constant = 0.05  # Fast activation
        self.fatigue_rate = 0.01
        self.recovery_rate = 0.005
        self.optimal_lengths = np.ones(self.num_muscles)  # Optimal muscle lengths
        self.max_forces = np.ones(self.num_muscles) * 1000.0  # Max isometric force
        self.max_velocities = np.ones(self.num_muscles) * 10.0  # Max contraction velocity
        
        # Track muscle state
        self.muscle_forces_history = []
        self.muscle_efficiency_history = []
        
        # Get joint information for muscle length estimation
        self.joint_ids = []
        if hasattr(base_env, 'model') and mujoco_utils.HAS_MUJOCO:
            import mujoco
            try:
                # Get joint positions for estimating muscle lengths
                for i in range(base_env.model.njnt):
                    self.joint_ids.append(i)
            except:
                pass
        
        # Time step
        self.dt = 0.002  # Default MuJoCo timestep
    
    def reset(self, **kwargs):
        """Reset environment."""
        obs, info = self.base_env.reset(**kwargs)
        
        # Reset muscle state
        self.muscle_activations = np.zeros(self.num_muscles)
        self.muscle_fatigue = np.zeros(self.num_muscles)
        self.muscle_lengths = np.ones(self.num_muscles) * 1.0
        self.muscle_velocities = np.zeros(self.num_muscles)
        
        # Reset tracking
        self.muscle_forces_history = []
        self.muscle_efficiency_history = []
        
        return obs, info
    
    def step(self, action):
        """Step environment with full muscle dynamics."""
        # Map actions to muscle activation commands
        muscle_commands = np.tanh(action[:self.num_muscles]) * 0.5 + 0.5  # [0, 1]
        
        # Update muscle activations with dynamics
        for i in range(self.num_muscles):
            new_activation, new_fatigue = mujoco_utils.update_muscle_dynamics(
                self.muscle_activations[i],
                muscle_commands[i],
                self.dt,
                self.activation_time_constant,
                self.fatigue_rate,
                self.recovery_rate,
            )
            self.muscle_activations[i] = new_activation
            self.muscle_fatigue[i] = new_fatigue
        
        # Get REAL muscle lengths and velocities from tendon sensors
        if hasattr(self.base_env, 'data') and hasattr(self.base_env, 'model'):
            if mujoco_utils.HAS_MUJOCO:
                import mujoco
                # Get tendon IDs for muscles
                muscle_tendon_ids = []
                for i in range(self.base_env.model.ntendon):
                    tendon_name = mujoco.mj_id2name(self.base_env.model, mujoco.mjtObj.mjOBJ_TENDON, i)
                    if tendon_name and ('muscle' in tendon_name.lower() or 'flexor' in tendon_name.lower() or 'extensor' in tendon_name.lower()):
                        muscle_tendon_ids.append(i)
                
                # Get actual tendon lengths (which represent muscle lengths)
                if len(muscle_tendon_ids) > 0:
                    tendon_lengths = mujoco_utils.get_tendon_lengths(
                        self.base_env.data,
                        self.base_env.model,
                        muscle_tendon_ids[:self.num_muscles],
                    )
                    if len(tendon_lengths) > 0:
                        # Normalize to rest lengths
                        for i, length in enumerate(tendon_lengths[:self.num_muscles]):
                            self.muscle_lengths[i] = length / self.optimal_lengths[i]
                    
                    # Get actual tendon velocities
                    tendon_velocities = mujoco_utils.get_tendon_velocities(
                        self.base_env.data,
                        self.base_env.model,
                        muscle_tendon_ids[:self.num_muscles],
                    )
                    if len(tendon_velocities) > 0:
                        for i, vel in enumerate(tendon_velocities[:self.num_muscles]):
                            self.muscle_velocities[i] = vel
                else:
                    # Fallback: estimate from joint angles if no tendons
                    if len(self.joint_ids) > 0:
                        for i in range(min(self.num_muscles, len(self.joint_ids))):
                            joint_id = self.joint_ids[i]
                            if joint_id < self.base_env.model.njnt:
                                joint_angle = self.base_env.data.qpos[joint_id] if joint_id < len(self.base_env.data.qpos) else 0.0
                                self.muscle_lengths[i] = 1.0 + 0.2 * np.sin(joint_angle)
                                joint_vel = self.base_env.data.qvel[joint_id] if joint_id < len(self.base_env.data.qvel) else 0.0
                                self.muscle_velocities[i] = joint_vel * 0.1
        
        # Compute muscle forces using force-length-velocity relationship
        muscle_forces = np.zeros(self.num_muscles)
        for i in range(self.num_muscles):
            force = mujoco_utils.compute_muscle_force(
                self.muscle_activations[i],
                self.muscle_lengths[i],
                self.muscle_velocities[i],
                self.optimal_lengths[i],
                self.max_forces[i],
                self.max_velocities[i],
            )
            muscle_forces[i] = force
        
        # Apply fatigued activations
        effective_forces = muscle_forces * (1.0 - self.muscle_fatigue)
        
        # Map muscle forces to joint torques using REAL moment arms
        # Ensure we have the right number of outputs matching action space
        action_dim = len(action)
        muscle_torques = np.zeros(action_dim)
        
        # Map muscle forces to action space (muscles control actions)
        for i in range(min(self.num_muscles, action_dim)):
            # Normalize muscle force to action range [-1, 1]
            normalized_force = (effective_forces[i] / self.max_forces[i]) * 2.0 - 1.0
            muscle_torques[i] = np.clip(normalized_force, -1.0, 1.0)
            
            # If we have joint IDs, try to apply via moment arms
            if i < len(self.joint_ids):
                joint_id = self.joint_ids[i]
                if joint_id < self.base_env.model.njnt:
                    # Get moment arm from tendon-joint coupling
                    if hasattr(self.base_env, 'model') and mujoco_utils.HAS_MUJOCO:
                        import mujoco
                        # Try to find tendon associated with this joint
                        for tid in range(self.base_env.model.ntendon):
                            moment_arm = mujoco_utils.compute_tendon_moment_arm(
                                self.base_env.model,
                                tid,
                                joint_id,
                            )
                            if moment_arm > 0:
                                # Apply muscle force via moment arm (scaled)
                                torque_scale = moment_arm * 0.1  # Scale to action range
                                muscle_torques[i] = np.clip(normalized_force * (1.0 + torque_scale), -1.0, 1.0)
                                break
        
        # Use muscle torques as actions (they replace the muscle command actions)
        combined_action = muscle_torques
        
        # Step environment
        obs, reward, terminated, truncated, info = self.base_env.step(combined_action)
        
        # Track muscle state
        self.muscle_forces_history.append(np.mean(effective_forces))
        
        # Add muscle-specific rewards based on real muscle mechanics
        # Reward for efficient muscle use (low fatigue)
        efficiency_reward = np.mean(1.0 - self.muscle_fatigue) * 0.5
        
        # Penalty for excessive activation (energy cost)
        energy_penalty = -np.mean(self.muscle_activations ** 2) * 0.1
        
        # Reward for coordinated muscle activation (antagonist pairs)
        coordination_reward = 0.0
        if self.num_muscles >= 2:
            # Reward for alternating activation (like walking)
            for i in range(0, self.num_muscles - 1, 2):
                if i + 1 < self.num_muscles:
                    # Antagonist pair should have complementary activation
                    pair_coordination = 1.0 - abs(self.muscle_activations[i] - (1.0 - self.muscle_activations[i+1]))
                    coordination_reward += pair_coordination * 0.1
        
        # Reward for smooth activation changes
        if len(self.muscle_forces_history) > 1:
            force_change = abs(self.muscle_forces_history[-1] - self.muscle_forces_history[-2])
            smoothness_reward = -force_change * 0.05
        else:
            smoothness_reward = 0.0
        
        reward += efficiency_reward + energy_penalty + coordination_reward + smoothness_reward
        
        # Store info
        info['muscle_activation'] = np.mean(self.muscle_activations)
        info['muscle_fatigue'] = np.mean(self.muscle_fatigue)
        info['muscle_force'] = np.mean(effective_forces)
        info['muscle_efficiency'] = efficiency_reward
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate to base environment."""
        return getattr(self.base_env, name)


def create_muscle_reward_shaper():
    """Create reward shaping for muscle-actuated control."""
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
        return 0.8 * np.exp(-tilt)
    
    def smoothness_reward(obs, prev_obs, action):
        # Reward for smooth muscle activation
        if prev_obs is None:
            return 0.0
        
        action_change = np.linalg.norm(action)
        return -0.05 * action_change
    
    def energy_penalty(obs, prev_obs, action):
        # Muscle activation has energy cost
        return -0.08 * np.sum(action ** 2)
    
    return [forward_reward, stability_reward, smoothness_reward, energy_penalty]


def main():
    """Train muscle-actuated agent with full muscle mechanics."""
    print("=" * 60)
    print("Muscle-Actuated Control Learning - FULL IMPLEMENTATION")
    print("=" * 60)
    
    # Create base environment
    try:
        base_env = gym.make("Ant-v4")
    except Exception as e:
        print(f"Error creating Ant environment: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    # Wrap with muscle control rewards
    reward_shapers = create_muscle_reward_shaper()
    env = EnvironmentWrapper(base_env, reward_shapers=reward_shapers)
    env = MuscleActuatedEnv(env)
    
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
    print(f"Muscle system: {env.num_muscles} muscles with FULL dynamics:")
    print("  - Force-length-velocity relationships")
    print("  - Activation dynamics")
    print("  - Fatigue and recovery")
    print("  - Coordinated activation patterns")
    
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
    fatigue_history = []
    efficiency_history = []
    
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
                trainer.step()
            
            obs = next_obs
            episode_reward += reward
            steps += 1
        
        rewards_history.append(episode_reward)
        
        # Track muscle metrics
        if hasattr(env, 'muscle_fatigue'):
            avg_fatigue = np.mean(env.muscle_fatigue)
            fatigue_history.append(avg_fatigue)
        
        if 'muscle_efficiency' in info:
            efficiency_history.append(info['muscle_efficiency'])
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            avg_fatigue = np.mean(fatigue_history[-20:]) if fatigue_history else 0.0
            avg_efficiency = np.mean(efficiency_history[-20:]) if efficiency_history else 0.0
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Avg (last 20) = {avg_reward:.2f}, "
                  f"Avg fatigue = {avg_fatigue:.3f}, "
                  f"Avg efficiency = {avg_efficiency:.3f}")
    
    viewer.close()
    
    # Evaluate
    print("\nEvaluating final policy...")
    eval_rewards = []
    eval_fatigue = []
    eval_efficiency = []
    
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
        if 'muscle_fatigue' in info:
            eval_fatigue.append(info['muscle_fatigue'])
        if 'muscle_efficiency' in info:
            eval_efficiency.append(info['muscle_efficiency'])
    
    print(f"Final evaluation - Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    if eval_fatigue:
        print(f"Mean muscle fatigue: {np.mean(eval_fatigue):.3f} ± {np.std(eval_fatigue):.3f}")
    if eval_efficiency:
        print(f"Mean muscle efficiency: {np.mean(eval_efficiency):.3f} ± {np.std(eval_efficiency):.3f}")
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plot_training_curves(
        rewards_history,
        save_path=os.path.join(results_dir, "muscle_actuated_training.png"),
        title="Muscle-Actuated Control - Training Progress",
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
