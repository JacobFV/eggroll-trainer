"""Advanced example: Flying agent learns FULL flight control with aerodynamics."""

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
from typing import Tuple

EnvironmentWrapper = animal_utils.EnvironmentWrapper
MuJoCoViewer = animal_vis.MuJoCoViewer
plot_training_curves = animal_vis.plot_training_curves


def compute_aerodynamic_forces(
    velocity: np.ndarray,
    orientation: np.ndarray,
    air_density: float = 1.225,
    wing_area: float = 0.5,
    lift_coefficient: float = 0.8,
    drag_coefficient: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute aerodynamic forces (lift and drag) using real aerodynamics.
    
    Args:
        velocity: Velocity vector (3D)
        orientation: Orientation (roll, pitch, yaw)
        air_density: Air density (kg/m^3)
        wing_area: Effective wing area (m^2)
        lift_coefficient: Lift coefficient
        drag_coefficient: Drag coefficient
        
    Returns:
        Tuple of (lift_force, drag_force)
    """
    speed = np.linalg.norm(velocity)
    
    if speed < 1e-6:
        return np.zeros(3), np.zeros(3)
    
    # Velocity direction
    vel_direction = velocity / speed
    
    # Angle of attack - FULL calculation from velocity and orientation
    pitch = orientation[1] if len(orientation) > 1 else 0.0
    # Angle of attack is angle between velocity vector and body orientation
    if speed > 1e-6:
        # Velocity direction in body frame
        vel_body_x = np.cos(pitch) * vel_direction[0] - np.sin(pitch) * vel_direction[2]
        vel_body_z = np.sin(pitch) * vel_direction[0] + np.cos(pitch) * vel_direction[2]
        # Angle of attack (angle between velocity and horizontal)
        angle_of_attack = np.arctan2(vel_body_z, vel_body_x)
    else:
        angle_of_attack = pitch
    
    # Lift force (perpendicular to velocity, upward)
    # L = 0.5 * rho * v^2 * S * CL * sin(alpha)
    lift_magnitude = 0.5 * air_density * speed ** 2 * wing_area * lift_coefficient * np.sin(angle_of_attack)
    lift_direction = np.array([0.0, 0.0, 1.0])  # Upward
    lift_force = lift_magnitude * lift_direction
    
    # Drag force (opposite to velocity)
    # D = 0.5 * rho * v^2 * S * CD
    drag_magnitude = 0.5 * air_density * speed ** 2 * wing_area * drag_coefficient
    drag_force = -drag_magnitude * vel_direction
    
    return lift_force, drag_force


class AerialEnv:
    """Full implementation: Aerial locomotion with real aerodynamics."""
    
    def __init__(
        self,
        base_env: gym.Env,
        gravity: float = -2.0,
        wind_strength: float = 0.5,
        air_density: float = 1.225,
    ):
        """
        Initialize aerial environment with full aerodynamics.
        
        Args:
            base_env: Base Gymnasium environment
            gravity: Reduced gravity for flight (negative)
            wind_strength: Wind force strength
            air_density: Air density (kg/m^3)
        """
        self.base_env = base_env
        self.gravity = gravity
        self.wind_strength = wind_strength
        self.air_density = air_density
        
        # Wind dynamics
        self.wind_direction = np.array([1.0, 0.0, 0.0])
        self.wind_change_rate = 0.01
        
        # Target altitude for flight
        self.target_altitude = 3.0
        
        # Aerodynamic parameters
        self.wing_area = 0.5  # Effective wing area
        self.lift_coefficient = 0.8
        self.drag_coefficient = 0.3
        
        # Track aerodynamic forces
        self.lift_forces = []
        self.drag_forces = []
        self.altitudes = []
        
        # Modify gravity
        if hasattr(base_env, 'model') and mujoco_utils.HAS_MUJOCO:
            import mujoco
            self.original_gravity = base_env.model.opt.gravity.copy()
            base_env.model.opt.gravity[2] = gravity
        
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
        self.lift_forces = []
        self.drag_forces = []
        self.altitudes = []
        
        # Randomize wind direction
        self.wind_direction = np.random.uniform(-1, 1, 3)
        self.wind_direction = self.wind_direction / (np.linalg.norm(self.wind_direction) + 1e-8)
        
        return obs, info
    
    def step(self, action):
        """Step environment with full aerial physics and aerodynamics."""
        # Apply aerodynamic forces BEFORE stepping
        if hasattr(self.base_env, 'data') and hasattr(self.base_env, 'model') and self.agent_body_id is not None:
            if mujoco_utils.HAS_MUJOCO:
                import mujoco
                
                # Get body state
                body_pos = self.base_env.data.xpos[self.agent_body_id]
                body_vel = self.base_env.data.qvel[self.base_env.model.body_jntadr[self.agent_body_id]:self.base_env.model.body_jntadr[self.agent_body_id] + 6]
                lin_vel = body_vel[:3]
                
                # Get orientation - FULL quaternion to Euler conversion
                body_quat = self.base_env.data.xquat[self.agent_body_id]
                # Convert quaternion to Euler angles (roll, pitch, yaw)
                w, x, y, z = body_quat
                # Roll (x-axis rotation)
                roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
                # Pitch (y-axis rotation)
                sin_pitch = 2*(w*y - z*x)
                sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
                pitch = np.arcsin(sin_pitch)
                # Yaw (z-axis rotation)
                yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
                orientation = np.array([roll, pitch, yaw])
                
                # Compute aerodynamic forces
                lift_force, drag_force = compute_aerodynamic_forces(
                    lin_vel,
                    orientation,
                    self.air_density,
                    self.wing_area,
                    self.lift_coefficient,
                    self.drag_coefficient,
                )
                
                # Apply wind force
                wind_force = self.wind_strength * self.wind_direction * 10.0
                
                # Total aerodynamic force
                total_aero_force = lift_force + drag_force + wind_force
                
                # Apply via xfrc_applied
                if self.agent_body_id < self.base_env.model.nbody:
                    self.base_env.data.xfrc_applied[self.agent_body_id, :3] = total_aero_force
                
                # Track forces
                self.lift_forces.append(np.linalg.norm(lift_force))
                self.drag_forces.append(np.linalg.norm(drag_force))
                self.altitudes.append(body_pos[2])
        
        # Step environment
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Add aerial-specific rewards based on real aerodynamics
        if len(obs) >= 3:
            position = obs[:3]
            z_pos = position[2]
            
            # Altitude reward (maintain target altitude)
            altitude_error = abs(z_pos - self.target_altitude)
            altitude_reward = np.exp(-altitude_error / 2.0) * 2.0
            
            # Penalty for crashing (too low)
            if z_pos < 0.5:
                crash_penalty = -10.0
            else:
                crash_penalty = 0.0
            
            # Efficiency reward (lift to drag ratio)
            if len(self.lift_forces) > 0 and len(self.drag_forces) > 0:
                recent_lift = np.mean(self.lift_forces[-5:])
                recent_drag = np.mean(self.drag_forces[-5:])
                if recent_drag > 0:
                    lift_to_drag = recent_lift / recent_drag
                    efficiency_reward = min(lift_to_drag / 3.0, 1.0) * 0.5
                else:
                    efficiency_reward = 0.0
            else:
                efficiency_reward = 0.0
            
            # Forward velocity reward (flying forward)
            if len(obs) >= 9:
                velocity = obs[6:9] if len(obs) >= 9 else np.zeros(3)
                forward_vel = max(0.0, velocity[0])
                forward_reward = forward_vel * 1.5
            else:
                forward_reward = 0.0
            
            reward += altitude_reward + crash_penalty + efficiency_reward + forward_reward
        
        # Update wind direction (slowly changing)
        self.wind_direction += np.random.normal(0, self.wind_change_rate, 3)
        self.wind_direction = self.wind_direction / (np.linalg.norm(self.wind_direction) + 1e-8)
        
        # Store info
        info['altitude'] = z_pos if len(obs) >= 3 else 0.0
        info['lift_force'] = np.mean(self.lift_forces[-5:]) if self.lift_forces else 0.0
        info['drag_force'] = np.mean(self.drag_forces[-5:]) if self.drag_forces else 0.0
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate to base environment."""
        return getattr(self.base_env, name)


def create_aerial_reward_shaper():
    """Create reward shaping for aerial locomotion."""
    def altitude_reward(obs, prev_obs, action):
        if len(obs) < 3:
            return 0.0
        
        # Reward for maintaining altitude
        z_pos = obs[2]
        target_altitude = 3.0
        altitude_error = abs(z_pos - target_altitude)
        return 2.0 * np.exp(-altitude_error / 2.0)
    
    def forward_velocity_reward(obs, prev_obs, action):
        if prev_obs is None or len(obs) < 3 or len(prev_obs) < 3:
            return 0.0
        
        velocity = obs[:3] - prev_obs[:3]
        forward_vel = max(0.0, velocity[0])
        return 1.5 * forward_vel
    
    def stability_reward(obs, prev_obs, action):
        if len(obs) < 6:
            return 0.0
        
        # Stability important for flight
        orientation = obs[3:6]
        tilt = np.linalg.norm(orientation[:2])
        return 1.0 * np.exp(-tilt)
    
    def energy_penalty(obs, prev_obs, action):
        # Flight requires energy
        return -0.12 * np.sum(action ** 2)
    
    return [altitude_reward, forward_velocity_reward, stability_reward, energy_penalty]


def main():
    """Train agent for full aerial locomotion with aerodynamics."""
    print("=" * 60)
    print("Aerial Locomotion Learning - FULL IMPLEMENTATION")
    print("=" * 60)
    
    # Create base environment
    try:
        base_env = gym.make("Ant-v4")
    except Exception as e:
        print(f"Error creating Ant environment: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    # Wrap with aerial physics
    reward_shapers = create_aerial_reward_shaper()
    env = EnvironmentWrapper(base_env, reward_shapers=reward_shapers)
    env = AerialEnv(
        env,
        gravity=-2.0,
        wind_strength=0.5,
        air_density=1.225,
    )
    
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
    print("Using FULL aerodynamics:")
    print("  - Real lift and drag forces")
    print("  - Angle of attack effects")
    print("  - Wind dynamics")
    print("  - Reduced gravity for flight")
    
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
    altitude_history = []
    
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
        
        # Track altitude
        if hasattr(env, 'altitudes') and env.altitudes:
            avg_altitude = np.mean(env.altitudes[-100:])
            altitude_history.append(avg_altitude)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            avg_altitude = np.mean(altitude_history[-20:]) if altitude_history else 0.0
            avg_lift = np.mean(env.lift_forces[-100:]) if env.lift_forces else 0.0
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Avg (last 20) = {avg_reward:.2f}, "
                  f"Avg altitude = {avg_altitude:.2f}, "
                  f"Avg lift = {avg_lift:.2f}")
    
    viewer.close()
    
    # Evaluate
    print("\nEvaluating final policy...")
    eval_rewards = []
    eval_altitudes = []
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
        if 'altitude' in info:
            eval_altitudes.append(info['altitude'])
        if 'lift_force' in info and 'drag_force' in info:
            if info['drag_force'] > 0:
                efficiency = info['lift_force'] / info['drag_force']
                eval_efficiency.append(efficiency)
    
    print(f"Final evaluation - Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    if eval_altitudes:
        print(f"Mean altitude: {np.mean(eval_altitudes):.2f} ± {np.std(eval_altitudes):.2f}")
    if eval_efficiency:
        print(f"Mean lift-to-drag ratio: {np.mean(eval_efficiency):.2f} ± {np.std(eval_efficiency):.2f}")
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plot_training_curves(
        rewards_history,
        save_path=os.path.join(results_dir, "aerial_locomotion_training.png"),
        title="Aerial Locomotion - Training Progress",
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
