"""Advanced example: Agent learns underwater swimming with FULL fluid dynamics."""

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


class UnderwaterEnv:
    """Full implementation: Underwater environment with real fluid dynamics."""
    
    def __init__(
        self,
        base_env: gym.Env,
        water_level: float = 2.0,
        fluid_density: float = 1000.0,
        drag_coefficient: float = 0.5,
        buoyancy_density: float = 500.0,
    ):
        """
        Initialize underwater environment with full fluid dynamics.
        
        Args:
            base_env: Base Gymnasium environment
            water_level: Z-coordinate of water surface
            fluid_density: Density of water (kg/m^3)
            drag_coefficient: Drag coefficient for fluid resistance
            buoyancy_density: Effective density of agent for buoyancy
        """
        self.base_env = base_env
        self.water_level = water_level
        self.fluid_density = fluid_density
        self.drag_coefficient = drag_coefficient
        self.buoyancy_density = buoyancy_density
        
        # Track fluid forces
        self.buoyancy_forces = []
        self.drag_forces = []
        self.depths = []
        
        # Get agent body ID
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
        
        # Modify gravity for underwater effect (reduced gravity)
        if hasattr(base_env, 'model') and mujoco_utils.HAS_MUJOCO:
            import mujoco
            # Store original gravity
            self.original_gravity = base_env.model.opt.gravity.copy()
            # Reduced gravity underwater (buoyancy effect)
            base_env.model.opt.gravity[2] = -4.9  # Half gravity
    
    def reset(self, **kwargs):
        """Reset environment."""
        obs, info = self.base_env.reset(**kwargs)
        
        # Reset tracking
        self.buoyancy_forces = []
        self.drag_forces = []
        self.depths = []
        
        return obs, info
    
    def step(self, action):
        """Step environment with full underwater physics."""
        # Apply fluid dynamics BEFORE stepping
        if hasattr(self.base_env, 'data') and hasattr(self.base_env, 'model') and self.agent_body_id is not None:
            if mujoco_utils.HAS_MUJOCO:
                import mujoco
                
                # Get current body state
                body_pos = self.base_env.data.xpos[self.agent_body_id]
                depth = self.water_level - body_pos[2]
                self.depths.append(depth)
                
                # Apply fluid dynamics forces
                fluid_force = mujoco_utils.apply_fluid_dynamics(
                    self.base_env.data,
                    self.base_env.model,
                    self.agent_body_id,
                    water_level=self.water_level,
                    fluid_density=self.fluid_density,
                    drag_coefficient=self.drag_coefficient,
                    buoyancy_density=self.buoyancy_density,
                )
                
                # Track forces
                buoyancy_force = fluid_force[2]  # Z-component
                drag_force = np.linalg.norm(fluid_force[:2])  # XY components
                self.buoyancy_forces.append(buoyancy_force)
                self.drag_forces.append(drag_force)
        
        # Step environment
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Add underwater-specific rewards based on real physics
        if len(obs) >= 3:
            position = obs[:3]
            z_pos = position[2]
            depth = self.water_level - z_pos
            
            # Buoyancy reward (optimal depth)
            optimal_depth = 1.0
            if 0.5 < depth < 1.5:
                buoyancy_reward = 1.0
            else:
                buoyancy_reward = -abs(depth - optimal_depth) * 0.5
            
            # Drag penalty (energy cost for movement)
            if len(obs) >= 9:
                velocity = obs[6:9] if len(obs) >= 9 else np.zeros(3)
                speed = np.linalg.norm(velocity)
                # Drag is proportional to speed^2
                drag_penalty = -self.drag_coefficient * speed ** 2 * 0.1
            else:
                drag_penalty = 0.0
            
            # Efficiency reward (forward movement per energy)
            if len(obs) >= 3 and hasattr(self, 'prev_pos'):
                forward_vel = max(0.0, position[0] - self.prev_pos[0])
                energy_cost = np.sum(action ** 2)
                if energy_cost > 0:
                    efficiency = forward_vel / (energy_cost + 0.01)
                    efficiency_reward = efficiency * 0.5
                else:
                    efficiency_reward = 0.0
            else:
                efficiency_reward = 0.0
                self.prev_pos = position.copy()
            
            reward += buoyancy_reward + drag_penalty + efficiency_reward
        
        # Store info
        info['depth'] = depth if len(obs) >= 3 else 0.0
        info['buoyancy_force'] = self.buoyancy_forces[-1] if self.buoyancy_forces else 0.0
        info['drag_force'] = self.drag_forces[-1] if self.drag_forces else 0.0
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate to base environment."""
        return getattr(self.base_env, name)


def create_underwater_reward_shaper():
    """Create reward shaping for underwater locomotion."""
    def forward_reward(obs, prev_obs, action):
        if prev_obs is None or len(obs) < 3 or len(prev_obs) < 3:
            return 0.0
        
        # Forward velocity in any direction (3D swimming)
        velocity = obs[:3] - prev_obs[:3]
        forward_vel = np.linalg.norm(velocity)
        return 2.0 * forward_vel  # Higher weight for swimming
    
    def depth_reward(obs, prev_obs, action):
        if len(obs) < 3:
            return 0.0
        
        # Reward for maintaining optimal depth (1.0 below surface)
        z_pos = obs[2]
        target_depth = 1.0
        depth_error = abs(z_pos - target_depth)
        return 0.5 * np.exp(-depth_error)
    
    def stability_reward(obs, prev_obs, action):
        if len(obs) < 6:
            return 0.0
        
        # Stability in water (less critical than on land)
        orientation = obs[3:6]
        tilt = np.linalg.norm(orientation[:2])
        return 0.3 * np.exp(-tilt)
    
    def energy_penalty(obs, prev_obs, action):
        # Higher energy cost underwater
        return -0.15 * np.sum(action ** 2)
    
    return [forward_reward, depth_reward, stability_reward, energy_penalty]


def main():
    """Train agent for underwater locomotion with full fluid dynamics."""
    print("=" * 60)
    print("Underwater Locomotion Learning - FULL IMPLEMENTATION")
    print("=" * 60)
    
    # Create base environment
    try:
        base_env = gym.make("Ant-v4")
    except Exception as e:
        print(f"Error creating Ant environment: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    # Wrap with underwater physics
    reward_shapers = create_underwater_reward_shaper()
    env = EnvironmentWrapper(base_env, reward_shapers=reward_shapers)
    env = UnderwaterEnv(
        env,
        water_level=2.0,
        fluid_density=1000.0,
        drag_coefficient=0.5,
        buoyancy_density=500.0,
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
    print("Using FULL underwater physics:")
    print("  - Real buoyancy forces")
    print("  - Actual drag forces (proportional to velocity^2)")
    print("  - Fluid density effects")
    print("  - Reduced gravity for underwater effect")
    
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
    depth_history = []
    
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
        
        # Track depth
        if hasattr(env, 'depths') and env.depths:
            avg_depth = np.mean(env.depths[-100:])
            depth_history.append(avg_depth)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            avg_depth = np.mean(depth_history[-20:]) if depth_history else 0.0
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Avg (last 20) = {avg_reward:.2f}, "
                  f"Avg depth = {avg_depth:.2f}")
    
    viewer.close()
    
    # Evaluate
    print("\nEvaluating final policy...")
    eval_rewards = []
    eval_depths = []
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
            
            if 'depth' in info:
                eval_depths.append(info['depth'])
        
        eval_rewards.append(episode_reward)
    
    print(f"Final evaluation - Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    if eval_depths:
        print(f"Mean depth: {np.mean(eval_depths):.2f} ± {np.std(eval_depths):.2f}")
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plot_training_curves(
        rewards_history,
        save_path=os.path.join(results_dir, "underwater_locomotion_training.png"),
        title="Underwater Locomotion - Training Progress",
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
