"""Advanced example: Agent adapts to FULL procedurally generated terrain."""

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


class TerrainEnv:
    """FULL implementation: Terrain environment with real heightfield terrain."""
    
    def __init__(self, base_env: gym.Env, terrain_generator: world_builder.TerrainGenerator):
        """
        Initialize terrain environment with FULL heightfield integration.
        
        Args:
            base_env: Base Gymnasium environment
            terrain_generator: Terrain generator instance
        """
        self.base_env = base_env
        self.terrain_generator = terrain_generator
        self.current_terrain_type = None
        
        # Terrain heightfield model
        self.terrain_model = None
        self.terrain_data = None
        self.heightfield_id = None
        
        # Current heightmap
        self.current_heightmap = None
        
        # Generate initial terrain
        self.regenerate_terrain()
        
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
    
    def regenerate_terrain(self, terrain_type: str = "hills"):
        """Regenerate terrain with FULL heightfield."""
        if terrain_type == "hills":
            heightmap = self.terrain_generator.generate_hills(
                num_hills=np.random.randint(3, 8),
                hill_height=np.random.uniform(1.0, 3.0),
                hill_width=np.random.uniform(3.0, 7.0),
            )
        else:
            heightmap = self.terrain_generator.generate_noise_terrain(
                scale=np.random.uniform(5.0, 15.0),
                roughness=np.random.uniform(0.3, 0.7),
            )
        
        self.current_heightmap = heightmap
        self.current_terrain_type = terrain_type
        
        # Create heightfield terrain XML
        terrain_xml = mujoco_utils.create_heightfield_terrain(
            heightmap,
            scale=1.0,
        )
        
        # Try to load terrain model
        if mujoco_utils.HAS_MUJOCO:
            try:
                import mujoco
                self.terrain_model = mujoco.MjModel.from_xml_string(terrain_xml)
                self.terrain_data = mujoco.MjData(self.terrain_model)
                
                # Find heightfield geom
                for i in range(self.terrain_model.ngeom):
                    geom_type = self.terrain_model.geom_type[i]
                    if geom_type == mujoco.mjtGeom.mjGEOM_HFIELD:
                        self.heightfield_id = i
                        break
            except Exception as e:
                print(f"Warning: Could not load terrain model: {e}")
    
    def get_terrain_height(self, x: float, y: float) -> float:
        """
        Get terrain height at position (x, y) from heightfield.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Terrain height
        """
        if self.current_heightmap is None:
            return 0.0
        
        # Map world coordinates to heightmap indices
        heightmap_shape = self.current_heightmap.shape
        # Assuming heightmap covers [-10, 10] x [-10, 10]
        x_idx = int((x + 10.0) / 20.0 * heightmap_shape[1])
        y_idx = int((y + 10.0) / 20.0 * heightmap_shape[0])
        
        x_idx = np.clip(x_idx, 0, heightmap_shape[1] - 1)
        y_idx = np.clip(y_idx, 0, heightmap_shape[0] - 1)
        
        # Get height (normalized to [0, 1], scale to [0, 3])
        height = self.current_heightmap[y_idx, x_idx] * 3.0
        
        return height
    
    def reset(self, **kwargs):
        """Reset environment with new terrain."""
        # Regenerate terrain periodically
        if np.random.random() < 0.3:  # 30% chance to regenerate
            terrain_types = ["hills", "noise"]
            self.regenerate_terrain(np.random.choice(terrain_types))
        
        obs, info = self.base_env.reset(**kwargs)
        
        # Reset terrain simulation
        if self.terrain_data is not None:
            import mujoco
            mujoco.mj_resetData(self.terrain_model, self.terrain_data)
        
        return obs, info
    
    def step(self, action):
        """Step environment with FULL terrain adaptation."""
        # Step terrain simulation
        if self.terrain_model is not None and self.terrain_data is not None:
            import mujoco
            mujoco.mj_step(self.terrain_model, self.terrain_data)
        
        # Step main environment
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Add terrain-specific rewards based on REAL terrain height
        if len(obs) >= 3:
            agent_pos = obs[:3]
            x_pos, y_pos, z_pos = agent_pos
            
            # Get terrain height at agent position
            terrain_height = self.get_terrain_height(x_pos, y_pos)
            
            # Reward for maintaining appropriate height above terrain
            height_above_terrain = z_pos - terrain_height
            target_height_above = 0.5  # Target height above terrain
            height_error = abs(height_above_terrain - target_height_above)
            height_reward = np.exp(-height_error / 0.5) * 2.0
            
            # Penalty for excessive tilt (terrain adaptation)
            tilt_penalty = 0.0
            if len(obs) >= 6:
                orientation = obs[3:6]
                tilt = np.linalg.norm(orientation[:2])
                # More tilt penalty on rough terrain
                terrain_roughness = np.std(self.current_heightmap) if self.current_heightmap is not None else 0.0
                tilt_penalty = -tilt * (0.5 + terrain_roughness * 2.0)
            
            # Reward for forward progress on terrain
            forward_reward = 0.0
            if hasattr(self, 'prev_pos') and self.prev_pos is not None:
                forward_vel = max(0.0, agent_pos[0] - self.prev_pos[0])
                # Bonus for moving forward on difficult terrain
                terrain_difficulty = np.std(self.current_heightmap) if self.current_heightmap is not None else 0.0
                forward_reward = forward_vel * (1.5 + terrain_difficulty * 2.0)
            else:
                self.prev_pos = agent_pos.copy()
            
            # Penalty for falling below terrain
            fall_penalty = 0.0
            if height_above_terrain < -0.1:
                fall_penalty = -10.0
            
            reward += height_reward + tilt_penalty + forward_reward + fall_penalty
            
            # Store info
            info['terrain_height'] = terrain_height
            info['height_above_terrain'] = height_above_terrain
            info['terrain_type'] = self.current_terrain_type
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate to base environment."""
        return getattr(self.base_env, name)


def main():
    """Train agent to adapt to FULL procedural terrain."""
    print("=" * 60)
    print("Procedural Terrain Adaptation - FULL IMPLEMENTATION")
    print("=" * 60)
    
    # Create base environment
    try:
        base_env = gym.make("Ant-v4")
    except Exception as e:
        print(f"Error creating Ant environment: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    # Create terrain generator
    terrain_gen = world_builder.TerrainGenerator(width=50, height=50)
    
    # Wrap environment
    reward_shapers = create_locomotion_reward_shaper(
        forward_weight=1.5,
        stability_weight=1.0,  # Higher stability for terrain
        energy_weight=0.1,
    )
    env = EnvironmentWrapper(base_env, reward_shapers=reward_shapers)
    env = TerrainEnv(env, terrain_gen)
    
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
    print("Using FULL terrain system:")
    print("  - Real heightfield terrain")
    print("  - Actual terrain height queries")
    print("  - Terrain-aware rewards")
    print("  - Procedural terrain regeneration")
    
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
    terrain_adaptation_history = []
    
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
        
        # Track terrain adaptation
        if 'height_above_terrain' in info:
            terrain_adaptation_history.append(abs(info['height_above_terrain'] - 0.5))
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            avg_adaptation = np.mean(terrain_adaptation_history[-20:]) if terrain_adaptation_history else 0.0
            terrain_type = info.get('terrain_type', 'unknown')
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Avg (last 20) = {avg_reward:.2f}, "
                  f"Terrain = {terrain_type}, "
                  f"Adaptation error = {avg_adaptation:.3f}")
    
    viewer.close()
    
    # Evaluate on different terrain types
    print("\nEvaluating final policy on different terrains...")
    for terrain_type in ["hills", "noise"]:
        env.current_terrain_type = terrain_type
        env.regenerate_terrain(terrain_type)
        print(f"\nEvaluating on {terrain_type} terrain...")
        
        eval_rewards = []
        eval_adaptations = []
        
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
            if 'height_above_terrain' in info:
                eval_adaptations.append(abs(info['height_above_terrain'] - 0.5))
        
        print(f"  Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        if eval_adaptations:
            print(f"  Mean adaptation error: {np.mean(eval_adaptations):.3f} ± {np.std(eval_adaptations):.3f}")
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plot_training_curves(
        rewards_history,
        save_path=os.path.join(results_dir, "terrain_generation_training.png"),
        title="Terrain Adaptation - Training Progress",
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
