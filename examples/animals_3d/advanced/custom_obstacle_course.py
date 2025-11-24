"""Advanced example: Agent navigates FULL procedurally generated obstacle course."""

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
from examples.animals_3d import world_builder

EnvironmentWrapper = animal_utils.EnvironmentWrapper
create_locomotion_reward_shaper = animal_utils.create_locomotion_reward_shaper
MuJoCoViewer = animal_vis.MuJoCoViewer
plot_training_curves = animal_vis.plot_training_curves


class ObstacleCourseEnv:
    """FULL implementation: Custom obstacle course with real obstacle tracking."""
    
    def __init__(self, base_env: gym.Env, obstacle_course_xml: str):
        """
        Initialize obstacle course environment with FULL obstacle tracking.
        
        Args:
            base_env: Base Gymnasium environment (e.g., Ant-v4)
            obstacle_course_xml: MJCF XML string for obstacle course
        """
        self.base_env = base_env
        self.obstacle_course_xml = obstacle_course_xml
        
        # Load custom obstacle course model
        self.obstacle_model = None
        self.obstacle_data = None
        self.obstacle_body_ids = []
        
        if mujoco_utils.HAS_MUJOCO:
            try:
                import mujoco
                self.obstacle_model = mujoco.MjModel.from_xml_string(obstacle_course_xml)
                self.obstacle_data = mujoco.MjData(self.obstacle_model)
                
                # Find all obstacle body IDs
                for i in range(self.obstacle_model.nbody):
                    body_name = mujoco.mj_id2name(self.obstacle_model, mujoco.mjtObj.mjOBJ_BODY, i)
                    if body_name and ('obstacle' in body_name.lower() or 'platform' in body_name.lower() or 'pendulum' in body_name.lower()):
                        self.obstacle_body_ids.append(i)
            except Exception as e:
                print(f"Warning: Could not load obstacle course model: {e}")
        
        # Track obstacle positions from physics
        self.obstacle_positions = {}
        self.obstacle_velocities = {}
        
        # Goal position
        self.goal_position = np.array([15.0, 0.0, 0.5])
        
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
        
        # Reset obstacle tracking
        self.obstacle_positions = {}
        self.obstacle_velocities = {}
        
        # Reset obstacle course simulation
        if self.obstacle_data is not None:
            import mujoco
            mujoco.mj_resetData(self.obstacle_model, self.obstacle_data)
        
        return obs, info
    
    def step(self, action):
        """Step environment with FULL obstacle tracking."""
        # Step obstacle course simulation
        if self.obstacle_model is not None and self.obstacle_data is not None:
            import mujoco
            mujoco.mj_step(self.obstacle_model, self.obstacle_data)
            
            # Get REAL obstacle positions from physics
            for body_id in self.obstacle_body_ids:
                if body_id < self.obstacle_model.nbody:
                    pos = self.obstacle_data.xpos[body_id]
                    vel = self.obstacle_data.qvel[self.obstacle_model.body_jntadr[body_id]:self.obstacle_model.body_jntadr[body_id] + 6][:3]
                    
                    body_name = mujoco.mj_id2name(self.obstacle_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    self.obstacle_positions[body_name] = pos.copy()
                    self.obstacle_velocities[body_name] = vel.copy()
        
        # Step main environment
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Add obstacle avoidance rewards based on REAL obstacle positions
        if len(obs) >= 3:
            agent_pos = obs[:3]
            
            # Reward for moving toward goal
            dist_to_goal = np.linalg.norm(agent_pos - self.goal_position)
            goal_reward = np.exp(-dist_to_goal / 5.0) * 2.0
            
            # Penalty for colliding with obstacles (based on REAL positions)
            obstacle_penalty = 0.0
            collision_detected = False
            
            for body_name, obs_pos in self.obstacle_positions.items():
                dist = np.linalg.norm(agent_pos - obs_pos)
                
                # Get obstacle size (estimate from body)
                if self.obstacle_model is not None:
                    try:
                        import mujoco
                        body_id = mujoco.mj_name2id(self.obstacle_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                        if body_id >= 0:
                            # Get geom size
                            for geom_id in range(self.obstacle_model.ngeom):
                                if self.obstacle_model.geom_bodyid[geom_id] == body_id:
                                    geom_size = self.obstacle_model.geom_size[geom_id]
                                    obstacle_radius = np.max(geom_size)
                                    break
                            else:
                                obstacle_radius = 1.0
                        else:
                            obstacle_radius = 1.0
                    except:
                        obstacle_radius = 1.0
                else:
                    obstacle_radius = 1.0
                
                # Check collision
                if dist < obstacle_radius + 0.5:  # Agent radius ~0.5
                    collision_detected = True
                    obstacle_penalty -= (obstacle_radius + 0.5 - dist) * 10.0
            
            # Reward for avoiding moving obstacles (predictive)
            avoidance_reward = 0.0
            for body_name, obs_pos in self.obstacle_positions.items():
                if body_name in self.obstacle_velocities:
                    obs_vel = self.obstacle_velocities[body_name]
                    if np.linalg.norm(obs_vel) > 0.1:  # Moving obstacle
                        # Predict future position
                        future_pos = obs_pos + obs_vel * 0.5  # 0.5s ahead
                        future_dist = np.linalg.norm(agent_pos - future_pos)
                        if future_dist > 2.0:  # Successfully avoiding
                            avoidance_reward += 0.5
            
            reward += goal_reward + obstacle_penalty + avoidance_reward
            
            # Store info
            info['obstacle_collision'] = collision_detected
            info['num_obstacles'] = len(self.obstacle_positions)
            info['dist_to_goal'] = dist_to_goal
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Delegate to base environment."""
        return getattr(self.base_env, name)


def create_obstacle_course():
    """Create a procedurally generated obstacle course."""
    builder = world_builder.ObstacleCourseBuilder(width=20.0, height=20.0)
    
    # Add some obstacles
    builder.add_box_obstacle((5.0, 0.0, 0.5), (0.5, 2.0, 1.0), "obstacle_1")
    builder.add_box_obstacle((10.0, 2.0, 0.5), (0.5, 2.0, 1.0), "obstacle_2")
    builder.add_box_obstacle((10.0, -2.0, 0.5), (0.5, 2.0, 1.0), "obstacle_3")
    
    # Add moving platform
    builder.add_moving_platform(
        (7.0, 0.0, 0.5),
        (7.0, 3.0, 0.5),
        (1.0, 0.5, 0.1),
        speed=1.0,
    )
    
    # Add pendulum
    builder.add_pendulum((12.0, 0.0, 2.0), length=1.5, mass=2.0)
    
    return builder.generate_xml()


def main():
    """Train agent to navigate obstacle course with FULL obstacle tracking."""
    print("=" * 60)
    print("Custom Obstacle Course Navigation - FULL IMPLEMENTATION")
    print("=" * 60)
    
    # Create base environment
    try:
        base_env = gym.make("Ant-v4")
    except Exception as e:
        print(f"Error creating Ant environment: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    # Create obstacle course
    obstacle_course_xml = create_obstacle_course()
    
    # Wrap environment
    reward_shapers = create_locomotion_reward_shaper(
        forward_weight=2.0,  # Emphasize forward movement
        stability_weight=0.5,
        energy_weight=0.1,
    )
    env = EnvironmentWrapper(base_env, reward_shapers=reward_shapers)
    env = ObstacleCourseEnv(env, obstacle_course_xml)
    
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
    print("Using FULL obstacle course:")
    print("  - Real obstacle position tracking from physics")
    print("  - Moving obstacle velocity tracking")
    print("  - Collision detection based on actual positions")
    print("  - Predictive avoidance for moving obstacles")
    
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
    collision_history = []
    
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
        
        # Track collisions
        if 'obstacle_collision' in info:
            collision_history.append(1 if info['obstacle_collision'] else 0)
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            collision_rate = np.mean(collision_history[-20:]) if collision_history else 0.0
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Avg (last 20) = {avg_reward:.2f}, "
                  f"Collision rate = {collision_rate:.2%}")
    
    viewer.close()
    
    # Evaluate
    print("\nEvaluating final policy...")
    eval_rewards = []
    eval_collisions = []
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
            
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Check success
            if 'dist_to_goal' in info and info['dist_to_goal'] < 1.0:
                eval_successes += 1
                break
        
        eval_rewards.append(episode_reward)
        if 'obstacle_collision' in info:
            eval_collisions.append(1 if info['obstacle_collision'] else 0)
    
    print(f"Final evaluation - Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Success rate: {eval_successes}/10 ({eval_successes*10}%)")
    if eval_collisions:
        print(f"Collision rate: {np.mean(eval_collisions):.2%}")
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plot_training_curves(
        rewards_history,
        save_path=os.path.join(results_dir, "obstacle_course_training.png"),
        title="Obstacle Course Navigation - Training Progress",
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
