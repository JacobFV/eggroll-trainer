"""Environment configurations for RL comparison."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import gymnasium as gym


@dataclass
class EnvironmentConfig:
    """Configuration for a Gymnasium environment."""
    name: str
    env_id: str
    max_episode_steps: int
    observation_space_type: str  # 'box', 'discrete', etc.
    action_space_type: str  # 'discrete', 'box'
    observation_dim: Optional[int] = None
    action_dim: Optional[int] = None
    continuous_actions: bool = False
    requires_mujoco: bool = False
    hyperparameters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and set dimensions if not provided."""
        if self.observation_dim is None or self.action_dim is None:
            try:
                env = gym.make(self.env_id)
                if self.observation_dim is None:
                    obs_space = env.observation_space
                    if hasattr(obs_space, 'shape'):
                        self.observation_dim = obs_space.shape[0] if len(obs_space.shape) == 1 else obs_space.shape
                    elif hasattr(obs_space, 'n'):
                        self.observation_dim = obs_space.n
                
                if self.action_dim is None:
                    action_space = env.action_space
                    if hasattr(action_space, 'n'):
                        self.action_dim = action_space.n
                        self.continuous_actions = False
                    elif hasattr(action_space, 'shape'):
                        self.action_dim = action_space.shape[0] if len(action_space.shape) == 1 else action_space.shape[0]
                        self.continuous_actions = True
                
                env.close()
            except Exception as e:
                print(f"Warning: Could not auto-detect dimensions for {self.env_id}: {e}")


# List of environments to test
ENVIRONMENTS: List[EnvironmentConfig] = [
    EnvironmentConfig(
        name="CartPole",
        env_id="CartPole-v1",
        max_episode_steps=500,
        observation_space_type="box",
        action_space_type="discrete",
        continuous_actions=False,
        hyperparameters={
            "learning_rate": 0.01,
            "population_size": 32,
            "sigma": 0.1,
            "episodes_per_generation": 10,
        },
    ),
    EnvironmentConfig(
        name="Acrobot",
        env_id="Acrobot-v1",
        max_episode_steps=500,
        observation_space_type="box",
        action_space_type="discrete",
        continuous_actions=False,
        hyperparameters={
            "learning_rate": 0.01,
            "population_size": 32,
            "sigma": 0.1,
            "episodes_per_generation": 10,
        },
    ),
    EnvironmentConfig(
        name="MountainCar",
        env_id="MountainCar-v0",
        max_episode_steps=200,
        observation_space_type="box",
        action_space_type="discrete",
        continuous_actions=False,
        hyperparameters={
            "learning_rate": 0.01,
            "population_size": 32,
            "sigma": 0.1,
            "episodes_per_generation": 10,
        },
    ),
    EnvironmentConfig(
        name="Pendulum",
        env_id="Pendulum-v1",
        max_episode_steps=200,
        observation_space_type="box",
        action_space_type="box",
        continuous_actions=True,
        hyperparameters={
            "learning_rate": 0.01,
            "population_size": 32,
            "sigma": 0.1,
            "episodes_per_generation": 10,
        },
    ),
    EnvironmentConfig(
        name="LunarLander",
        env_id="LunarLander-v2",
        max_episode_steps=1000,
        observation_space_type="box",
        action_space_type="discrete",
        continuous_actions=False,
        hyperparameters={
            "learning_rate": 0.001,
            "population_size": 64,
            "sigma": 0.05,
            "episodes_per_generation": 5,
        },
    ),
    EnvironmentConfig(
        name="LunarLanderContinuous",
        env_id="LunarLander-v2",
        max_episode_steps=1000,
        observation_space_type="box",
        action_space_type="box",
        continuous_actions=True,
        # Note: LunarLander-v2 actually has discrete actions, but we'll treat it as continuous for testing
        hyperparameters={
            "learning_rate": 0.001,
            "population_size": 64,
            "sigma": 0.05,
            "episodes_per_generation": 5,
        },
    ),
    EnvironmentConfig(
        name="BipedalWalker",
        env_id="BipedalWalker-v3",
        max_episode_steps=1600,
        observation_space_type="box",
        action_space_type="box",
        continuous_actions=True,
        requires_mujoco=False,
        hyperparameters={
            "learning_rate": 0.001,
            "population_size": 64,
            "sigma": 0.05,
            "episodes_per_generation": 5,
        },
    ),
    EnvironmentConfig(
        name="HalfCheetah",
        env_id="HalfCheetah-v4",
        max_episode_steps=1000,
        observation_space_type="box",
        action_space_type="box",
        continuous_actions=True,
        requires_mujoco=True,
        hyperparameters={
            "learning_rate": 0.001,
            "population_size": 128,
            "sigma": 0.05,
            "episodes_per_generation": 3,
        },
    ),
    EnvironmentConfig(
        name="Ant",
        env_id="Ant-v4",
        max_episode_steps=1000,
        observation_space_type="box",
        action_space_type="box",
        continuous_actions=True,
        requires_mujoco=True,
        hyperparameters={
            "learning_rate": 0.001,
            "population_size": 128,
            "sigma": 0.05,
            "episodes_per_generation": 3,
        },
    ),
]


def get_environment_config(env_id: str) -> Optional[EnvironmentConfig]:
    """Get environment configuration by ID."""
    for config in ENVIRONMENTS:
        if config.env_id == env_id:
            return config
    return None


def get_environments(require_mujoco: bool = False, quick_mode: bool = False) -> List[EnvironmentConfig]:
    """
    Get list of environments to test.
    
    Args:
        require_mujoco: If True, only return environments that require MuJoCo
        quick_mode: If True, return only a subset of simpler environments
        
    Returns:
        List of environment configurations
    """
    if quick_mode:
        # Return only simple environments for quick testing
        return [
            env for env in ENVIRONMENTS
            if env.name in ["CartPole", "Pendulum"] and not env.requires_mujoco
        ]
    
    if require_mujoco:
        return [env for env in ENVIRONMENTS if env.requires_mujoco]
    else:
        return [env for env in ENVIRONMENTS if not env.requires_mujoco]

