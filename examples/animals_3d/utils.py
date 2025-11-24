"""Utility functions for 3D animal RL examples."""

import numpy as np
import gymnasium as gym
from typing import Callable, Dict, List, Tuple, Optional
import torch
import torch.nn as nn


class RewardShaper:
    """Reward shaping utilities for locomotion tasks."""
    
    @staticmethod
    def forward_velocity_reward(
        obs: np.ndarray,
        prev_obs: Optional[np.ndarray],
        x_pos_idx: int = 0,
    ) -> float:
        """
        Reward for forward velocity.
        
        Args:
            obs: Current observation
            prev_obs: Previous observation
            x_pos_idx: Index of x position in observation
            
        Returns:
            Reward value
        """
        if prev_obs is None:
            return 0.0
        
        velocity = obs[x_pos_idx] - prev_obs[x_pos_idx]
        return max(0.0, velocity)  # Only reward forward movement
    
    @staticmethod
    def stability_reward(
        obs: np.ndarray,
        orientation_indices: Tuple[int, int, int] = (3, 4, 5),
        target_orientation: Optional[np.ndarray] = None,
    ) -> float:
        """
        Reward for maintaining stable orientation.
        
        Args:
            obs: Current observation
            orientation_indices: Indices of orientation angles (roll, pitch, yaw)
            target_orientation: Target orientation (default: upright)
            
        Returns:
            Reward value
        """
        if target_orientation is None:
            target_orientation = np.array([0.0, 0.0, 0.0])
        
        # Use list indexing to avoid tuple indexing issues
        orientation = np.array([obs[i] for i in orientation_indices])
        error = np.linalg.norm(orientation - target_orientation)
        return np.exp(-error)  # Exponential decay with error
    
    @staticmethod
    def energy_penalty(
        action: np.ndarray,
        penalty_coeff: float = 0.1,
    ) -> float:
        """
        Penalty for energy consumption (action magnitude).
        
        Args:
            action: Action vector
            penalty_coeff: Penalty coefficient
            
        Returns:
            Penalty value (negative)
        """
        return -penalty_coeff * np.sum(action ** 2)
    
    @staticmethod
    def height_reward(
        obs: np.ndarray,
        z_pos_idx: int = 2,
        target_height: float = 1.0,
    ) -> float:
        """
        Reward for maintaining target height.
        
        Args:
            obs: Current observation
            z_pos_idx: Index of z position
            target_height: Target height
            
        Returns:
            Reward value
        """
        height = obs[z_pos_idx]
        error = abs(height - target_height)
        return np.exp(-error)


class ObservationPreprocessor:
    """Observation preprocessing utilities."""
    
    @staticmethod
    def normalize(obs: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Normalize observation."""
        return (obs - mean) / (std + 1e-8)
    
    @staticmethod
    def add_velocity_info(
        obs: np.ndarray,
        prev_obs: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Add velocity information to observation.
        
        Args:
            obs: Current observation
            prev_obs: Previous observation
            
        Returns:
            Extended observation with velocity
        """
        if prev_obs is None:
            velocity = np.zeros_like(obs)
        else:
            velocity = obs - prev_obs
        
        return np.concatenate([obs, velocity])
    
    @staticmethod
    def extract_key_features(
        obs: np.ndarray,
        position_indices: Tuple[int, ...] = (0, 1, 2),
        velocity_indices: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
        """
        Extract key features from observation.
        
        Args:
            obs: Full observation
            position_indices: Indices of position features
            velocity_indices: Indices of velocity features
            
        Returns:
            Extracted features
        """
        features = [obs[i] for i in position_indices]
        
        if velocity_indices is not None:
            features.extend([obs[i] for i in velocity_indices])
        
        return np.array(features)


class MultiAgentCoordinator:
    """Utilities for multi-agent coordination."""
    
    @staticmethod
    def compute_distances(
        positions: List[np.ndarray],
    ) -> np.ndarray:
        """
        Compute pairwise distances between agents.
        
        Args:
            positions: List of agent positions
            
        Returns:
            Distance matrix
        """
        n = len(positions)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    @staticmethod
    def compute_centroid(positions: List[np.ndarray]) -> np.ndarray:
        """
        Compute centroid of agent positions.
        
        Args:
            positions: List of agent positions
            
        Returns:
            Centroid position
        """
        return np.mean(positions, axis=0)
    
    @staticmethod
    def separation_reward(
        positions: List[np.ndarray],
        min_distance: float = 0.5,
        max_distance: float = 2.0,
    ) -> List[float]:
        """
        Compute separation rewards for each agent.
        
        Args:
            positions: List of agent positions
            min_distance: Minimum desired distance
            max_distance: Maximum desired distance
            
        Returns:
            List of rewards, one per agent
        """
        n = len(positions)
        rewards = []
        
        for i in range(n):
            agent_reward = 0.0
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < min_distance:
                        # Too close - penalty
                        agent_reward -= (min_distance - dist) * 10
                    elif dist > max_distance:
                        # Too far - small penalty
                        agent_reward -= (dist - max_distance) * 0.1
                    else:
                        # Good distance - reward
                        agent_reward += 1.0
            
            rewards.append(agent_reward / (n - 1))
        
        return rewards
    
    @staticmethod
    def alignment_reward(
        velocities: List[np.ndarray],
    ) -> float:
        """
        Compute alignment reward (for flocking behavior).
        
        Args:
            velocities: List of agent velocities
            
        Returns:
            Alignment reward
        """
        if len(velocities) < 2:
            return 0.0
        
        # Normalize velocities
        normalized = [v / (np.linalg.norm(v) + 1e-8) for v in velocities]
        
        # Compute average alignment
        avg_vel = np.mean(normalized, axis=0)
        avg_norm = np.linalg.norm(avg_vel)
        
        return avg_norm  # Higher when velocities are aligned


class EnvironmentWrapper(gym.Wrapper):
    """Wrapper for adding reward shaping and observation preprocessing."""
    
    def __init__(
        self,
        env: gym.Env,
        reward_shapers: Optional[List[Callable]] = None,
        obs_preprocessor: Optional[Callable] = None,
    ):
        """
        Initialize wrapper.
        
        Args:
            env: Base environment
            reward_shapers: List of reward shaping functions
            obs_preprocessor: Observation preprocessing function
        """
        super().__init__(env)
        self.reward_shapers = reward_shapers or []
        self.obs_preprocessor = obs_preprocessor
        self.prev_obs = None
    
    def reset(self, **kwargs):
        """Reset environment."""
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        
        if self.obs_preprocessor:
            obs = self.obs_preprocessor(obs, self.prev_obs)
        
        return obs, info
    
    def step(self, action):
        """Step environment with reward shaping."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Apply reward shaping
        shaped_reward = reward
        for shaper in self.reward_shapers:
            shaped_reward += shaper(obs, self.prev_obs, action)
        
        # Preprocess observation
        if self.obs_preprocessor:
            obs = self.obs_preprocessor(obs, self.prev_obs)
        
        self.prev_obs = obs
        
        return obs, shaped_reward, terminated, truncated, info


def create_locomotion_reward_shaper(
    forward_weight: float = 1.0,
    stability_weight: float = 0.5,
    energy_weight: float = 0.1,
) -> List[Callable]:
    """
    Create reward shaping functions for locomotion.
    
    Args:
        forward_weight: Weight for forward velocity reward
        stability_weight: Weight for stability reward
        energy_weight: Weight for energy penalty
        
    Returns:
        List of reward shaping functions
    """
    def forward_reward(obs, prev_obs, action):
        if prev_obs is None:
            return 0.0
        return forward_weight * RewardShaper.forward_velocity_reward(obs, prev_obs)
    
    def stability_reward_fn(obs, prev_obs, action):
        return stability_weight * RewardShaper.stability_reward(obs, orientation_indices=(3, 4, 5))
    
    def energy_penalty_fn(obs, prev_obs, action):
        return energy_weight * RewardShaper.energy_penalty(action)
    
    return [forward_reward, stability_reward_fn, energy_penalty_fn]

