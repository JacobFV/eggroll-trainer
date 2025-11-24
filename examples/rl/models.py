"""RL model architectures for policy, value, and Q-networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PolicyNetwork(nn.Module):
    """Policy network for discrete or continuous action spaces."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
        continuous: bool = False,
        action_std: float = 0.5,
    ):
        """
        Initialize policy network.
        
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
            continuous: If True, outputs mean and std for continuous actions
            action_std: Standard deviation for continuous actions (if not learned)
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.action_std = action_std
        
        # Build network layers
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        
        if continuous:
            # Output mean and log_std for continuous actions
            self.mean_head = nn.Linear(input_dim, action_dim)
            self.log_std_head = nn.Linear(input_dim, action_dim)
        else:
            # Output logits for discrete actions
            self.action_head = nn.Linear(input_dim, action_dim)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.network(obs)
        
        if self.continuous:
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, -20, 2)  # Clamp for numerical stability
            return mean, log_std
        else:
            logits = self.action_head(features)
            return logits
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return deterministic action
            
        Returns:
            Tuple of (action, log_prob)
        """
        if self.continuous:
            mean, log_std = self.forward(obs)
            std = torch.exp(log_std)
            
            if deterministic:
                action = mean
                log_prob = torch.zeros(obs.shape[0], device=obs.device)
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                # Clamp action to valid range (assume [-1, 1] for now)
                action = torch.clamp(action, -1.0, 1.0)
            
            return action, log_prob
        else:
            logits = self.forward(obs)
            
            if deterministic:
                action = logits.argmax(dim=-1)
                log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(1)).squeeze(1)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            return action, log_prob
    
    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action given observation."""
        if self.continuous:
            mean, log_std = self.forward(obs)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            return dist.log_prob(action).sum(dim=-1)
        else:
            logits = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(action)


class ValueNetwork(nn.Module):
    """Value network for estimating state values."""
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
    ):
        """
        Initialize value network.
        
        Args:
            obs_dim: Observation space dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()
        self.obs_dim = obs_dim
        
        # Build network layers
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.value_head = nn.Linear(input_dim, 1)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.network(obs)
        value = self.value_head(features)
        return value.squeeze(-1)


class QNetwork(nn.Module):
    """Q-network for DQN (discrete actions only)."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation: str = "relu",
    ):
        """
        Initialize Q-network.
        
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function ('relu', 'tanh', 'elu')
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        input_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.q_head = nn.Linear(input_dim, action_dim)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.network(obs)
        q_values = self.q_head(features)
        return q_values
    
    def get_action(self, obs: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            obs: Observation tensor
            epsilon: Epsilon for epsilon-greedy exploration
            
        Returns:
            Action tensor
        """
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (obs.shape[0],), device=obs.device)
        else:
            q_values = self.forward(obs)
            return q_values.argmax(dim=-1)

