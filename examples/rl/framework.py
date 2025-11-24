"""RL training framework with support for SGD, ES, and EGGROLL optimizers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from collections import deque
import gymnasium as gym

from eggroll_trainer import VanillaESTrainer, EGGROLLTrainer
from .models import PolicyNetwork, ValueNetwork, QNetwork


class ReplayBuffer:
    """Simple replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample batch of experiences."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.BoolTensor(np.array(dones)),
        )
    
    def __len__(self):
        return len(self.buffer)


class OptimizerWrapper:
    """Base class for optimizer wrappers."""
    
    def __init__(self, model: nn.Module, learning_rate: float, **kwargs):
        self.model = model
        self.learning_rate = learning_rate
    
    def step(self, loss: torch.Tensor):
        """Perform optimization step."""
        raise NotImplementedError
    
    def zero_grad(self):
        """Zero gradients."""
        pass


class SGDOptimizer(OptimizerWrapper):
    """SGD optimizer wrapper."""
    
    def __init__(self, model: nn.Module, learning_rate: float, momentum: float = 0.9, **kwargs):
        super().__init__(model, learning_rate)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    def step(self, loss: torch.Tensor):
        """Perform SGD step."""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()


class ESOptimizer(OptimizerWrapper):
    """ES optimizer wrapper using VanillaESTrainer."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        fitness_fn: Callable[[nn.Module], float],
        population_size: int = 32,
        sigma: float = 0.1,
        **kwargs
    ):
        super().__init__(model, learning_rate)
        self.fitness_fn = fitness_fn
        self.trainer = VanillaESTrainer(
            model=model,
            fitness_fn=fitness_fn,
            population_size=population_size,
            learning_rate=learning_rate,
            sigma=sigma,
        )
    
    def step(self, loss: torch.Tensor):
        """Perform ES step (ignores loss, uses fitness function)."""
        self.trainer.train_step()
    
    def zero_grad(self):
        """No-op for ES."""
        pass


class EGGROLLOptimizer(OptimizerWrapper):
    """EGGROLL optimizer wrapper."""
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        fitness_fn: Callable[[nn.Module], float],
        population_size: int = 32,
        sigma: float = 0.1,
        rank: int = 1,
        **kwargs
    ):
        super().__init__(model, learning_rate)
        self.fitness_fn = fitness_fn
        self.trainer = EGGROLLTrainer(
            model=model,
            fitness_fn=fitness_fn,
            population_size=population_size,
            learning_rate=learning_rate,
            sigma=sigma,
            rank=rank,
        )
    
    def step(self, loss: torch.Tensor):
        """Perform EGGROLL step (ignores loss, uses fitness function)."""
        self.trainer.train_step()
    
    def zero_grad(self):
        """No-op for EGGROLL."""
        pass


class RLTrainer:
    """Base class for RL trainers."""
    
    def __init__(
        self,
        env: gym.Env,
        policy: nn.Module,
        optimizer_type: str = "sgd",
        learning_rate: float = 0.01,
        device: torch.device = None,
        **optimizer_kwargs
    ):
        """
        Initialize RL trainer.
        
        Args:
            env: Gymnasium environment
            policy: Policy network
            optimizer_type: 'sgd', 'es', or 'eggroll'
            learning_rate: Learning rate
            device: Device to run on
            **optimizer_kwargs: Additional optimizer arguments
        """
        self.env = env
        self.policy = policy
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = self.policy.to(self.device)
        
        # Create fitness function for ES/EGGROLL
        self.fitness_fn = self._create_fitness_fn(optimizer_kwargs.get("episodes_per_generation", 10))
        
        # Create optimizer
        if optimizer_type == "sgd":
            self.optimizer = SGDOptimizer(self.policy, learning_rate, **optimizer_kwargs)
        elif optimizer_type == "es":
            self.optimizer = ESOptimizer(
                self.policy, learning_rate, self.fitness_fn, **optimizer_kwargs
            )
        elif optimizer_type == "eggroll":
            self.optimizer = EGGROLLOptimizer(
                self.policy, learning_rate, self.fitness_fn, **optimizer_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        self.optimizer_type = optimizer_type
    
    def _create_fitness_fn(self, num_episodes: int = 10) -> Callable[[nn.Module], float]:
        """Create fitness function for ES/EGGROLL."""
        def fitness_fn(model: nn.Module) -> float:
            """Evaluate model by running episodes."""
            model.eval()
            total_reward = 0.0
            
            with torch.no_grad():
                for _ in range(num_episodes):
                    obs, _ = self.env.reset()
                    episode_reward = 0.0
                    done = False
                    steps = 0
                    max_steps = 1000
                    
                    while not done and steps < max_steps:
                        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                        
                        # Handle both PolicyNetwork and QNetwork
                        if isinstance(model, QNetwork):
                            action = model.get_action(obs_tensor, epsilon=0.0)
                            action_np = action.cpu().numpy()[0]
                        else:
                            action, _ = model.get_action(obs_tensor, deterministic=False)
                            action_np = action.cpu().numpy()[0]
                        
                        if isinstance(action_np, np.ndarray):
                            obs, reward, terminated, truncated, _ = self.env.step(action_np)
                        else:
                            obs, reward, terminated, truncated, _ = self.env.step(int(action_np))
                        
                        done = terminated or truncated
                        episode_reward += reward
                        steps += 1
                    
                    total_reward += episode_reward
            
            return total_reward / num_episodes
        
        return fitness_fn
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        raise NotImplementedError
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate policy."""
        self.policy.eval()
        rewards = []
        
        with torch.no_grad():
            for _ in range(num_episodes):
                obs, _ = self.env.reset()
                episode_reward = 0.0
                done = False
                steps = 0
                max_steps = 1000
                
                while not done and steps < max_steps:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    
                    # Handle both PolicyNetwork and QNetwork
                    if isinstance(self.policy, QNetwork):
                        action = self.policy.get_action(obs_tensor, epsilon=0.0)
                        action_np = action.cpu().numpy()[0]
                    else:
                        action, _ = self.policy.get_action(obs_tensor, deterministic=True)
                        action_np = action.cpu().numpy()[0]
                    
                    if isinstance(action_np, np.ndarray):
                        obs, reward, terminated, truncated, _ = self.env.step(action_np)
                    else:
                        obs, reward, terminated, truncated, _ = self.env.step(int(action_np))
                    
                    done = terminated or truncated
                    episode_reward += reward
                    steps += 1
                
                rewards.append(episode_reward)
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
        }


class REINFORCETrainer(RLTrainer):
    """REINFORCE (vanilla policy gradient) trainer."""
    
    def train_step(self) -> Dict[str, float]:
        """Perform REINFORCE training step."""
        if self.optimizer_type in ["es", "eggroll"]:
            # Use ES/EGGROLL optimizer
            self.optimizer.step(torch.tensor(0.0))  # Dummy loss
            return {"loss": 0.0}
        
        # SGD-based REINFORCE
        self.policy.train()
        obs_list, action_list, reward_list = [], [], []
        
        # Collect episode
        obs, _ = self.env.reset()
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob = self.policy.get_action(obs_tensor, deterministic=False)
            
            obs_list.append(obs_tensor)
            action_list.append(action)
            
            if isinstance(action.cpu().numpy()[0], np.ndarray):
                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy()[0])
            else:
                obs, reward, terminated, truncated, _ = self.env.step(int(action.cpu().numpy()[0]))
            
            done = terminated or truncated
            reward_list.append(reward)
            steps += 1
        
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(reward_list):
            G = reward + 0.99 * G  # Discount factor
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize
        
        # Compute loss
        log_probs = []
        for obs_t, action_t in zip(obs_list, action_list):
            log_prob = self.policy.log_prob(obs_t, action_t)
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs)
        loss = -(log_probs * returns).mean()
        
        # Optimize
        self.optimizer.step(loss)
        
        return {"loss": loss.item(), "episode_reward": sum(reward_list)}


class PPOTrainer(RLTrainer):
    """PPO (Proximal Policy Optimization) trainer."""
    
    def __init__(self, *args, clip_epsilon: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_epsilon = clip_epsilon
    
    def train_step(self) -> Dict[str, float]:
        """Perform PPO training step."""
        if self.optimizer_type in ["es", "eggroll"]:
            # Use ES/EGGROLL optimizer
            self.optimizer.step(torch.tensor(0.0))
            return {"loss": 0.0}
        
        # SGD-based PPO
        self.policy.train()
        obs_list, action_list, reward_list, old_log_probs = [], [], [], []
        
        # Collect episode
        obs, _ = self.env.reset()
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob = self.policy.get_action(obs_tensor, deterministic=False)
            
            obs_list.append(obs_tensor)
            action_list.append(action)
            old_log_probs.append(log_prob.detach())
            
            if isinstance(action.cpu().numpy()[0], np.ndarray):
                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy()[0])
            else:
                obs, reward, terminated, truncated, _ = self.env.step(int(action.cpu().numpy()[0]))
            
            done = terminated or truncated
            reward_list.append(reward)
            steps += 1
        
        # Compute returns and advantages (simplified - no value function)
        returns = []
        G = 0
        for reward in reversed(reward_list):
            G = reward + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - returns.mean()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute new log probs
        new_log_probs = []
        for obs_t, action_t in zip(obs_list, action_list):
            log_prob = self.policy.log_prob(obs_t, action_t)
            new_log_probs.append(log_prob)
        
        new_log_probs = torch.stack(new_log_probs)
        old_log_probs = torch.stack(old_log_probs)
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        self.optimizer.step(loss)
        
        return {"loss": loss.item(), "episode_reward": sum(reward_list)}


class ActorCriticTrainer(RLTrainer):
    """Actor-Critic trainer."""
    
    def __init__(self, *args, value_network: Optional[nn.Module] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if value_network is None:
            obs_dim = self.env.observation_space.shape[0]
            self.value_network = ValueNetwork(obs_dim).to(self.device)
        else:
            self.value_network = value_network.to(self.device)
        
        # Value network always uses SGD
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=kwargs.get("learning_rate", 0.01))
    
    def train_step(self) -> Dict[str, float]:
        """Perform Actor-Critic training step."""
        if self.optimizer_type in ["es", "eggroll"]:
            # Policy via ES/EGGROLL, value via SGD
            self.optimizer.step(torch.tensor(0.0))
            # Still train value network with SGD
            self._train_value_network()
            return {"loss": 0.0}
        
        # Both via SGD
        self.policy.train()
        self.value_network.train()
        
        obs_list, action_list, reward_list = [], [], []
        
        # Collect episode
        obs, _ = self.env.reset()
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, log_prob = self.policy.get_action(obs_tensor, deterministic=False)
            
            obs_list.append(obs_tensor)
            action_list.append(action)
            
            if isinstance(action.cpu().numpy()[0], np.ndarray):
                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy()[0])
            else:
                obs, reward, terminated, truncated, _ = self.env.step(int(action.cpu().numpy()[0]))
            
            done = terminated or truncated
            reward_list.append(reward)
            steps += 1
        
        # Compute returns
        returns = []
        G = 0
        for reward in reversed(reward_list):
            G = reward + 0.99 * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Compute values and advantages
        values = []
        for obs_t in obs_list:
            value = self.value_network(obs_t)
            values.append(value)
        
        values = torch.stack(values).squeeze()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy loss
        log_probs = []
        for obs_t, action_t in zip(obs_list, action_list):
            log_prob = self.policy.log_prob(obs_t, action_t)
            log_probs.append(log_prob)
        
        log_probs = torch.stack(log_probs)
        policy_loss = -(log_probs * advantages).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Optimize
        self.optimizer.step(policy_loss)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "episode_reward": sum(reward_list),
        }
    
    def _train_value_network(self):
        """Train value network (used when policy uses ES/EGGROLL)."""
        # Simplified: just run one episode and update value network
        obs_list, reward_list = [], []
        
        obs, _ = self.env.reset()
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action, _ = self.policy.get_action(obs_tensor, deterministic=False)
            
            obs_list.append(obs_tensor)
            
            if isinstance(action.cpu().numpy()[0], np.ndarray):
                obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy()[0])
            else:
                obs, reward, terminated, truncated, _ = self.env.step(int(action.cpu().numpy()[0]))
            
            done = terminated or truncated
            reward_list.append(reward)
            steps += 1
        
        if len(obs_list) > 0:
            returns = []
            G = 0
            for reward in reversed(reward_list):
                G = reward + 0.99 * G
                returns.insert(0, G)
            
            returns = torch.FloatTensor(returns).to(self.device)
            
            values = torch.stack([self.value_network(obs_t) for obs_t in obs_list]).squeeze()
            value_loss = F.mse_loss(values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()


class A2CTrainer(ActorCriticTrainer):
    """A2C (Advantage Actor-Critic) trainer - same as Actor-Critic."""
    pass


class DQNTrainer(RLTrainer):
    """DQN trainer."""
    
    def __init__(self, *args, target_update_freq: int = 100, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure policy is QNetwork
        if not isinstance(self.policy, QNetwork):
            obs_dim = self.env.observation_space.shape[0]
            action_dim = self.env.action_space.n
            self.policy = QNetwork(obs_dim, action_dim, hidden_dims=(64, 64)).to(self.device)
        
        self.target_network = QNetwork(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
        ).to(self.device)
        self.target_network.load_state_dict(self.policy.state_dict())
        self.target_update_freq = target_update_freq
        self.step_count = 0
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = kwargs.get("batch_size", 32)
        self.gamma = kwargs.get("gamma", 0.99)
        self.epsilon = kwargs.get("epsilon_start", 1.0)
        self.epsilon_end = kwargs.get("epsilon_end", 0.01)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.995)
    
    def train_step(self) -> Dict[str, float]:
        """Perform DQN training step."""
        if self.optimizer_type in ["es", "eggroll"]:
            # Use ES/EGGROLL optimizer
            self.optimizer.step(torch.tensor(0.0))
            return {"loss": 0.0}
        
        # SGD-based DQN
        self.policy.train()
        
        # Collect experience
        obs, _ = self.env.reset()
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.policy.get_action(obs_tensor, epsilon=self.epsilon)
            
            if isinstance(action.cpu().numpy()[0], np.ndarray):
                next_obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy()[0])
            else:
                next_obs, reward, terminated, truncated, _ = self.env.step(int(action.cpu().numpy()[0]))
            
            done = terminated or truncated
            self.replay_buffer.push(obs, action.cpu().item(), reward, next_obs, done)
            obs = next_obs
            steps += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Train from replay buffer
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "episode_reward": 0.0}
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute Q values
        q_values = self.policy(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (1 - dones.float()) * self.gamma * next_q_value
        
        # Compute loss
        loss = F.mse_loss(q_value, target_q_value)
        
        # Optimize
        self.optimizer.step(loss)
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.policy.state_dict())
        
        return {"loss": loss.item(), "episode_reward": rewards.sum().item()}

