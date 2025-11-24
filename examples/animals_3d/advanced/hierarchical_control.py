"""Advanced example: FULL Hierarchical RL with Option-Critic architecture."""

import sys
import os

# Add project root to path (when running with uv run from repo root)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

from examples.rl.models import PolicyNetwork, ValueNetwork

from examples.animals_3d import utils as animal_utils
from examples.animals_3d import visualization as animal_vis

EnvironmentWrapper = animal_utils.EnvironmentWrapper
create_locomotion_reward_shaper = animal_utils.create_locomotion_reward_shaper
plot_training_curves = animal_vis.plot_training_curves


class OptionPolicy(nn.Module):
    """Policy over options (meta-policy)."""
    
    def __init__(self, obs_dim: int, num_options: int, hidden_dims: tuple = (128, 128)):
        super().__init__()
        self.num_options = num_options
        
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.logits = nn.Linear(prev_dim, num_options)
    
    def forward(self, obs: torch.Tensor):
        features = self.network(obs)
        return self.logits(features)
    
    def get_option(self, obs: torch.Tensor, deterministic: bool = False):
        logits = self.forward(obs)
        if deterministic:
            return logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return dist.sample(), dist.log_prob(dist.sample())


class TerminationFunction(nn.Module):
    """Termination function for options (beta)."""
    
    def __init__(self, obs_dim: int, hidden_dims: tuple = (64, 64)):
        super().__init__()
        
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.network = nn.Sequential(*layers)
        self.termination = nn.Linear(prev_dim, 1)
    
    def forward(self, obs: torch.Tensor):
        features = self.network(obs)
        return torch.sigmoid(self.termination(features))


class HierarchicalRL:
    """Full Option-Critic hierarchical RL implementation."""
    
    def __init__(
        self,
        env: gym.Env,
        device: torch.device,
        num_options: int = 4,
    ):
        """
        Initialize hierarchical RL system with Option-Critic architecture.
        
        Args:
            env: Environment
            device: Device to run on
            num_options: Number of options (sub-policies)
        """
        self.env = env
        self.device = device
        self.num_options = num_options
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        # Option policy (pi_Omega)
        self.option_policy = OptionPolicy(obs_dim, num_options).to(device)
        
        # Sub-policies (pi_omega for each option)
        self.sub_policies = torch.nn.ModuleList([
            PolicyNetwork(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dims=(256, 256),
                continuous=True,
            ).to(device)
            for _ in range(num_options)
        ])
        
        # Termination functions (beta_omega for each option)
        self.termination_functions = torch.nn.ModuleList([
            TerminationFunction(obs_dim).to(device)
            for _ in range(num_options)
        ])
        
        # Value functions
        # Q_U: value of executing option omega in state s
        self.option_values = torch.nn.ModuleList([
            ValueNetwork(obs_dim, hidden_dims=(256, 256)).to(device)
            for _ in range(num_options)
        ])
        
        # Q_Omega: value of being in state s and following option policy
        self.meta_value = ValueNetwork(obs_dim, hidden_dims=(128, 128)).to(device)
        
        # Current option
        self.current_option = 0
        self.option_start_step = 0
    
    def get_action(self, obs: torch.Tensor, step: int, deterministic: bool = False):
        """
        Get action from hierarchical policy using Option-Critic.
        
        Args:
            obs: Observation tensor
            step: Current step
            deterministic: Whether to use deterministic selection
            
        Returns:
            Tuple of (action, log_prob, option_info)
        """
        # Check if current option should terminate
        if step > self.option_start_step:
            termination_prob = self.termination_functions[self.current_option](obs)
            should_terminate = torch.rand(1).item() < termination_prob.item()
            
            if should_terminate:
                # Select new option
                option, option_log_prob = self.option_policy.get_option(obs, deterministic)
                self.current_option = option.item() if isinstance(option, torch.Tensor) else option
                self.option_start_step = step
        
        # Get action from current option's policy
        sub_policy = self.sub_policies[self.current_option]
        action, log_prob = sub_policy.get_action(obs, deterministic=deterministic)
        
        # Get termination probability for current option
        termination_prob = self.termination_functions[self.current_option](obs)
        
        return action, log_prob, {
            'option': self.current_option,
            'termination_prob': termination_prob.item(),
        }
    
    def get_option_value(self, obs: torch.Tensor, option: int):
        """Get Q_U(s, omega) - value of executing option omega in state s."""
        return self.option_values[option](obs)
    
    def get_meta_value(self, obs: torch.Tensor):
        """Get Q_Omega(s) - value of following option policy from state s."""
        return self.meta_value(obs)


def main():
    """Train hierarchical RL system with full Option-Critic."""
    print("=" * 60)
    print("Hierarchical RL Control - FULL Option-Critic Implementation")
    print("=" * 60)
    
    try:
        env = gym.make("Ant-v4")
    except Exception as e:
        print(f"Error creating Ant environment: {e}")
        print("Make sure MuJoCo is installed: pip install 'gymnasium[mujoco]'")
        return
    
    # Wrap with reward shaping
    reward_shapers = create_locomotion_reward_shaper(
        forward_weight=1.5,
        stability_weight=0.5,
        energy_weight=0.1,
    )
    env = EnvironmentWrapper(env, reward_shapers=reward_shapers)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hierarchical_rl = HierarchicalRL(env, device, num_options=4)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Device: {device}")
    print("Using FULL Option-Critic HRL:")
    print("  - Option policy (pi_Omega) for selecting options")
    print("  - Sub-policies (pi_omega) for each option")
    print("  - Termination functions (beta_omega) for each option")
    print("  - Option value functions (Q_U)")
    print("  - Meta value function (Q_Omega)")
    
    # Optimizers
    option_optimizer = torch.optim.Adam(
        list(hierarchical_rl.option_policy.parameters()) +
        list(hierarchical_rl.meta_value.parameters()),
        lr=3e-4,
    )
    
    sub_optimizers = [
        torch.optim.Adam(
            list(hierarchical_rl.sub_policies[i].parameters()) +
            list(hierarchical_rl.option_values[i].parameters()) +
            list(hierarchical_rl.termination_functions[i].parameters()),
            lr=3e-4,
        )
        for i in range(hierarchical_rl.num_options)
    ]
    
    # Training loop
    num_episodes = 200
    rewards_history = []
    option_usage = {i: 0 for i in range(hierarchical_rl.num_options)}
    
    print(f"\nTraining for {num_episodes} episodes...")
    print("-" * 60)
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        max_steps = 1000
        
        # Initialize option
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        option, _ = hierarchical_rl.option_policy.get_option(obs_tensor, deterministic=False)
        hierarchical_rl.current_option = option.item()
        hierarchical_rl.option_start_step = 0
        option_usage[hierarchical_rl.current_option] += 1
        
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_log_probs = []
        episode_options = []
        episode_terminations = []
        
        while not done and steps < max_steps:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, log_prob, option_info = hierarchical_rl.get_action(obs_tensor, steps, deterministic=False)
            action_np = action.detach().cpu().numpy()[0]
            
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            episode_obs.append(obs_tensor)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)
            episode_options.append(option_info['option'])
            episode_terminations.append(option_info['termination_prob'])
            
            obs = next_obs
            episode_reward += reward
            steps += 1
        
        rewards_history.append(episode_reward)
        
        # Compute returns
        returns = []
        G = 0
        gamma = 0.99
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Train option policy and meta value
        if len(episode_obs) > 0:
            obs_batch = torch.cat(episode_obs)
            returns_batch = returns
            
            # Option policy loss
            option_logits = hierarchical_rl.option_policy(obs_batch)
            option_values = hierarchical_rl.meta_value(obs_batch).squeeze()
            
            # Advantage
            advantages = returns_batch - option_values.detach()
            
            # Option policy gradient
            option_dist = torch.distributions.Categorical(logits=option_logits)
            option_log_probs = option_dist.log_prob(torch.tensor(episode_options).to(device))
            option_policy_loss = -(option_log_probs * advantages).mean()
            
            # Meta value loss
            meta_value_loss = F.mse_loss(option_values, returns_batch)
            
            # Update option policy and meta value
            option_optimizer.zero_grad()
            (option_policy_loss + meta_value_loss).backward()
            option_optimizer.step()
            
            # Train sub-policies, option values, and termination functions
            for i in range(hierarchical_rl.num_options):
                # Get steps where this option was active
                option_mask = torch.tensor([opt == i for opt in episode_options]).to(device)
                
                if option_mask.sum() > 0:
                    option_obs = obs_batch[option_mask]
                    option_actions = torch.stack([episode_actions[j] for j in range(len(episode_actions)) if episode_options[j] == i])
                    option_log_probs = torch.stack([episode_log_probs[j] for j in range(len(episode_log_probs)) if episode_options[j] == i])
                    option_returns = returns_batch[option_mask]
                    option_terminations = torch.tensor([episode_terminations[j] for j in range(len(episode_terminations)) if episode_options[j] == i]).to(device)
                    
                    # Option value
                    option_q_values = hierarchical_rl.option_values[i](option_obs).squeeze()
                    option_value_loss = F.mse_loss(option_q_values, option_returns)
                    
                    # Sub-policy loss
                    option_advantages = option_returns - option_q_values.detach()
                    sub_policy_loss = -(option_log_probs * option_advantages).mean()
                    
                    # Termination function loss - FULL Option-Critic termination gradient
                    # Option-Critic termination gradient: beta(s) * (Q_U(s,omega) - V_Omega(s))
                    option_q_values_at_termination = hierarchical_rl.option_values[i](option_obs).squeeze()
                    meta_values_at_termination = hierarchical_rl.meta_value(option_obs).squeeze()
                    termination_advantage = option_q_values_at_termination - meta_values_at_termination.detach()
                    # Termination loss: encourage termination when Q_U < V_Omega (bad option)
                    termination_loss = -torch.mean(option_terminations * termination_advantage)
                    
                    # Update
                    sub_optimizers[i].zero_grad()
                    (option_value_loss + sub_policy_loss + termination_loss).backward()
                    sub_optimizers[i].step()
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean(rewards_history[-20:])
            option_usage_str = ", ".join([f"Opt{i}:{option_usage[i]}" for i in range(hierarchical_rl.num_options)])
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, "
                  f"Avg (last 20) = {avg_reward:.2f}, "
                  f"Options: [{option_usage_str}]")
    
    # Evaluate
    print("\nEvaluating final policy...")
    eval_rewards = []
    eval_option_usage = {i: 0 for i in range(hierarchical_rl.num_options)}
    
    for _ in range(5):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0
        
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        option, _ = hierarchical_rl.option_policy.get_option(obs_tensor, deterministic=True)
        hierarchical_rl.current_option = option.item()
        hierarchical_rl.option_start_step = 0
        eval_option_usage[hierarchical_rl.current_option] += 1
        
        while not done and steps < 500:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action, _, option_info = hierarchical_rl.get_action(obs_tensor, steps, deterministic=True)
            action_np = action.detach().cpu().numpy()[0]
            
            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        eval_rewards.append(episode_reward)
    
    print(f"Final evaluation - Mean reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    print(f"Option usage: {eval_option_usage}")
    
    # Plot results
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(_script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    
    plot_training_curves(
        rewards_history,
        save_path=os.path.join(results_dir, "hierarchical_control_training.png"),
        title="Hierarchical RL Control - Training Progress",
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
