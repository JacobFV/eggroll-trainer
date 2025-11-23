# Custom Trainers

Creating your own Evolution Strategy algorithm.

## Overview

Subclass `ESTrainer` to implement custom ES algorithms like:
- CMA-ES (Covariance Matrix Adaptation)
- Natural ES
- OpenAI ES
- Your own variants

## Basic Template

```python
from eggroll_trainer import ESTrainer
import torch

class MyESTrainer(ESTrainer):
    def sample_perturbations(self, population_size):
        """Sample perturbations for population."""
        param_dim = self.current_params.shape[0]
        # Your sampling logic here
        perturbations = torch.randn(
            population_size,
            param_dim,
            device=self.device
        ) * self.sigma
        return perturbations
    
    def compute_update(self, perturbations, fitnesses):
        """Compute parameter update from perturbations and fitnesses."""
        # Your update logic here
        weights = (fitnesses - fitnesses.mean()) / (
            fitnesses.std() + 1e-8
        )
        update = (weights[:, None] * perturbations).mean(dim=0)
        return update
```

## Example: CMA-ES (Simplified)

```python
class CMAESTrainer(ESTrainer):
    """Simplified Covariance Matrix Adaptation ES."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize covariance matrix
        param_dim = self.current_params.shape[0]
        self.covariance = torch.eye(
            param_dim,
            device=self.device
        )
    
    def sample_perturbations(self, population_size):
        """Sample from multivariate Gaussian."""
        param_dim = self.current_params.shape[0]
        # Cholesky decomposition
        L = torch.linalg.cholesky(
            self.covariance + 1e-6 * torch.eye(
                param_dim,
                device=self.device
            )
        )
        # Sample standard normal
        noise = torch.randn(
            population_size,
            param_dim,
            device=self.device
        )
        # Transform to multivariate Gaussian
        perturbations = noise @ L.T * self.sigma
        return perturbations
    
    def compute_update(self, perturbations, fitnesses):
        """CMA-ES update with covariance adaptation."""
        # Fitness-weighted average
        weights = (fitnesses - fitnesses.mean()) / (
            fitnesses.std() + 1e-8
        )
        update = (weights[:, None] * perturbations).mean(dim=0)
        
        # Update covariance matrix
        self.covariance = 0.9 * self.covariance + 0.1 * (
            update.outer(update) + 1e-6 * torch.eye(
                self.covariance.shape[0],
                device=self.device
            )
        )
        
        return update
```

## Example: Natural ES

```python
class NaturalESTrainer(ESTrainer):
    """Natural Evolution Strategy."""
    
    def sample_perturbations(self, population_size):
        """Sample standard Gaussian perturbations."""
        param_dim = self.current_params.shape[0]
        return torch.randn(
            population_size,
            param_dim,
            device=self.device
        ) * self.sigma
    
    def compute_update(self, perturbations, fitnesses):
        """Natural ES update (fitness-weighted average)."""
        # Rank-based weighting
        ranks = torch.argsort(fitnesses, descending=True)
        weights = torch.zeros_like(fitnesses)
        weights[ranks[:len(ranks)//2]] = 1.0  # Top half
        
        # Normalize
        weights = weights / (weights.sum() + 1e-8)
        
        # Weighted average
        update = (weights[:, None] * perturbations).sum(dim=0)
        return update
```

## Example: Adaptive Sigma

```python
class AdaptiveEGGROLLTrainer(EGGROLLTrainer):
    """EGGROLL with adaptive sigma."""
    
    def __init__(self, *args, sigma_decay=0.99, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma_decay = sigma_decay
        self.initial_sigma = self.sigma
    
    def train_step(self):
        """Override to adapt sigma."""
        # Standard training step
        super().train_step()
        
        # Adapt sigma based on fitness variance
        if len(self.history['fitness']) > 10:
            recent_fitness = self.history['fitness'][-10:]
            fitness_std = torch.tensor(recent_fitness).std().item()
            
            if fitness_std < 0.01:  # Low variance
                self.sigma *= 1.1  # Increase exploration
            else:
                self.sigma *= self.sigma_decay  # Decrease exploration
            
            # Clamp sigma
            self.sigma = max(0.01, min(0.5, self.sigma))
```

## Testing Your Custom Trainer

```python
# Test your custom trainer
def test_custom_trainer():
    model = SimpleModel()
    
    def fitness_fn(model):
        # Simple fitness function
        return torch.randn(1).item()
    
    trainer = MyESTrainer(
        model=model,
        fitness_fn=fitness_fn,
        population_size=50,
        learning_rate=0.01,
        sigma=0.1,
    )
    
    # Train
    trainer.train(num_generations=10)
    
    # Check that fitness improved
    initial_fitness = trainer.history['fitness'][0]
    final_fitness = trainer.history['fitness'][-1]
    
    print(f"Initial fitness: {initial_fitness:.4f}")
    print(f"Final fitness: {final_fitness:.4f}")
    
    assert final_fitness >= initial_fitness, "Fitness should improve"
```

## Tips

1. **Start simple** - Begin with basic Gaussian perturbations
2. **Test thoroughly** - Verify your algorithm works on simple tasks
3. **Compare baselines** - Compare against SimpleESTrainer or EGGROLLTrainer
4. **Document** - Add docstrings explaining your algorithm
5. **Profile** - Check performance bottlenecks

## Next Steps

- See [User Guide](../user-guide/advanced-usage.md) for more advanced techniques
- Check [API Reference](../api-reference/base.md) for ESTrainer details
- Read [Research](../research/index.md) for algorithm inspiration

