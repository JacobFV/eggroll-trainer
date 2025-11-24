# Advanced Usage

Advanced techniques and customization options.

## Custom ES Algorithms

Create your own ES algorithm by subclassing `ESTrainer`:

```python
from eggroll_trainer import ESTrainer
import torch

class CMAESTrainer(ESTrainer):
    """Covariance Matrix Adaptation ES (simplified)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize CMA-ES specific state
        self.covariance = torch.eye(
            self.current_params.shape[0], 
            device=self.device
        )
    
    def sample_perturbations(self, population_size):
        """Sample from multivariate Gaussian."""
        param_dim = self.current_params.shape[0]
        # Sample from N(0, covariance)
        L = torch.linalg.cholesky(self.covariance)
        noise = torch.randn(population_size, param_dim, device=self.device)
        return noise @ L.T * self.sigma
    
    def compute_update(self, perturbations, fitnesses):
        """CMA-ES update (simplified)."""
        # Fitness-weighted average
        weights = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
        update = (weights[:, None] * perturbations).mean(dim=0)
        
        # Update covariance (simplified)
        self.covariance = 0.9 * self.covariance + 0.1 * (
            update.outer(update) + 1e-6 * torch.eye(
                self.covariance.shape[0], 
                device=self.device
            )
        )
        
        return update
```

## Device Management

### Automatic Device Detection

By default, trainers use the model's device:

```python
model = MyModel().to('cuda')
trainer = EGGROLLTrainer(model.parameters(), model=model, ...)
# Automatically uses CUDA
```

### Explicit Device Specification

```python
trainer = EGGROLLTrainer(
    model.parameters(),
    model=model,
    fitness_fn=fitness_fn,
    device=torch.device('cuda'),
    ...
)
```

### Multi-GPU (Future)

For multi-GPU, you can distribute population evaluation:

```python
# Example: Distribute population across GPUs
def distributed_fitness(model, population, devices):
    # Split population across devices
    # Evaluate in parallel
    # Gather results
    pass
```

## Hyperparameter Tuning

### Learning Rate

Start with 0.01 and adjust:

```python
# Too high: May overshoot
trainer = EGGROLLTrainer(..., learning_rate=0.1)

# Too low: Slow convergence
trainer = EGGROLLTrainer(..., learning_rate=0.001)

# Good: Typical range
trainer = EGGROLLTrainer(..., learning_rate=0.01)
```

### Sigma (Perturbation Scale)

Tune based on parameter scale:

```python
# Large sigma: More exploration
trainer = EGGROLLTrainer(..., sigma=0.1)

# Small sigma: More exploitation
trainer = EGGROLLTrainer(..., sigma=0.01)

# Adaptive sigma (custom implementation)
class AdaptiveEGGROLLTrainer(EGGROLLTrainer):
    def step(self, closure=None):
        # Adjust sigma based on fitness variance
        if fitness_std < threshold:
            self.sigma *= 1.1  # Increase exploration
        else:
            self.sigma *= 0.9  # Decrease exploration
        super().step(closure)
```

### Population Size

Larger populations = better but slower:

```python
# Small: Fast but noisy
trainer = EGGROLLTrainer(..., population_size=64)

# Medium: Good balance
trainer = EGGROLLTrainer(..., population_size=256)

# Large: Best quality, still efficient with EGGROLL
trainer = EGGROLLTrainer(..., population_size=1024)
```

## Early Stopping

Implement early stopping based on fitness:

```python
class EarlyStoppingTrainer(EGGROLLTrainer):
    def __init__(self, patience=10, min_delta=1e-4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patience = patience
        self.min_delta = min_delta
        self.best_fitness = float('-inf')
        self.wait = 0
    
    def train(self, num_generations):
        for gen in range(num_generations):
            self.step()
            current_fitness = max(self.history['fitness'])
            
            if current_fitness > self.best_fitness + self.min_delta:
                self.best_fitness = current_fitness
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"Early stopping at generation {gen}")
                    break
```

## Checkpointing

Save and load trainer state:

```python
# Save
torch.save({
    'model_state_dict': trainer.model.state_dict(),
    'history': trainer.history,
    'generation': len(trainer.history['fitness']),
}, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
trainer.model.load_state_dict(checkpoint['model_state_dict'])
trainer.history = checkpoint['history']
```

## Custom Training Loop

For more control, use `step()` directly:

```python
trainer = EGGROLLTrainer(...)

for generation in range(100):
    # Custom logic before step
    if generation % 10 == 0:
        evaluate_on_validation_set(trainer.model)
    
    # Training step
    trainer.step()
    
    # Custom logic after step
    if generation % 10 == 0:
        print(f"Generation {generation}: "
              f"Fitness = {trainer.history['fitness'][-1]:.4f}")
```

## Fitness Normalization

EGGROLL supports fitness normalization:

```python
# Global normalization (default)
trainer = EGGROLLTrainer(..., group_size=0)

# Group-based normalization
trainer = EGGROLLTrainer(..., group_size=32)
# Normalizes within groups of 32 population members
```

## Noise Reuse (Antithetic Sampling)

Reduce variance by reusing noise:

```python
# Standard (no reuse)
trainer = EGGROLLTrainer(..., noise_reuse=0)

# Antithetic sampling (use +ε and -ε)
trainer = EGGROLLTrainer(..., noise_reuse=2)
# Evaluates 2x models but reuses noise
```

## Freezing Parameters

Freeze non-matrix parameters:

```python
# Only update matrix parameters (LoRA-style)
trainer = EGGROLLTrainer(..., freeze_nonlora=True)
# Biases and 1D parameters are frozen
```

## Debugging

### Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

trainer = EGGROLLTrainer(...)
trainer.train(num_generations=10)
```

### Inspect Perturbations

```python
class DebugTrainer(EGGROLLTrainer):
    def step(self, closure=None):
        # Inspect perturbations before update
        perturbations = self._sample_perturbations()
        print(f"Perturbation stats: mean={perturbations.mean()}, "
              f"std={perturbations.std()}")
        super().step(closure)
```

### Monitor Fitness Distribution

```python
import numpy as np

# After training
fitnesses = trainer.history['fitness']
print(f"Fitness stats:")
print(f"  Mean: {np.mean(fitnesses):.4f}")
print(f"  Std: {np.std(fitnesses):.4f}")
print(f"  Min: {np.min(fitnesses):.4f}")
print(f"  Max: {np.max(fitnesses):.4f}")
```

## Performance Tips

1. **Use large populations** - EGGROLL makes this efficient
2. **Cache evaluation data** - Pre-load data subsets
3. **Use appropriate rank** - rank=1 is often sufficient
4. **Batch evaluations** - If possible, batch fitness evaluations
5. **Use GPU** - Move model and data to GPU

## Next Steps

- See [Examples](../examples/index.md) for real-world usage
- Check [API Reference](../api-reference/index.md) for details
- Read [Research](../research/index.md) for algorithm details

