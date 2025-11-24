# Trainers

Guide to using ESTrainer, VanillaESTrainer, and EGGROLLTrainer.

## ESTrainer (Base Class)

Abstract base class for implementing custom ES algorithms.

### When to Use

- Implementing a custom ES algorithm
- Research and experimentation
- Understanding ES internals

### Required Methods

Subclasses must implement:

```python
class MyESTrainer(ESTrainer):
    def sample_perturbations(self, population_size):
        """Sample perturbations for population."""
        # Return tensor of shape (population_size, param_dim)
        pass
    
    def compute_update(self, perturbations, fitnesses):
        """Compute parameter update from perturbations and fitnesses."""
        # Return update tensor of shape (param_dim,)
        pass
```

### Example

```python
from eggroll_trainer import ESTrainer
import torch

class CustomESTrainer(ESTrainer):
    def sample_perturbations(self, population_size):
        param_dim = self.current_params.shape[0]
        return torch.randn(
            population_size, 
            param_dim, 
            device=self.device
        ) * self.sigma
    
    def compute_update(self, perturbations, fitnesses):
        # Fitness-weighted average
        weights = (fitnesses - fitnesses.mean()) / (fitnesses.std() + 1e-8)
        return (weights[:, None] * perturbations).mean(dim=0)
```

## VanillaESTrainer

Vanilla ES with full-rank Gaussian perturbations.

### When to Use

- Small models (< 10K parameters)
- Understanding ES basics
- Baseline comparisons

### Usage

```python
from eggroll_trainer import VanillaESTrainer

trainer = VanillaESTrainer(
    model=model,
    fitness_fn=fitness_fn,
    population_size=50,      # Smaller populations OK for small models
    learning_rate=0.01,
    sigma=0.1,
    seed=42,
)

trainer.train(num_generations=100)
```

### Characteristics

- ✅ Simple and straightforward
- ✅ Good for small models
- ❌ Memory intensive for large models
- ❌ Slower than EGGROLL for matrix parameters

## EGGROLLTrainer (Recommended)

Advanced ES with low-rank perturbations.

### When to Use

- **Most use cases** - This is the recommended trainer
- Large models with matrix parameters
- Need for large population sizes
- Memory/computation constraints

### Usage

```python
from eggroll_trainer import EGGROLLTrainer

trainer = EGGROLLTrainer(
    model=model,
    fitness_fn=fitness_fn,
    population_size=256,      # Large populations are efficient!
    learning_rate=0.01,
    sigma=0.1,
    rank=1,                   # Low-rank rank (1 is often sufficient)
    noise_reuse=0,            # 0 = no reuse, 2 = antithetic sampling
    group_size=0,             # 0 = global normalization
    freeze_nonlora=False,      # If True, only update matrix params
    seed=42,
)

trainer.train(num_generations=100)
```

### Parameters

#### `rank` (int, default: 1)

Rank of low-rank perturbations. Controls memory/computation tradeoff:

- **rank=1**: Minimum memory, fastest (recommended)
- **rank=2-4**: Better expressivity, still efficient
- **rank>>1**: Approaches full-rank (not recommended)

#### `noise_reuse` (int, default: 0)

Number of evaluations to reuse noise:

- **0**: No reuse (standard)
- **2**: Antithetic sampling (use +ε and -ε)
- **>2**: Multiple reuses (rarely needed)

Antithetic sampling can reduce variance but requires 2x evaluations.

#### `group_size` (int, default: 0)

Size of groups for fitness normalization:

- **0**: Global normalization (all population members)
- **>0**: Group-based normalization (can improve stability)

#### `freeze_nonlora` (bool, default: False)

If True, only apply LoRA updates to 2D parameters (matrices):

- **False**: Update all parameters (recommended)
- **True**: Only update matrix parameters (biases frozen)

### Characteristics

- ✅ **100x speedup** over full-rank for large models
- ✅ Memory efficient
- ✅ Handles large population sizes
- ✅ Per-layer updates
- ✅ Supports fitness normalization

## Training Process

### Basic Training

```python
trainer.train(num_generations=100)
```

### With Progress Tracking

```python
def log_callback(generation, fitness_history):
    if generation % 10 == 0:
        print(f"Generation {generation}: "
              f"Mean fitness = {fitness_history[-1]:.4f}")

trainer.train(
    num_generations=100,
    log_every=10,
    callback=log_callback
)
```

### Accessing History

```python
# After training
history = trainer.history
print(f"Best fitness: {max(history['fitness'])}")
print(f"Mean fitness: {history['mean_fitness']}")
print(f"Best model fitness: {history['best_fitness']}")
```

### Getting Best Model

```python
best_model = trainer.get_best_model()
# Use best_model for inference
```

## Comparison Table

| Feature | ESTrainer | VanillaESTrainer | EGGROLLTrainer |
|---------|-----------|------------------|----------------|
| **Type** | Abstract base | Full-rank ES | Low-rank ES |
| **Memory** | Depends on impl | O(mn) | O(r(m+n)) |
| **Speed** | Depends on impl | Baseline | ~100x faster |
| **Population Size** | Flexible | Small (<100) | Large (256+) |
| **Use Case** | Custom algos | Small models | **Most cases** |

## Tips

1. **Start with EGGROLLTrainer** - It's the most efficient
2. **Use large populations** - EGGROLL makes this feasible (256-1024)
3. **Start with rank=1** - Often sufficient and fastest
4. **Tune sigma** - Start with 0.1, adjust based on problem scale
5. **Monitor fitness** - Use `trainer.history` to track progress

## Next Steps

- Learn about [Fitness Functions](fitness-functions.md)
- See [Advanced Usage](advanced-usage.md) for customization
- Check [Examples](../examples/index.md) for real-world usage

