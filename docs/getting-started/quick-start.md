# Quick Start

Get started with Eggroll Trainer in just a few minutes!

## Minimal Example

Here's a complete example that trains a simple model:

```python
import torch
import torch.nn as nn
from eggroll_trainer import EGGROLLTrainer

# 1. Define your model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# 2. Define a fitness function (higher is better)
def fitness_fn(model):
    # Example: evaluate on random data
    x = torch.randn(32, 10)
    y_pred = model(x)
    # Simple fitness: maximize output magnitude
    return y_pred.abs().mean().item()

# 3. Create trainer
model = SimpleModel()
trainer = EGGROLLTrainer(
    model.parameters(),
    model=model,
    fitness_fn=fitness_fn,
    population_size=256,
    learning_rate=0.01,
    sigma=0.1,
    rank=1,
    seed=42,
)

# 4. Train!
trainer.train(num_generations=100)

# 5. Get best model
best_model = trainer.get_best_model()
```

## What You Can Build

Eggroll Trainer works great for reinforcement learning in 3D environments:

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
  <img src="../assets/images/ant_walk.png" alt="Ant Locomotion" width="180" style="border-radius: 8px;">
  <img src="../assets/images/halfcheetah_run.png" alt="HalfCheetah Running" width="180" style="border-radius: 8px;">
  <img src="../assets/images/humanoid_stand.png" alt="Humanoid Walking" width="180" style="border-radius: 8px;">
</div>

## Understanding the Code

### Model

Any PyTorch `nn.Module` works! The trainer will optimize all trainable parameters.

### Fitness Function

The fitness function takes a model and returns a scalar score. **Higher is better**. For loss minimization, return `-loss`.

```python
def fitness_fn(model):
    # Evaluate model on your task
    loss = compute_loss(model)
    return -loss  # Convert to maximization
```

### Trainer Parameters

- **`population_size`**: Number of perturbed models evaluated per generation (larger = better but slower)
- **`learning_rate`**: Step size for parameter updates
- **`sigma`**: Standard deviation of perturbations
- **`rank`**: Rank of low-rank perturbations (1 is often sufficient)

## Running the Example

Save the code above to `quick_start.py` and run:

```bash
python quick_start.py
```

You should see output like:

```
Training EGGROLLTrainer...
Generation 0: Mean fitness = 0.1234
Generation 10: Mean fitness = 0.2345
...
Generation 100: Mean fitness = 0.4567
Training complete!
```

## Next Steps

- Learn about [Core Concepts](../user-guide/core-concepts.md)
- See more [Examples](../examples/index.md)
- Read the [User Guide](../user-guide/index.md)

## Common Patterns

### Classification Task

```python
def classification_fitness(model, data_loader):
    correct = 0
    total = 0
    for x, y in data_loader:
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += len(y)
    accuracy = correct / total
    return accuracy  # Higher is better
```

### Regression Task

```python
def regression_fitness(model, data_loader):
    total_loss = 0
    count = 0
    for x, y in data_loader:
        y_pred = model(x)
        loss = nn.functional.mse_loss(y_pred, y)
        total_loss += loss.item()
        count += 1
    avg_loss = total_loss / count
    return -avg_loss  # Convert loss to fitness
```

### Parameter Matching

```python
def parameter_fitness(model, target_params):
    current_params = torch.cat([p.flatten() for p in model.parameters()])
    distance = (current_params - target_params).norm()
    return -distance.item()  # Minimize distance
```

## Tips

1. **Start with `population_size=256`** - EGGROLL is efficient with large populations
2. **Use `rank=1`** - Often sufficient and fastest
3. **Tune `sigma`** - Start with 0.1, adjust based on your problem scale
4. **Monitor fitness** - Use `trainer.history` to track progress

## Getting Help

- Check the [API Reference](../api-reference/index.md)
- See [Examples](../examples/index.md)
- Open an [issue](https://github.com/JacobFV/eggroll-trainer/issues) on GitHub

