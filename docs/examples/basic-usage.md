# Basic Usage Example

A simple example comparing `VanillaESTrainer` and `EGGROLLTrainer`.

## Overview

This example demonstrates:
- Using both VanillaESTrainer and EGGROLLTrainer
- Parameter matching task
- Comparing performance

## Code

```python
import torch
import torch.nn as nn
from eggroll_trainer import VanillaESTrainer, EGGROLLTrainer

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Create target parameters
target_model = SimpleModel()
target_params = torch.cat([
    p.flatten() for p in target_model.parameters()
])

# Fitness function: minimize distance to target
def fitness_fn(model):
    current_params = torch.cat([
        p.flatten() for p in model.parameters()
    ])
    distance = (current_params - target_params).norm()
    return -distance.item()  # Higher is better (negate distance)

# Test VanillaESTrainer
print("Training with VanillaESTrainer...")
simple_model = SimpleModel()
simple_trainer = VanillaESTrainer(
    model=simple_model,
    fitness_fn=fitness_fn,
    population_size=50,
    learning_rate=0.01,
    sigma=0.1,
    seed=42,
)
simple_trainer.train(num_generations=50)

# Test EGGROLLTrainer
print("\nTraining with EGGROLLTrainer...")
eggroll_model = SimpleModel()
eggroll_trainer = EGGROLLTrainer(
    model=eggroll_model,
    fitness_fn=fitness_fn,
    population_size=256,  # Larger population!
    learning_rate=0.01,
    sigma=0.1,
    rank=1,
    seed=42,
)
eggroll_trainer.train(num_generations=50)

# Compare results
simple_best = max(simple_trainer.history['fitness'])
eggroll_best = max(eggroll_trainer.history['fitness'])

print(f"\nVanillaESTrainer best fitness: {simple_best:.4f}")
print(f"EGGROLLTrainer best fitness: {eggroll_best:.4f}")
```

## Running

```bash
python examples/basic_example.py
```

## Expected Output

```
Training with VanillaESTrainer...
Generation 0: Mean fitness = -1.2345
...
Generation 50: Mean fitness = -0.1234

Training with EGGROLLTrainer...
Generation 0: Mean fitness = -1.2345
...
Generation 50: Mean fitness = -0.0567

VanillaESTrainer best fitness: -0.1234
EGGROLLTrainer best fitness: -0.0567
```

## Key Points

1. **Both trainers work** - They optimize the same objective
2. **EGGROLL can use larger populations** - More efficient
3. **Fitness function** - Returns negative distance (higher is better)
4. **Parameter matching** - Simple task to verify training works

## Next Steps

- See [MNIST Classification](mnist-classification.md) for a real-world example
- Learn about [Fitness Functions](../user-guide/fitness-functions.md)
- Check [User Guide](../user-guide/index.md) for details

