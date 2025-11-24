# Examples

Walkthroughs and tutorials for using Eggroll Trainer.

## Available Examples

- **[Basic Usage](basic-usage.md)** - Simple example with both trainers
- **[MNIST Classification](mnist-classification.md)** - Full EGGROLL vs SGD comparison
- **[Custom Trainers](custom-trainers.md)** - Creating your own ES algorithm

## 3D Reinforcement Learning

Train agents in 3D MuJoCo environments:

<div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0;">
  <img src="../assets/images/ant_walk.png" alt="Ant Locomotion" width="180" style="border-radius: 8px;">
  <img src="../assets/images/halfcheetah_run.png" alt="HalfCheetah Running" width="180" style="border-radius: 8px;">
  <img src="../assets/images/humanoid_stand.png" alt="Humanoid Walking" width="180" style="border-radius: 8px;">
  <img src="../assets/images/hopper.png" alt="Hopper Locomotion" width="180" style="border-radius: 8px;">
  <img src="../assets/images/walker2d.png" alt="Walker2d" width="180" style="border-radius: 8px;">
  <img src="../assets/images/swimmer.png" alt="Swimmer" width="180" style="border-radius: 8px;">
  <img src="../assets/images/reacher.png" alt="Reacher" width="180" style="border-radius: 8px;">
</div>

See `examples/animals_3d/` for 3D RL examples including:
- Simple locomotion (Ant, HalfCheetah, Humanoid)
- Multi-agent scenarios (flocking, predator-prey)
- Advanced physics (underwater, aerial, cloth simulation)

## Running Examples

All examples are in the `examples/` directory:

```bash
# Basic example
python examples/basic_example.py

# MNIST comparison
python examples/mnist_comparison.py

# Multi-architecture comparison
python examples/run_all_comparisons.py
```

## Example Structure

Each example demonstrates:

1. **Model Definition** - PyTorch model architecture
2. **Fitness Function** - How to evaluate model performance
3. **Trainer Setup** - Configuring the trainer
4. **Training Loop** - Running the training
5. **Results** - Analyzing performance

## Quick Example

```python
import torch
import torch.nn as nn
from eggroll_trainer import EGGROLLTrainer

# Model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

# Fitness function
def fitness_fn(model):
    x = torch.randn(32, 10)
    y_pred = model(x)
    return y_pred.abs().mean().item()

# Train
model = SimpleModel()
trainer = EGGROLLTrainer(
    model.parameters(),
    model=model,
    fitness_fn=fitness_fn,
    population_size=256,
    learning_rate=0.01,
    sigma=0.1,
)

trainer.train(num_generations=100)
```

## Next Steps

- Start with [Basic Usage](basic-usage.md)
- See [User Guide](../user-guide/index.md) for detailed explanations
- Check [API Reference](../api-reference/index.md) for API details

