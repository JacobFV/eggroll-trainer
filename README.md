# Eggroll Trainer

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A library for Evolution Strategy (ES) trainers in PyTorch, including the **EGGROLL** algorithm.

## Installation

```bash
# Using uv (recommended)
uv sync
uv run python examples/eggroll_example.py

# For examples that need matplotlib (like mnist_comparison.py):
uv sync --extra dev

# Or using pip
pip install -e .

# With dev dependencies (for examples with plots):
pip install -e ".[dev]"
```

## What is EGGROLL?

**EGGROLL** (Evolution Guided General Optimization via Low-rank Learning) is a novel ES algorithm that provides a **hundredfold increase in training speed** over naïve evolution strategies by using low-rank perturbations instead of full-rank ones.

Key innovation: For matrix parameters W ∈ R^(m×n), EGGROLL samples low-rank matrices A ∈ R^(m×r), B ∈ R^(n×r) where r << min(m,n), forming perturbations A @ B.T. This reduces:
- Memory: O(mn) → O(r(m+n))
- Computation: O(mn) → O(r(m+n))

Yet still achieves high-rank updates through population averaging!

Based on: [Evolution Strategies at the Hyperscale](https://eshyperscale.github.io/)

## Usage

### EGGROLL Trainer (Recommended)

```python
import torch
import torch.nn as nn
from eggroll_trainer import EGGROLLTrainer

# Define a model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)  # Matrix: uses LoRA updates
        self.fc2 = nn.Linear(20, 1)   # Matrix: uses LoRA updates
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Define fitness function (higher is better)
def fitness_fn(model):
    # Your evaluation logic
    return torch.randn(1).item()

# Create EGGROLL trainer
model = SimpleModel()
trainer = EGGROLLTrainer(
    model=model,
    fitness_fn=fitness_fn,
    population_size=256,      # Large populations are efficient!
    learning_rate=0.01,
    sigma=0.1,
    rank=1,                   # Low-rank rank (1 is often sufficient)
    noise_reuse=0,            # 0 = no reuse, 2 = antithetic sampling
    group_size=0,             # 0 = global normalization
    freeze_nonlora=False,     # If True, only update matrix params
    seed=42,
)

# Train
trainer.train(num_generations=100)
```

### Base ES Trainer

For custom ES algorithms, subclass `ESTrainer`:

```python
from eggroll_trainer import ESTrainer
import torch

class MyESTrainer(ESTrainer):
    def sample_perturbations(self, population_size):
        param_dim = self.current_params.shape[0]
        return torch.randn(population_size, param_dim, device=self.device)
    
    def compute_update(self, perturbations, fitnesses):
        weights = (fitnesses - fitnesses.mean()) / fitnesses.std()
        return (weights[:, None] * perturbations).mean(dim=0)
```

## Architecture

### EGGROLLTrainer

The `EGGROLLTrainer` implements the actual EGGROLL algorithm:
- **Low-rank perturbations** for 2D parameters (matrices): Uses A @ B.T where A ∈ R^(m×r), B ∈ R^(n×r)
- **Full-rank perturbations** for 1D/3D+ parameters (biases, etc.)
- **Per-layer updates**: Handles each parameter tensor independently
- **Fitness normalization**: Supports global or group-based normalization
- **Noise reuse**: Optional antithetic sampling for efficiency

### ESTrainer (Base Class)

The base `ESTrainer` class provides:
- Parameter flattening/unflattening utilities
- Training loop framework
- Fitness evaluation infrastructure
- History tracking

Subclasses implement:
- `sample_perturbations()`: How to sample perturbations
- `compute_update()`: How to compute parameter updates from fitnesses

## Examples

See the `examples/` directory:
- `basic_example.py` ⭐ **START HERE** - Side-by-side comparison of SimpleESTrainer and EGGROLLTrainer
- `mnist_comparison.py` - Full EGGROLL vs SGD comparison on MNIST with plots
- `run_all_comparisons.py` - Multi-architecture comparison (CNN, Transformer, MLP)
- `comparison_framework.py` - Reusable framework for comparing optimizers
- `models.py` - Shared model architectures
- `utils.py` - Shared utility functions
- Test suites: `test_comprehensive.py`, `test_eggroll.py`, `test_mnist_eggroll.py`

See `examples/README.md` for detailed documentation.

## Key Features

- ✅ **EGGROLL algorithm** - Low-rank perturbations for massive speedup
- ✅ **PyTorch native** - Works with any PyTorch model
- ✅ **Flexible** - Supports custom ES algorithms via subclassing
- ✅ **Efficient** - Optimized for large population sizes
- ✅ **Well-tested** - Comprehensive test suite included

## References

- [EGGROLL Paper/Blog](https://eshyperscale.github.io/)
- [GitHub Implementation](https://github.com/ESHyperscale/HyperscaleES)

