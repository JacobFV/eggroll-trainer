# Eggroll Trainer

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://jacobfv.github.io/eggroll-trainer/)

**Eggroll Trainer** is a library for Evolution Strategy (ES) trainers in PyTorch, featuring the **EGGROLL** algorithm—a novel ES method that provides a **hundredfold increase in training speed** over naïve evolution strategies.

## Features

- 🔄 **EGGROLL Algorithm** - Low-rank perturbations for massive speedup
- 🧩 **PyTorch Native** - Works seamlessly with any PyTorch model
- 🎯 **Flexible Architecture** - Easy to subclass for custom ES algorithms
- 💾 **Memory Efficient** - Optimized for large population sizes
- ⚡ **High Performance** - Near-linear complexity in sequence length
- 🧪 **Well-Tested** - Comprehensive test suite included

## What is EGGROLL?

**EGGROLL** (Evolution Guided General Optimization via Low-rank Learning) is a novel ES algorithm that uses **low-rank perturbations** instead of full-rank ones, dramatically reducing memory and computation requirements.

### Key Innovation

For matrix parameters W ∈ R^(m×n), EGGROLL samples low-rank matrices:
- A ∈ R^(m×r), B ∈ R^(n×r) where r << min(m,n)
- Forms perturbations as A @ B.T

This reduces:
- **Memory**: O(mn) → O(r(m+n))
- **Computation**: O(mn) → O(r(m+n))

Yet still achieves high-rank updates through population averaging!

## Quick Start

```python
import torch
import torch.nn as nn
from eggroll_trainer import EGGROLLTrainer

# Define a model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

# Define fitness function (higher is better)
def fitness_fn(model):
    # Your evaluation logic here
    return torch.randn(1).item()

# Create EGGROLL trainer
model = SimpleModel()
trainer = EGGROLLTrainer(
    model.parameters(),
    model=model,
    fitness_fn=fitness_fn,
    population_size=256,      # Large populations are efficient!
    learning_rate=0.01,
    sigma=0.1,
    rank=1,                   # Low-rank rank (1 is often sufficient)
    seed=42,
)

# Train
trainer.train(num_generations=100)
```

## Installation

=== "pip"

    ```bash
    pip install eggroll-trainer
    ```

=== "uv"

    ```bash
    uv add eggroll-trainer
    ```

For examples with plotting:

```bash
pip install "eggroll-trainer[examples]"
# or
uv add eggroll-trainer --extra examples
```

For development/contributing, see [Contributing Guide](../CONTRIBUTING.md).

## Documentation

- **[Getting Started](getting-started/index.md)** - Installation and quick start guide
- **[User Guide](user-guide/index.md)** - Core concepts, trainers, and advanced usage
- **[API Reference](api-reference/index.md)** - Complete API documentation
- **[Examples](examples/index.md)** - Walkthroughs and tutorials
- **[Research](research/index.md)** - Algorithm details and benchmarks

## Key Concepts

### Evolution Strategies

Evolution Strategies (ES) are a class of black-box optimization algorithms that:
- Evaluate multiple perturbed versions of a model (population)
- Use fitness scores to guide parameter updates
- Don't require gradients—perfect for non-differentiable objectives

### EGGROLL Advantages

1. **Low-Rank Perturbations**: For matrices, uses A @ B.T instead of full noise
2. **Per-Layer Updates**: Handles each parameter tensor independently
3. **Fitness Normalization**: Supports global or group-based normalization
4. **Noise Reuse**: Optional antithetic sampling for efficiency

## References

- [EGGROLL Paper/Blog](https://eshyperscale.github.io/)
- [GitHub Implementation](https://github.com/ESHyperscale/HyperscaleES)

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/JacobFV/eggroll-trainer/blob/main/LICENSE) file for details.

