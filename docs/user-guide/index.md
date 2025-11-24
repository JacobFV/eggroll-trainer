# User Guide

Learn how to use Eggroll Trainer effectively.

## Sections

- **[Core Concepts](core-concepts.md)** - Understanding Evolution Strategies and EGGROLL
- **[Trainers](trainers.md)** - Using ESTrainer and EGGROLLTrainer
- **[Fitness Functions](fitness-functions.md)** - Writing effective fitness functions
- **[Advanced Usage](advanced-usage.md)** - Custom algorithms, device management, tuning

## Overview

Eggroll Trainer provides three main classes:

1. **ESTrainer** - Base class for custom ES algorithms
2. **VanillaESTrainer** - Vanilla ES with full-rank perturbations
3. **EGGROLLTrainer** - Advanced ES with low-rank perturbations (recommended)

Most users should start with `EGGROLLTrainer` for best performance.

## Quick Reference

### Basic Training Loop

```python
from eggroll_trainer import EGGROLLTrainer

trainer = EGGROLLTrainer(
    model=model,
    fitness_fn=fitness_fn,
    population_size=256,
    learning_rate=0.01,
    sigma=0.1,
)

trainer.train(num_generations=100)
best_model = trainer.get_best_model()
```

### Accessing Training History

```python
# After training
history = trainer.history
print(f"Best fitness: {max(history['fitness'])}")
print(f"Mean fitness over time: {history['mean_fitness']}")
```

### Custom ES Algorithm

```python
from eggroll_trainer import ESTrainer

class MyESTrainer(ESTrainer):
    def sample_perturbations(self, population_size):
        # Your perturbation sampling logic
        pass
    
    def compute_update(self, perturbations, fitnesses):
        # Your update computation logic
        pass
```

## Next Steps

Start with [Core Concepts](core-concepts.md) to understand the fundamentals.

