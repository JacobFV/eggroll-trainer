# API Reference

Complete API documentation for Eggroll Trainer.

## Modules

- **[ESTrainer](base.md)** - Base class for Evolution Strategy trainers
- **[SimpleESTrainer](simple.md)** - Simple ES with full-rank perturbations
- **[EGGROLLTrainer](eggroll.md)** - EGGROLL algorithm with low-rank perturbations

## Quick Import

```python
from eggroll_trainer import ESTrainer, SimpleESTrainer, EGGROLLTrainer
```

## Class Hierarchy

```
ESTrainer (abstract)
├── SimpleESTrainer
└── EGGROLLTrainer
```

## Common Patterns

### Basic Usage

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

### Custom ES Algorithm

```python
from eggroll_trainer import ESTrainer

class MyESTrainer(ESTrainer):
    def sample_perturbations(self, population_size):
        # Your implementation
        pass
    
    def compute_update(self, perturbations, fitnesses):
        # Your implementation
        pass
```

## See Also

- [User Guide](../user-guide/index.md) - Usage guide
- [Examples](../examples/index.md) - Code examples
- [Research](../research/index.md) - Algorithm details

