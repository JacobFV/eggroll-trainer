# SimpleESTrainer

Simple Evolution Strategy trainer with full-rank Gaussian perturbations.

::: eggroll_trainer.simple.SimpleESTrainer
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Usage

```python
from eggroll_trainer import SimpleESTrainer

trainer = SimpleESTrainer(
    model=model,
    fitness_fn=fitness_fn,
    population_size=50,
    learning_rate=0.01,
    sigma=0.1,
    seed=42,
)

trainer.train(num_generations=100)
```

## When to Use

- Small models (< 10K parameters)
- Understanding ES basics
- Baseline comparisons

## Characteristics

- ✅ Simple and straightforward
- ✅ Good for small models
- ❌ Memory intensive for large models
- ❌ Slower than EGGROLL for matrix parameters

## See Also

- [EGGROLLTrainer](eggroll.md) - Recommended for most use cases
- [User Guide](../user-guide/trainers.md) - Detailed usage guide

