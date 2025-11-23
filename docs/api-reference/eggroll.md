# EGGROLLTrainer

EGGROLL (Evolution Guided General Optimization via Low-rank Learning) trainer.

::: eggroll_trainer.eggroll.EGGROLLTrainer
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Usage

```python
from eggroll_trainer import EGGROLLTrainer

trainer = EGGROLLTrainer(
    model=model,
    fitness_fn=fitness_fn,
    population_size=256,
    learning_rate=0.01,
    sigma=0.1,
    rank=1,
    noise_reuse=0,
    group_size=0,
    freeze_nonlora=False,
    seed=42,
)

trainer.train(num_generations=100)
```

## Key Parameters

### `rank` (int, default: 1)

Rank of low-rank perturbations. Controls memory/computation tradeoff:

- **rank=1**: Minimum memory, fastest (recommended)
- **rank=2-4**: Better expressivity, still efficient
- **rank>>1**: Approaches full-rank (not recommended)

### `noise_reuse` (int, default: 0)

Number of evaluations to reuse noise:

- **0**: No reuse (standard)
- **2**: Antithetic sampling (use +ε and -ε)
- **>2**: Multiple reuses (rarely needed)

### `group_size` (int, default: 0)

Size of groups for fitness normalization:

- **0**: Global normalization (all population members)
- **>0**: Group-based normalization (can improve stability)

### `freeze_nonlora` (bool, default: False)

If True, only apply LoRA updates to 2D parameters (matrices):

- **False**: Update all parameters (recommended)
- **True**: Only update matrix parameters (biases frozen)

## Characteristics

- ✅ **100x speedup** over full-rank for large models
- ✅ Memory efficient
- ✅ Handles large population sizes
- ✅ Per-layer updates
- ✅ Supports fitness normalization

## See Also

- [User Guide](../user-guide/trainers.md) - Detailed usage guide
- [Research](../research/eggroll-algorithm.md) - Algorithm details

