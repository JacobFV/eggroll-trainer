# EGGROLL Algorithm

Deep dive into the EGGROLL algorithm implementation.

## Overview

**EGGROLL** (Evolution Guided General Optimization via Low-rank Learning) uses low-rank perturbations to achieve massive speedups while maintaining high-rank updates through population averaging.

## Key Innovation

### Low-Rank Perturbations

For matrix parameters W ∈ R^(m×n):

**Standard ES:**
```python
N = torch.randn(m, n) * sigma  # Full-rank noise
W_perturbed = W + N
```
- Memory: O(mn)
- Computation: O(mn)

**EGGROLL:**
```python
A = torch.randn(m, r) * sigma  # Low-rank factors
B = torch.randn(n, r) * sigma
W_perturbed = W + A @ B.T
```
- Memory: O(r(m+n))
- Computation: O(r(m+n))
- Speedup: ~100x for typical r=1

## Algorithm Details

### Per-Layer Updates

EGGROLL handles each parameter tensor independently:

1. **2D parameters (matrices)**: Use low-rank perturbations (A @ B.T)
2. **1D/3D+ parameters**: Use full-rank perturbations (standard Gaussian)

### Perturbation Sampling

```python
# For matrix W ∈ R^(m×n) with rank r
A = torch.randn(m, r, device=device) * sigma
B = torch.randn(n, r, device=device) * sigma
perturbation = A @ B.T  # Shape: (m, n)
```

### Update Computation

```python
# Fitness-weighted average of perturbations
weights = normalize_fitnesses(fitnesses)
update = sum(weights[i] * perturbations[i] for i in range(population_size))
```

### Fitness Normalization

EGGROLL supports:
- **Global normalization**: Normalize across all population members
- **Group normalization**: Normalize within groups (can improve stability)

```python
# Global normalization
fitnesses_normalized = (fitnesses - fitnesses.mean()) / (fitnesses.std() + eps)

# Group normalization
for group in groups:
    group_fitnesses = fitnesses[group]
    fitnesses_normalized[group] = (
        (group_fitnesses - group_fitnesses.mean()) / 
        (group_fitnesses.std() + eps)
    )
```

## Implementation Details

### Parameter Classification

EGGROLL classifies parameters:

```python
def _get_lora_params(self):
    """Get 2D parameters (matrices) for LoRA updates."""
    lora_params = {}
    for name, param in self.model.named_parameters():
        if param.dim() == 2:  # Matrix
            lora_params[name] = param
    return lora_params

def _get_full_rank_params(self):
    """Get non-2D parameters for full-rank updates."""
    full_params = {}
    for name, param in self.model.named_parameters():
        if param.dim() != 2:  # Not a matrix
            full_params[name] = param
    return full_params
```

### Low-Rank Update

```python
def _compute_lora_update(self, A, B, fitnesses):
    """Compute low-rank update for matrix parameter."""
    # Fitness-weighted average
    weights = normalize_fitnesses(fitnesses)
    
    # Weighted sum of A @ B.T perturbations
    A_weighted = sum(weights[i] * A[i] for i in range(population_size))
    B_weighted = sum(weights[i] * B[i] for i in range(population_size))
    
    # Update is A_weighted @ B_weighted.T
    update = A_weighted @ B_weighted.T
    return update
```

### Full-Rank Update

```python
def _compute_full_update(self, perturbations, fitnesses):
    """Compute full-rank update for non-matrix parameters."""
    weights = normalize_fitnesses(fitnesses)
    update = (weights[:, None] * perturbations).mean(dim=0)
    return update
```

## Why It Works

### Rank Analysis

Even with rank-1 perturbations, EGGROLL achieves high-rank updates:

1. **Population averaging**: Multiple rank-1 perturbations combine
2. **Fitness weighting**: Better perturbations contribute more
3. **Per-layer independence**: Each layer updated separately

### Theoretical Justification

The update can be written as:

```
ΔW = Σᵢ wᵢ (Aᵢ @ Bᵢᵀ)
```

This is equivalent to:

```
ΔW = (Σᵢ wᵢ Aᵢ) @ (Σᵢ wᵢ Bᵢ)ᵀ
```

While each term is rank-1, the combination can have higher effective rank.

## Hyperparameters

### Rank (r)

Controls memory/computation tradeoff:

- **r = 1**: Minimum memory, fastest (recommended)
- **r = 2-4**: Better expressivity, still efficient
- **r >> 1**: Approaches full-rank (not recommended)

### Noise Reuse

Number of evaluations to reuse noise:

- **0**: No reuse (standard)
- **2**: Antithetic sampling (use +ε and -ε)
- **>2**: Multiple reuses (rarely needed)

### Group Size

Size of groups for fitness normalization:

- **0**: Global normalization
- **>0**: Group-based normalization

## Performance Characteristics

### Memory Complexity

For a model with:
- M matrix parameters of size (mᵢ, nᵢ)
- Total matrix parameters: P = Σᵢ mᵢnᵢ

**Standard ES**: O(P · population_size)  
**EGGROLL**: O(r · Σᵢ(mᵢ + nᵢ) · population_size)

For typical models, EGGROLL uses **~100x less memory**.

### Computation Complexity

**Standard ES**: O(P · population_size)  
**EGGROLL**: O(r · Σᵢ(mᵢ + nᵢ) · population_size)

For typical models, EGGROLL is **~100x faster**.

## References

- [EGGROLL Blog](https://eshyperscale.github.io/)
- [GitHub Implementation](https://github.com/ESHyperscale/HyperscaleES)

## Next Steps

- See [Benchmarks](benchmarks.md) for performance data
- Check [User Guide](../user-guide/core-concepts.md) for usage
- Read [API Reference](../api-reference/eggroll.md) for implementation

