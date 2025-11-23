# Core Concepts

Understanding Evolution Strategies and the EGGROLL algorithm.

## Evolution Strategies Overview

Evolution Strategies (ES) are a class of **black-box optimization algorithms** that:

- Don't require gradients
- Work with non-differentiable objectives
- Evaluate multiple perturbed versions of a model (population)
- Use fitness scores to guide parameter updates

### How ES Works

1. **Initialize** a model with parameters θ
2. **Sample perturbations** ε₁, ε₂, ..., εₙ for n population members
3. **Evaluate fitness** f(θ + εᵢ) for each perturbed model
4. **Compute update** Δθ based on fitness-weighted perturbations
5. **Update parameters** θ ← θ + α·Δθ (where α is learning rate)
6. **Repeat** until convergence

### Advantages

- ✅ No gradients needed
- ✅ Works with discrete/discontinuous objectives
- ✅ Naturally parallelizable
- ✅ Robust to noise

### Disadvantages

- ❌ Requires many evaluations (population size)
- ❌ Can be slower than gradient-based methods
- ❌ Less sample-efficient for smooth objectives

## EGGROLL Algorithm

**EGGROLL** (Evolution Guided General Optimization via Low-rank Learning) addresses the main limitation of ES: **computational cost**.

### The Problem

Standard ES requires:
- **Memory**: O(mn) for each matrix parameter W ∈ R^(m×n)
- **Computation**: O(mn) to apply perturbations

For large models, this becomes prohibitive.

### The Solution: Low-Rank Perturbations

Instead of sampling full-rank noise N ∈ R^(m×n), EGGROLL samples:
- A ∈ R^(m×r), B ∈ R^(n×r) where r << min(m,n)
- Forms perturbation as A @ B.T

This reduces:
- **Memory**: O(mn) → O(r(m+n))
- **Computation**: O(mn) → O(r(m+n))

### How It Still Works

Even with low-rank perturbations, EGGROLL achieves high-rank updates through:
1. **Population averaging**: Multiple low-rank perturbations combine
2. **Per-layer updates**: Each parameter tensor handled independently
3. **Fitness weighting**: Better perturbations contribute more

### When to Use EGGROLL

✅ **Use EGGROLL when:**
- You have large models with many matrix parameters
- You need to train with large population sizes
- Memory/computation is a bottleneck
- You want gradient-free optimization

❌ **Consider alternatives when:**
- Your model is very small (< 1K parameters)
- You can compute gradients efficiently
- You need maximum sample efficiency

## Key Terminology

### Population Size

Number of perturbed models evaluated per generation. Larger populations:
- Provide better gradient estimates
- Are more robust to noise
- Require more computation

EGGROLL makes large populations feasible (256-1024+).

### Sigma (σ)

Standard deviation of perturbations. Controls exploration vs exploitation:
- **Large σ**: More exploration, slower convergence
- **Small σ**: More exploitation, may get stuck

Typical range: 0.01 - 0.1

### Learning Rate (α)

Step size for parameter updates. Similar to SGD:
- **Large α**: Faster but may overshoot
- **Small α**: Slower but more stable

Typical range: 0.001 - 0.1

### Rank (r)

Rank of low-rank perturbations. Controls memory/computation tradeoff:
- **r = 1**: Minimum memory, fastest
- **r = 2-4**: Better expressivity, still efficient
- **r >> 1**: Approaches full-rank (not recommended)

Default: r = 1 (often sufficient)

### Fitness Function

Function that evaluates model performance. **Higher is better**.

For loss minimization:
```python
def fitness_fn(model):
    loss = compute_loss(model)
    return -loss  # Convert to maximization
```

## Comparison: Full-Rank vs Low-Rank

### Full-Rank ES (SimpleESTrainer)

```python
# For W ∈ R^(m×n), sample N ∈ R^(m×n)
N = torch.randn(m, n) * sigma
W_perturbed = W + N
```

**Memory**: O(mn) per population member  
**Computation**: O(mn) per evaluation

### Low-Rank ES (EGGROLLTrainer)

```python
# For W ∈ R^(m×n), sample A ∈ R^(m×r), B ∈ R^(n×r)
A = torch.randn(m, r) * sigma
B = torch.randn(n, r) * sigma
W_perturbed = W + A @ B.T
```

**Memory**: O(r(m+n)) per population member  
**Computation**: O(r(m+n)) per evaluation

**Speedup**: ~100x for typical models!

## Next Steps

- Learn about [Trainers](trainers.md) - How to use the classes
- See [Fitness Functions](fitness-functions.md) - Writing evaluation functions
- Check [Advanced Usage](advanced-usage.md) - Customization and tuning

