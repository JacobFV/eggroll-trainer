# Background

Evolution Strategies history, theory, and motivation.

## History

Evolution Strategies (ES) were developed in the 1960s-1970s as a class of optimization algorithms inspired by biological evolution. Key milestones:

- **1960s**: Ingo Rechenberg and Hans-Paul Schwefel develop ES
- **1990s**: CMA-ES (Covariance Matrix Adaptation) improves performance
- **2010s**: ES applied to deep learning (OpenAI ES, Natural ES)
- **2020s**: EGGROLL introduces low-rank perturbations for efficiency

## Theory

### Basic ES Algorithm

1. **Initialize** parameters θ₀
2. **Sample perturbations** ε₁, ..., εₙ ~ N(0, σ²I)
3. **Evaluate fitness** f(θ + εᵢ) for each perturbation
4. **Compute update**:
   ```
   Δθ = α · Σᵢ wᵢ εᵢ
   ```
   where wᵢ are fitness weights
5. **Update** θ ← θ + Δθ
6. **Repeat** until convergence

### Why ES Works

ES approximates the gradient of the expected fitness:

```
∇_θ E[f(θ + ε)] ≈ E[ε · f(θ + ε)] / σ²
```

This is estimated via Monte Carlo sampling:

```
∇_θ E[f(θ + ε)] ≈ (1/n) Σᵢ εᵢ f(θ + εᵢ) / σ²
```

### Advantages

- ✅ **No gradients needed** - Works with non-differentiable objectives
- ✅ **Robust to noise** - Naturally handles stochastic objectives
- ✅ **Parallelizable** - Population evaluations are independent
- ✅ **Global search** - Can escape local minima

### Disadvantages

- ❌ **Sample inefficient** - Requires many evaluations
- ❌ **Memory intensive** - Stores perturbations for all parameters
- ❌ **Slow convergence** - Compared to gradient-based methods

## EGGROLL Innovation

EGGROLL addresses the main limitations:

### Problem: Memory and Computation

For a matrix parameter W ∈ R^(m×n):
- **Memory**: O(mn) per population member
- **Computation**: O(mn) to apply perturbations

For large models, this becomes prohibitive.

### Solution: Low-Rank Perturbations

Instead of full-rank noise N ∈ R^(m×n), use:
- A ∈ R^(m×r), B ∈ R^(n×r) where r << min(m,n)
- Form perturbation as A @ B.T

This reduces:
- **Memory**: O(mn) → O(r(m+n))
- **Computation**: O(mn) → O(r(m+n))

### How It Still Works

Even with low-rank perturbations, EGGROLL achieves high-rank updates through:

1. **Population averaging**: Multiple low-rank perturbations combine
2. **Per-layer updates**: Each parameter tensor handled independently
3. **Fitness weighting**: Better perturbations contribute more

## Comparison with Other Methods

### ES vs Gradient Descent

| Aspect | ES | Gradient Descent |
|--------|----|------------------|
| **Gradients** | Not needed | Required |
| **Sample efficiency** | Lower | Higher |
| **Non-differentiable** | Works | Fails |
| **Parallelization** | Easy | Harder |
| **Global search** | Better | Worse |

### EGGROLL vs Standard ES

| Aspect | Standard ES | EGGROLL |
|--------|-------------|---------|
| **Memory** | O(mn) | O(r(m+n)) |
| **Computation** | O(mn) | O(r(m+n)) |
| **Speedup** | 1x | ~100x |
| **Population size** | Limited | Large (256+) |

## Applications

ES and EGGROLL are useful for:

- **Reinforcement Learning** - Policy optimization
- **Non-differentiable objectives** - Discrete/combinatorial problems
- **Robust optimization** - Noisy or adversarial settings
- **Hyperparameter optimization** - Black-box tuning
- **Neural architecture search** - Discrete architectures

## References

- Rechenberg, I. (1973). *Evolutionsstrategie: Optimierung technischer Systeme nach Prinzipien der biologischen Evolution*
- Hansen, N. (2016). *The CMA Evolution Strategy: A Tutorial*
- Salimans, T., et al. (2017). *Evolution Strategies as a Scalable Alternative to Reinforcement Learning*
- [EGGROLL Blog](https://eshyperscale.github.io/)

## Next Steps

- See [EGGROLL Algorithm](eggroll-algorithm.md) for implementation details
- Check [Benchmarks](benchmarks.md) for performance analysis

