# Research

Background, algorithm details, and benchmarks.

## Sections

- **[Background](background.md)** - Evolution Strategies history and theory
- **[EGGROLL Algorithm](eggroll-algorithm.md)** - Deep dive into the algorithm
- **[Benchmarks](benchmarks.md)** - Performance comparisons and speedups

## Overview

Eggroll Trainer implements the **EGGROLL** algorithm, a novel Evolution Strategy that achieves **100x speedup** over naïve ES methods through low-rank perturbations.

## Key References

- [EGGROLL Paper/Blog](https://eshyperscale.github.io/)
- [GitHub Implementation](https://github.com/ESHyperscale/HyperscaleES)

## Key Innovation

**Low-rank perturbations** reduce memory and computation:

- **Memory**: O(mn) → O(r(m+n)) for matrices W ∈ R^(m×n)
- **Computation**: O(mn) → O(r(m+n))
- **Speedup**: ~100x for typical models

Yet still achieves high-rank updates through population averaging!

## Next Steps

- Read [Background](background.md) for ES theory
- See [EGGROLL Algorithm](eggroll-algorithm.md) for details
- Check [Benchmarks](benchmarks.md) for performance data

