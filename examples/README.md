# Examples

## Running Examples

```bash
# From the project root
uv run python examples/<example_name>.py

# For examples that need matplotlib (like mnist_comparison.py):
uv sync --extra dev
uv run python examples/mnist_comparison.py
```

## Available Examples

### Basic Examples

- **`basic_example.py`** ⭐ **START HERE** - Side-by-side comparison of SimpleESTrainer and EGGROLLTrainer
  - Shows both trainers on the same simple task
  - Demonstrates the difference between full-rank and low-rank perturbations
  - Quick to run (~30 seconds)

### MNIST Examples

- **`mnist_comparison.py`** ⭐ **RECOMMENDED** - Full comparison between EGGROLL and SGD on MNIST
  - Trains both methods on MNIST
  - Generates comparison plots (`mnist_comparison.png`)
  - Shows accuracy over time
  - Includes timing comparisons
  - Use `--quick` flag for faster testing
  
- **`run_all_comparisons.py`** - Comprehensive multi-architecture comparison
  - Compares EGGROLL vs SGD across multiple architectures:
    - Simple CNN
    - Baby Transformer
    - MLP Classifier
  - Generates plots for each architecture
  - Use `--quick` flag for faster testing

### Framework & Utilities

- **`comparison_framework.py`** - Generalized comparison framework
  - Reusable framework for comparing EGGROLL with any optimizer
  - Handles training, evaluation, and plotting
  - Used by `mnist_comparison.py` and `run_all_comparisons.py`

- **`models.py`** - Shared model architectures
  - `SimpleModel`, `TinyModel`, `MNISTNet`, `SimpleMNISTNet`
  - `SimpleCNN`, `BabyTransformer`, `MLPClassifier`
  - Used across all examples

- **`utils.py`** - Shared utility functions
  - `create_parameter_distance_fitness_fn()` - For simple parameter-based fitness
  - `create_accuracy_fitness_fn()` - For classification tasks
  - `evaluate_model()` - Model evaluation helper

### Test Suites

- **`test_comprehensive.py`** - Comprehensive test suite for base ES trainers
- **`test_eggroll.py`** - Test suite for EGGROLL trainer
- **`test_mnist_eggroll.py`** - Lightweight test for EGGROLL on MNIST

## Quick Start

```bash
# Start with the basic example (shows both trainers)
uv run python examples/basic_example.py

# Run MNIST comparison (requires matplotlib)
uv sync --extra dev
uv run python examples/mnist_comparison.py

# Run comprehensive multi-architecture comparison
uv run python examples/run_all_comparisons.py --quick

# Run tests
uv run python examples/test_eggroll.py
```

## Output Files

- `mnist_comparison.png` - Generated comparison plots (created by mnist_comparison.py)
- `*_comparison.png` - Architecture-specific comparison plots (created by run_all_comparisons.py)

## Architecture

The examples are organized with:
- **Shared code** in `models.py` and `utils.py` to avoid duplication
- **Comparison framework** for reusable comparison logic
- **Focused examples** that demonstrate specific use cases
- **Test suites** that verify functionality
