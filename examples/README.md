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

- **`simple_example.py`** - Basic ES trainer demo with SimpleESTrainer
- **`eggroll_example.py`** - EGGROLL trainer demo on a simple model

### MNIST Examples

- **`mnist_comparison.py`** ⭐ **RECOMMENDED** - Full comparison between EGGROLL and SGD
  - Trains both methods on MNIST
  - Generates comparison plots
  - Shows accuracy over time
  - Includes timing comparisons
  
- **`mnist_eggroll.py`** - EGGROLL-only MNIST training (legacy, see comparison above)
- **`test_mnist_eggroll.py`** - Lightweight test for EGGROLL on MNIST

### Test Suites

- **`test_comprehensive.py`** - Comprehensive test suite for base ES trainers
- **`test_eggroll.py`** - Test suite for EGGROLL trainer

## Quick Start

```bash
# Run EGGROLL vs SGD comparison (recommended)
uv run python examples/mnist_comparison.py

# Run simple EGGROLL example
uv run python examples/eggroll_example.py

# Run tests
uv run python examples/test_eggroll.py
```

## Output Files

- `mnist_comparison.png` - Generated comparison plots (created by mnist_comparison.py)
