# Installation

Install Eggroll Trainer using your preferred package manager.

## Using pip

```bash
pip install eggroll-trainer
```

## Using uv

```bash
uv add eggroll-trainer
```

## From Source (Development)

For development and contributions, clone the repository:

```bash
git clone https://github.com/JacobFV/eggroll-trainer.git
cd eggroll-trainer
uv sync
# or
pip install -e .
```

See the [Contributing Guide](../../CONTRIBUTING.md) for more details.

## Optional Dependencies

### Examples with Plotting

For examples that generate plots (like `mnist_comparison.py`):

```bash
pip install "eggroll-trainer[examples]"
# or
uv sync --extra examples
```

### Development Dependencies

For development and testing:

```bash
pip install "eggroll-trainer[dev]"
# or
uv sync --extra dev
```

## Verify Installation

Test that everything works:

```python
import torch
from eggroll_trainer import EGGROLLTrainer, ESTrainer, SimpleESTrainer

print("Eggroll Trainer installed successfully!")
```

## Platform-Specific Notes

### macOS (M3/Apple Silicon)

PyTorch should automatically detect and use MPS (Metal Performance Shaders) for GPU acceleration. If you encounter issues, ensure you have the latest PyTorch version:

```bash
pip install --upgrade torch torchvision
```

### CUDA Support

For CUDA support, install PyTorch with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Troubleshooting

### Import Errors

If you see import errors, ensure you're using Python 3.12+:

```bash
python --version  # Should be 3.12 or higher
```

### PyTorch Not Found

Install PyTorch separately if needed:

```bash
pip install torch>=2.0.0 torchvision>=0.15.0
```

## Next Steps

Once installed, check out the [Quick Start Guide](quick-start.md) to run your first example!

