# Contributing to Eggroll Trainer

Thank you for your interest in contributing to Eggroll Trainer!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/JacobFV/eggroll-trainer.git
cd eggroll-trainer
```

2. Install dependencies:
```bash
uv sync --extra dev
```

3. Run tests:
```bash
uv run python examples/test_comprehensive.py
uv run python examples/test_eggroll.py
```

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to public functions and classes
- Keep functions focused and modular

## Submitting Changes

1. Create a branch for your changes
2. Make your changes with tests
3. Ensure all tests pass
4. Update documentation if needed
5. Submit a pull request with a clear description

## Adding New ES Algorithms

To add a new ES algorithm:

1. Create a new file in `eggroll_trainer/` or subclass `ESTrainer`
2. Implement the required abstract methods
3. Add tests in `examples/test_*.py`
4. Add an example in `examples/`
5. Update `eggroll_trainer/__init__.py` to export your class

## Questions?

Feel free to open an issue for questions or discussions!

