"""Eggroll Trainer - A library for Evolution Strategy trainers in PyTorch."""

from eggroll_trainer.base import ESTrainer
from eggroll_trainer.simple import SimpleESTrainer
from eggroll_trainer.eggroll import EGGROLLTrainer

__all__ = ["ESTrainer", "SimpleESTrainer", "EGGROLLTrainer"]
__version__ = "0.1.0"
