"""Basic example demonstrating both VanillaESTrainer and EGGROLLTrainer.

This example shows how to use both the vanilla ES trainer and the EGGROLL trainer
on the same simple task, allowing for easy comparison.
"""

import torch
import sys
import os

# Get script directory and parent directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)

# Add parent directory to path for eggroll_trainer imports
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from eggroll_trainer import ESTrainer, VanillaESTrainer, EGGROLLTrainer

# Import from local examples directory
# When running as script: examples/models.py
# When imported: examples.models
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from models import SimpleModel
from utils import create_parameter_distance_fitness_fn


def main():
    """Run basic ES training examples."""
    print("=" * 70)
    print("Basic ES Trainer Examples")
    print("=" * 70)
    
    # Create model
    print("\nCreating model...")
    model = SimpleModel()
    
    # Create fitness function (parameter distance)
    print("Creating fitness function...")
    fitness_fn = create_parameter_distance_fitness_fn(noise_std=0.1, use_dict=False)
    
    # ========== VanillaESTrainer Example ==========
    print("\n" + "=" * 70)
    print("1. VanillaESTrainer (Vanilla ES with full-rank perturbations)")
    print("=" * 70)
    
    simple_model = SimpleModel()
    simple_trainer = VanillaESTrainer(
        simple_model.parameters(),
        model=simple_model,
        fitness_fn=fitness_fn,
        population_size=20,
        learning_rate=0.01,
        sigma=0.1,
        seed=42,
    )
    
    print("\nTraining for 10 generations...")
    simple_results = simple_trainer.train(num_generations=10, verbose=True)
    
    print(f"\n✓ VanillaESTrainer complete!")
    print(f"  Best fitness: {simple_results['best_fitness']:.4f}")
    
    # ========== EGGROLLTrainer Example ==========
    print("\n" + "=" * 70)
    print("2. EGGROLLTrainer (Low-rank perturbations)")
    print("=" * 70)
    
    eggroll_model = SimpleModel()
    eggroll_fitness_fn = create_parameter_distance_fitness_fn(noise_std=0.1, use_dict=True)
    
    print("\nCreating EGGROLL trainer...")
    print("  - Population size: 32")
    print("  - Rank: 1 (low-rank perturbations)")
    print("  - Sigma: 0.1")
    print("  - Learning rate: 0.01")
    
    eggroll_trainer = EGGROLLTrainer(
        eggroll_model.parameters(),
        model=eggroll_model,
        fitness_fn=eggroll_fitness_fn,
        population_size=32,
        learning_rate=0.01,
        sigma=0.1,
        rank=1,
        noise_reuse=0,
        group_size=0,
        freeze_nonlora=False,
        seed=42,
    )
    
    print("\nTraining for 10 generations...")
    print("(EGGROLL uses low-rank perturbations for matrix parameters)")
    print("-" * 70)
    
    eggroll_results = eggroll_trainer.train(num_generations=10, verbose=True)
    
    print(f"\n✓ EGGROLLTrainer complete!")
    print(f"  Best fitness: {eggroll_results['best_fitness']:.4f}")
    
    # ========== Comparison ==========
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"VanillaESTrainer best fitness: {simple_results['best_fitness']:.4f}")
    print(f"EGGROLLTrainer best fitness:  {eggroll_results['best_fitness']:.4f}")
    print(f"\nDifference: {eggroll_results['best_fitness'] - simple_results['best_fitness']:+.4f}")
    
    # Show parameter statistics for EGGROLL
    print("\nEGGROLL parameter statistics:")
    best_model = eggroll_trainer.get_best_model()
    for name, param in best_model.named_parameters():
        if len(param.shape) == 2:
            print(f"  {name}: shape {param.shape} (matrix - used LoRA updates)")
        else:
            print(f"  {name}: shape {param.shape} (non-matrix - used full-rank updates)")


if __name__ == "__main__":
    main()

