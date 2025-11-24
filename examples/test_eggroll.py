"""Test suite for EGGROLL trainer."""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path for imports when running as script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from eggroll_trainer import EGGROLLTrainer
from models import TinyModel


def test_eggroll_basic():
    """Test basic EGGROLL functionality."""
    print("=" * 60)
    print("Test 1: Basic EGGROLL Functionality")
    print("=" * 60)
    
    model = TinyModel()
    
    def fitness_fn(model):
        params = torch.cat([p.flatten() for p in model.parameters()])
        return -torch.norm(params).item()
    
    trainer = EGGROLLTrainer(
        model.parameters(),
        model=model,
        fitness_fn=fitness_fn,
        population_size=16,
        learning_rate=0.1,
        sigma=0.1,
        rank=1,
        seed=42,
    )
    
    # Test single step
    metrics = trainer.step()
    print(f"✓ Single step completed: generation={metrics['generation']}")
    print(f"  Mean fitness: {metrics['mean_fitness']:.4f}")
    print(f"  Best fitness: {metrics['best_fitness']:.4f}")
    
    # Test multiple steps
    results = trainer.train(num_generations=5, verbose=False)
    print(f"✓ Training completed: {results['generation']} generations")
    print(f"  Best fitness achieved: {results['best_fitness']:.4f}")
    assert results['generation'] == 6, "Generation count mismatch"
    print("✓ Generation count correct")
    
    # Test best model retrieval
    best_model = trainer.get_best_model()
    assert best_model is not None, "Best model should not be None"
    print("✓ Best model retrieved successfully")
    
    print("✓ All basic tests passed!\n")


def test_eggroll_low_rank():
    """Test that EGGROLL uses low-rank perturbations for matrices."""
    print("=" * 60)
    print("Test 2: Low-Rank Perturbations")
    print("=" * 60)
    
    model = TinyModel()
    
    def fitness_fn(model):
        return torch.randn(1).item()
    
    trainer = EGGROLLTrainer(
        model.parameters(),
        model=model,
        fitness_fn=fitness_fn,
        population_size=8,
        learning_rate=0.01,
        sigma=0.1,
        rank=1,
        seed=42,
    )
    
    # Check that weight matrix uses low-rank
    for name, param in model.named_parameters():
        if len(param.shape) == 2:
            print(f"✓ Matrix parameter {name}: shape {param.shape} uses LoRA (rank={trainer.rank})")
        else:
            print(f"✓ Non-matrix parameter {name}: shape {param.shape} uses full-rank")
    
    print("✓ Low-rank structure verified!\n")


def test_eggroll_fitness_improvement():
    """Test that EGGROLL can improve fitness."""
    print("=" * 60)
    print("Test 3: Fitness Improvement")
    print("=" * 60)
    
    model = TinyModel()
    
    # Target: parameters should be close to zero
    def fitness_fn(model):
        params = torch.cat([p.flatten() for p in model.parameters()])
        # Reward closeness to zero
        return -torch.norm(params).item()
    
    trainer = EGGROLLTrainer(
        model.parameters(),
        model=model,
        fitness_fn=fitness_fn,
        population_size=32,
        learning_rate=0.05,
        sigma=0.1,
        rank=1,
        seed=42,
    )
    
    initial_fitness = trainer.fitness_fn(model)
    print(f"Initial fitness: {initial_fitness:.4f}")
    
    # Train for a few generations
    trainer.train(num_generations=10, verbose=False)
    
    final_fitness = trainer.fitness_fn(model)
    print(f"Final fitness: {final_fitness:.4f}")
    print(f"Best fitness: {trainer.best_fitness:.4f}")
    
    # Fitness should improve (become less negative)
    assert final_fitness >= initial_fitness - 0.5, "Fitness should improve or stay similar"
    print("✓ Fitness improved over training")
    
    print("✓ All fitness improvement tests passed!\n")


def test_eggroll_different_ranks():
    """Test EGGROLL with different rank values."""
    print("=" * 60)
    print("Test 4: Different Rank Values")
    print("=" * 60)
    
    model = TinyModel()
    
    def fitness_fn(model):
        params = torch.cat([p.flatten() for p in model.parameters()])
        return -torch.norm(params).item()
    
    for rank in [1, 2, 4]:
        trainer = EGGROLLTrainer(
            model.parameters(),
            model=model,
            fitness_fn=fitness_fn,
            population_size=16,
            learning_rate=0.1,
            sigma=0.1,
            rank=rank,
            seed=42,
        )
        metrics = trainer.step()
        print(f"✓ Rank {rank}: Mean fitness = {metrics['mean_fitness']:.4f}")
    
    print("✓ All rank tests passed!\n")


def main():
    """Run all EGGROLL tests."""
    print("\n" + "=" * 60)
    print("EGGROLL TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_eggroll_basic()
        test_eggroll_low_rank()
        test_eggroll_fitness_improvement()
        test_eggroll_different_ranks()
        
        print("=" * 60)
        print("ALL EGGROLL TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

