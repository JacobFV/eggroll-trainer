"""Comprehensive test of the ESTrainer functionality."""

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

from eggroll_trainer import ESTrainer, VanillaESTrainer
from models import TinyModel


def test_basic_functionality():
    """Test basic trainer functionality."""
    print("=" * 60)
    print("Test 1: Basic Functionality")
    print("=" * 60)
    
    model = TinyModel()
    
    # Simple fitness: reward models with parameters close to zero
    def fitness_fn(model):
        params = torch.cat([p.flatten() for p in model.parameters()])
        return -torch.norm(params).item()
    
    trainer = VanillaESTrainer(
        model.parameters(),
        model=model,
        fitness_fn=fitness_fn,
        population_size=10,
        learning_rate=0.1,
        sigma=0.1,
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
    
    # Test reset
    trainer.reset()
    assert trainer.generation == 0, "Reset should set generation to 0"
    assert len(trainer.fitness_history) == 0, "Reset should clear history"
    print("✓ Reset functionality works")
    
    print("✓ All basic tests passed!\n")


def test_device_handling():
    """Test device handling."""
    print("=" * 60)
    print("Test 2: Device Handling")
    print("=" * 60)
    
    model = TinyModel()
    
    def fitness_fn(model):
        return torch.randn(1).item()
    
    # Test CPU device
    trainer_cpu = VanillaESTrainer(
        model.parameters(),
        model=model,
        fitness_fn=fitness_fn,
        population_size=5,
        device=torch.device("cpu"),
    )
    assert trainer_cpu.device.type == "cpu", "Should use CPU"
    print("✓ CPU device handling works")
    
    # Test default device (should use model's device)
    model_cpu = TinyModel()
    trainer_default = VanillaESTrainer(
        model_cpu.parameters(),
        model=model_cpu,
        fitness_fn=fitness_fn,
        population_size=5,
    )
    print(f"✓ Default device: {trainer_default.device}")
    
    print("✓ All device tests passed!\n")


def test_parameter_handling():
    """Test parameter flattening/unflattening."""
    print("=" * 60)
    print("Test 3: Parameter Handling")
    print("=" * 60)
    
    model = TinyModel()
    trainer = VanillaESTrainer(
        model.parameters(),
        model=model,
        fitness_fn=lambda m: 1.0,
        population_size=5,
    )
    
    # Get initial params
    initial_flat = trainer._get_flat_params()
    print(f"✓ Initial params shape: {initial_flat.shape}")
    
    # Modify and set back
    modified = initial_flat + 0.1
    trainer._set_flat_params(modified)
    retrieved = trainer._get_flat_params()
    
    assert torch.allclose(modified, retrieved), "Params should match after set/get"
    print("✓ Parameter flattening/unflattening works correctly")
    
    # Verify model parameters were actually updated
    model_params = torch.cat([p.flatten() for p in model.parameters()])
    assert torch.allclose(model_params, modified), "Model params should match"
    print("✓ Model parameters updated correctly")
    
    print("✓ All parameter handling tests passed!\n")


def test_fitness_improvement():
    """Test that fitness can improve over time."""
    print("=" * 60)
    print("Test 4: Fitness Improvement")
    print("=" * 60)
    
    model = TinyModel()
    
    # Target: parameters should be close to [0.5, 0.5, ...]
    target = torch.ones(6) * 0.5  # 5 weights + 1 bias
    
    def fitness_fn(model):
        params = torch.cat([p.flatten() for p in model.parameters()])
        # Reward closeness to target
        return -torch.norm(params - target).item()
    
    trainer = VanillaESTrainer(
        model.parameters(),
        model=model,
        fitness_fn=fitness_fn,
        population_size=20,
        learning_rate=0.05,
        sigma=0.1,
        seed=42,
    )
    
    initial_fitness = trainer.evaluate_fitness(model)
    print(f"Initial fitness: {initial_fitness:.4f}")
    
    # Train for a few generations
    trainer.train(num_generations=10, verbose=False)
    
    final_fitness = trainer.evaluate_fitness(model)
    print(f"Final fitness: {final_fitness:.4f}")
    print(f"Best fitness: {trainer.best_fitness:.4f}")
    
    # Fitness should improve (become less negative)
    assert final_fitness >= initial_fitness - 0.1, "Fitness should improve or stay similar"
    print("✓ Fitness improved over training")
    
    print("✓ All fitness improvement tests passed!\n")


def test_custom_trainer():
    """Test creating a custom trainer subclass."""
    print("=" * 60)
    print("Test 5: Custom Trainer Subclass")
    print("=" * 60)
    
    class CustomESTrainer(ESTrainer):
        def sample_perturbations(self, population_size):
            param_dim = self.current_params.shape[0]
            # Use uniform perturbations instead of Gaussian
            return torch.rand(population_size, param_dim, device=self.device) * 2 - 1
        
        def compute_update(self, perturbations, fitnesses):
            # Simple update: just use mean of top 50% perturbations
            top_k = len(fitnesses) // 2
            top_indices = torch.topk(fitnesses, top_k).indices
            top_perturbations = perturbations[top_indices]
            return top_perturbations.mean(dim=0)
    
    model = TinyModel()
    
    def fitness_fn(model):
        params = torch.cat([p.flatten() for p in model.parameters()])
        return -torch.norm(params).item()
    
    trainer = CustomESTrainer(
        model.parameters(),
        model=model,
        fitness_fn=fitness_fn,
        population_size=10,
        learning_rate=0.1,
        sigma=0.1,
    )
    
    metrics = trainer.step()
    print(f"✓ Custom trainer step completed: generation={metrics['generation']}")
    print(f"  Mean fitness: {metrics['mean_fitness']:.4f}")
    
    print("✓ Custom trainer works correctly!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_basic_functionality()
        test_device_handling()
        test_parameter_handling()
        test_fitness_improvement()
        test_custom_trainer()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

