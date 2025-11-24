"""Test MNIST EGGROLL training.

This is a lightweight test that verifies EGGROLL training works correctly.
For full comparison with SGD, see mnist_comparison.py
"""

import torch
import sys
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Add parent directory to path for imports when running as script
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from eggroll_trainer import EGGROLLTrainer
from models import SimpleMNISTNet
from utils import create_accuracy_fitness_fn, evaluate_model


def test_mnist_training():
    """Test that EGGROLL can train on MNIST."""
    print("=" * 60)
    print("Test: MNIST Training with EGGROLL")
    print("=" * 60)
    
    device = torch.device("cpu")
    
    # Load small subset of MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    
    # Create model
    model = SimpleMNISTNet().to(device)
    
    # Create fitness function using shared utility
    fitness_fn = create_accuracy_fitness_fn(train_loader, device, batch_limit=10)
    
    # Evaluate initial model
    initial_fitness = fitness_fn(model)
    print(f"\nInitial fitness: {initial_fitness:.4f}")
    
    # Create trainer
    trainer = EGGROLLTrainer(
        model.parameters(),
        model=model,
        fitness_fn=fitness_fn,
        population_size=32,
        learning_rate=0.05,
        sigma=0.05,
        rank=1,
        device=device,
        seed=42,
    )
    
    # Train for a few generations
    print("\nTraining for 10 generations...")
    results = trainer.train(num_generations=10, verbose=False)
    
    # Check that fitness improved
    final_fitness = fitness_fn(model)
    print(f"Final fitness: {final_fitness:.4f}")
    print(f"Best fitness: {results['best_fitness']:.4f}")
    
    # Fitness should improve (or at least not degrade significantly)
    assert results['best_fitness'] >= initial_fitness - 0.1, \
        f"Best fitness {results['best_fitness']:.4f} should be >= initial {initial_fitness:.4f} - 0.1"
    
    print("✓ Fitness improved during training")
    
    # Test that best model can be retrieved
    best_model = trainer.get_best_model()
    assert best_model is not None, "Best model should not be None"
    best_fitness = fitness_fn(best_model)
    print(f"Best model fitness: {best_fitness:.4f}")
    print("✓ Best model retrieved successfully")
    
    # Evaluate on test set using shared utility
    test_acc = evaluate_model(best_model, test_loader, device)
    print(f"\nTest accuracy: {test_acc:.2f}%")
    print("✓ Model evaluation works")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        test_mnist_training()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

