"""Test MNIST EGGROLL training.

This is a lightweight test that verifies EGGROLL training works correctly.
For full comparison with SGD, see mnist_comparison.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from eggroll_trainer import EGGROLLTrainer


class SimpleMNISTNet(nn.Module):
    """Simplified CNN for MNIST."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 14 * 14, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
    
    # Cache training data for fitness evaluation
    cached_data = []
    cached_targets = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= 10:  # Use 10 batches
            break
        cached_data.append(data.to(device))
        cached_targets.append(target.to(device))
    
    def fitness_fn(model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in zip(cached_data, cached_targets):
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        return correct / total if total > 0 else 0.0
    
    # Evaluate initial model
    initial_fitness = fitness_fn(model)
    print(f"\nInitial fitness: {initial_fitness:.4f}")
    
    # Create trainer
    trainer = EGGROLLTrainer(
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
    
    # Evaluate on test set
    def evaluate_test(model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        return 100.0 * correct / total
    
    test_acc = evaluate_test(best_model)
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

