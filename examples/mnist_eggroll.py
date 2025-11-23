"""MNIST classifier trained with EGGROLL.

NOTE: For a comprehensive comparison with SGD including plots,
see mnist_comparison.py instead.

This file is kept for backward compatibility but mnist_comparison.py
is recommended for new usage.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from eggroll_trainer import EGGROLLTrainer
import numpy as np


class MNISTNet(nn.Module):
    """Simple CNN for MNIST classification."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def create_fitness_fn(train_loader, device, batch_limit=20):
    """
    Create a fitness function that evaluates model accuracy on training data.
    
    Args:
        train_loader: DataLoader for training data
        device: Device to run on
        batch_limit: Maximum number of batches to evaluate (for speed)
    
    Returns:
        Fitness function that returns accuracy (higher is better)
    """
    # Cache a subset of data for faster evaluation
    cached_data = []
    cached_targets = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx >= batch_limit:
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
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    return fitness_fn


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
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
    
    accuracy = 100.0 * correct / total
    return accuracy


def main():
    """Train MNIST classifier with EGGROLL."""
    print("=" * 60)
    print("MNIST Classification with EGGROLL")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
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
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0  # Set to 0 for compatibility
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = MNISTNet().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create fitness function
    print("\nCreating fitness function...")
    fitness_fn = create_fitness_fn(train_loader, device, batch_limit=5)
    
    # Evaluate initial model
    initial_acc = evaluate_model(model, test_loader, device)
    print(f"\nInitial test accuracy: {initial_acc:.2f}%")
    
    # Create EGGROLL trainer
    print("\nCreating EGGROLL trainer...")
    print("  - Population size: 64")
    print("  - Rank: 1 (low-rank perturbations)")
    print("  - Sigma: 0.05")
    print("  - Learning rate: 0.05")
    
    trainer = EGGROLLTrainer(
        model=model,
        fitness_fn=fitness_fn,
        population_size=64,
        learning_rate=0.05,
        sigma=0.05,
        rank=1,
        noise_reuse=0,
        group_size=0,
        freeze_nonlora=False,
        device=device,
        seed=42,
    )
    
    # Train with periodic test evaluation
    print("\n" + "=" * 60)
    print("Training with EGGROLL...")
    print("=" * 60)
    
    best_test_acc = initial_acc
    
    for gen in range(30):
        metrics = trainer.train_step()
        
        # Evaluate on test set every 5 generations
        if (gen + 1) % 5 == 0:
            test_acc = evaluate_model(trainer.model, test_loader, device)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            print(
                f"Generation {metrics['generation']}: "
                f"Mean Fitness = {metrics['mean_fitness']:.4f}, "
                f"Best Fitness = {metrics['best_fitness']:.4f}, "
                f"Test Acc = {test_acc:.2f}%"
            )
        else:
            print(
                f"Generation {metrics['generation']}: "
                f"Mean Fitness = {metrics['mean_fitness']:.4f}, "
                f"Best Fitness = {metrics['best_fitness']:.4f}"
            )
    
    results = {
        "generation": trainer.generation,
        "best_fitness": trainer.best_fitness,
        "fitness_history": trainer.fitness_history,
    }
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Best fitness achieved: {results['best_fitness']:.4f}")
    print(f"Final generation: {results['generation']}")
    
    # Evaluate best model
    print("\nEvaluating best model on test set...")
    best_model = trainer.get_best_model()
    best_model.eval()
    
    test_accuracy = evaluate_model(best_model, test_loader, device)
    print(f"\n{'='*60}")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    print(f"{'='*60}")
    
    # Compare with initial accuracy
    improvement = test_accuracy - initial_acc
    print(f"\nImprovement: {improvement:+.2f}%")
    
    # Show some predictions
    print("\nSample predictions:")
    best_model.eval()
    with torch.no_grad():
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        images, labels = images.to(device)[:10], labels[:10]
        outputs = best_model(images)
        preds = outputs.argmax(dim=1)
        
        print("True labels:", labels.tolist())
        print("Predictions:", preds.cpu().tolist())
        print("Correct:", (preds.cpu() == labels).sum().item(), "/", len(labels))


if __name__ == "__main__":
    main()

