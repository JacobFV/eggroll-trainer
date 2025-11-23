"""MNIST classifier comparison: EGGROLL vs SGD.

This example trains the same model using both EGGROLL and SGD,
compares their performance, and generates visualization plots.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from eggroll_trainer import EGGROLLTrainer
import copy
import time

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: uv sync --extra dev")


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


def create_fitness_fn(train_loader, device, batch_limit=50):
    """
    Create a fitness function that evaluates model accuracy on training data.
    
    Uses cached subset for faster evaluation during EGGROLL training.
    """
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
        return correct / total if total > 0 else 0.0
    
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
    return 100.0 * correct / total


def train_sgd(model, train_loader, test_loader, device, num_epochs=20):
    """Train model using SGD optimizer."""
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    train_acc_history = []
    test_acc_history = []
    loss_history = []
    
    print(f"\nTraining SGD for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        train_acc = 100.0 * correct / total
        test_acc = evaluate_model(model, test_loader, device)
        
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        loss_history.append(running_loss / len(train_loader))
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Acc = {train_acc:.2f}%, "
              f"Test Acc = {test_acc:.2f}%, "
              f"Loss = {loss_history[-1]:.4f}")
    
    return {
        'train_acc': train_acc_history,
        'test_acc': test_acc_history,
        'loss': loss_history,
    }


def train_eggroll(model, train_loader, test_loader, device, num_generations=100):
    """Train model using EGGROLL."""
    fitness_fn = create_fitness_fn(train_loader, device, batch_limit=50)
    
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
    
    train_acc_history = []
    test_acc_history = []
    fitness_history = []
    
    print(f"\nTraining EGGROLL for {num_generations} generations...")
    
    for gen in range(num_generations):
        metrics = trainer.train_step()
        
        # Evaluate periodically
        if (gen + 1) % 10 == 0 or gen == 0:
            train_acc = fitness_fn(trainer.model) * 100.0
            test_acc = evaluate_model(trainer.model, test_loader, device)
            
            train_acc_history.append((gen + 1, train_acc))
            test_acc_history.append((gen + 1, test_acc))
            fitness_history.append((gen + 1, metrics['mean_fitness']))
            
            if (gen + 1) % 10 == 0:
                print(f"Generation {gen+1}/{num_generations}: "
                      f"Train Acc = {train_acc:.2f}%, "
                      f"Test Acc = {test_acc:.2f}%, "
                      f"Fitness = {metrics['mean_fitness']:.4f}")
    
    return {
        'train_acc': train_acc_history,
        'test_acc': test_acc_history,
        'fitness': fitness_history,
    }


def plot_comparison(sgd_results, eggroll_results, save_path='mnist_comparison.png'):
    """Generate comparison plots."""
    if not HAS_MATPLOTLIB:
        print("Skipping plot generation (matplotlib not installed)")
        print("Install with: uv sync --extra dev")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Test accuracy comparison
    ax1 = axes[0, 0]
    sgd_epochs = range(1, len(sgd_results['test_acc']) + 1)
    eggroll_gens, eggroll_test_acc = zip(*eggroll_results['test_acc']) if eggroll_results['test_acc'] else ([], [])
    
    ax1.plot(sgd_epochs, sgd_results['test_acc'], 'b-o', label='SGD', linewidth=2, markersize=6)
    ax1.plot(eggroll_gens, eggroll_test_acc, 'r-s', label='EGGROLL', linewidth=2, markersize=6)
    ax1.set_xlabel('Epochs / Generations', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Train accuracy comparison
    ax2 = axes[0, 1]
    sgd_train_acc = sgd_results['train_acc']
    eggroll_gens, eggroll_train_acc = zip(*eggroll_results['train_acc']) if eggroll_results['train_acc'] else ([], [])
    
    ax2.plot(sgd_epochs, sgd_train_acc, 'b-o', label='SGD', linewidth=2, markersize=6)
    ax2.plot(eggroll_gens, eggroll_train_acc, 'r-s', label='EGGROLL', linewidth=2, markersize=6)
    ax2.set_xlabel('Epochs / Generations', fontsize=12)
    ax2.set_ylabel('Train Accuracy (%)', fontsize=12)
    ax2.set_title('Train Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])
    
    # SGD Loss
    ax3 = axes[1, 0]
    ax3.plot(sgd_epochs, sgd_results['loss'], 'b-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Epochs', fontsize=12)
    ax3.set_ylabel('Loss', fontsize=12)
    ax3.set_title('SGD Training Loss', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # EGGROLL Fitness
    ax4 = axes[1, 1]
    eggroll_gens, eggroll_fitness = zip(*eggroll_results['fitness']) if eggroll_results['fitness'] else ([], [])
    ax4.plot(eggroll_gens, eggroll_fitness, 'r-s', linewidth=2, markersize=6)
    ax4.set_xlabel('Generations', fontsize=12)
    ax4.set_ylabel('Mean Fitness', fontsize=12)
    ax4.set_title('EGGROLL Mean Fitness', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to {save_path}")
    plt.close()


def main():
    """Main comparison function."""
    import sys
    
    # Check for quick mode
    quick_mode = '--quick' in sys.argv or '-q' in sys.argv
    
    print("=" * 70)
    print("MNIST Classification: EGGROLL vs SGD Comparison")
    if quick_mode:
        print("(QUICK MODE - Reduced epochs/generations for testing)")
    print("=" * 70)
    
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    
    # Evaluate initial model
    initial_model = MNISTNet().to(device)
    initial_acc = evaluate_model(initial_model, test_loader, device)
    print(f"\nInitial test accuracy: {initial_acc:.2f}%")
    
    # Train SGD
    print("\n" + "=" * 70)
    print("SGD Training")
    print("=" * 70)
    sgd_model = copy.deepcopy(initial_model)
    sgd_start = time.time()
    sgd_epochs = 5 if quick_mode else 20
    sgd_results = train_sgd(sgd_model, train_loader, test_loader, device, num_epochs=sgd_epochs)
    sgd_time = time.time() - sgd_start
    
    # Train EGGROLL
    print("\n" + "=" * 70)
    print("EGGROLL Training")
    print("=" * 70)
    eggroll_model = copy.deepcopy(initial_model)
    eggroll_start = time.time()
    eggroll_generations = 50 if quick_mode else 200
    eggroll_results = train_eggroll(eggroll_model, train_loader, test_loader, device, num_generations=eggroll_generations)
    eggroll_time = time.time() - eggroll_start
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Results")
    print("=" * 70)
    
    sgd_final_acc = evaluate_model(sgd_model, test_loader, device)
    eggroll_final_acc = evaluate_model(eggroll_model, test_loader, device)
    
    print(f"\nSGD:")
    print(f"  Final test accuracy: {sgd_final_acc:.2f}%")
    print(f"  Training time: {sgd_time:.2f} seconds")
    print(f"  Best test accuracy: {max(sgd_results['test_acc']):.2f}%")
    
    print(f"\nEGGROLL:")
    print(f"  Final test accuracy: {eggroll_final_acc:.2f}%")
    print(f"  Training time: {eggroll_time:.2f} seconds")
    if eggroll_results['test_acc']:
        eggroll_best = max(acc for _, acc in eggroll_results['test_acc'])
        print(f"  Best test accuracy: {eggroll_best:.2f}%")
    
    print(f"\nComparison:")
    print(f"  Accuracy difference: {eggroll_final_acc - sgd_final_acc:+.2f}%")
    print(f"  Time ratio (EGGROLL/SGD): {eggroll_time / sgd_time:.2f}x")
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison(sgd_results, eggroll_results, save_path='mnist_comparison.png')
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

