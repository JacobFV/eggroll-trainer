"""MNIST classifier comparison: EGGROLL vs SGD.

This example trains the same model using both EGGROLL and SGD,
compares their performance, and generates visualization plots.

Uses the comparison_framework for cleaner code.
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

from comparison_framework import ComparisonFramework, ModelConfig, TrainingConfig
from models import MNISTNet
from utils import evaluate_model

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Install with: uv sync --extra dev")


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
    
    # Create model config
    model_config = ModelConfig(
        name="MNIST CNN",
        model_class=MNISTNet,
        model_kwargs={}
    )
    
    # Create training config
    config = TrainingConfig(
        num_epochs=20,
        num_generations=200,
        learning_rate=0.05,
        population_size=64,
        sigma=0.05,
        rank=1,
        eval_every=10,
        quick_mode=quick_mode,
    )
    
    if quick_mode:
        config.num_epochs = 5
        config.num_generations = 50
    
    # Evaluate initial model
    initial_model = MNISTNet().to(device)
    initial_acc = evaluate_model(initial_model, test_loader, device)
    print(f"\nInitial test accuracy: {initial_acc:.2f}%")
    
    # Run comparison using framework
    framework = ComparisonFramework(
        model_config=model_config,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        config=config,
    )
    
    results = framework.run_comparison()
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison(
        results['sgd_results'],
        results['eggroll_results'],
        save_path='mnist_comparison.png'
    )
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
