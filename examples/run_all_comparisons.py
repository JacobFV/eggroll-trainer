"""Run comprehensive comparisons across multiple architectures and tasks."""

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
from models import SimpleCNN, BabyTransformer, MLPClassifier


# ==================== Dataset Setup ====================

def get_mnist_loaders(batch_size=64):
    """Get MNIST data loaders."""
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


# ==================== Main Comparison Runner ====================

def run_comparisons(quick_mode=False):
    """Run comparisons for all model architectures."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    config = TrainingConfig(
        num_epochs=20,
        num_generations=200,
        quick_mode=quick_mode,
    )
    
    if quick_mode:
        config.num_epochs = 5
        config.num_generations = 50
    
    # Define model configurations
    model_configs = [
        ModelConfig(
            name="Simple CNN",
            model_class=SimpleCNN,
            model_kwargs={'num_classes': 10, 'input_channels': 1}
        ),
        ModelConfig(
            name="Baby Transformer",
            model_class=BabyTransformer,
            model_kwargs={'vocab_size': 100, 'd_model': 128, 'nhead': 4, 'num_layers': 2, 'num_classes': 10, 'seq_len': 28*28}
        ),
        ModelConfig(
            name="MLP Classifier",
            model_class=MLPClassifier,
            model_kwargs={'input_dim': 784, 'hidden_dims': [256, 128], 'num_classes': 10}
        ),
    ]
    
    all_results = []
    
    for model_config in model_configs:
        print("\n" + "=" * 70)
        print(f"Running comparison for: {model_config.name}")
        print("=" * 70)
        
        framework = ComparisonFramework(
            model_config=model_config,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            config=config,
        )
        
        results = framework.run_comparison()
        framework.plot_comparison()
        
        all_results.append({
            'name': model_config.name,
            'results': results,
        })
        
        print(f"\n✓ Completed {model_config.name}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for result in all_results:
        name = result['name']
        sgd_results = result['results']['sgd_results']
        eggroll_results = result['results']['eggroll_results']
        
        sgd_best = max(sgd_results['test_acc'])
        eggroll_best = max(acc for _, acc in eggroll_results['test_acc']) if eggroll_results['test_acc'] else 0
        
        print(f"\n{name}:")
        print(f"  SGD Best Test Acc: {sgd_best:.2f}%")
        print(f"  EGGROLL Best Test Acc: {eggroll_best:.2f}%")
        print(f"  Difference: {eggroll_best - sgd_best:+.2f}%")
        print(f"  Time Ratio: {eggroll_results['time'] / sgd_results['time']:.2f}x")


if __name__ == "__main__":
    quick_mode = '--quick' in sys.argv or '-q' in sys.argv
    
    if quick_mode:
        print("Running in QUICK MODE (reduced epochs/generations)")
    
    run_comparisons(quick_mode=quick_mode)

