"""Run comprehensive comparisons across multiple architectures and tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from comparison_framework import ComparisonFramework, ModelConfig, TrainingConfig
import sys


# ==================== Model Architectures ====================

class SimpleCNN(nn.Module):
    """Simple CNN for image classification."""
    
    def __init__(self, num_classes=10, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class BabyTransformer(nn.Module):
    """Small transformer for sequence classification."""
    
    def __init__(self, vocab_size=100, d_model=128, nhead=4, num_layers=2, num_classes=10, seq_len=28):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model, num_classes)
        self.seq_len = seq_len
    
    def forward(self, x):
        # x shape: (batch, channels, height, width) for MNIST
        # Convert to sequence: flatten spatial dims
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # (batch, 784)
        x = (x * 99).long().clamp(0, 99)  # Quantize to vocab_size
        
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x


class MLPClassifier(nn.Module):
    """Simple MLP classifier."""
    
    def __init__(self, input_dim=784, hidden_dims=[256, 128], num_classes=10):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


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

