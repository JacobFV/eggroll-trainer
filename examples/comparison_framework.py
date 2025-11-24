"""Generalized comparison framework for EGGROLL vs SGD.

This framework allows easy comparison of EGGROLL with traditional optimizers
across different model architectures and tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from eggroll_trainer import EGGROLLTrainer
import copy
import time
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class TrainingConfig:
    """Configuration for training."""
    num_epochs: int = 20
    num_generations: int = 200
    batch_size: int = 64
    learning_rate: float = 0.01
    population_size: int = 64
    sigma: float = 0.05
    rank: int = 1
    eval_every: int = 5
    quick_mode: bool = False


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    model_class: type
    model_kwargs: Dict[str, Any]
    name: str


class ComparisonFramework:
    """Framework for comparing EGGROLL with traditional optimizers."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        config: Optional[TrainingConfig] = None,
    ):
        self.model_config = model_config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config or TrainingConfig()
        
        # Results storage
        self.results = {
            'sgd': {'train_acc': [], 'test_acc': [], 'loss': [], 'time': 0},
            'eggroll': {'train_acc': [], 'test_acc': [], 'fitness': [], 'time': 0},
        }
    
    def create_fitness_fn(self, batch_limit: int = 50) -> Callable:
        """Create fitness function for EGGROLL."""
        import sys
        import os
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        if _script_dir not in sys.path:
            sys.path.insert(0, _script_dir)
        from utils import create_accuracy_fitness_fn
        return create_accuracy_fitness_fn(self.train_loader, self.device, batch_limit)
    
    def evaluate_model(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluate model on train and test sets."""
        import sys
        import os
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        if _script_dir not in sys.path:
            sys.path.insert(0, _script_dir)
        from utils import evaluate_model as eval_model
        
        # Evaluate on train subset
        def eval_train():
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(self.train_loader):
                    if batch_idx >= 20:
                        break
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    if isinstance(output, tuple):
                        output = output[0]
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            return 100.0 * correct / total if total > 0 else 0.0
        
        train_acc = eval_train()
        test_acc = eval_model(model, self.test_loader, self.device)
        return train_acc, test_acc
    
    def train_sgd(
        self,
        model: nn.Module,
        optimizer_class=optim.SGD,
        optimizer_kwargs: Optional[Dict] = None,
    ) -> Dict:
        """Train model using SGD or other optimizer."""
        if optimizer_kwargs is None:
            optimizer_kwargs = {'lr': self.config.learning_rate, 'momentum': 0.9}
        
        model.train()
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        criterion = nn.CrossEntropyLoss()
        
        train_acc_history = []
        test_acc_history = []
        loss_history = []
        
        num_epochs = 5 if self.config.quick_mode else self.config.num_epochs
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                if isinstance(output, tuple):
                    output = output[0]
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            
            train_acc = 100.0 * correct / total
            test_acc = self.evaluate_model(model)[1]
            
            train_acc_history.append(train_acc)
            test_acc_history.append(test_acc)
            loss_history.append(running_loss / len(self.train_loader))
            
            if (epoch + 1) % self.config.eval_every == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}: "
                      f"Train Acc = {train_acc:.2f}%, "
                      f"Test Acc = {test_acc:.2f}%, "
                      f"Loss = {loss_history[-1]:.4f}")
        
        elapsed_time = time.time() - start_time
        
        return {
            'train_acc': train_acc_history,
            'test_acc': test_acc_history,
            'loss': loss_history,
            'time': elapsed_time,
        }
    
    def train_eggroll(self, model: nn.Module) -> Dict:
        """Train model using EGGROLL."""
        fitness_fn = self.create_fitness_fn(batch_limit=50)
        
        trainer = EGGROLLTrainer(
            model.parameters(),
            model=model,
            fitness_fn=fitness_fn,
            population_size=self.config.population_size,
            learning_rate=self.config.learning_rate,
            sigma=self.config.sigma,
            rank=self.config.rank,
            noise_reuse=0,
            group_size=0,
            freeze_nonlora=False,
            device=self.device,
            seed=42,
        )
        
        train_acc_history = []
        test_acc_history = []
        fitness_history = []
        
        num_generations = 50 if self.config.quick_mode else self.config.num_generations
        
        start_time = time.time()
        
        for gen in range(num_generations):
            metrics = trainer.step()
            
            if (gen + 1) % self.config.eval_every == 0 or gen == 0:
                train_acc, test_acc = self.evaluate_model(trainer.model)
                
                train_acc_history.append((gen + 1, train_acc))
                test_acc_history.append((gen + 1, test_acc))
                fitness_history.append((gen + 1, metrics['mean_fitness']))
                
                if (gen + 1) % self.config.eval_every == 0:
                    print(f"  Generation {gen+1}/{num_generations}: "
                          f"Train Acc = {train_acc:.2f}%, "
                          f"Test Acc = {test_acc:.2f}%, "
                          f"Fitness = {metrics['mean_fitness']:.4f}")
        
        elapsed_time = time.time() - start_time
        
        return {
            'train_acc': train_acc_history,
            'test_acc': test_acc_history,
            'fitness': fitness_history,
            'time': elapsed_time,
        }
    
    def run_comparison(self) -> Dict:
        """Run full comparison between SGD and EGGROLL."""
        print("=" * 70)
        print(f"Comparison: {self.model_config.name}")
        print("=" * 70)
        
        # Create initial models
        initial_model = self.model_config.model_class(**self.model_config.model_kwargs).to(self.device)
        initial_train_acc, initial_test_acc = self.evaluate_model(initial_model)
        print(f"\nInitial - Train Acc: {initial_train_acc:.2f}%, Test Acc: {initial_test_acc:.2f}%")
        
        # Train SGD
        print(f"\n{'='*70}")
        print("SGD Training")
        print(f"{'='*70}")
        sgd_model = copy.deepcopy(initial_model)
        sgd_results = self.train_sgd(sgd_model)
        self.results['sgd'] = sgd_results
        
        # Train EGGROLL
        print(f"\n{'='*70}")
        print("EGGROLL Training")
        print(f"{'='*70}")
        eggroll_model = copy.deepcopy(initial_model)
        eggroll_results = self.train_eggroll(eggroll_model)
        self.results['eggroll'] = eggroll_results
        
        # Final evaluation
        print(f"\n{'='*70}")
        print("Final Results")
        print(f"{'='*70}")
        
        sgd_final_train, sgd_final_test = self.evaluate_model(sgd_model)
        eggroll_final_train, eggroll_final_test = self.evaluate_model(eggroll_model)
        
        print(f"\nSGD:")
        print(f"  Final Train Acc: {sgd_final_train:.2f}%")
        print(f"  Final Test Acc: {sgd_final_test:.2f}%")
        print(f"  Best Test Acc: {max(sgd_results['test_acc']):.2f}%")
        print(f"  Training Time: {sgd_results['time']:.2f}s")
        
        print(f"\nEGGROLL:")
        print(f"  Final Train Acc: {eggroll_final_train:.2f}%")
        print(f"  Final Test Acc: {eggroll_final_test:.2f}%")
        if eggroll_results['test_acc']:
            eggroll_best = max(acc for _, acc in eggroll_results['test_acc'])
            print(f"  Best Test Acc: {eggroll_best:.2f}%")
        print(f"  Training Time: {eggroll_results['time']:.2f}s")
        
        print(f"\nComparison:")
        print(f"  Test Acc Difference: {eggroll_final_test - sgd_final_test:+.2f}%")
        print(f"  Time Ratio (EGGROLL/SGD): {eggroll_results['time'] / sgd_results['time']:.2f}x")
        
        return {
            'sgd_model': sgd_model,
            'eggroll_model': eggroll_model,
            'sgd_results': sgd_results,
            'eggroll_results': eggroll_results,
        }
    
    def plot_comparison(self, save_path: Optional[str] = None):
        """Generate comparison plots."""
        if not HAS_MATPLOTLIB:
            print("Skipping plot generation (matplotlib not installed)")
            return
        
        if save_path is None:
            save_path = f"{self.model_config.name.lower().replace(' ', '_')}_comparison.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        sgd_results = self.results['sgd']
        eggroll_results = self.results['eggroll']
        
        # Test accuracy comparison
        ax1 = axes[0, 0]
        sgd_epochs = range(1, len(sgd_results['test_acc']) + 1)
        eggroll_gens, eggroll_test_acc = zip(*eggroll_results['test_acc']) if eggroll_results['test_acc'] else ([], [])
        
        ax1.plot(sgd_epochs, sgd_results['test_acc'], 'b-o', label='SGD', linewidth=2, markersize=6)
        ax1.plot(eggroll_gens, eggroll_test_acc, 'r-s', label='EGGROLL', linewidth=2, markersize=6)
        ax1.set_xlabel('Epochs / Generations', fontsize=12)
        ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax1.set_title(f'{self.model_config.name} - Test Accuracy', fontsize=14, fontweight='bold')
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
        ax2.set_title(f'{self.model_config.name} - Train Accuracy', fontsize=14, fontweight='bold')
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

