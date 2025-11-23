"""Example demonstrating the EGGROLL trainer."""

import torch
import torch.nn as nn
from eggroll_trainer import EGGROLLTrainer


class SimpleModel(nn.Module):
    """A simple neural network with both matrix and non-matrix parameters."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # Matrix: will use LoRA
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Matrix: will use LoRA
        self.fc3 = nn.Linear(hidden_dim, output_dim)   # Matrix: will use LoRA
        self.relu = nn.ReLU()
        # Bias terms are 1D, will use full-rank updates
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_fitness_fn(target_params=None, noise_std=0.1):
    """
    Create a fitness function that rewards models close to target parameters.
    """
    if target_params is None:
        # Create random target parameters
        model = SimpleModel()
        target_params = {}
        for name, param in model.named_parameters():
            target_params[name] = param.data.clone()
    
    def fitness_fn(model):
        # Get current model parameters
        total_distance = 0.0
        for name, param in model.named_parameters():
            if name in target_params:
                distance = torch.norm(param - target_params[name])
                total_distance += distance.item()
        
        # Add some noise to make it more realistic
        noise = torch.randn(1).item() * noise_std
        
        # Return negative distance + noise (higher is better)
        return -total_distance + noise
    
    return fitness_fn


def main():
    """Run EGGROLL training example."""
    print("=" * 60)
    print("EGGROLL Trainer Example")
    print("=" * 60)
    
    print("\nCreating model...")
    model = SimpleModel()
    
    print("Creating fitness function...")
    fitness_fn = create_fitness_fn()
    
    print("\nCreating EGGROLL trainer...")
    print("  - Population size: 32")
    print("  - Rank: 1 (low-rank perturbations)")
    print("  - Sigma: 0.1")
    print("  - Learning rate: 0.01")
    
    trainer = EGGROLLTrainer(
        model=model,
        fitness_fn=fitness_fn,
        population_size=32,
        learning_rate=0.01,
        sigma=0.1,
        rank=1,  # Low-rank rank
        noise_reuse=0,  # No noise reuse
        group_size=0,  # Global normalization
        freeze_nonlora=False,
        seed=42,
    )
    
    print("\nTraining for 10 generations...")
    print("(EGGROLL uses low-rank perturbations for matrix parameters)")
    print("-" * 60)
    
    results = trainer.train(num_generations=10, verbose=True)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best fitness: {results['best_fitness']:.4f}")
    print(f"Final generation: {results['generation']}")
    
    # Get best model
    best_model = trainer.get_best_model()
    print("\nBest model retrieved successfully!")
    
    # Show parameter statistics
    print("\nParameter statistics:")
    for name, param in best_model.named_parameters():
        if len(param.shape) == 2:
            print(f"  {name}: shape {param.shape} (matrix - used LoRA updates)")
        else:
            print(f"  {name}: shape {param.shape} (non-matrix - used full-rank updates)")


if __name__ == "__main__":
    main()

