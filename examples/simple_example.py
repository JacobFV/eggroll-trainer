"""Simple example demonstrating the ESTrainer."""

import torch
import torch.nn as nn
from eggroll_trainer import SimpleESTrainer


class SimpleModel(nn.Module):
    """A simple neural network for testing."""
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def create_fitness_fn(target_params=None, noise_std=0.1):
    """
    Create a fitness function that rewards models close to target parameters.
    
    This is a simple test fitness function that evaluates how close
    the model parameters are to a target set of parameters.
    """
    if target_params is None:
        # Create random target parameters
        model = SimpleModel()
        target_params = torch.cat([p.flatten() for p in model.parameters()])
    
    def fitness_fn(model):
        # Get current model parameters
        current_params = torch.cat([p.flatten() for p in model.parameters()])
        
        # Compute negative distance (closer = higher fitness)
        distance = torch.norm(current_params - target_params)
        
        # Add some noise to make it more realistic
        noise = torch.randn(1).item() * noise_std
        
        # Return negative distance + noise (higher is better)
        return -distance.item() + noise
    
    return fitness_fn


def main():
    """Run a simple ES training example."""
    print("Creating model...")
    model = SimpleModel()
    
    print("Creating fitness function...")
    fitness_fn = create_fitness_fn()
    
    print("Creating trainer...")
    trainer = SimpleESTrainer(
        model=model,
        fitness_fn=fitness_fn,
        population_size=20,
        learning_rate=0.01,
        sigma=0.1,
        seed=42,
    )
    
    print("Training for 10 generations...")
    results = trainer.train(num_generations=10, verbose=True)
    
    print("\nTraining complete!")
    print(f"Best fitness: {results['best_fitness']:.4f}")
    print(f"Final generation: {results['generation']}")
    
    # Get best model
    best_model = trainer.get_best_model()
    print("\nBest model retrieved successfully!")


if __name__ == "__main__":
    main()

