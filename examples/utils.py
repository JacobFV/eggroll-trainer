"""Shared utility functions for examples."""

import torch
from torch.utils.data import DataLoader
from typing import Callable, Dict, Optional


def create_parameter_distance_fitness_fn(
    target_params: Optional[Dict[str, torch.Tensor]] = None,
    noise_std: float = 0.1,
    use_dict: bool = False,
) -> Callable:
    """
    Create a fitness function that rewards models close to target parameters.
    
    Args:
        target_params: Target parameters dict (if None, creates random target)
        noise_std: Standard deviation of noise to add
        use_dict: If True, uses dict-based parameter access (for EGGROLL)
                  If False, uses flattened tensor (for base ESTrainer)
    
    Returns:
        Fitness function that returns accuracy (higher is better)
    """
    if target_params is None:
        # Will be set when model is provided
        target_params_holder = None
    else:
        target_params_holder = target_params
    
    def fitness_fn(model):
        if target_params_holder is None:
            # Create target from current model
            if use_dict:
                target = {}
                for name, param in model.named_parameters():
                    target[name] = param.data.clone()
            else:
                target = torch.cat([p.flatten() for p in model.parameters()])
        else:
            target = target_params_holder
        
        if use_dict:
            # Dict-based (for EGGROLL)
            total_distance = 0.0
            for name, param in model.named_parameters():
                if name in target:
                    distance = torch.norm(param - target[name])
                    total_distance += distance.item()
            distance = total_distance
        else:
            # Flattened tensor (for base ESTrainer)
            current_params = torch.cat([p.flatten() for p in model.parameters()])
            distance = torch.norm(current_params - target).item()
        
        # Add noise to make it more realistic
        noise = torch.randn(1).item() * noise_std
        
        # Return negative distance + noise (higher is better)
        return -distance + noise
    
    return fitness_fn


def create_accuracy_fitness_fn(
    train_loader: DataLoader,
    device: torch.device,
    batch_limit: int = 50,
) -> Callable:
    """
    Create a fitness function that evaluates model accuracy on training data.
    
    Uses cached subset for faster evaluation during EGGROLL training.
    
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
                if isinstance(output, tuple):
                    output = output[0]  # Handle models that return (logits, ...)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    return fitness_fn


def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """
    Evaluate model on test set.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to run on
    
    Returns:
        Test accuracy as percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if isinstance(output, tuple):
                output = output[0]
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return 100.0 * correct / total if total > 0 else 0.0

