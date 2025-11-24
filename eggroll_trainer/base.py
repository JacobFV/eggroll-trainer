"""Base Evolution Strategy trainer class."""

import abc
from typing import Callable, Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.optim
from torch import Tensor


class ESTrainer(torch.optim.Optimizer, abc.ABC):
    """
    Base class for Evolution Strategy trainers.
    
    This class provides a framework for implementing various ES algorithms.
    Subclasses should override the abstract methods to implement specific
    ES variants (e.g., CMA-ES, OpenAI ES, Natural ES, etc.).
    
    Example:
        class VanillaESTrainer(ESTrainer):
            def sample_perturbations(self, population_size):
                # Return perturbations for each member of the population
                pass
            
            def compute_update(self, perturbations, fitnesses):
                # Compute parameter update from perturbations and fitnesses
                pass
    """
    
    def __init__(
        self,
        params,
        model: nn.Module,
        fitness_fn: Callable[[nn.Module], float],
        population_size: int = 50,
        learning_rate: float = 0.01,
        sigma: float = 0.1,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the ES trainer.
        
        Args:
            params: Iterable of parameters to optimize (for optimizer compatibility).
                   Typically model.parameters().
            model: PyTorch model to train. Parameters will be optimized.
            fitness_fn: Function that takes a model and returns a fitness score
                        (higher is better). Should handle model evaluation.
            population_size: Number of perturbed models to evaluate per generation
            learning_rate: Learning rate for parameter updates
            sigma: Standard deviation for parameter perturbations
            device: Device to run training on (defaults to model's device)
            seed: Random seed for reproducibility
        """
        defaults = dict(
            learning_rate=learning_rate,
            population_size=population_size,
            sigma=sigma,
        )
        super().__init__(params, defaults)
        
        self.model = model
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
            self.model = self.model.to(self.device)
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Extract initial parameters
        self.initial_params = self._get_flat_params()
        self.current_params = self.initial_params.clone()
        
        # Training history
        self.generation = 0
        self.fitness_history: List[float] = []
        self.best_fitness: Optional[float] = None
        self.best_params: Optional[Tensor] = None
    
    def _get_flat_params(self) -> Tensor:
        """Flatten model parameters into a single vector."""
        params = []
        for param in self.model.parameters():
            params.append(param.data.flatten())
        return torch.cat(params)
    
    def _set_flat_params(self, flat_params: Tensor) -> None:
        """Set model parameters from a flattened vector."""
        idx = 0
        for param in self.model.parameters():
            param_size = param.numel()
            param.data = flat_params[idx:idx + param_size].view(param.shape)
            idx += param_size
    
    def _clone_model(self) -> nn.Module:
        """
        Create a deep copy of the model.
        
        Uses state_dict copying which works for most PyTorch models.
        Subclasses can override if they need custom cloning logic.
        """
        import copy
        return copy.deepcopy(self.model)
    
    @abc.abstractmethod
    def sample_perturbations(self, population_size: int) -> Tensor:
        """
        Sample perturbations for the population.
        
        Args:
            population_size: Number of perturbations to sample
            
        Returns:
            Tensor of shape (population_size, param_dim) containing perturbations
        """
        pass
    
    @abc.abstractmethod
    def compute_update(
        self,
        perturbations: Tensor,
        fitnesses: Tensor,
    ) -> Tensor:
        """
        Compute parameter update from perturbations and fitnesses.
        
        Args:
            perturbations: Tensor of shape (population_size, param_dim)
            fitnesses: Tensor of shape (population_size,) containing fitness scores
            
        Returns:
            Tensor of shape (param_dim,) containing the parameter update
        """
        pass
    
    def evaluate_fitness(self, model: nn.Module) -> float:
        """
        Evaluate fitness of a model.
        
        Args:
            model: Model to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        return self.fitness_fn(model)
    
    def step(self, closure=None):
        """
        Perform one optimization step (generation).
        
        This method provides compatibility with PyTorch optimizer interface.
        For ES algorithms, the fitness function is provided at initialization,
        not per-step. The closure parameter is ignored for ES.
        
        Args:
            closure: Optional callable (ignored for ES, fitness_fn used instead)
        
        Returns:
            Dictionary containing training metrics
        """
        # Sample perturbations
        perturbations = self.sample_perturbations(self.population_size)
        
        # Evaluate population
        fitnesses = []
        for i in range(self.population_size):
            # Create perturbed parameters
            perturbed_params = self.current_params + self.sigma * perturbations[i]
            
            # Set model parameters
            self._set_flat_params(perturbed_params)
            
            # Evaluate fitness
            fitness = self.evaluate_fitness(self.model)
            fitnesses.append(fitness)
        
        fitnesses_tensor = torch.tensor(
            fitnesses,
            device=self.device,
            dtype=torch.float32,
        )
        
        # Compute update
        update = self.compute_update(perturbations, fitnesses_tensor)
        
        # Update parameters
        self.current_params = self.current_params + self.learning_rate * update
        self._set_flat_params(self.current_params)
        
        # Track best fitness
        best_fitness = fitnesses_tensor.max().item()
        if self.best_fitness is None or best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_params = self.current_params.clone()
        
        # Update history
        self.generation += 1
        mean_fitness = fitnesses_tensor.mean().item()
        self.fitness_history.append(mean_fitness)
        
        return {
            "generation": self.generation,
            "mean_fitness": mean_fitness,
            "best_fitness": best_fitness,
            "std_fitness": fitnesses_tensor.std().item(),
        }
    
    def zero_grad(self, set_to_none: bool = False):
        """
        Zero gradients (no-op for ES algorithms).
        
        This method exists for optimizer interface compatibility.
        ES algorithms don't use gradients, so this is a no-op.
        
        Args:
            set_to_none: If True, set gradients to None (ignored for ES)
        """
        pass
    
    def train(self, num_generations: int, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the model for multiple generations.
        
        Args:
            num_generations: Number of generations to train
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing final training state
        """
        for gen in range(num_generations):
            metrics = self.step()
            
            if verbose:
                print(
                    f"Generation {metrics['generation']}: "
                    f"Mean Fitness = {metrics['mean_fitness']:.4f}, "
                    f"Best Fitness = {metrics['best_fitness']:.4f}"
                )
        
        return {
            "generation": self.generation,
            "best_fitness": self.best_fitness,
            "fitness_history": self.fitness_history,
        }
    
    def reset(self) -> None:
        """Reset trainer to initial state."""
        self.current_params = self.initial_params.clone()
        self._set_flat_params(self.current_params)
        self.generation = 0
        self.fitness_history = []
        self.best_fitness = None
        self.best_params = None
    
    def get_best_model(self) -> nn.Module:
        """
        Get a copy of the model with the best parameters found.
        
        Returns:
            A new model instance with the best parameters loaded
        """
        best_model = self._clone_model().to(self.device)
        
        if self.best_params is not None:
            # Temporarily save current params
            temp_params = self.current_params.clone()
            # Set best params to model
            self._set_flat_params(self.best_params)
            # Load best params into cloned model
            best_model.load_state_dict(self.model.state_dict())
            # Restore current params
            self._set_flat_params(temp_params)
        
        return best_model

