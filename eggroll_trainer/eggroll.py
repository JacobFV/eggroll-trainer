"""
EGGROLL (Evolution Guided General Optimization via Low-rank Learning) trainer.

Based on the EGGROLL algorithm from:
https://eshyperscale.github.io/
https://github.com/ESHyperscale/HyperscaleES

Key innovation: Uses low-rank perturbations (A @ B.T) instead of full-rank
perturbations, reducing memory from O(mn) to O(r(m+n)) and computation
from O(mn) to O(r(m+n)) while still achieving high-rank updates through
population averaging.
"""

import torch
import torch.nn as nn
import torch.optim
from typing import Callable, Dict, List, Optional, Tuple, Any
from torch import Tensor
from collections import OrderedDict
import copy


class EGGROLLTrainer(torch.optim.Optimizer):
    """
    EGGROLL trainer implementing the actual EGGROLL algorithm.
    
    Unlike the base ESTrainer which works with flattened parameters,
    EGGROLL works per-layer with low-rank perturbations for efficiency.
    
    Key features:
    - Low-rank perturbations: For matrices W ∈ R^(m×n), samples A ∈ R^(m×r), B ∈ R^(n×r)
      where r << min(m,n), forming perturbation A @ B.T
    - Per-layer updates: Handles each parameter tensor independently
    - Noise reuse: Can reuse noise across multiple evaluations (antithetic sampling)
    - Group normalization: Supports fitness normalization within groups
    
    Subclasses torch.optim.Optimizer for compatibility with PyTorch optimizer interface.
    Use model.parameters() as the first argument, similar to standard optimizers.
    """
    
    def __init__(
        self,
        params,
        model: nn.Module,
        fitness_fn: Callable[[nn.Module], float],
        population_size: int = 256,
        learning_rate: float = 0.01,
        sigma: float = 0.1,
        rank: int = 1,
        noise_reuse: int = 0,
        group_size: int = 0,
        freeze_nonlora: bool = False,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the EGGROLL trainer.
        
        Args:
            params: Iterable of parameters to optimize (for optimizer compatibility).
                   Typically model.parameters().
            model: PyTorch model to train. Parameters will be optimized.
            fitness_fn: Function that takes a model and returns a fitness score
                        (higher is better). Should handle model evaluation.
            population_size: Number of population members
            learning_rate: Learning rate for parameter updates
            sigma: Standard deviation for perturbations
            rank: Rank of low-rank perturbations (default: 1)
            noise_reuse: Number of evaluations to reuse noise (0 = no reuse, 2 = antithetic)
            group_size: Size of groups for fitness normalization (0 = global normalization)
            freeze_nonlora: If True, only apply LoRA updates to linear layers
            device: Device to run on
            seed: Random seed
        """
        defaults = dict(
            learning_rate=learning_rate,
            population_size=population_size,
            sigma=sigma,
            rank=rank,
            noise_reuse=noise_reuse,
            group_size=group_size,
            freeze_nonlora=freeze_nonlora,
        )
        super().__init__(params, defaults)
        
        self.model = model
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.rank = rank
        self.noise_reuse = noise_reuse
        self.group_size = group_size
        self.freeze_nonlora = freeze_nonlora
        
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
            self.model = self.model.to(self.device)
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # Extract parameter structure
        self.param_names = []
        self.param_shapes = []
        self.param_dims = []
        self.is_matrix = []  # True for 2D tensors (can use LoRA)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.param_names.append(name)
                self.param_shapes.append(param.shape)
                self.param_dims.append(param.numel())
                # Matrices (2D tensors) can use low-rank updates
                self.is_matrix.append(len(param.shape) == 2)
        
        # Initialize optimizer state (we'll use SGD-style updates)
        self.optimizer_state = {}
        
        # Training state
        self.generation = 0
        self.fitness_history: List[float] = []
        self.best_fitness: Optional[float] = None
        self.best_state_dict: Optional[Dict] = None
        
        # PRNG state for noise generation
        self.rng = torch.Generator(device=self.device)
        if seed is not None:
            self.rng.manual_seed(seed)
    
    def _get_lora_params(self, param: Tensor, thread_id: int, epoch: int) -> Tuple[Tensor, Tensor]:
        """
        Generate low-rank LoRA parameters A and B for a matrix parameter.
        
        For parameter W ∈ R^(m×n), generates:
        - A ∈ R^(m×r)
        - B ∈ R^(n×r)
        such that perturbation is A @ B.T
        
        Args:
            param: Parameter tensor of shape (m, n)
            thread_id: Thread/population member ID
            epoch: Current epoch (for noise reuse)
            
        Returns:
            Tuple of (A, B) tensors
        """
        m, n = param.shape
        
        # Handle noise reuse
        if self.noise_reuse > 0:
            true_epoch = epoch // self.noise_reuse
            true_thread_id = thread_id // 2
            # Antithetic sampling: alternate signs
            sign = 1.0 if (thread_id % 2 == 0) else -1.0
        else:
            true_epoch = epoch
            true_thread_id = thread_id
            sign = 1.0
        
        # Generate random vectors for low-rank decomposition
        # Sample (m+n) * rank values, split into A and B
        key = (true_epoch * 1000000 + true_thread_id) % (2**31)
        self.rng.manual_seed(key)
        
        lora_params = torch.randn(
            (m + n) * self.rank,
            device=self.device,
            dtype=param.dtype,
            generator=self.rng,
        )
        
        # Split into A and B
        B_flat = lora_params[:n * self.rank]  # n * r
        A_flat = lora_params[n * self.rank:]  # m * r
        
        B = B_flat.view(n, self.rank)  # n × r
        A = A_flat.view(m, self.rank)  # m × r
        
        # Scale by sigma / sqrt(rank) for proper variance
        scale = self.sigma / (self.rank ** 0.5)
        return A * scale * sign, B
    
    def _get_full_rank_params(self, param: Tensor, thread_id: int, epoch: int) -> Tensor:
        """
        Generate full-rank perturbation for non-matrix parameters.
        
        Args:
            param: Parameter tensor
            thread_id: Thread/population member ID
            epoch: Current epoch (for noise reuse)
            
        Returns:
            Perturbation tensor of same shape as param
        """
        if self.freeze_nonlora:
            return torch.zeros_like(param)
        
        # Handle noise reuse
        if self.noise_reuse > 0:
            true_epoch = epoch // self.noise_reuse
            true_thread_id = thread_id // 2
            sign = 1.0 if (thread_id % 2 == 0) else -1.0
        else:
            true_epoch = epoch
            true_thread_id = thread_id
            sign = 1.0
        
        key = (true_epoch * 1000000 + true_thread_id) % (2**31)
        self.rng.manual_seed(key)
        
        perturbation = torch.randn(
            param.shape,
            device=self.device,
            dtype=param.dtype,
            generator=self.rng,
        )
        
        return perturbation * self.sigma * sign
    
    def _apply_perturbations(self, thread_id: int, epoch: int) -> Dict[str, Tensor]:
        """
        Generate perturbations for all parameters for a given thread.
        
        Returns:
            Dictionary mapping parameter names to perturbed parameter values
        """
        perturbations = {}
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            idx = self.param_names.index(name)
            is_mat = self.is_matrix[idx]
            
            if is_mat:
                # Low-rank perturbation for matrices
                A, B = self._get_lora_params(param, thread_id, epoch)
                # Perturbation is A @ B.T
                perturbation = A @ B.T
                perturbations[name] = param + perturbation
            else:
                # Full-rank perturbation for non-matrices
                perturbation = self._get_full_rank_params(param, thread_id, epoch)
                perturbations[name] = param + perturbation
        
        return perturbations
    
    def _convert_fitnesses(self, raw_scores: Tensor) -> Tensor:
        """
        Convert raw scores to normalized fitnesses.
        
        Supports group normalization if group_size > 0.
        
        Args:
            raw_scores: Raw fitness scores of shape (population_size,)
            
        Returns:
            Normalized fitnesses of same shape
        """
        if self.group_size == 0:
            # Global normalization
            mean = raw_scores.mean()
            std = raw_scores.std() + 1e-8
            return (raw_scores - mean) / std
        else:
            # Group normalization
            num_groups = self.population_size // self.group_size
            scores_reshaped = raw_scores.view(num_groups, self.group_size)
            group_means = scores_reshaped.mean(dim=1, keepdim=True)
            group_stds = scores_reshaped.std(dim=1, keepdim=True) + 1e-8
            normalized = (scores_reshaped - group_means) / group_stds
            return normalized.view(-1)
    
    def _compute_lora_update(
        self,
        param: Tensor,
        fitnesses: Tensor,
        epoch: int,
    ) -> Tensor:
        """
        Compute EGGROLL update for a matrix parameter using low-rank structure.
        
        This reconstructs the low-rank perturbations and computes the weighted
        average update, which results in a high-rank update despite using
        low-rank perturbations.
        
        Args:
            param: Parameter tensor of shape (m, n)
            fitnesses: Normalized fitnesses of shape (population_size,)
            epoch: Current epoch
            
        Returns:
            Update tensor of shape (m, n)
        """
        m, n = param.shape
        pop = self.population_size
        
        # Reconstruct all A and B matrices
        A_list = []
        B_list = []
        
        for thread_id in range(pop):
            A, B = self._get_lora_params(param, thread_id, epoch)
            A_list.append(A)  # m × r
            B_list.append(B)  # n × r
        
        # Stack: A_stack shape (pop, m, r), B_stack shape (pop, n, r)
        A_stack = torch.stack(A_list, dim=0)  # pop × m × r
        B_stack = torch.stack(B_list, dim=0)  # pop × n × r
        
        # Weight by fitnesses: fitnesses shape (pop,)
        # Broadcast fitnesses: (pop, 1, 1) for A_stack, (pop, 1, 1) for B_stack
        fitness_broadcast = fitnesses.view(pop, 1, 1)
        A_weighted = A_stack * fitness_broadcast  # pop × m × r
        
        # Compute update: sum over population of A_weighted @ B.T
        # Using einsum: 'nmi,njr->ij' where n=pop, m=m, i=r, j=n, r=r
        # Actually: we want sum_n (A_weighted[n] @ B_stack[n].T)
        # Which is: sum_n sum_r A_weighted[n, i, r] * B_stack[n, j, r]
        # einsum: 'nir,njr->ij'
        update = torch.einsum('nir,njr->ij', A_weighted, B_stack) / pop
        
        return update
    
    def _compute_full_update(
        self,
        param: Tensor,
        fitnesses: Tensor,
        epoch: int,
    ) -> Tensor:
        """
        Compute EGGROLL update for a non-matrix parameter.
        
        Args:
            param: Parameter tensor
            fitnesses: Normalized fitnesses of shape (population_size,)
            epoch: Current epoch
            
        Returns:
            Update tensor of same shape as param
        """
        if self.freeze_nonlora:
            return torch.zeros_like(param)
        
        pop = self.population_size
        
        # Reconstruct all perturbations
        perturbations = []
        for thread_id in range(pop):
            pert = self._get_full_rank_params(param, thread_id, epoch)
            perturbations.append(pert)
        
        # Stack: shape (pop, *param.shape)
        pert_stack = torch.stack(perturbations, dim=0)
        
        # Weight by fitnesses and average
        fitness_broadcast = fitnesses.view((pop,) + (1,) * len(param.shape))
        update = (pert_stack * fitness_broadcast).mean(dim=0)
        
        return update
    
    def step(self, closure=None):
        """
        Perform one optimization step.
        
        This method provides compatibility with PyTorch optimizer interface.
        For ES algorithms, the fitness function is provided at initialization,
        not per-step. The closure parameter is ignored for ES.
        
        Args:
            closure: Optional callable (ignored for ES, fitness_fn used instead)
        
        Returns:
            Dictionary with training metrics
        """
        # Evaluate population
        raw_scores = []
        
        # Save original state
        original_state_dict = {name: param.data.clone() for name, param in self.model.named_parameters()}
        
        for thread_id in range(self.population_size):
            # Apply perturbations
            perturbed_params = self._apply_perturbations(thread_id, self.generation)
            
            # Set model parameters
            for name, value in perturbed_params.items():
                param_ref = dict(self.model.named_parameters())[name]
                param_ref.data.copy_(value)
            
            # Evaluate fitness
            fitness = self.fitness_fn(self.model)
            raw_scores.append(fitness)
        
        # Restore original parameters
        for name, value in original_state_dict.items():
            param_ref = dict(self.model.named_parameters())[name]
            param_ref.data.copy_(value)
        
        raw_scores_tensor = torch.tensor(
            raw_scores,
            device=self.device,
            dtype=torch.float32,
        )
        
        # Convert to normalized fitnesses
        fitnesses = self._convert_fitnesses(raw_scores_tensor)
        
        # Compute updates for each parameter
        updates = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            idx = self.param_names.index(name)
            is_mat = self.is_matrix[idx]
            
            if is_mat:
                update = self._compute_lora_update(param, fitnesses, self.generation)
            else:
                update = self._compute_full_update(param, fitnesses, self.generation)
            
            updates[name] = update
        
        # Apply updates
        for name, param in self.model.named_parameters():
            if name in updates:
                # Scale by learning rate and population size
                scale = self.learning_rate * (self.population_size ** 0.5)
                param.data = param.data + scale * updates[name]
        
        # Track best fitness
        best_fitness = raw_scores_tensor.max().item()
        if self.best_fitness is None or best_fitness > self.best_fitness:
            self.best_fitness = best_fitness
            self.best_state_dict = copy.deepcopy(self.model.state_dict())
        
        # Update history
        self.generation += 1
        mean_fitness = raw_scores_tensor.mean().item()
        self.fitness_history.append(mean_fitness)
        
        return {
            "generation": self.generation,
            "mean_fitness": mean_fitness,
            "best_fitness": best_fitness,
            "std_fitness": raw_scores_tensor.std().item(),
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
        Train for multiple generations.
        
        Args:
            num_generations: Number of generations to train
            verbose: Whether to print progress
            
        Returns:
            Dictionary with final training state
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
    
    def get_best_model(self) -> nn.Module:
        """
        Get a copy of the model with the best parameters found.
        
        Returns:
            New model instance with best parameters
        """
        best_model = copy.deepcopy(self.model)
        if self.best_state_dict is not None:
            best_model.load_state_dict(self.best_state_dict)
        return best_model

