"""Simple ES trainer implementation as an example."""

import torch
from torch import Tensor
from eggroll_trainer.base import ESTrainer


class SimpleESTrainer(ESTrainer):
    """
    Simple Evolution Strategy trainer using Gaussian perturbations
    and fitness-weighted updates.
    
    This is a basic implementation that can be used as a reference
    for creating more sophisticated ES algorithms.
    """
    
    def sample_perturbations(self, population_size: int) -> Tensor:
        """
        Sample Gaussian perturbations.
        
        Args:
            population_size: Number of perturbations to sample
            
        Returns:
            Tensor of shape (population_size, param_dim) with Gaussian noise
        """
        param_dim = self.current_params.shape[0]
        return torch.randn(
            population_size,
            param_dim,
            device=self.device,
            dtype=self.current_params.dtype,
        )
    
    def compute_update(
        self,
        perturbations: Tensor,
        fitnesses: Tensor,
    ) -> Tensor:
        """
        Compute fitness-weighted update.
        
        Args:
            perturbations: Tensor of shape (population_size, param_dim)
            fitnesses: Tensor of shape (population_size,) with fitness scores
            
        Returns:
            Tensor of shape (param_dim,) with parameter update
        """
        # Normalize fitnesses (center around zero)
        normalized_fitnesses = fitnesses - fitnesses.mean()
        
        # Compute weighted average of perturbations
        # Higher fitness -> larger contribution to update
        weights = normalized_fitnesses / (normalized_fitnesses.std() + 1e-8)
        update = (weights[:, None] * perturbations).mean(dim=0)
        
        return update

