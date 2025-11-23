# ESTrainer

Base class for Evolution Strategy trainers.

::: eggroll_trainer.base.ESTrainer
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Abstract Methods

Subclasses must implement:

### `sample_perturbations`

```python
def sample_perturbations(self, population_size: int) -> Tensor:
    """
    Sample perturbations for the population.
    
    Args:
        population_size: Number of population members
        
    Returns:
        Tensor of shape (population_size, param_dim) containing
        perturbations for each population member
    """
    pass
```

### `compute_update`

```python
def compute_update(
    self,
    perturbations: Tensor,
    fitnesses: Tensor,
) -> Tensor:
    """
    Compute parameter update from perturbations and fitnesses.
    
    Args:
        perturbations: Tensor of shape (population_size, param_dim)
        fitnesses: Tensor of shape (population_size,) with fitness scores
        
    Returns:
        Tensor of shape (param_dim,) containing parameter update
    """
    pass
```

## Example

```python
from eggroll_trainer import ESTrainer
import torch

class CustomESTrainer(ESTrainer):
    def sample_perturbations(self, population_size):
        param_dim = self.current_params.shape[0]
        return torch.randn(
            population_size,
            param_dim,
            device=self.device
        ) * self.sigma
    
    def compute_update(self, perturbations, fitnesses):
        # Fitness-weighted average
        weights = (fitnesses - fitnesses.mean()) / (
            fitnesses.std() + 1e-8
        )
        return (weights[:, None] * perturbations).mean(dim=0)
```

