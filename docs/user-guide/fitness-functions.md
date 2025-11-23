# Fitness Functions

Writing effective fitness functions for Evolution Strategies.

## Overview

A **fitness function** evaluates how well a model performs. In ES, **higher fitness is better**.

```python
def fitness_fn(model: nn.Module) -> float:
    """Evaluate model and return fitness score."""
    # Your evaluation logic
    return score  # Higher is better!
```

## Key Principles

### 1. Higher is Better

ES maximizes fitness, so:
- ✅ For accuracy: return accuracy directly
- ✅ For rewards: return reward directly
- ❌ For losses: return `-loss` (negate it)

### 2. Deterministic (When Possible)

Fitness should be deterministic for reproducibility:

```python
# Good: Deterministic
def fitness_fn(model):
    model.eval()
    with torch.no_grad():
        accuracy = evaluate_on_test_set(model)
    return accuracy

# Bad: Non-deterministic (unless seeded)
def fitness_fn(model):
    return torch.randn(1).item()  # Random!
```

### 3. Efficient

Fitness is called many times (population_size × generations), so keep it fast:

```python
# Good: Fast evaluation
def fitness_fn(model):
    # Use cached data subset
    return evaluate_on_subset(model, cached_data)

# Bad: Slow evaluation
def fitness_fn(model):
    # Evaluate on full dataset every time
    return evaluate_on_full_dataset(model)  # Too slow!
```

## Common Patterns

### Classification

```python
def classification_fitness(model, data_loader):
    """Fitness = accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    accuracy = correct / total
    return accuracy  # Higher is better
```

### Regression

```python
def regression_fitness(model, data_loader):
    """Fitness = negative MSE loss."""
    model.eval()
    total_loss = 0
    count = 0
    with torch.no_grad():
        for x, y in data_loader:
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y)
            total_loss += loss.item()
            count += 1
    avg_loss = total_loss / count
    return -avg_loss  # Convert loss to fitness
```

### Parameter Matching

```python
def parameter_fitness(model, target_params):
    """Fitness = negative distance to target."""
    current_params = torch.cat([p.flatten() for p in model.parameters()])
    distance = (current_params - target_params).norm()
    return -distance.item()  # Minimize distance
```

### Reinforcement Learning

```python
def rl_fitness(model, env, num_episodes=10):
    """Fitness = average episode reward."""
    model.eval()
    total_reward = 0
    with torch.no_grad():
        for _ in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = model(obs).argmax()
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
            total_reward += episode_reward
    return total_reward / num_episodes  # Higher is better
```

## Advanced Techniques

### Cached Data Subsets

For large datasets, cache a subset for fast evaluation:

```python
# Pre-cache data subset
cached_subset = []
for i, (x, y) in enumerate(train_loader):
    if i >= 10:  # Use first 10 batches
        break
    cached_subset.append((x, y))

def fitness_fn(model):
    """Fast evaluation on cached subset."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in cached_subset:
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)
    return correct / total
```

### Multi-Objective Fitness

Combine multiple objectives:

```python
def multi_objective_fitness(model):
    accuracy = compute_accuracy(model)
    efficiency = compute_efficiency(model)  # e.g., inference time
    # Weighted combination
    fitness = 0.8 * accuracy + 0.2 * efficiency
    return fitness
```

### Fitness Shaping

Apply transformations to improve learning:

```python
def shaped_fitness(model):
    raw_fitness = compute_raw_fitness(model)
    # Rank-based shaping (can help with outliers)
    return rank_transform(raw_fitness)
```

## Closure Pattern

Use closures to capture data/state:

```python
def create_fitness_fn(data_loader, device):
    """Factory function for fitness with captured data."""
    def fitness_fn(model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        return correct / total
    return fitness_fn

# Usage
fitness_fn = create_fitness_fn(train_loader, device)
trainer = EGGROLLTrainer(model=model, fitness_fn=fitness_fn, ...)
```

## Common Pitfalls

### ❌ Returning Loss Instead of Fitness

```python
# Bad: Returning loss (lower is better)
def fitness_fn(model):
    loss = compute_loss(model)
    return loss  # ES will minimize this!

# Good: Negate loss
def fitness_fn(model):
    loss = compute_loss(model)
    return -loss  # ES will maximize this (minimize loss)
```

### ❌ Non-Deterministic Evaluation

```python
# Bad: Random evaluation
def fitness_fn(model):
    return torch.randn(1).item()

# Good: Deterministic evaluation
def fitness_fn(model):
    model.eval()
    with torch.no_grad():
        return evaluate_deterministic(model)
```

### ❌ Too Slow

```python
# Bad: Full dataset every time
def fitness_fn(model):
    return evaluate_on_full_dataset(model)  # Too slow!

# Good: Cached subset
cached_data = load_subset()
def fitness_fn(model):
    return evaluate_on_subset(model, cached_data)  # Fast!
```

### ❌ Modifying Model State

```python
# Bad: Modifying model during evaluation
def fitness_fn(model):
    model.train()  # Don't change model state!
    # ... evaluation ...

# Good: Use eval mode
def fitness_fn(model):
    model.eval()  # Set eval mode
    with torch.no_grad():
        # ... evaluation ...
```

## Tips

1. **Keep it fast** - Fitness is called many times
2. **Use cached data** - Pre-load evaluation data
3. **Be deterministic** - For reproducibility
4. **Higher is better** - Negate losses
5. **Use eval mode** - `model.eval()` for inference
6. **No gradients** - Use `torch.no_grad()` context

## Next Steps

- See [Examples](../examples/index.md) for real-world fitness functions
- Learn [Advanced Usage](advanced-usage.md) for optimization tips
- Check [API Reference](../api-reference/index.md) for details

