# MNIST Classification Example

Full example comparing EGGROLL vs SGD on MNIST classification.

## Overview

This example demonstrates:
- Training a CNN on MNIST
- Comparing EGGROLL vs SGD
- Using cached data subsets for efficiency
- Generating comparison plots

## Model Architecture

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 9216)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

## Fitness Function with Cached Subsets

```python
# Pre-cache a subset of training data
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
    return correct / total  # Accuracy (higher is better)
```

## EGGROLL Training

```python
from eggroll_trainer import EGGROLLTrainer

model = MNISTNet()
trainer = EGGROLLTrainer(
    model.parameters(),
    model=model,
    fitness_fn=fitness_fn,
    population_size=256,
    learning_rate=0.01,
    sigma=0.1,
    rank=1,
    seed=42,
)

trainer.train(num_generations=100)
```

## SGD Comparison

```python
import torch.optim as optim

model_sgd = MNISTNet()
optimizer = optim.SGD(model_sgd.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in train_loader:
        optimizer.zero_grad()
        logits = model_sgd(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
```

## Running

```bash
python examples/mnist_comparison.py
```

This will:
1. Train both EGGROLL and SGD
2. Generate comparison plots
3. Print final accuracies

## Results

Typical results:
- **EGGROLL**: ~85-90% accuracy after 100 generations
- **SGD**: ~95%+ accuracy after 10 epochs

Note: EGGROLL is gradient-free and may need more evaluations to match SGD.

## Key Concepts

### Cached Subsets

For efficiency, we evaluate on a cached subset of data rather than the full dataset:

```python
# Pre-cache data
cached_subset = load_subset(train_loader, num_batches=10)

# Fast evaluation
def fitness_fn(model):
    return evaluate_on_subset(model, cached_subset)
```

This makes fitness evaluation fast enough for ES (called many times).

### Fitness vs Loss

- **SGD**: Minimizes loss (lower is better)
- **EGGROLL**: Maximizes fitness (higher is better)

For classification, we use accuracy as fitness.

## Full Example

See `examples/mnist_comparison.py` for the complete code with:
- Data loading
- Model definition
- Training loops
- Plotting
- Evaluation

## Next Steps

- See [Custom Trainers](custom-trainers.md) for creating your own
- Learn about [Fitness Functions](../user-guide/fitness-functions.md)
- Check [Research](../research/benchmarks.md) for performance analysis

