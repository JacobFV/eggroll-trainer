# RL Optimizer Experiments Guide

This directory contains a comprehensive experimental framework for comparing SGD, ES, and EGGROLL optimizers in reinforcement learning.

## Quick Start

### Run a Quick Test Experiment

```bash
cd examples/rl
python quick_experiment.py
```

This runs a small-scale experiment with:
- 2 environments (CartPole, Pendulum)
- 1 RL method (REINFORCE)
- 3 optimizers (SGD, ES, EGGROLL)
- 3 trials per configuration
- 50 training steps

### Run Full Experiments

```bash
python run_experiments.py --help
```

Example: Run comprehensive experiments with 5 trials and 100 steps:

```bash
python run_experiments.py \
    --environments CartPole-v1 Pendulum-v1 Acrobot-v1 \
    --methods reinforce ppo \
    --optimizers sgd es eggroll \
    --trials 5 \
    --steps 100 \
    --skip-mujoco
```

### Analyze Existing Results

If you already have results saved:

```bash
python run_experiments.py \
    --analyze-only \
    --results-file results/comparison_results.json
```

## Experiment Structure

### What Gets Tested

1. **Environments**: Various Gymnasium environments (CartPole, Pendulum, Acrobot, etc.)
2. **RL Methods**: REINFORCE, PPO, Actor-Critic, A2C, DQN
3. **Optimizers**: SGD, ES (Evolution Strategies), EGGROLL

### Output Files

After running experiments, you'll find:

- `results/comparison_results.json`: Raw experimental data
- `results/statistical_analysis.json`: Statistical analysis results
- `results/scientific_report.md`: Generated scientific report
- `results/plots/`: Visualizations including:
  - `overall_comparison.png`: Overall optimizer performance
  - `performance_by_environment.png`: Performance broken down by environment
  - `sample_learning_curves.png`: Sample learning curves
  - Individual plots for each experiment

## Scientific Analysis

The framework performs:

1. **Statistical Significance Testing**
   - t-test (for normal distributions)
   - Mann-Whitney U test (for non-normal distributions)
   - Effect size calculation (Cohen's d)

2. **Performance Rankings**
   - Overall optimizer rankings
   - Environment-specific rankings
   - Method-specific rankings

3. **Comprehensive Visualizations**
   - Learning curves with confidence intervals
   - Bar charts comparing final performance
   - Environment-specific breakdowns

4. **Scientific Report Generation**
   - Executive summary
   - Methodology description
   - Results tables
   - Key findings
   - Conclusions

## Command Line Options

### Basic Options

- `--environments`: List of environment IDs to test
- `--methods`: RL methods to test (reinforce, ppo, actor_critic, a2c, dqn)
- `--optimizers`: Optimizers to test (sgd, es, eggroll)
- `--trials`: Number of trials per configuration (default: 5)
- `--steps`: Number of training steps (default: 100)

### Mode Options

- `--quick`: Use reduced settings for quick testing
- `--skip-mujoco`: Skip MuJoCo environments (useful if not installed)
- `--analyze-only`: Only analyze existing results, don't run new experiments
- `--results-dir`: Directory to save results (default: examples/rl/results)
- `--results-file`: Path to existing results file for analysis

## Example Workflows

### 1. Quick Verification

```bash
python quick_experiment.py
```

### 2. Medium-Scale Experiment

```bash
python run_experiments.py \
    --environments CartPole-v1 Pendulum-v1 \
    --methods reinforce ppo \
    --trials 5 \
    --steps 100
```

### 3. Full-Scale Experiment

```bash
python run_experiments.py \
    --methods reinforce ppo actor_critic a2c dqn \
    --trials 10 \
    --steps 200
```

### 4. Analyze Previous Results

```bash
python run_experiments.py \
    --analyze-only \
    --results-file results/comparison_results.json
```

## Understanding Results

### Learning Curves

Learning curves show how performance improves over training steps. Look for:
- **Convergence speed**: How quickly each optimizer reaches good performance
- **Final performance**: The peak performance achieved
- **Stability**: Consistency across trials (narrow confidence intervals)

### Statistical Significance

The analysis reports:
- **p-value**: Probability that differences are due to chance (< 0.05 = significant)
- **Effect size**: Magnitude of difference (Cohen's d)
  - Small: |d| < 0.5
  - Medium: 0.5 ≤ |d| < 0.8
  - Large: |d| ≥ 0.8

### Performance Rankings

Rankings show which optimizer performs best:
- **Overall**: Across all environments and methods
- **By Environment**: Which optimizer works best for specific tasks
- **By Method**: Which optimizer works best for specific RL algorithms

## Tips for Good Science

1. **Use Multiple Trials**: At least 5 trials per configuration for statistical power
2. **Control Random Seeds**: Seeds are set per trial for reproducibility
3. **Report All Results**: Include both successful and failed experiments
4. **Check Assumptions**: Verify normality assumptions for statistical tests
5. **Consider Effect Sizes**: Statistical significance doesn't always mean practical significance

## Troubleshooting

### Missing Dependencies

Install required packages:

```bash
pip install numpy scipy matplotlib seaborn gymnasium torch
```

### MuJoCo Environments

If you get errors about MuJoCo:
- Use `--skip-mujoco` flag, or
- Install MuJoCo: `pip install mujoco`

### Box2D Environments

If you get errors about Box2D:
- Install: `pip install swig` then `pip install "gymnasium[box2d]"`

### Memory Issues

For large experiments:
- Reduce `--trials` or `--steps`
- Use `--quick` mode
- Test fewer environments at once

## Next Steps

After running experiments:

1. Review the scientific report in `results/scientific_report.md`
2. Examine plots in `results/plots/`
3. Analyze statistical significance results
4. Formulate hypotheses based on findings
5. Design follow-up experiments to test hypotheses

## Contributing

When adding new experiments:
- Document hyperparameters used
- Report random seeds
- Include statistical analysis
- Generate visualizations
- Write up findings

