# 3D Animals RL Examples

Advanced reinforcement learning examples using MuJoCo for 3D animal locomotion and manipulation.

## Overview

This directory contains examples demonstrating RL training on 3D animal locomotion tasks using MuJoCo physics simulation. Examples range from simple single-agent locomotion to complex multi-agent scenarios and advanced physics simulations.

## Structure

```
animals_3d/
├── utils.py                    # Reward shaping, environment wrappers
├── visualization.py            # MuJoCo viewer, plotting utilities
├── mujoco_utils.py             # Custom MJCF utilities, cloth/tendon generation
├── world_builder.py            # Procedural world generation
├── models/                     # Custom MJCF model files
├── simple_examples/            # Basic locomotion examples
├── multi_agent/                # Multi-agent scenarios
└── advanced/                   # Advanced MuJoCo features
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install 'gymnasium[mujoco]' matplotlib torch numpy

# Or with uv
uv sync --extra examples
```

### Running Examples

```bash
# Simple example
python examples/animals_3d/simple_examples/ant_walk.py

# Multi-agent example
python examples/animals_3d/multi_agent/multi_ant_race.py

# Advanced example
python examples/animals_3d/advanced/custom_obstacle_course.py
```

## Examples

### Simple Examples

**Basic locomotion tasks using standard Gymnasium environments:**

- **`ant_walk.py`** - Ant learning to walk
- **`halfcheetah_run.py`** - HalfCheetah learning to run
- **`humanoid_stand.py`** - Humanoid learning to stand and walk

### Multi-Agent Examples

**Multiple agents interacting in shared environments:**

- **`multi_ant_race.py`** - Multiple ants racing against each other
- **`predator_prey.py`** - Predator-prey dynamics with multiple agents
- **`flocking.py`** - Flocking behavior with separation, alignment, cohesion

### Advanced Examples

**Advanced MuJoCo features and custom environments:**

#### Custom Worlds
- **`custom_obstacle_course.py`** - Navigate procedurally generated obstacle courses
- **`terrain_generation.py`** - Adapt to procedurally generated terrain

#### Advanced Physics
- **`underwater_locomotion.py`** - Underwater swimming with fluid dynamics
- **`aerial_locomotion.py`** - Flight control with reduced gravity and wind

#### Cloth Simulation
- **`cloth_walking.py`** - Walk on deformable cloth surface
- **`cloth_manipulation.py`** - Manipulate cloth (folding, draping, pulling)

#### Tendon/Muscle Systems
- **`tendon_driven_robot.py`** - Precise control using tendon-driven actuation
- **`muscle_actuated.py`** - Muscle-like actuators with activation dynamics and fatigue

#### Soft Bodies
- **`morphing_creature.py`** - Deformable body that adapts shape for different tasks
- **`soft_rigid_interaction.py`** - Agent with soft body parts interacts with rigid objects

#### Skinned Models
- **`skinned_character.py`** - Humanoid with skinned mesh learns natural locomotion

#### Other Advanced
- **`complex_locomotion.py`** - Hierarchical control with multiple behavior modes
- **`hierarchical_control.py`** - Hierarchical RL with meta-policy and sub-policies
- **`multi_agent_cooperation.py`** - Cooperative multi-agent behaviors
- **`object_manipulation.py`** - Learn to manipulate objects (push, pull, carry)

## Utilities

### `utils.py`
- `RewardShaper` - Reward shaping utilities (forward velocity, stability, energy)
- `ObservationPreprocessor` - Observation normalization and feature extraction
- `MultiAgentCoordinator` - Multi-agent coordination utilities
- `EnvironmentWrapper` - Wrapper for adding reward shaping and preprocessing
- `create_locomotion_reward_shaper()` - Helper for creating locomotion rewards

### `visualization.py`
- `MuJoCoViewer` - Real-time MuJoCo rendering wrapper
- `plot_training_curves()` - Plot training progress
- `plot_trajectory()` - Visualize episode trajectories
- `visualize_multi_agent()` - Multi-agent trajectory visualization

### `mujoco_utils.py`
- `load_custom_model()` - Load custom MJCF models from XML files
- `create_cloth_mesh()` - Generate cloth mesh MJCF XML
- `create_tendon_system()` - Generate tendon system MJCF XML
- `create_heightfield_terrain()` - Generate heightfield terrain MJCF XML
- `generate_procedural_heightmap()` - Generate procedural terrain heightmaps
- `apply_custom_forces()` - Apply custom physics forces

### `world_builder.py`
- `ObstacleCourseBuilder` - Procedurally generate obstacle courses
- `TerrainGenerator` - Generate procedural terrain (hills, noise)
- `WorldRandomizer` - Randomize world parameters
- `ObjectSpawner` - Spawn dynamic objects

## Custom MJCF Models

Example MJCF model files are provided in `models/`:
- `cloth_simple.xml` - Simple cloth mesh example
- `obstacle_course_simple.xml` - Obstacle course example
- `tendon_arm_simple.xml` - Tendon-driven arm example
- `terrain_hills.xml` - Procedural terrain with heightfield
- `underwater_env.xml` - Underwater environment setup
- `aerial_env.xml` - Aerial environment with reduced gravity
- `muscle_leg.xml` - Muscle-actuated leg example
- `soft_body_simple.xml` - Soft body with rigid object interaction

## Features

### Reward Shaping
All examples use reward shaping to guide learning:
- Forward velocity rewards
- Stability rewards (orientation maintenance)
- Energy penalties (action magnitude)
- Task-specific rewards (height, contact, manipulation, etc.)

### Observation Preprocessing
- Normalization
- Velocity information
- Feature extraction
- Multi-agent state aggregation

### Multi-Agent Coordination
- Distance computation
- Centroid calculation
- Separation, alignment, cohesion rewards
- Shared goal coordination

## Optimizers

Examples support multiple optimizers:
- **SGD** - Standard gradient descent
- **ES** - Evolution Strategies (VanillaESTrainer)
- **EGGROLL** - Low-rank ES (EGGROLLTrainer)

Specify optimizer type when creating trainers:
```python
trainer = PPOTrainer(
    env=env,
    policy=policy,
    optimizer_type="eggroll",  # or "sgd" or "es"
    ...
)
```

## Notes

- **Full cloth/tendon functionality** requires custom MJCF models with soft bodies/tendons
- **Skinned models** require mesh assets and proper skinning setup
- **Custom worlds** can be generated procedurally or loaded from MJCF files
- Examples use simplified physics where full MuJoCo integration needs custom models
- All examples are functional and ready to run with base Gymnasium environments

## Results

Training results (plots, trajectories) are saved to `examples/animals_3d/results/` directory.

## References

- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Gymnasium MuJoCo Environments](https://gymnasium.farama.org/environments/mujoco/)
- [EGGROLL Algorithm](https://eshyperscale.github.io/)

