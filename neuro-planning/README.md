# neuro-planning

Hierarchical planning and temporal abstraction for AGI systems.

## Overview

neuro-planning provides hierarchical planning capabilities including:

- **Goal Hierarchy**: MAXQ-style task decomposition with value function decomposition
- **Temporal Abstraction**: Options framework for multi-step action sequences
- **Skill Library**: Reusable skills with composition and transfer
- **Subgoal Discovery**: Automatic discovery of subgoals from trajectories
- **Planning Search**: Hierarchical MCTS with world model imagination

## Installation

```bash
cd neuro-planning
pip install -r requirements.txt
```

## Quick Start

```python
from neuro_planning import (
    PlanningIntegration,
    PlanningConfig,
    Goal,
    Skill,
    SkillType,
)
import numpy as np

# Create planning system
config = PlanningConfig(
    state_dim=64,
    action_dim=4,
    mcts_simulations=100,
)
planner = PlanningIntegration(config=config)

# Set up world model for imagination
planner.set_world_model(
    transition_fn=lambda s, a: s + 0.1 * a,
    reward_fn=lambda s, a, ns: -np.linalg.norm(ns - goal_state),
    terminal_fn=lambda s: np.linalg.norm(s - goal_state) < 0.1,
)

# Set goal with hierarchical decomposition
goal = Goal(name="navigate_to_target")
planner.set_goal(goal, decomposition={
    "navigate_to_target": ["plan_path", "execute_path"],
    "execute_path": ["move_forward", "avoid_obstacles"],
})

# Plan from current state
state = np.zeros(64)
goal_state = np.ones(64)
plan = planner.plan(state, goal_state=goal_state)

# Execute plan
while True:
    action = planner.get_action(state)
    if action is None:
        break
    # Execute action in environment
    # state = env.step(action)
```

## Components

### MAXQDecomposition

Hierarchical task decomposition based on MAXQ:

```python
from neuro_planning import MAXQDecomposition, Goal

decomp = MAXQDecomposition(discount=0.99)

root_goal = Goal(name="taxi_task")
tree = decomp.decompose(root_goal, {
    "taxi_task": ["get_passenger", "put_passenger"],
    "get_passenger": ["navigate", "pickup"],
    "put_passenger": ["navigate", "dropoff"],
})

# Select subtask using learned values
subtask = decomp.select_subtask("taxi_task", state, epsilon=0.1)

# Update values from experience
decomp.update_value("navigate", state, reward, next_state)
```

### OptionsFramework

Temporal abstraction with options (macro-actions):

```python
from neuro_planning import OptionsFramework

options = OptionsFramework(discount=0.99)

# Create option (initiation, policy, termination)
options.create_option(
    name="go_to_door",
    initiation_set=lambda s: not at_door(s),
    policy=lambda s: move_toward_door(s),
    termination=lambda s: at_door(s),
)

# Select and execute options
option_name = options.select_option(state, epsilon=0.1)
options.initiate_option(option_name, state)

while True:
    action, terminated = options.step_option(state)
    if terminated:
        break
```

### SkillLibrary

Reusable skills with composition and transfer:

```python
from neuro_planning import SkillLibrary, Skill, SkillType

library = SkillLibrary(embedding_dim=64)

# Register skills
skill = Skill(
    name="grasp_object",
    skill_type=SkillType.PRIMITIVE,
    preconditions=lambda s: object_reachable(s),
    effects=lambda s: holding_object(s),
    policy=lambda s, params: grasp_action(s),
    termination=lambda s, params: is_grasping(s),
)
library.register_skill(skill)

# Compose skills
library.compose_skills(["reach", "grasp", "lift"], "pick_up")

# Transfer to new domain
library.transfer_skill("grasp_object", "new_robot")

# Retrieve applicable skills
applicable = library.retrieve_applicable(state)
```

### SubgoalDiscovery

Automatic subgoal identification:

```python
from neuro_planning import SubgoalDiscovery, Trajectory

discovery = SubgoalDiscovery(state_dim=64)

# Collect trajectories
trajectories = [
    Trajectory(states=[...], actions=[...], rewards=[...], success=True)
    for _ in range(100)
]

# Discover subgoals
subgoals = discovery.discover_subgoals(
    trajectories,
    methods=["bottleneck", "termination", "cluster"],
)

# Verify utility
for sg in subgoals:
    utility = discovery.verify_subgoal_utility(sg, test_trajectories)
```

### HierarchicalMCTS

Planning search with imagination:

```python
from neuro_planning import HierarchicalMCTS, MCTSConfig, WorldModelAdapter

config = MCTSConfig(
    num_simulations=100,
    exploration_weight=1.41,
    max_depth=50,
)

world_model = WorldModelAdapter(
    transition_fn=predict_next_state,
    reward_fn=predict_reward,
    terminal_fn=is_terminal,
    uncertainty_fn=model_uncertainty,
)

mcts = HierarchicalMCTS(
    config=config,
    world_model=world_model,
    action_space=[0, 1, 2, 3],
    option_names=["navigate", "pickup"],
)

result = mcts.search(state, goal=goal_state)
print(f"Best actions: {result.actions}")
print(f"Expected value: {result.expected_value}")
```

## Integration Points

- **neuro-module-17-world-model**: Provides imagination for MCTS rollouts
- **neuro-module-13-executive**: Goal maintenance and executive control
- **neuro-module-06-motivation**: Dopamine reward signals

## Tests

```bash
python -m pytest tests/ -v
```

82 tests covering:
- Goal hierarchy and MAXQ decomposition
- Options framework and temporal abstraction
- Skill library composition and transfer
- Subgoal discovery methods
- MCTS planning search
- End-to-end integration

## License

MIT
