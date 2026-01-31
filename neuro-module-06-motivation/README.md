# Module 6: Motivation - Why Humans Act

Implementation of the motivation systems that drive behavior, based on neuroscience research.

## Core Insight

Humans are **NOT** primarily reward-maximizers. Instead, they have an "irreducible desire to act and to move" - maximizing **action-state path entropy** to expand possibilities.

Evidence:
- Newborn motor babbling (purposeless movement without reward)
- Children choosing harder games over easy success
- Information seeking even when useless
- Post-satiation exploration (continuing after needs met)

## Components

### 1. Intrinsic Motivation (`src/intrinsic_motivation.py`)

Path entropy maximization system with multiple intrinsic drives.

```python
from src import PathEntropyMaximizer, DriveType

maximizer = PathEntropyMaximizer(state_dim=4, action_dim=2)

# Observe state-action transitions
maximizer.observe(state, action, extrinsic_reward=0.0)

# Get intrinsic motivation level
motivation = maximizer.compute_intrinsic_motivation()

# Compute intrinsic value of an action
value = maximizer.compute_action_value(proposed_action)

# Get action that maximizes path entropy
suggested = maximizer.suggest_action(current_state)

# Check individual drives
drives = maximizer.get_drive_levels()
# {DriveType.EXPLORATION: 0.7, DriveType.MASTERY: 0.5, ...}
```

### 2. Dopamine System (`src/dopamine_system.py`)

Dopamine does NOT signal pleasure. It signals:
- **Prediction Error**: Surprise (actual - expected)
- **Incentive Salience**: "Wanting" (not "liking")
- **Benefit/Cost Ratio**: Potentiates sensitivity to benefits vs costs

```python
from src import DopamineSystem

dopamine = DopamineSystem(state_dim=4, cue_dim=3)

# Process a transition
signal = dopamine.process_transition(
    state=current_state,
    action=action_taken,
    reward=reward_received,
    next_state=resulting_state,
    cue=reward_cue  # Optional
)

print(signal.prediction_error)      # RPE: +ve = better than expected
print(signal.incentive_salience)    # How much we "want" the cue
print(signal.benefit_cost_ratio)    # Net value assessment

# Get motivation metrics
motivation = dopamine.get_motivation_level()
exploration = dopamine.get_exploration_bonus()
```

### 3. Curiosity Drive (`src/curiosity_drive.py`)

Neural mechanisms of curiosity and information-seeking.

```python
from src import CuriosityModule

curiosity = CuriosityModule(state_dim=4)

# Process a stimulus
result = curiosity.process_stimulus(
    stimulus=sensory_input,
    action=action_taken
)

print(result['novelty'])          # How novel is this?
print(result['info_value'])       # Intrinsic value of information
print(result['memory_strength'])  # How strongly to encode in memory

# Decide whether to explore
should_explore = curiosity.should_explore(
    dopamine_level=0.6,
    uncertainty=0.4
)

# Get exploration bonus for a state
bonus = curiosity.get_exploration_bonus(potential_state)
```

### 4. Homeostatic Regulation (`src/homeostatic_regulation.py`)

State-dependent valuation where value depends on internal needs.

```python
from src import HomeostaticState, NeedBasedValuation, NeedType

state = HomeostaticState()

# Update over time (natural depletion)
state.update(dt=1.0)

# Consume resources
satisfaction = state.consume_resource(NeedType.ENERGY, amount=0.5)

# Get drives
drives = state.get_all_drives()
dominant_need, urgency = state.get_dominant_drive()

# Value depends on need state
valuation = NeedBasedValuation(state)
food_value = valuation.compute_value('food', amount=1.0)
# Value is HIGH when hungry, LOW when satiated
```

### 5. Effort Valuation (`src/effort_valuation.py`)

Models effort as cost AND potential value-add.

```python
from src import EffortCostModel, ParadoxicalEffort, MotivationalTransform

# Basic effort cost
cost_model = EffortCostModel()
cost = cost_model.compute_cost(effort_profile, context={'deadline_pressure': 0.8})

# Paradoxical effort value
paradox = ParadoxicalEffort()
value_added = paradox.compute_effort_value(
    effort_expended=0.7,
    task_difficulty=0.6,
    skill_level=0.5,
    success=True
)
# Can be POSITIVE (effort adds value via effort justification)

# Motivation transforms effort experience
transform = MotivationalTransform()
transform.set_context(deadline=0.9, importance=0.8, autonomy=0.7)
transformed_cost = transform.transform_effort_cost(raw_cost)
# High motivation makes effort feel less costly
```

## Key Properties

### Path Entropy Maximization
- **Stochastic behavior**: Complex rather than rigid patterns
- **Resource-seeking**: Enables future diversity
- **Risk avoidance**: Protects possibility space
- **Hierarchical goals**: Above temporary drives

### Dopamine Functions
| Signal | Function |
|--------|----------|
| Prediction error | Surprise (learning signal) |
| Incentive salience | "Wanting" (motivational pull) |
| Benefit/cost ratio | Decision weighting |

### Motivation's Dual Effects
Motivation can both **enhance** and **impair** decision-making:
- **Enhance**: Faster goal-directed responses under reward
- **Impair**: Over-reliance on habitual responses
- **Transform**: Effort becomes valued (not costly) under pressure

## Installation

```bash
cd neuro-module-06-motivation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Testing

```bash
pytest tests/ -v
```

## References

- Intrinsic motivation: https://arxiv.org/html/2601.10276
- Dopamine system: https://pmc.ncbi.nlm.nih.gov/articles/PMC11602011/
- Curiosity: https://www.sciencedirect.com/science/article/pii/S0166223623002400
