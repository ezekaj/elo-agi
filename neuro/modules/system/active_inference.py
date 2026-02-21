"""
Active Inference Controller: Central decision-making via Expected Free Energy.

Implements the active inference framework where action selection
minimizes Expected Free Energy (EFE), balancing:
- Epistemic value (information gain / curiosity)
- Pragmatic value (goal achievement / reward)

Based on:
- Friston's Active Inference framework
- Expected Free Energy (arXiv:2504.14898)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
import time

from .config import SystemConfig


@dataclass
class Policy:
    """A sequence of actions (a policy)."""
    actions: List[np.ndarray]
    efe: float = 0.0  # Expected Free Energy
    epistemic_value: float = 0.0  # Information gain
    pragmatic_value: float = 0.0  # Goal achievement
    probability: float = 0.0  # Policy probability after softmax

    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class EFEResult:
    """Result of EFE computation."""
    total_efe: float
    epistemic: float  # Ambiguity reduction
    pragmatic: float  # Risk/goal-related
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class BeliefState:
    """Current belief state for inference."""
    state: np.ndarray  # Current state estimate
    precision: np.ndarray  # Confidence in each state dimension
    goals: np.ndarray  # Desired/preferred states
    timestamp: float = field(default_factory=time.time)


class ActiveInferenceController:
    """
    Central controller using Active Inference.

    The controller:
    1. Maintains beliefs about the world state
    2. Evaluates policies by their Expected Free Energy
    3. Selects actions that minimize EFE
    4. Balances exploration (epistemic) and exploitation (pragmatic)

    EFE = Ambiguity + Risk
        = E[H[P(o|s,π)]] + E[D_KL[P(o|π) || P(o|C)]]

    Where:
    - Ambiguity: Expected uncertainty in observations
    - Risk: Divergence from preferred observations
    """

    def __init__(
        self,
        config: Optional[SystemConfig] = None,
        world_model: Optional[Any] = None,
    ):
        self.config = config or SystemConfig()
        self.world_model = world_model

        # Belief state
        self.belief = BeliefState(
            state=np.zeros(self.config.input_dim),
            precision=np.ones(self.config.input_dim),
            goals=np.zeros(self.config.input_dim),
        )

        # Policy space
        self._policies: List[Policy] = []
        self._current_policy: Optional[Policy] = None
        self._policy_step: int = 0

        # Statistics
        self._efe_history: List[EFEResult] = []
        self._action_count: int = 0

    def set_goals(self, goals: np.ndarray) -> None:
        """Set the preferred/goal states."""
        self.belief.goals = self._resize(goals, self.config.input_dim)

    def set_world_model(self, world_model: Any) -> None:
        """Set the world model for prediction."""
        self.world_model = world_model

    def update_belief(self, observation: np.ndarray) -> None:
        """
        Update belief state based on new observation.

        Implements Bayesian belief update with precision weighting.
        """
        obs = self._resize(observation, self.config.input_dim)

        # Prediction error
        error = obs - self.belief.state

        # Precision-weighted update
        learning_rate = self.config.learning_rate
        self.belief.state += learning_rate * self.belief.precision * error

        # Update precision based on error magnitude
        error_magnitude = np.abs(error)
        self.belief.precision = 1.0 / (1.0 + error_magnitude)

        self.belief.timestamp = time.time()

    def compute_efe(self, policy: Policy) -> EFEResult:
        """
        Compute Expected Free Energy for a policy.

        EFE = Epistemic Value + Pragmatic Value
            = -Information Gain - Expected Utility
        """
        epistemic = self._compute_epistemic_value(policy)
        pragmatic = self._compute_pragmatic_value(policy)

        # EFE is negative of value (we minimize EFE)
        total_efe = -epistemic - pragmatic

        result = EFEResult(
            total_efe=total_efe,
            epistemic=epistemic,
            pragmatic=pragmatic,
            components={
                'information_gain': epistemic,
                'goal_alignment': pragmatic,
            }
        )

        policy.efe = total_efe
        policy.epistemic_value = epistemic
        policy.pragmatic_value = pragmatic

        return result

    def _compute_epistemic_value(self, policy: Policy) -> float:
        """
        Compute epistemic value (information gain).

        Measures expected reduction in uncertainty about the world.
        """
        if not policy.actions:
            return 0.0

        # Predict states under policy
        predicted_states = self._predict_trajectory(policy)

        if not predicted_states:
            return 0.0

        # Information gain = reduction in entropy
        # Approximate by variance reduction
        current_entropy = np.sum(np.log(1.0 / (self.belief.precision + 1e-8)))

        # Expected entropy after observations
        expected_precisions = []
        for state in predicted_states:
            # Predicted precision improves with each observation
            pred_precision = self.belief.precision * 1.1  # Expect some improvement
            expected_precisions.append(pred_precision)

        if expected_precisions:
            mean_precision = np.mean(expected_precisions, axis=0)
            expected_entropy = np.sum(np.log(1.0 / (mean_precision + 1e-8)))
            info_gain = current_entropy - expected_entropy
        else:
            info_gain = 0.0

        # Scale by exploration weight
        return float(info_gain * self.config.exploration_weight)

    def _compute_pragmatic_value(self, policy: Policy) -> float:
        """
        Compute pragmatic value (goal achievement).

        Measures expected alignment with preferred states.
        """
        if not policy.actions:
            return 0.0

        # Predict final state under policy
        predicted_states = self._predict_trajectory(policy)

        if not predicted_states:
            # Use action directly as proxy
            final_state = policy.actions[-1]
        else:
            final_state = predicted_states[-1]

        # Goal alignment (negative KL divergence from preferences)
        # Simplified as cosine similarity to goals
        goal = self.belief.goals
        final = self._resize(final_state, len(goal))

        norm_goal = np.linalg.norm(goal)
        norm_final = np.linalg.norm(final)

        if norm_goal > 0 and norm_final > 0:
            alignment = np.dot(goal, final) / (norm_goal * norm_final)
        else:
            alignment = 0.0

        # Scale by exploitation weight (1 - exploration)
        return float(alignment * (1 - self.config.exploration_weight))

    def _predict_trajectory(self, policy: Policy) -> List[np.ndarray]:
        """Predict state trajectory under policy."""
        if self.world_model is None:
            return []

        states = []
        current = self.belief.state.copy()

        for action in policy.actions:
            try:
                # Use world model to predict next state
                if hasattr(self.world_model, 'predict'):
                    next_state = self.world_model.predict(current, action)
                elif hasattr(self.world_model, 'imagine'):
                    trajectory = self.world_model.imagine([action], horizon=1)
                    if trajectory:
                        next_state = trajectory[-1].next_state
                    else:
                        next_state = current + action * 0.1
                else:
                    # Simple linear prediction
                    next_state = current + action * 0.1

                states.append(next_state)
                current = next_state

            except Exception:
                # Fallback to simple prediction
                next_state = current + action * 0.1
                states.append(next_state)
                current = next_state

        return states

    def generate_policies(self, n_policies: int = 10) -> List[Policy]:
        """
        Generate candidate policies.

        Creates diverse action sequences for evaluation.
        """
        policies = []

        action_dim = self.config.output_dim
        horizon = self.config.efe_horizon

        for _ in range(n_policies):
            # Generate action sequence
            actions = []
            for _ in range(horizon):
                # Random action with goal bias
                random_action = np.random.randn(action_dim) * 0.5
                goal_direction = self.belief.goals[:action_dim] - self.belief.state[:action_dim]
                goal_direction = goal_direction / (np.linalg.norm(goal_direction) + 1e-8)

                # Mix random and goal-directed
                mix = np.random.rand()
                action = mix * random_action + (1 - mix) * goal_direction
                action = np.tanh(action)  # Bound actions

                actions.append(action)

            policy = Policy(actions=actions)
            policies.append(policy)

        self._policies = policies
        return policies

    def select_action(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Select action by minimizing Expected Free Energy.

        Args:
            state: Current observation (updates belief if provided)

        Returns:
            Selected action vector
        """
        # Update belief if observation provided
        if state is not None:
            self.update_belief(state)

        # Check if we're executing an existing policy
        if self._current_policy is not None and self._policy_step < len(self._current_policy):
            action = self._current_policy.actions[self._policy_step]
            self._policy_step += 1
            return action

        # Generate new policies
        policies = self.generate_policies(self.config.efe_samples)

        # Evaluate each policy
        for policy in policies:
            result = self.compute_efe(policy)
            self._efe_history.append(result)

        # Convert EFE to probabilities (softmax)
        efes = np.array([p.efe for p in policies])
        # Lower EFE = higher probability
        probs = self._softmax(-efes / self.config.temperature)

        for i, policy in enumerate(policies):
            policy.probability = probs[i]

        # Sample policy (or take argmin for deterministic)
        if np.random.rand() < 0.1:  # 10% exploration
            selected_idx = np.random.choice(len(policies))
        else:
            selected_idx = np.argmin(efes)

        self._current_policy = policies[selected_idx]
        self._policy_step = 1  # Return first action

        action = self._current_policy.actions[0]
        self._action_count += 1

        return action

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))  # Numerical stability
        return exp_x / (np.sum(exp_x) + 1e-8)

    def _resize(self, vec: np.ndarray, target_dim: int) -> np.ndarray:
        """Resize vector to target dimension."""
        if len(vec) == target_dim:
            return vec
        elif len(vec) < target_dim:
            return np.pad(vec, (0, target_dim - len(vec)))
        else:
            return vec[:target_dim]

    def get_current_policy(self) -> Optional[Policy]:
        """Get the currently executing policy."""
        return self._current_policy

    def get_belief_state(self) -> BeliefState:
        """Get current belief state."""
        return self.belief

    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics."""
        recent_efes = self._efe_history[-10:] if self._efe_history else []

        return {
            'action_count': self._action_count,
            'current_policy_step': self._policy_step,
            'n_policies_evaluated': len(self._policies),
            'exploration_weight': self.config.exploration_weight,
            'temperature': self.config.temperature,
            'mean_efe': float(np.mean([e.total_efe for e in recent_efes])) if recent_efes else 0.0,
            'mean_epistemic': float(np.mean([e.epistemic for e in recent_efes])) if recent_efes else 0.0,
            'mean_pragmatic': float(np.mean([e.pragmatic for e in recent_efes])) if recent_efes else 0.0,
        }

    def reset(self) -> None:
        """Reset controller state."""
        self.belief = BeliefState(
            state=np.zeros(self.config.input_dim),
            precision=np.ones(self.config.input_dim),
            goals=np.zeros(self.config.input_dim),
        )
        self._policies = []
        self._current_policy = None
        self._policy_step = 0
        self._efe_history = []
        self._action_count = 0
