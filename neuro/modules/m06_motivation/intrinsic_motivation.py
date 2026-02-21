"""
Intrinsic Motivation: Action-State Path Entropy Maximization

Implements the fundamental insight that humans are NOT primarily reward-maximizers.
Instead, humans have an "irreducible desire to act and to move" - maximizing
action-state path entropy to expand possibilities.

Evidence for this view:
- Newborn motor babbling (purposeless movement without reward)
- Children choosing harder games over easy success
- Information seeking even when useless
- Post-satiation exploration (continuing after needs met)

Key properties of path entropy maximization:
- Stochastic behavior (complex rather than rigid patterns)
- Resource-seeking (enables future diversity)
- Risk avoidance (protects possibility space)
- Hierarchical goals (above temporary drives)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import scipy.stats as stats


class DriveType(Enum):
    """Types of intrinsic drives"""
    EXPLORATION = "exploration"      # Seek new states
    MASTERY = "mastery"              # Expand action repertoire
    AUTONOMY = "autonomy"            # Maintain control/options
    CHALLENGE = "challenge"          # Seek optimal difficulty


@dataclass
class ActionState:
    """A point in action-state space"""
    state: np.ndarray
    action: Optional[np.ndarray] = None
    timestamp: float = 0.0
    novelty: float = 0.0
    reachability: float = 1.0  # How accessible from current position


@dataclass
class PossibilityMetrics:
    """Metrics about the current possibility space"""
    volume: float                    # Size of reachable space
    diversity: float                 # Entropy of visited states
    expansion_rate: float            # How fast space is growing
    constraint_level: float          # How constrained we are
    future_options: int              # Estimated future possibilities


class PossibilitySpace:
    """Represents the space of possible action-state trajectories.

    The core insight: organisms seek to maximize the volume and diversity
    of their possibility space, not just immediate rewards.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_length: int = 1000,
        expansion_weight: float = 1.0
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_length = history_length
        self.expansion_weight = expansion_weight

        # Track visited states
        self.visited_states: deque = deque(maxlen=history_length)
        self.visited_actions: deque = deque(maxlen=history_length)

        # Estimate of reachable space boundaries
        self.state_min = np.full(state_dim, np.inf)
        self.state_max = np.full(state_dim, -np.inf)

        # Density estimation for diversity
        self.state_bins = 20
        self.state_histogram = np.zeros([self.state_bins] * min(state_dim, 3))

        # Current position
        self.current_state = np.zeros(state_dim)

    def observe(self, state: np.ndarray, action: Optional[np.ndarray] = None) -> None:
        """Record a visited state-action pair."""
        self.visited_states.append(state.copy())
        if action is not None:
            self.visited_actions.append(action.copy())

        # Update boundaries
        self.state_min = np.minimum(self.state_min, state)
        self.state_max = np.maximum(self.state_max, state)

        # Update histogram for diversity estimation
        self._update_histogram(state)

        self.current_state = state

    def _update_histogram(self, state: np.ndarray) -> None:
        """Update state space histogram for diversity estimation."""
        if len(self.visited_states) < 2:
            return

        # Normalize state to bin indices
        range_vec = self.state_max - self.state_min + 1e-8
        normalized = (state - self.state_min) / range_vec
        normalized = np.clip(normalized, 0, 0.999)

        # Use first 3 dimensions for histogram
        dims = min(self.state_dim, 3)
        indices = tuple((normalized[:dims] * self.state_bins).astype(int))

        try:
            self.state_histogram[indices] += 1
        except IndexError:
            pass

    def compute_volume(self) -> float:
        """Compute volume of explored possibility space."""
        if len(self.visited_states) < 2:
            return 0.0

        range_vec = self.state_max - self.state_min
        range_vec = np.maximum(range_vec, 1e-8)

        # Volume as product of ranges
        volume = np.prod(range_vec)

        # Scale by number of samples
        coverage = len(self.visited_states) / self.history_length

        return volume * coverage

    def compute_diversity(self) -> float:
        """Compute entropy of state distribution (diversity of experiences)."""
        if len(self.visited_states) < 10:
            return 0.0

        # Normalize histogram to probability distribution
        total = np.sum(self.state_histogram)
        if total == 0:
            return 0.0

        probs = self.state_histogram.flatten() / total
        probs = probs[probs > 0]  # Remove zeros for log

        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Normalize by max possible entropy
        max_entropy = np.log(len(probs))
        if max_entropy > 0:
            entropy /= max_entropy

        return entropy

    def compute_expansion_rate(self, window: int = 50) -> float:
        """Compute how fast the possibility space is expanding."""
        if len(self.visited_states) < window * 2:
            return 0.0

        states = list(self.visited_states)

        # Compare recent vs older volume
        old_states = np.array(states[-window*2:-window])
        new_states = np.array(states[-window:])

        old_volume = np.prod(np.ptp(old_states, axis=0) + 1e-8)
        new_volume = np.prod(np.ptp(new_states, axis=0) + 1e-8)

        expansion = (new_volume - old_volume) / (old_volume + 1e-8)
        return np.clip(expansion, -1.0, 1.0)

    def estimate_future_options(self, horizon: int = 10) -> int:
        """Estimate number of distinct states reachable in horizon steps."""
        if len(self.visited_states) < 20:
            return horizon

        # Estimate transition spread from history
        states = np.array(list(self.visited_states)[-100:])
        if len(states) < 2:
            return horizon

        transitions = np.diff(states, axis=0)
        transition_std = np.std(transitions, axis=0)

        # Estimated reachable volume grows with sqrt(horizon)
        reachable_radius = transition_std * np.sqrt(horizon)
        reachable_volume = np.prod(2 * reachable_radius + 1e-8)

        # Discretize to count of states
        cell_size = np.mean(transition_std) + 1e-8
        n_options = int(reachable_volume / (cell_size ** self.state_dim))

        return max(1, min(n_options, 10000))

    def get_metrics(self) -> PossibilityMetrics:
        """Get comprehensive metrics about possibility space."""
        return PossibilityMetrics(
            volume=self.compute_volume(),
            diversity=self.compute_diversity(),
            expansion_rate=self.compute_expansion_rate(),
            constraint_level=1.0 - self.compute_diversity(),
            future_options=self.estimate_future_options()
        )

    def compute_action_value(self, proposed_action: np.ndarray) -> float:
        """Compute intrinsic value of an action based on possibility expansion.

        Actions that expand the possibility space have higher value.
        """
        if len(self.visited_states) < 10:
            return 0.0

        # Estimate resulting state
        states = np.array(list(self.visited_states)[-20:])
        actions = list(self.visited_actions)[-20:] if self.visited_actions else None

        if actions and len(actions) >= 10:
            # Learn rough action->state_change mapping
            actions_arr = np.array(actions[-10:])
            state_changes = np.diff(states[-11:], axis=0)

            # Simple linear estimate
            mean_change = np.mean(state_changes, axis=0)
            predicted_state = self.current_state + mean_change
        else:
            predicted_state = self.current_state + np.random.randn(self.state_dim) * 0.1

        # Value = how much this expands possibility space
        current_diversity = self.compute_diversity()

        # Temporarily add predicted state
        self.visited_states.append(predicted_state)
        self._update_histogram(predicted_state)
        new_diversity = self.compute_diversity()

        # Remove temporary addition
        self.visited_states.pop()

        expansion_value = new_diversity - current_diversity
        novelty_value = self._compute_novelty(predicted_state)

        return self.expansion_weight * expansion_value + 0.5 * novelty_value

    def _compute_novelty(self, state: np.ndarray) -> float:
        """Compute novelty of a state based on distance from visited states."""
        if len(self.visited_states) < 5:
            return 1.0

        states = np.array(list(self.visited_states)[-100:])
        distances = np.linalg.norm(states - state, axis=1)

        min_dist = np.min(distances)
        mean_dist = np.mean(distances)

        # Novelty high when far from all visited states
        novelty = min_dist / (mean_dist + 1e-8)
        return np.clip(novelty, 0, 1)

    def reset(self) -> None:
        """Reset the possibility space tracking."""
        self.visited_states.clear()
        self.visited_actions.clear()
        self.state_min = np.full(self.state_dim, np.inf)
        self.state_max = np.full(self.state_dim, -np.inf)
        self.state_histogram = np.zeros_like(self.state_histogram)


class ActionDiversityTracker:
    """Tracks and encourages diversity in action selection.

    Implements the observation that organisms exhibit "stochastic behavior"
    with complex rather than rigid patterns.
    """

    def __init__(
        self,
        action_dim: int,
        history_length: int = 500,
        diversity_target: float = 0.7
    ):
        self.action_dim = action_dim
        self.history_length = history_length
        self.diversity_target = diversity_target

        self.action_history: deque = deque(maxlen=history_length)
        self.action_counts = {}  # Discretized action counts

        # For continuous actions
        self.n_bins = 10

    def record_action(self, action: np.ndarray) -> None:
        """Record an action taken."""
        self.action_history.append(action.copy())

        # Discretize for counting
        key = self._discretize(action)
        self.action_counts[key] = self.action_counts.get(key, 0) + 1

    def _discretize(self, action: np.ndarray) -> tuple:
        """Discretize continuous action for counting."""
        # Assume actions in [-1, 1] range
        bins = ((action + 1) / 2 * self.n_bins).astype(int)
        bins = np.clip(bins, 0, self.n_bins - 1)
        return tuple(bins)

    def compute_diversity(self) -> float:
        """Compute current action diversity (entropy)."""
        if len(self.action_history) < 10:
            return 1.0

        total = sum(self.action_counts.values())
        if total == 0:
            return 1.0

        probs = np.array(list(self.action_counts.values())) / total
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Normalize by max entropy
        max_entropy = np.log(float(self.n_bins ** self.action_dim))
        return entropy / max_entropy if max_entropy > 0 else 0.0

    def get_diversity_bonus(self, proposed_action: np.ndarray) -> float:
        """Get bonus for taking a less-frequently-used action."""
        if len(self.action_history) < 10:
            return 0.0

        key = self._discretize(proposed_action)
        count = self.action_counts.get(key, 0)
        total = len(self.action_history)

        # Bonus inversely proportional to how often we've taken this action
        frequency = count / total

        # High bonus for rare actions
        bonus = 1.0 - frequency

        return bonus

    def suggest_exploration_direction(self) -> np.ndarray:
        """Suggest an action direction that increases diversity."""
        if len(self.action_history) < 10:
            return np.random.randn(self.action_dim)

        # Find least-used action region
        min_count = float('inf')
        min_key = None

        # Sample random keys and find underexplored region
        for _ in range(20):
            random_action = np.random.rand(self.action_dim) * 2 - 1
            key = self._discretize(random_action)
            count = self.action_counts.get(key, 0)
            if count < min_count:
                min_count = count
                min_key = key

        if min_key is None:
            return np.random.randn(self.action_dim)

        # Convert back to continuous action
        exploration_target = (np.array(min_key) + 0.5) / self.n_bins * 2 - 1
        return exploration_target

    def reset(self) -> None:
        """Reset tracking."""
        self.action_history.clear()
        self.action_counts = {}


class IntrinsicDrive:
    """Represents a specific intrinsic drive (exploration, mastery, etc.).

    Each drive contributes to the overall motivation signal.
    """

    def __init__(
        self,
        drive_type: DriveType,
        base_strength: float = 1.0,
        satiation_rate: float = 0.1,
        recovery_rate: float = 0.05
    ):
        self.drive_type = drive_type
        self.base_strength = base_strength
        self.satiation_rate = satiation_rate
        self.recovery_rate = recovery_rate

        # Current drive level (0 = satiated, 1 = maximum drive)
        self.level = 0.5

        # History of satisfaction events
        self.satisfaction_history: List[float] = []

    def update(self, satisfaction: float, dt: float = 1.0) -> float:
        """Update drive level based on satisfaction received.

        Args:
            satisfaction: How much this drive was satisfied (0-1)
            dt: Time step

        Returns:
            New drive level
        """
        # Satisfaction reduces drive
        self.level -= self.satiation_rate * satisfaction * dt

        # Natural recovery (drive increases over time without satisfaction)
        self.level += self.recovery_rate * (1.0 - self.level) * dt

        self.level = np.clip(self.level, 0.0, 1.0)
        self.satisfaction_history.append(satisfaction)

        return self.level

    def get_motivation(self) -> float:
        """Get current motivation from this drive."""
        # Motivation is drive level scaled by base strength
        return self.level * self.base_strength

    def get_deprivation(self) -> float:
        """Get deprivation level (how long since satisfaction)."""
        if not self.satisfaction_history:
            return 1.0

        recent = self.satisfaction_history[-20:]
        mean_satisfaction = np.mean(recent)
        return 1.0 - mean_satisfaction


class PathEntropyMaximizer:
    """The core intrinsic motivation system.

    Implements the principle that organisms maximize action-state path entropy
    rather than cumulative reward. This explains:
    - Exploration without immediate reward
    - Preference for having options
    - Resource-seeking (enables future possibilities)
    - Risk avoidance (protects possibility space)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        entropy_weight: float = 1.0,
        diversity_weight: float = 0.5,
        risk_aversion: float = 0.3
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight
        self.risk_aversion = risk_aversion

        # Core components
        self.possibility_space = PossibilitySpace(state_dim, action_dim)
        self.action_tracker = ActionDiversityTracker(action_dim)

        # Intrinsic drives
        self.drives = {
            DriveType.EXPLORATION: IntrinsicDrive(DriveType.EXPLORATION, 1.0),
            DriveType.MASTERY: IntrinsicDrive(DriveType.MASTERY, 0.8),
            DriveType.AUTONOMY: IntrinsicDrive(DriveType.AUTONOMY, 1.2),
            DriveType.CHALLENGE: IntrinsicDrive(DriveType.CHALLENGE, 0.6),
        }

        # Track path entropy
        self.path_history: List[Tuple[np.ndarray, np.ndarray]] = []
        self.max_path_length = 1000

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        extrinsic_reward: float = 0.0
    ) -> None:
        """Observe a state-action transition."""
        self.possibility_space.observe(state, action)
        self.action_tracker.record_action(action)

        self.path_history.append((state.copy(), action.copy()))
        if len(self.path_history) > self.max_path_length:
            self.path_history.pop(0)

        # Update drives based on what was satisfied
        self._update_drives(state, action, extrinsic_reward)

    def _update_drives(
        self,
        state: np.ndarray,
        action: np.ndarray,
        extrinsic_reward: float
    ) -> None:
        """Update intrinsic drives based on transition."""
        # Exploration: satisfied by novelty
        novelty = self.possibility_space._compute_novelty(state)
        self.drives[DriveType.EXPLORATION].update(novelty)

        # Mastery: satisfied by action diversity
        diversity = self.action_tracker.compute_diversity()
        self.drives[DriveType.MASTERY].update(diversity)

        # Autonomy: satisfied by expansion of possibilities
        metrics = self.possibility_space.get_metrics()
        expansion = max(0, metrics.expansion_rate)
        self.drives[DriveType.AUTONOMY].update(expansion)

        # Challenge: satisfied by optimal difficulty (not too easy, not too hard)
        # Approximate with distance traveled
        if len(self.path_history) >= 2:
            distance = np.linalg.norm(state - self.path_history[-2][0])
            optimal_distance = 0.3  # Target difficulty
            challenge_satisfaction = 1.0 - abs(distance - optimal_distance) / optimal_distance
            challenge_satisfaction = np.clip(challenge_satisfaction, 0, 1)
        else:
            challenge_satisfaction = 0.5
        self.drives[DriveType.CHALLENGE].update(challenge_satisfaction)

    def compute_intrinsic_motivation(self) -> float:
        """Compute total intrinsic motivation level."""
        total = 0.0
        for drive in self.drives.values():
            total += drive.get_motivation()
        return total / len(self.drives)

    def compute_action_value(
        self,
        proposed_action: np.ndarray,
        expected_state: Optional[np.ndarray] = None
    ) -> float:
        """Compute intrinsic value of a proposed action.

        Value comes from:
        1. Path entropy expansion
        2. Action diversity
        3. Possibility space expansion
        4. Risk assessment
        """
        value = 0.0

        # Path entropy contribution
        path_entropy_value = self.possibility_space.compute_action_value(proposed_action)
        value += self.entropy_weight * path_entropy_value

        # Action diversity contribution
        diversity_bonus = self.action_tracker.get_diversity_bonus(proposed_action)
        value += self.diversity_weight * diversity_bonus

        # Drive-weighted contributions
        exploration_drive = self.drives[DriveType.EXPLORATION].get_motivation()
        value += 0.3 * exploration_drive * diversity_bonus

        # Risk penalty (avoid actions that might constrain future possibilities)
        if expected_state is not None:
            risk = self._assess_risk(expected_state)
            value -= self.risk_aversion * risk

        return value

    def _assess_risk(self, expected_state: np.ndarray) -> float:
        """Assess risk that a state constrains future possibilities."""
        metrics = self.possibility_space.get_metrics()

        # Risk higher near boundaries of explored space
        if len(self.possibility_space.visited_states) < 10:
            return 0.0

        # Distance from center of explored space
        states = np.array(list(self.possibility_space.visited_states))
        center = np.mean(states, axis=0)
        max_dist = np.max(np.linalg.norm(states - center, axis=1))

        dist_from_center = np.linalg.norm(expected_state - center)
        relative_dist = dist_from_center / (max_dist + 1e-8)

        # Risk high at edges (might lead to dead ends)
        if relative_dist > 0.9:
            return 0.5

        # Risk also high if space is already constrained
        if metrics.constraint_level > 0.7:
            return 0.3

        return 0.0

    def compute_path_entropy(self, horizon: int = 50) -> float:
        """Compute entropy of recent path through state-action space."""
        if len(self.path_history) < horizon:
            return 0.0

        recent_path = self.path_history[-horizon:]

        # State entropy
        states = np.array([p[0] for p in recent_path])
        state_cov = np.cov(states.T)
        if np.ndim(state_cov) == 0:
            state_cov = np.array([[state_cov]])
        state_entropy = 0.5 * np.log(np.linalg.det(state_cov + 1e-8 * np.eye(self.state_dim)) + 1e-10)

        # Action entropy
        actions = np.array([p[1] for p in recent_path])
        action_cov = np.cov(actions.T)
        if np.ndim(action_cov) == 0:
            action_cov = np.array([[action_cov]])
        action_entropy = 0.5 * np.log(np.linalg.det(action_cov + 1e-8 * np.eye(self.action_dim)) + 1e-10)

        return state_entropy + action_entropy

    def suggest_action(self, current_state: np.ndarray) -> np.ndarray:
        """Suggest an action that maximizes intrinsic motivation."""
        # Sample candidate actions
        n_candidates = 20
        candidates = [np.random.randn(self.action_dim) * 0.5 for _ in range(n_candidates)]

        # Add diversity-seeking direction
        exploration_dir = self.action_tracker.suggest_exploration_direction()
        candidates.append(exploration_dir)

        # Evaluate each
        best_action = candidates[0]
        best_value = float('-inf')

        for action in candidates:
            # Clip to valid range
            action = np.clip(action, -1, 1)
            value = self.compute_action_value(action)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def get_drive_levels(self) -> Dict[DriveType, float]:
        """Get current level of each intrinsic drive."""
        return {dt: drive.level for dt, drive in self.drives.items()}

    def get_metrics(self) -> Dict[str, float]:
        """Get comprehensive metrics about motivation state."""
        metrics = self.possibility_space.get_metrics()
        return {
            "intrinsic_motivation": self.compute_intrinsic_motivation(),
            "path_entropy": self.compute_path_entropy(),
            "possibility_volume": metrics.volume,
            "state_diversity": metrics.diversity,
            "action_diversity": self.action_tracker.compute_diversity(),
            "expansion_rate": metrics.expansion_rate,
            "future_options": metrics.future_options,
            "exploration_drive": self.drives[DriveType.EXPLORATION].level,
            "mastery_drive": self.drives[DriveType.MASTERY].level,
            "autonomy_drive": self.drives[DriveType.AUTONOMY].level,
            "challenge_drive": self.drives[DriveType.CHALLENGE].level,
        }

    def reset(self) -> None:
        """Reset the motivation system."""
        self.possibility_space.reset()
        self.action_tracker.reset()
        self.path_history = []
        for drive in self.drives.values():
            drive.level = 0.5
            drive.satisfaction_history = []
