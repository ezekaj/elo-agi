"""
Subgoal Discovery

Automatic discovery of subgoals from trajectories using:
- Bottleneck state detection
- Option termination identification
- State abstraction clustering
- Graph-based subgoal mining
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum
import numpy as np
from collections import defaultdict


class SubgoalType(Enum):
    """Types of discovered subgoals."""

    BOTTLENECK = "bottleneck"
    LANDMARK = "landmark"
    OPTION_TERMINATION = "option_termination"
    STATE_CLUSTER = "state_cluster"
    REWARD_PEAK = "reward_peak"


@dataclass
class Subgoal:
    """A discovered subgoal."""

    name: str
    subgoal_type: SubgoalType
    state_representation: np.ndarray
    utility: float = 0.0
    reachability: float = 0.0
    frequency: int = 0
    associated_states: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def matches(self, state: np.ndarray, threshold: float = 0.9) -> bool:
        """Check if a state matches this subgoal."""
        norm1 = np.linalg.norm(self.state_representation)
        norm2 = np.linalg.norm(state)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return False

        similarity = float(np.dot(self.state_representation, state) / (norm1 * norm2))
        return similarity >= threshold

    def update_statistics(self, visited: bool, reached_goal: bool) -> None:
        """Update subgoal statistics after trajectory."""
        if visited:
            self.frequency += 1
            if reached_goal:
                self.utility = self.utility * 0.9 + 0.1


@dataclass
class Trajectory:
    """A trajectory of states, actions, and rewards."""

    states: List[np.ndarray]
    actions: List[Any]
    rewards: List[float]
    terminal: bool = False
    success: bool = False

    @property
    def length(self) -> int:
        return len(self.states)

    def get_returns(self, discount: float = 0.99) -> List[float]:
        """Compute discounted returns from each state."""
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + discount * G
            returns.insert(0, G)
        return returns


class BottleneckDetector:
    """
    Detects bottleneck states in trajectories.

    Bottlenecks are states that many successful trajectories pass through,
    making them natural subgoal candidates.
    """

    def __init__(
        self,
        state_dim: int = 64,
        similarity_threshold: float = 0.8,
        min_frequency: int = 3,
        random_seed: Optional[int] = None,
    ):
        self.state_dim = state_dim
        self.similarity_threshold = similarity_threshold
        self.min_frequency = min_frequency
        self._rng = np.random.default_rng(random_seed)

        self._state_visits: Dict[str, int] = defaultdict(int)
        self._state_centroids: Dict[str, np.ndarray] = {}
        self._success_visits: Dict[str, int] = defaultdict(int)

    def _state_to_key(self, state: np.ndarray) -> str:
        """Convert state to hashable key via discretization."""
        discretized = np.round(state * 10).astype(int)
        return str(discretized.tobytes())

    def _find_or_create_centroid(self, state: np.ndarray) -> str:
        """Find matching centroid or create new one."""
        state_norm = np.linalg.norm(state)
        if state_norm < 1e-8:
            key = "zero"
            if key not in self._state_centroids:
                self._state_centroids[key] = np.zeros(len(state))
            return key

        normalized = state / state_norm

        for key, centroid in self._state_centroids.items():
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-8:
                similarity = float(np.dot(normalized, centroid / centroid_norm))
                if similarity >= self.similarity_threshold:
                    n = self._state_visits[key]
                    self._state_centroids[key] = (centroid * n + state) / (n + 1)
                    return key

        key = self._state_to_key(state)
        self._state_centroids[key] = state.copy()
        return key

    def process_trajectory(self, trajectory: Trajectory) -> None:
        """Process a trajectory to update visit counts."""
        for state in trajectory.states:
            key = self._find_or_create_centroid(state)
            self._state_visits[key] += 1

            if trajectory.success:
                self._success_visits[key] += 1

    def identify_bottlenecks(
        self,
        trajectories: List[Trajectory],
        top_k: int = 10,
    ) -> List[Subgoal]:
        """
        Identify bottleneck states from trajectories.

        Returns list of subgoals ranked by bottleneck score.
        """
        for traj in trajectories:
            self.process_trajectory(traj)

        successful_trajs = [t for t in trajectories if t.success]
        total_successful = len(successful_trajs)

        bottlenecks = []

        for key, visits in self._state_visits.items():
            if visits < self.min_frequency:
                continue

            success_visits = self._success_visits[key]

            if total_successful > 0:
                success_rate = success_visits / total_successful
            else:
                success_rate = 0.0

            all_rate = visits / len(trajectories) if trajectories else 0.0
            bottleneck_score = success_rate * np.log1p(visits)

            centroid = self._state_centroids.get(key, np.zeros(self.state_dim))

            subgoal = Subgoal(
                name=f"bottleneck_{len(bottlenecks)}",
                subgoal_type=SubgoalType.BOTTLENECK,
                state_representation=centroid,
                utility=bottleneck_score,
                reachability=all_rate,
                frequency=visits,
                metadata={
                    "success_rate": success_rate,
                    "key": key,
                },
            )
            bottlenecks.append(subgoal)

        bottlenecks.sort(key=lambda s: s.utility, reverse=True)
        return bottlenecks[:top_k]

    def statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "unique_states": len(self._state_visits),
            "total_visits": sum(self._state_visits.values()),
            "success_states": len(self._success_visits),
        }


class OptionTerminationDetector:
    """
    Identifies natural option termination points.

    Looks for states where:
    - Value function has local maxima
    - Action distribution changes significantly
    - Trajectory curvature is high
    """

    def __init__(
        self,
        state_dim: int = 64,
        value_threshold: float = 0.1,
        action_change_threshold: float = 0.5,
        random_seed: Optional[int] = None,
    ):
        self.state_dim = state_dim
        self.value_threshold = value_threshold
        self.action_change_threshold = action_change_threshold
        self._rng = np.random.default_rng(random_seed)

        self._termination_candidates: List[Tuple[np.ndarray, float]] = []

    def identify_value_peaks(
        self,
        trajectory: Trajectory,
        discount: float = 0.99,
    ) -> List[Tuple[int, float]]:
        """
        Find states with local value maxima.

        Returns list of (index, value) tuples.
        """
        returns = trajectory.get_returns(discount)

        if len(returns) < 3:
            return []

        peaks = []
        for i in range(1, len(returns) - 1):
            if returns[i] > returns[i - 1] + self.value_threshold:
                if returns[i] > returns[i + 1] + self.value_threshold:
                    peaks.append((i, returns[i]))

        return peaks

    def identify_action_changes(
        self,
        trajectory: Trajectory,
        action_encoder: Optional[Callable[[Any], np.ndarray]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Find states where action distribution changes significantly.

        Returns list of (index, change_magnitude) tuples.
        """
        if action_encoder is None:
            return []

        if len(trajectory.actions) < 3:
            return []

        changes = []
        prev_action_enc = action_encoder(trajectory.actions[0])

        for i in range(1, len(trajectory.actions) - 1):
            curr_action_enc = action_encoder(trajectory.actions[i])
            next_action_enc = action_encoder(trajectory.actions[i + 1])

            prev_change = np.linalg.norm(curr_action_enc - prev_action_enc)
            np.linalg.norm(next_action_enc - curr_action_enc)

            if prev_change > self.action_change_threshold:
                changes.append((i, prev_change))

            prev_action_enc = curr_action_enc

        return changes

    def identify_option_terminations(
        self,
        trajectories: List[Trajectory],
        discount: float = 0.99,
        action_encoder: Optional[Callable[[Any], np.ndarray]] = None,
    ) -> List[Subgoal]:
        """
        Identify natural option termination points.

        Returns list of subgoals at termination points.
        """
        termination_states = defaultdict(list)

        for traj in trajectories:
            value_peaks = self.identify_value_peaks(traj, discount)
            for idx, value in value_peaks:
                state = traj.states[idx]
                key = str(np.round(state * 5).astype(int).tobytes())
                termination_states[key].append((state, value, "value_peak"))

            if action_encoder:
                action_changes = self.identify_action_changes(traj, action_encoder)
                for idx, magnitude in action_changes:
                    state = traj.states[idx]
                    key = str(np.round(state * 5).astype(int).tobytes())
                    termination_states[key].append((state, magnitude, "action_change"))

        subgoals = []
        for key, observations in termination_states.items():
            if len(observations) < 2:
                continue

            states = [o[0] for o in observations]
            scores = [o[1] for o in observations]
            types = [o[2] for o in observations]

            centroid = np.mean(states, axis=0)
            avg_score = np.mean(scores)

            subgoal = Subgoal(
                name=f"termination_{len(subgoals)}",
                subgoal_type=SubgoalType.OPTION_TERMINATION,
                state_representation=centroid,
                utility=avg_score,
                frequency=len(observations),
                metadata={
                    "detection_types": list(set(types)),
                },
            )
            subgoals.append(subgoal)

        subgoals.sort(key=lambda s: s.utility * s.frequency, reverse=True)
        return subgoals

    def statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        return {
            "candidates_tracked": len(self._termination_candidates),
        }


class SubgoalDiscovery:
    """
    Main subgoal discovery system combining multiple methods.

    Integrates bottleneck detection, option termination,
    and other subgoal discovery mechanisms.
    """

    def __init__(
        self,
        state_dim: int = 64,
        random_seed: Optional[int] = None,
    ):
        self.state_dim = state_dim
        self._rng = np.random.default_rng(random_seed)

        self._bottleneck_detector = BottleneckDetector(
            state_dim=state_dim,
            random_seed=random_seed,
        )
        self._termination_detector = OptionTerminationDetector(
            state_dim=state_dim,
            random_seed=random_seed,
        )

        self._discovered_subgoals: Dict[str, Subgoal] = {}
        self._trajectories_processed = 0

    def discover_subgoals(
        self,
        trajectories: List[Trajectory],
        methods: Optional[List[str]] = None,
        top_k_per_method: int = 5,
    ) -> List[Subgoal]:
        """
        Discover subgoals from trajectories using multiple methods.

        Args:
            trajectories: List of trajectories to analyze
            methods: List of methods to use ("bottleneck", "termination", "cluster")
            top_k_per_method: Max subgoals to return per method

        Returns:
            List of discovered subgoals
        """
        if methods is None:
            methods = ["bottleneck", "termination"]

        all_subgoals = []

        if "bottleneck" in methods:
            bottlenecks = self._bottleneck_detector.identify_bottlenecks(
                trajectories, top_k=top_k_per_method
            )
            all_subgoals.extend(bottlenecks)

        if "termination" in methods:
            terminations = self._termination_detector.identify_option_terminations(trajectories)
            all_subgoals.extend(terminations[:top_k_per_method])

        if "cluster" in methods:
            cluster_subgoals = self._discover_by_clustering(trajectories, top_k_per_method)
            all_subgoals.extend(cluster_subgoals)

        for subgoal in all_subgoals:
            if subgoal.name not in self._discovered_subgoals:
                self._discovered_subgoals[subgoal.name] = subgoal

        self._trajectories_processed += len(trajectories)

        return all_subgoals

    def _discover_by_clustering(
        self,
        trajectories: List[Trajectory],
        top_k: int,
    ) -> List[Subgoal]:
        """Discover subgoals via state clustering."""
        all_states = []
        for traj in trajectories:
            all_states.extend(traj.states)

        if len(all_states) < top_k:
            return []

        states_array = np.array(all_states)
        n_samples = min(1000, len(states_array))
        indices = self._rng.choice(len(states_array), n_samples, replace=False)
        sampled = states_array[indices]

        centroids = sampled[self._rng.choice(n_samples, top_k, replace=False)]

        for _ in range(10):
            distances = np.zeros((n_samples, top_k))
            for i, centroid in enumerate(centroids):
                distances[:, i] = np.linalg.norm(sampled - centroid, axis=1)

            assignments = np.argmin(distances, axis=1)

            new_centroids = np.zeros_like(centroids)
            for i in range(top_k):
                mask = assignments == i
                if np.any(mask):
                    new_centroids[i] = sampled[mask].mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        subgoals = []
        for i, centroid in enumerate(centroids):
            mask = assignments == i
            cluster_size = np.sum(mask)

            subgoal = Subgoal(
                name=f"cluster_{i}",
                subgoal_type=SubgoalType.STATE_CLUSTER,
                state_representation=centroid,
                utility=cluster_size / n_samples,
                frequency=int(cluster_size),
            )
            subgoals.append(subgoal)

        return subgoals

    def verify_subgoal_utility(
        self,
        subgoal: Subgoal,
        task_distribution: List[Trajectory],
        threshold: float = 0.5,
    ) -> float:
        """
        Verify a subgoal's utility on a task distribution.

        Returns utility score [0, 1].
        """
        successful_with_subgoal = 0
        successful_without_subgoal = 0
        total_with = 0
        total_without = 0

        for traj in task_distribution:
            visited_subgoal = any(subgoal.matches(state) for state in traj.states)

            if visited_subgoal:
                total_with += 1
                if traj.success:
                    successful_with_subgoal += 1
            else:
                total_without += 1
                if traj.success:
                    successful_without_subgoal += 1

        rate_with = successful_with_subgoal / total_with if total_with > 0 else 0
        rate_without = successful_without_subgoal / total_without if total_without > 0 else 0

        utility = rate_with - rate_without

        subgoal.utility = max(0, utility)
        return utility

    def get_subgoal(self, name: str) -> Optional[Subgoal]:
        """Get a discovered subgoal by name."""
        return self._discovered_subgoals.get(name)

    def list_subgoals(
        self,
        subgoal_type: Optional[SubgoalType] = None,
        min_utility: float = 0.0,
    ) -> List[Subgoal]:
        """List discovered subgoals with optional filters."""
        result = []

        for subgoal in self._discovered_subgoals.values():
            if subgoal_type and subgoal.subgoal_type != subgoal_type:
                continue
            if subgoal.utility < min_utility:
                continue
            result.append(subgoal)

        result.sort(key=lambda s: s.utility, reverse=True)
        return result

    def prune_subgoals(self, min_utility: float = 0.1, min_frequency: int = 2) -> int:
        """
        Remove low-utility subgoals.

        Returns number of subgoals removed.
        """
        to_remove = []

        for name, subgoal in self._discovered_subgoals.items():
            if subgoal.utility < min_utility or subgoal.frequency < min_frequency:
                to_remove.append(name)

        for name in to_remove:
            del self._discovered_subgoals[name]

        return len(to_remove)

    def statistics(self) -> Dict[str, Any]:
        """Get discovery statistics."""
        type_counts = defaultdict(int)
        for sg in self._discovered_subgoals.values():
            type_counts[sg.subgoal_type.value] += 1

        return {
            "total_subgoals": len(self._discovered_subgoals),
            "by_type": dict(type_counts),
            "trajectories_processed": self._trajectories_processed,
            "bottleneck_stats": self._bottleneck_detector.statistics(),
            "termination_stats": self._termination_detector.statistics(),
        }
