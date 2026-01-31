"""
Importance-Weighted Experience Replay

Implements sophisticated replay mechanisms for continual learning:
- Prioritized replay based on TD error
- Task-balanced sampling
- Importance weighting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np


class ReplayStrategy(Enum):
    """Replay sampling strategies."""
    UNIFORM = "uniform"
    PRIORITIZED = "prioritized"
    TASK_BALANCED = "task_balanced"
    RESERVOIR = "reservoir"


@dataclass
class ReplayConfig:
    """Configuration for experience replay."""
    buffer_size: int = 10000
    strategy: ReplayStrategy = ReplayStrategy.PRIORITIZED
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_epsilon: float = 1e-6
    min_samples_per_task: int = 10


@dataclass
class Experience:
    """A single experience in the replay buffer."""
    state: np.ndarray
    action: Any
    reward: float
    next_state: np.ndarray
    done: bool
    task_id: str
    priority: float = 1.0
    td_error: float = 0.0
    timestamp: int = 0


class ImportanceWeightedReplay:
    """
    Experience replay with importance weighting.

    Features:
    - Prioritized sampling based on TD error
    - Task-balanced sampling across tasks
    - Importance sampling weights for unbiased updates
    """

    def __init__(
        self,
        config: Optional[ReplayConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or ReplayConfig()
        self._rng = np.random.default_rng(random_seed)

        self._buffer: List[Experience] = []
        self._task_indices: Dict[str, List[int]] = {}
        self._priorities: np.ndarray = np.array([])

        self._timestep = 0
        self._total_samples = 0

    def add(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        task_id: str,
        td_error: Optional[float] = None,
    ) -> int:
        """
        Add an experience to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
            task_id: Task identifier
            td_error: Optional TD error for priority

        Returns:
            Index of the added experience
        """
        if td_error is None:
            td_error = abs(reward) + self.config.priority_epsilon

        priority = self._compute_priority(td_error)

        exp = Experience(
            state=np.asarray(state),
            action=action,
            reward=reward,
            next_state=np.asarray(next_state),
            done=done,
            task_id=task_id,
            priority=priority,
            td_error=td_error,
            timestamp=self._timestep,
        )

        if len(self._buffer) < self.config.buffer_size:
            idx = len(self._buffer)
            self._buffer.append(exp)
            self._priorities = np.append(self._priorities, priority)
        else:
            idx = self._select_replacement_index()
            old_exp = self._buffer[idx]
            if old_exp.task_id in self._task_indices:
                if idx in self._task_indices[old_exp.task_id]:
                    self._task_indices[old_exp.task_id].remove(idx)

            self._buffer[idx] = exp
            self._priorities[idx] = priority

        if task_id not in self._task_indices:
            self._task_indices[task_id] = []
        self._task_indices[task_id].append(idx)

        self._timestep += 1
        self._total_samples += 1

        return idx

    def _compute_priority(self, td_error: float) -> float:
        """Compute priority from TD error."""
        return (abs(td_error) + self.config.priority_epsilon) ** self.config.priority_alpha

    def _select_replacement_index(self) -> int:
        """Select index to replace when buffer is full."""
        if self.config.strategy == ReplayStrategy.RESERVOIR:
            return self._rng.integers(0, len(self._buffer))

        min_priority_idx = np.argmin(self._priorities)
        return int(min_priority_idx)

    def compute_importance(
        self,
        experience: Experience,
        td_error: float,
    ) -> float:
        """
        Compute importance weight for an experience.

        Args:
            experience: The experience
            td_error: Current TD error

        Returns:
            Importance weight
        """
        priority = self._compute_priority(td_error)
        n = len(self._buffer)

        if n == 0:
            return 1.0

        prob = priority / (np.sum(self._priorities) + 1e-8)
        weight = (1.0 / (n * prob + 1e-8)) ** self.config.priority_beta

        max_weight = (1.0 / (n * (self.config.priority_epsilon ** self.config.priority_alpha) + 1e-8)) ** self.config.priority_beta
        normalized_weight = weight / (max_weight + 1e-8)

        return float(normalized_weight)

    def sample_batch(
        self,
        batch_size: int,
        task_balance: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Experience, float]]:
        """
        Sample a batch of experiences with importance weights.

        Args:
            batch_size: Number of experiences to sample
            task_balance: Optional dict mapping task_id to sampling weight

        Returns:
            List of (experience, importance_weight) tuples
        """
        if len(self._buffer) == 0:
            return []

        batch_size = min(batch_size, len(self._buffer))

        if self.config.strategy == ReplayStrategy.UNIFORM:
            indices = self._sample_uniform(batch_size)
        elif self.config.strategy == ReplayStrategy.PRIORITIZED:
            indices = self._sample_prioritized(batch_size)
        elif self.config.strategy == ReplayStrategy.TASK_BALANCED:
            indices = self._sample_task_balanced(batch_size, task_balance)
        else:
            indices = self._sample_uniform(batch_size)

        batch = []
        for idx in indices:
            exp = self._buffer[idx]
            weight = self.compute_importance(exp, exp.td_error)
            batch.append((exp, weight))

        return batch

    def _sample_uniform(self, batch_size: int) -> List[int]:
        """Sample uniformly from buffer."""
        return self._rng.choice(len(self._buffer), size=batch_size, replace=False).tolist()

    def _sample_prioritized(self, batch_size: int) -> List[int]:
        """Sample proportional to priorities."""
        probs = self._priorities / (np.sum(self._priorities) + 1e-8)
        indices = self._rng.choice(
            len(self._buffer),
            size=batch_size,
            replace=False,
            p=probs,
        )
        return indices.tolist()

    def _sample_task_balanced(
        self,
        batch_size: int,
        task_balance: Optional[Dict[str, float]],
    ) -> List[int]:
        """Sample with task balancing."""
        if task_balance is None:
            num_tasks = len(self._task_indices)
            if num_tasks == 0:
                return []
            task_balance = {tid: 1.0 / num_tasks for tid in self._task_indices}

        indices = []
        total_weight = sum(task_balance.get(tid, 0) for tid in self._task_indices)

        for task_id, weight in task_balance.items():
            if task_id not in self._task_indices:
                continue

            task_indices = self._task_indices[task_id]
            if not task_indices:
                continue

            task_count = int(batch_size * weight / (total_weight + 1e-8))
            task_count = min(task_count, len(task_indices))

            if task_count > 0:
                selected = self._rng.choice(task_indices, size=task_count, replace=False)
                indices.extend(selected.tolist())

        remaining = batch_size - len(indices)
        if remaining > 0 and len(self._buffer) > len(indices):
            available = [i for i in range(len(self._buffer)) if i not in indices]
            if available:
                extra = self._rng.choice(available, size=min(remaining, len(available)), replace=False)
                indices.extend(extra.tolist())

        return indices

    def update_priorities(
        self,
        indices: List[int],
        td_errors: List[float],
    ) -> None:
        """
        Update priorities after learning.

        Args:
            indices: Buffer indices to update
            td_errors: New TD errors
        """
        for idx, td_error in zip(indices, td_errors):
            if 0 <= idx < len(self._buffer):
                priority = self._compute_priority(td_error)
                self._priorities[idx] = priority
                self._buffer[idx].td_error = td_error
                self._buffer[idx].priority = priority

    def get_task_experiences(self, task_id: str) -> List[Experience]:
        """Get all experiences for a task."""
        if task_id not in self._task_indices:
            return []
        return [self._buffer[idx] for idx in self._task_indices[task_id]]

    def get_task_count(self, task_id: str) -> int:
        """Get number of experiences for a task."""
        return len(self._task_indices.get(task_id, []))

    def get_all_tasks(self) -> List[str]:
        """Get all task IDs in the buffer."""
        return list(self._task_indices.keys())

    def clear_task(self, task_id: str) -> int:
        """Clear all experiences for a task."""
        if task_id not in self._task_indices:
            return 0

        indices = set(self._task_indices[task_id])
        count = len(indices)

        new_buffer = []
        new_priorities = []
        new_task_indices: Dict[str, List[int]] = {}

        for i, exp in enumerate(self._buffer):
            if i not in indices:
                new_idx = len(new_buffer)
                new_buffer.append(exp)
                new_priorities.append(self._priorities[i])

                if exp.task_id not in new_task_indices:
                    new_task_indices[exp.task_id] = []
                new_task_indices[exp.task_id].append(new_idx)

        self._buffer = new_buffer
        self._priorities = np.array(new_priorities) if new_priorities else np.array([])
        self._task_indices = new_task_indices

        return count

    def __len__(self) -> int:
        """Return buffer size."""
        return len(self._buffer)

    def statistics(self) -> Dict[str, Any]:
        """Get replay buffer statistics."""
        task_counts = {tid: len(indices) for tid, indices in self._task_indices.items()}

        return {
            "buffer_size": len(self._buffer),
            "max_size": self.config.buffer_size,
            "num_tasks": len(self._task_indices),
            "total_samples_added": self._total_samples,
            "strategy": self.config.strategy.value,
            "task_counts": task_counts,
            "avg_priority": float(np.mean(self._priorities)) if len(self._priorities) > 0 else 0.0,
            "max_priority": float(np.max(self._priorities)) if len(self._priorities) > 0 else 0.0,
        }
