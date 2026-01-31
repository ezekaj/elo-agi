"""
Experience Buffer: Stores experiences for learning.

Provides replay buffer functionality for:
- Experience replay
- Priority sampling
- Memory consolidation (Module 05 integration)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Iterator
import numpy as np
from collections import deque
import time


@dataclass
class Experience:
    """A single experience tuple."""
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: float = 1.0  # For prioritized replay


@dataclass
class Episode:
    """A complete episode of experiences."""
    experiences: List[Experience] = field(default_factory=list)
    total_reward: float = 0.0
    length: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def add(self, exp: Experience) -> None:
        """Add experience to episode."""
        self.experiences.append(exp)
        self.total_reward += exp.reward
        self.length += 1

    def finalize(self) -> None:
        """Mark episode as complete."""
        self.end_time = time.time()


class ExperienceBuffer:
    """
    Replay buffer for storing and sampling experiences.

    Features:
    - Fixed-size circular buffer
    - Uniform and prioritized sampling
    - Episode-based access
    - TD-error based priority updates
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,  # Importance sampling exponent
        beta_increment: float = 0.001,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self._buffer: deque = deque(maxlen=capacity)
        self._priorities: deque = deque(maxlen=capacity)
        self._episodes: List[Episode] = []
        self._current_episode: Optional[Episode] = None

        self._total_added = 0
        self._max_priority = 1.0

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a new experience to the buffer."""
        exp = Experience(
            observation=observation.copy(),
            action=action.copy(),
            reward=reward,
            next_observation=next_observation.copy(),
            done=done,
            info=info or {},
            priority=self._max_priority,
        )

        self._buffer.append(exp)
        self._priorities.append(self._max_priority ** self.alpha)
        self._total_added += 1

        # Track episode
        if self._current_episode is None:
            self._current_episode = Episode()

        self._current_episode.add(exp)

        if done:
            self._current_episode.finalize()
            self._episodes.append(self._current_episode)
            self._current_episode = None

            # Limit episode storage
            if len(self._episodes) > 1000:
                self._episodes = self._episodes[-500:]

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample a batch of experiences.

        Returns:
            Tuple of (experiences, indices, importance weights)
        """
        if len(self._buffer) < batch_size:
            batch_size = len(self._buffer)

        # Prioritized sampling
        priorities = np.array(self._priorities)
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self._buffer), size=batch_size, p=probs, replace=False)
        experiences = [self._buffer[i] for i in indices]

        # Compute importance sampling weights
        weights = (len(self._buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return experiences, indices, weights

    def sample_uniform(self, batch_size: int) -> List[Experience]:
        """Sample uniformly (without prioritization)."""
        if len(self._buffer) < batch_size:
            batch_size = len(self._buffer)

        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[i] for i in indices]

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self._priorities):
                self._priorities[idx] = (priority + 1e-6) ** self.alpha
                self._max_priority = max(self._max_priority, priority)

    def sample_episode(self) -> Optional[Episode]:
        """Sample a complete episode."""
        if not self._episodes:
            return None
        return np.random.choice(self._episodes)

    def sample_recent(self, n: int) -> List[Experience]:
        """Sample n most recent experiences."""
        n = min(n, len(self._buffer))
        return [self._buffer[-i - 1] for i in range(n)]

    def get_episode(self, idx: int) -> Optional[Episode]:
        """Get episode by index."""
        if 0 <= idx < len(self._episodes):
            return self._episodes[idx]
        return None

    def get_recent_episodes(self, n: int) -> List[Episode]:
        """Get n most recent complete episodes."""
        return self._episodes[-n:] if self._episodes else []

    def clear(self) -> None:
        """Clear all experiences."""
        self._buffer.clear()
        self._priorities.clear()
        self._episodes.clear()
        self._current_episode = None
        self._max_priority = 1.0

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self) -> Iterator[Experience]:
        return iter(self._buffer)

    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        rewards = [exp.reward for exp in self._buffer]
        episode_rewards = [ep.total_reward for ep in self._episodes]

        return {
            'size': len(self._buffer),
            'capacity': self.capacity,
            'total_added': self._total_added,
            'num_episodes': len(self._episodes),
            'current_episode_length': self._current_episode.length if self._current_episode else 0,
            'mean_reward': float(np.mean(rewards)) if rewards else 0.0,
            'std_reward': float(np.std(rewards)) if rewards else 0.0,
            'mean_episode_reward': float(np.mean(episode_rewards)) if episode_rewards else 0.0,
            'mean_episode_length': float(np.mean([ep.length for ep in self._episodes])) if self._episodes else 0.0,
            'beta': self.beta,
            'max_priority': self._max_priority,
        }


class SequenceBuffer(ExperienceBuffer):
    """
    Buffer optimized for sequence-based learning.

    Stores experiences as contiguous sequences for
    temporal learning (e.g., RNNs, transformers).
    """

    def __init__(
        self,
        capacity: int = 100000,
        sequence_length: int = 32,
        overlap: int = 8,
    ):
        super().__init__(capacity)
        self.sequence_length = sequence_length
        self.overlap = overlap

    def sample_sequences(self, batch_size: int) -> List[List[Experience]]:
        """Sample sequences of experiences."""
        sequences = []

        for _ in range(batch_size):
            # Sample start index
            max_start = len(self._buffer) - self.sequence_length
            if max_start <= 0:
                continue

            start = np.random.randint(0, max_start)
            seq = [self._buffer[start + i] for i in range(self.sequence_length)]

            # Check for episode boundaries
            valid = True
            for i in range(len(seq) - 1):
                if seq[i].done and i < len(seq) - 1:
                    valid = False
                    break

            if valid:
                sequences.append(seq)

        return sequences


class ConsolidationBuffer(ExperienceBuffer):
    """
    Buffer with memory consolidation for Module 05 integration.

    Supports:
    - Sleep-like consolidation phases
    - Important memory preservation
    - Forgetting of less relevant experiences
    """

    def __init__(
        self,
        capacity: int = 100000,
        consolidation_threshold: int = 10000,
        importance_decay: float = 0.99,
    ):
        super().__init__(capacity)
        self.consolidation_threshold = consolidation_threshold
        self.importance_decay = importance_decay
        self._consolidated: List[Experience] = []
        self._last_consolidation = 0

    def consolidate(self) -> Dict[str, Any]:
        """
        Run consolidation to identify important memories.

        Returns statistics about consolidation.
        """
        if len(self._buffer) < 100:
            return {'consolidated': 0, 'forgotten': 0}

        # Identify high-priority experiences
        high_priority = []
        low_priority_count = 0

        for exp in self._buffer:
            if exp.priority > np.median(list(self._priorities)):
                high_priority.append(exp)
            else:
                low_priority_count += 1

        # Move to consolidated storage
        self._consolidated.extend(high_priority[-100:])  # Keep top 100
        if len(self._consolidated) > 1000:
            self._consolidated = self._consolidated[-500:]

        # Decay priorities
        for i in range(len(self._priorities)):
            self._priorities[i] *= self.importance_decay

        self._last_consolidation = self._total_added

        return {
            'consolidated': len(high_priority),
            'forgotten': low_priority_count,
            'total_consolidated': len(self._consolidated),
        }

    def should_consolidate(self) -> bool:
        """Check if consolidation should run."""
        return (self._total_added - self._last_consolidation) >= self.consolidation_threshold

    def sample_consolidated(self, n: int) -> List[Experience]:
        """Sample from consolidated memories."""
        if not self._consolidated:
            return []
        n = min(n, len(self._consolidated))
        indices = np.random.choice(len(self._consolidated), size=n, replace=False)
        return [self._consolidated[i] for i in indices]
