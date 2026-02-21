"""
Task Inference for Continual Learning

Implements automatic task boundary detection and identification:
- Task change detection from state distributions
- Task ID inference from context
- Task embedding creation
- Similar task merging
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import numpy as np


class TaskChangeMethod(Enum):
    """Methods for detecting task changes."""
    DISTRIBUTION_SHIFT = "distribution_shift"
    PERFORMANCE_DROP = "performance_drop"
    CONTEXT_CHANGE = "context_change"
    REWARD_STRUCTURE = "reward_structure"


@dataclass
class TaskInferenceConfig:
    """Configuration for task inference."""
    change_threshold: float = 0.5
    embedding_dim: int = 64
    min_samples_for_task: int = 10
    similarity_threshold: float = 0.8
    use_multiple_detectors: bool = True
    ema_decay: float = 0.95
    # Number of standard deviations for anomaly detection (default: 3 sigma rule)
    anomaly_sigma_threshold: float = 3.0
    # Window size for distribution shift detection
    distribution_window_old: int = 50
    distribution_window_new: int = 10


@dataclass
class TaskInfo:
    """Information about a detected task."""
    task_id: str
    embedding: np.ndarray
    exemplars: List[np.ndarray]
    first_seen: int
    last_seen: int
    sample_count: int
    performance_history: List[float]


class TaskInference:
    """
    Detects task boundaries and identifies tasks in continual learning.

    Capabilities:
    - Detect when the task has changed
    - Infer task IDs from state observations
    - Create task embeddings for clustering
    - Merge similar tasks to reduce fragmentation
    """

    def __init__(
        self,
        config: Optional[TaskInferenceConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or TaskInferenceConfig()
        self._rng = np.random.default_rng(random_seed)

        self._tasks: Dict[str, TaskInfo] = {}
        self._current_task: Optional[str] = None
        self._task_counter = 0

        self._state_buffer: List[np.ndarray] = []
        self._state_statistics: Dict[str, Dict[str, float]] = {}

        self._timestep = 0
        self._task_changes = 0

    def detect_task_change(
        self,
        current_state: np.ndarray,
        recent_states: Optional[List[np.ndarray]] = None,
    ) -> bool:
        """
        Detect if a task change has occurred.

        Args:
            current_state: Current state observation
            recent_states: Optional list of recent states for context

        Returns:
            True if task change detected
        """
        current_state = np.asarray(current_state)

        self._state_buffer.append(current_state)
        if len(self._state_buffer) > 100:
            self._state_buffer = self._state_buffer[-100:]

        if self._current_task is None:
            task_id = self._create_new_task(current_state)
            self._current_task = task_id
            self._task_changes += 1
            return True

        if len(self._state_buffer) < self.config.min_samples_for_task:
            return False

        change_detected = False

        if self.config.use_multiple_detectors:
            dist_change = self._detect_distribution_shift(current_state)
            perf_change = self._detect_performance_drop()
            context_change = self._detect_context_change(current_state, recent_states)

            change_scores = [dist_change, perf_change, context_change]
            change_detected = np.mean(change_scores) > self.config.change_threshold
        else:
            change_detected = self._detect_distribution_shift(current_state) > self.config.change_threshold

        if change_detected:
            self._task_changes += 1

        self._timestep += 1
        return change_detected

    def _detect_distribution_shift(self, current_state: np.ndarray) -> float:
        """
        Detect distribution shift using KL divergence approximation.

        Uses Gaussian approximation KL divergence:
        KL(P||Q) ≈ log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²) / (2σ_q²) - 0.5

        Args:
            current_state: Current state observation

        Returns:
            Shift score in [0, 1]
        """
        window_new = self.config.distribution_window_new
        min_old_samples = self.config.distribution_window_old // 2

        if len(self._state_buffer) < min_old_samples + window_new:
            return 0.0

        old_states = np.array(self._state_buffer[:-window_new])
        new_states = np.array(self._state_buffer[-window_new:])

        old_mean = np.mean(old_states, axis=0)
        new_mean = np.mean(new_states, axis=0)

        old_std = np.std(old_states, axis=0) + 1e-8
        new_std = np.std(new_states, axis=0) + 1e-8

        # KL divergence for Gaussian: log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²)/(2σ_q²) - 0.5
        kl_approx = np.mean(
            np.log(new_std / old_std) +
            (old_std**2 + (old_mean - new_mean)**2) / (2 * new_std**2) - 0.5
        )

        # Use sigmoid instead of tanh for bounded [0,1] output
        shift_score = 1.0 / (1.0 + np.exp(-kl_approx))
        return float(np.clip(shift_score, 0, 1))

    def _detect_performance_drop(self) -> float:
        """Detect performance drop in current task."""
        if self._current_task is None:
            return 0.0

        task = self._tasks.get(self._current_task)
        if task is None or len(task.performance_history) < 5:
            return 0.0

        recent_perf = np.mean(task.performance_history[-5:])
        overall_perf = np.mean(task.performance_history)

        if overall_perf < 1e-8:
            return 0.0

        drop_ratio = (overall_perf - recent_perf) / (overall_perf + 1e-8)
        return float(np.clip(drop_ratio, 0, 1))

    def _detect_context_change(
        self,
        current_state: np.ndarray,
        recent_states: Optional[List[np.ndarray]],
    ) -> float:
        """
        Detect context change from state patterns using z-score anomaly detection.

        Uses the configurable sigma threshold (default 3-sigma rule) to determine
        what constitutes an anomalous state.

        Args:
            current_state: Current state observation
            recent_states: List of recent state observations

        Returns:
            Context change score in [0, 1]
        """
        if recent_states is None or len(recent_states) < 5:
            return 0.0

        recent = np.array(recent_states[-5:])
        recent_mean = np.mean(recent, axis=0)
        recent_std = np.std(recent, axis=0) + 1e-8

        # Z-score: how many standard deviations from mean
        current_diff = np.abs(current_state - recent_mean)
        z_scores = current_diff / recent_std

        # Average z-score normalized by sigma threshold
        # If avg z-score >= sigma_threshold, score = 1.0
        mean_z_score = np.mean(z_scores)
        anomaly_score = mean_z_score / self.config.anomaly_sigma_threshold
        return float(np.clip(anomaly_score, 0, 1))

    def infer_task_id(
        self,
        state: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Infer the task ID for a given state.

        Args:
            state: State observation
            context: Optional context information

        Returns:
            Inferred task ID
        """
        state = np.asarray(state)

        if not self._tasks:
            return self._create_new_task(state)

        best_task = None
        best_similarity = -1.0

        state_embedding = self._create_state_embedding(state)

        for task_id, task in self._tasks.items():
            similarity = self._compute_similarity(state_embedding, task.embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_task = task_id

        if best_similarity < self.config.similarity_threshold:
            return self._create_new_task(state)

        self._update_task(best_task, state)
        return best_task

    def _create_new_task(self, state: np.ndarray) -> str:
        """Create a new task from a state."""
        task_id = f"task_{self._task_counter}"
        self._task_counter += 1

        embedding = self._create_state_embedding(state)

        task = TaskInfo(
            task_id=task_id,
            embedding=embedding,
            exemplars=[state.copy()],
            first_seen=self._timestep,
            last_seen=self._timestep,
            sample_count=1,
            performance_history=[],
        )

        self._tasks[task_id] = task
        return task_id

    def _update_task(self, task_id: str, state: np.ndarray) -> None:
        """Update task with new state observation."""
        task = self._tasks[task_id]
        task.last_seen = self._timestep
        task.sample_count += 1

        if len(task.exemplars) < 100:
            task.exemplars.append(state.copy())
        elif self._rng.random() < 0.1:
            idx = self._rng.integers(0, len(task.exemplars))
            task.exemplars[idx] = state.copy()

        new_embedding = self._create_state_embedding(state)
        alpha = self.config.ema_decay
        task.embedding = alpha * task.embedding + (1 - alpha) * new_embedding

    def _create_state_embedding(self, state: np.ndarray) -> np.ndarray:
        """Create an embedding for a state."""
        state = state.flatten()

        if len(state) >= self.config.embedding_dim:
            indices = np.linspace(0, len(state) - 1, self.config.embedding_dim).astype(int)
            embedding = state[indices]
        else:
            embedding = np.zeros(self.config.embedding_dim)
            embedding[:len(state)] = state

        norm = np.linalg.norm(embedding) + 1e-8
        return embedding / norm

    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        dot = np.dot(emb1, emb2)
        return float(np.clip(dot, -1, 1))

    def create_task_embedding(
        self,
        exemplars: List[np.ndarray],
    ) -> np.ndarray:
        """
        Create a task embedding from exemplar states.

        Args:
            exemplars: List of exemplar states

        Returns:
            Task embedding vector
        """
        if not exemplars:
            return np.zeros(self.config.embedding_dim)

        embeddings = [self._create_state_embedding(np.asarray(e)) for e in exemplars]
        mean_embedding = np.mean(embeddings, axis=0)

        norm = np.linalg.norm(mean_embedding) + 1e-8
        return mean_embedding / norm

    def merge_similar_tasks(
        self,
        threshold: Optional[float] = None,
    ) -> int:
        """
        Merge similar tasks to reduce fragmentation.

        Args:
            threshold: Similarity threshold for merging

        Returns:
            Number of tasks merged
        """
        if threshold is None:
            threshold = self.config.similarity_threshold

        merged_count = 0
        task_ids = list(self._tasks.keys())

        merged_into: Dict[str, str] = {}

        for i, task_id1 in enumerate(task_ids):
            if task_id1 in merged_into:
                continue

            for task_id2 in task_ids[i + 1:]:
                if task_id2 in merged_into:
                    continue

                task1 = self._tasks[task_id1]
                task2 = self._tasks[task_id2]

                similarity = self._compute_similarity(task1.embedding, task2.embedding)

                if similarity > threshold:
                    self._merge_tasks(task_id1, task_id2)
                    merged_into[task_id2] = task_id1
                    merged_count += 1

        for task_id in merged_into:
            del self._tasks[task_id]

        return merged_count

    def _merge_tasks(self, target_id: str, source_id: str) -> None:
        """Merge source task into target task."""
        target = self._tasks[target_id]
        source = self._tasks[source_id]

        total_samples = target.sample_count + source.sample_count
        target_weight = target.sample_count / total_samples
        source_weight = source.sample_count / total_samples

        target.embedding = (
            target_weight * target.embedding +
            source_weight * source.embedding
        )

        combined = target.exemplars + source.exemplars
        if len(combined) > 100:
            indices = self._rng.choice(len(combined), 100, replace=False)
            target.exemplars = [combined[i] for i in indices]
        else:
            target.exemplars = combined

        target.first_seen = min(target.first_seen, source.first_seen)
        target.last_seen = max(target.last_seen, source.last_seen)
        target.sample_count = total_samples
        target.performance_history.extend(source.performance_history)

    def record_performance(self, task_id: str, performance: float) -> None:
        """Record performance for a task."""
        if task_id in self._tasks:
            self._tasks[task_id].performance_history.append(performance)

    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """Get information about a task."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[str]:
        """Get all task IDs."""
        return list(self._tasks.keys())

    def get_current_task(self) -> Optional[str]:
        """Get current task ID."""
        return self._current_task

    def set_current_task(self, task_id: str) -> None:
        """Set current task ID."""
        if task_id in self._tasks:
            self._current_task = task_id

    def statistics(self) -> Dict[str, Any]:
        """Get task inference statistics."""
        task_sizes = [t.sample_count for t in self._tasks.values()]

        return {
            "total_tasks": len(self._tasks),
            "current_task": self._current_task,
            "task_changes": self._task_changes,
            "timestep": self._timestep,
            "avg_task_size": float(np.mean(task_sizes)) if task_sizes else 0.0,
            "buffer_size": len(self._state_buffer),
            "tasks": {
                tid: {
                    "sample_count": t.sample_count,
                    "first_seen": t.first_seen,
                    "last_seen": t.last_seen,
                }
                for tid, t in self._tasks.items()
            },
        }
