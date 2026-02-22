"""
Continual Learning with Elastic Weight Consolidation (EWC)

Prevents catastrophic forgetting when learning new tasks by:
1. Computing Fisher information (importance) of weights
2. Penalizing changes to important weights
3. Balancing plasticity and stability
"""

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class TaskMemory:
    """Memory of a learned task."""

    task_id: str
    optimal_weights: Dict[str, np.ndarray]
    fisher_matrix: Dict[str, np.ndarray]
    performance: float
    timestamp: datetime = field(default_factory=datetime.now)
    samples_seen: int = 0


@dataclass
class LearningState:
    """Current learning state."""

    weights: Dict[str, np.ndarray]
    gradients: Dict[str, np.ndarray]
    task_id: str
    iteration: int


class EWCLearner:
    """
    Elastic Weight Consolidation for continual learning.

    Prevents catastrophic forgetting by:
    1. Computing Fisher information matrix after each task
    2. Penalizing changes to important weights (high Fisher)
    3. Consolidating knowledge across tasks
    """

    def __init__(
        self, ewc_lambda: float = 1000.0, learning_rate: float = 0.01, storage_path: str = None
    ):
        self.ewc_lambda = ewc_lambda
        self.learning_rate = learning_rate
        self.storage_path = Path(storage_path or "~/.neuro/ewc").expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Current weights
        self.weights: Dict[str, np.ndarray] = {}

        # Task memories
        self.task_memories: Dict[str, TaskMemory] = {}

        # Global Fisher matrix (accumulated)
        self.global_fisher: Dict[str, np.ndarray] = {}

        # Training history
        self.training_history: List[Dict[str, Any]] = []

        # Load previous state
        self._load()

    def initialize_weights(self, weight_shapes: Dict[str, Tuple[int, ...]]) -> None:
        """Initialize weights with random values."""
        for name, shape in weight_shapes.items():
            if name not in self.weights:
                self.weights[name] = np.random.randn(*shape) * 0.01

    def _compute_fisher(
        self, data_samples: List[np.ndarray], forward_fn, num_samples: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute Fisher information matrix.

        Approximates diagonal Fisher using gradients.
        """
        fisher = {name: np.zeros_like(w) for name, w in self.weights.items()}

        for sample in data_samples[:num_samples]:
            # Get gradients for this sample
            gradients = self._compute_gradients(sample, forward_fn)

            # Accumulate squared gradients
            for name in fisher:
                if name in gradients:
                    fisher[name] += gradients[name] ** 2

        # Average
        for name in fisher:
            fisher[name] /= len(data_samples[:num_samples])

        return fisher

    def _compute_gradients(self, sample: np.ndarray, forward_fn) -> Dict[str, np.ndarray]:
        """
        Compute gradients using numerical differentiation.

        In practice, you'd use automatic differentiation.
        """
        gradients = {}
        epsilon = 1e-5

        for name, weights in self.weights.items():
            grad = np.zeros_like(weights)

            # Compute numerical gradient
            for idx in np.ndindex(weights.shape):
                # Forward pass with +epsilon
                weights[idx] += epsilon
                loss_plus = forward_fn(sample, self.weights)

                # Forward pass with -epsilon
                weights[idx] -= 2 * epsilon
                loss_minus = forward_fn(sample, self.weights)

                # Restore
                weights[idx] += epsilon

                # Gradient
                grad[idx] = (loss_plus - loss_minus) / (2 * epsilon)

            gradients[name] = grad

        return gradients

    def ewc_loss(self, task_ids: List[str] = None) -> float:
        """
        Compute EWC loss for preserving old tasks.

        Penalizes deviations from optimal weights for previous tasks,
        weighted by Fisher information (importance).
        """
        if task_ids is None:
            task_ids = list(self.task_memories.keys())

        ewc_loss = 0.0

        for task_id in task_ids:
            if task_id not in self.task_memories:
                continue

            memory = self.task_memories[task_id]

            for name in self.weights:
                if name in memory.optimal_weights and name in memory.fisher_matrix:
                    diff = self.weights[name] - memory.optimal_weights[name]
                    fisher = memory.fisher_matrix[name]
                    ewc_loss += np.sum(fisher * (diff**2))

        return self.ewc_lambda * ewc_loss / 2

    def learn_task(
        self,
        task_id: str,
        data_samples: List[np.ndarray],
        forward_fn,
        loss_fn,
        epochs: int = 100,
        batch_size: int = 32,
    ) -> float:
        """
        Learn a new task while preserving knowledge of old tasks.

        Args:
            task_id: Unique identifier for this task
            data_samples: Training data
            forward_fn: Function that computes output given input and weights
            loss_fn: Function that computes loss given predictions and targets
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Final loss on the task
        """
        print(f"  [EWC] Learning task '{task_id}'...")

        final_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0

            # Mini-batch training
            np.random.shuffle(data_samples)
            for i in range(0, len(data_samples), batch_size):
                batch = data_samples[i : i + batch_size]

                # Compute task loss
                task_loss = sum(loss_fn(sample, self.weights) for sample in batch)
                task_loss /= len(batch)

                # Compute EWC penalty
                ewc_penalty = self.ewc_loss()

                # Total loss
                total_loss = task_loss + ewc_penalty
                epoch_loss += total_loss

                # Compute gradients and update
                for name in self.weights:
                    # Simple gradient descent
                    grad = np.zeros_like(self.weights[name])

                    # Gradient from task loss (numerical)
                    for sample in batch:
                        sample_grad = self._compute_gradients(sample, loss_fn)
                        if name in sample_grad:
                            grad += sample_grad[name] / len(batch)

                    # Gradient from EWC penalty
                    for prev_task_id, memory in self.task_memories.items():
                        if name in memory.fisher_matrix:
                            diff = self.weights[name] - memory.optimal_weights[name]
                            grad += self.ewc_lambda * memory.fisher_matrix[name] * diff

                    # Update weights
                    self.weights[name] -= self.learning_rate * grad

            final_loss = epoch_loss

            if epoch % 10 == 0:
                print(f"    Epoch {epoch}: loss={final_loss:.4f}")

        # Compute Fisher information for this task
        fisher = self._compute_fisher(data_samples, forward_fn)

        # Store task memory
        self.task_memories[task_id] = TaskMemory(
            task_id=task_id,
            optimal_weights={name: w.copy() for name, w in self.weights.items()},
            fisher_matrix=fisher,
            performance=1.0 - final_loss,
            samples_seen=len(data_samples),
        )

        # Update global Fisher
        for name in fisher:
            if name not in self.global_fisher:
                self.global_fisher[name] = fisher[name].copy()
            else:
                self.global_fisher[name] = 0.5 * self.global_fisher[name] + 0.5 * fisher[name]

        # Record history
        self.training_history.append(
            {"task_id": task_id, "final_loss": final_loss, "timestamp": datetime.now().isoformat()}
        )

        print(f"  [EWC] Completed task '{task_id}', performance={1.0 - final_loss:.2%}")

        return final_loss

    def evaluate_task(self, task_id: str, test_samples: List[np.ndarray], loss_fn) -> float:
        """Evaluate current performance on a task."""
        if not test_samples:
            return 0.0

        total_loss = sum(loss_fn(sample, self.weights) for sample in test_samples)
        return total_loss / len(test_samples)

    def evaluate_all_tasks(
        self, test_data: Dict[str, List[np.ndarray]], loss_fn
    ) -> Dict[str, float]:
        """Evaluate performance on all learned tasks."""
        results = {}
        for task_id, samples in test_data.items():
            results[task_id] = self.evaluate_task(task_id, samples, loss_fn)
        return results

    def get_weight_importance(self, name: str) -> Optional[np.ndarray]:
        """Get importance (Fisher information) for a weight."""
        return self.global_fisher.get(name)

    def save(self) -> None:
        """Save learner state to disk."""
        # Save weights
        np.savez(self.storage_path / "weights.npz", **{name: w for name, w in self.weights.items()})

        # Save Fisher matrices
        np.savez(
            self.storage_path / "global_fisher.npz",
            **{name: f for name, f in self.global_fisher.items()},
        )

        # Save task memories (metadata only)
        memories_data = {
            task_id: {
                "task_id": mem.task_id,
                "performance": mem.performance,
                "timestamp": mem.timestamp.isoformat(),
                "samples_seen": mem.samples_seen,
            }
            for task_id, mem in self.task_memories.items()
        }

        with open(self.storage_path / "memories.json", "w") as f:
            json.dump(memories_data, f, indent=2)

        # Save history
        with open(self.storage_path / "history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)

    def _load(self) -> None:
        """Load learner state from disk."""
        # Load weights
        weights_file = self.storage_path / "weights.npz"
        if weights_file.exists():
            try:
                data = np.load(weights_file)
                self.weights = {name: data[name] for name in data.files}
            except Exception:
                pass

        # Load global Fisher
        fisher_file = self.storage_path / "global_fisher.npz"
        if fisher_file.exists():
            try:
                data = np.load(fisher_file)
                self.global_fisher = {name: data[name] for name in data.files}
            except Exception:
                pass

        # Load history
        history_file = self.storage_path / "history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    self.training_history = json.load(f)
            except Exception:
                pass

    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            "num_weights": len(self.weights),
            "total_parameters": sum(w.size for w in self.weights.values()),
            "tasks_learned": len(self.task_memories),
            "ewc_lambda": self.ewc_lambda,
            "training_iterations": len(self.training_history),
            "storage_path": str(self.storage_path),
        }


# Simplified version for integration
class SimpleContinualLearner:
    """
    Simplified continual learner for knowledge retention.

    Tracks what was learned and prevents overwriting.
    """

    def __init__(self):
        self.knowledge: Dict[str, Any] = {}
        self.importance: Dict[str, float] = {}
        self.access_count: Dict[str, int] = defaultdict(int)

    def learn(self, key: str, value: Any, importance: float = 0.5) -> None:
        """Learn a new piece of knowledge."""
        if key in self.knowledge:
            # Blend old and new based on importance
            old_importance = self.importance.get(key, 0.5)
            if importance > old_importance:
                self.knowledge[key] = value
                self.importance[key] = importance
        else:
            self.knowledge[key] = value
            self.importance[key] = importance

    def recall(self, key: str) -> Optional[Any]:
        """Recall learned knowledge."""
        self.access_count[key] += 1
        return self.knowledge.get(key)

    def get_important_knowledge(self, threshold: float = 0.7) -> Dict[str, Any]:
        """Get highly important knowledge."""
        return {k: v for k, v in self.knowledge.items() if self.importance.get(k, 0) >= threshold}

    def consolidate(self) -> int:
        """Consolidate frequently accessed knowledge."""
        consolidated = 0
        for key in self.knowledge:
            if self.access_count[key] > 3:
                old_importance = self.importance.get(key, 0.5)
                self.importance[key] = min(1.0, old_importance * 1.2)
                consolidated += 1
        return consolidated


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("CONTINUAL LEARNING TEST")
    print("=" * 60)

    learner = SimpleContinualLearner()

    # Learn some facts
    learner.learn("AI_definition", "Artificial intelligence simulates human intelligence", 0.8)
    learner.learn("ML_definition", "Machine learning learns from data", 0.7)
    learner.learn("Python_usage", "Python is used for AI development", 0.6)

    # Recall
    print("\nRecalling 'AI_definition':", learner.recall("AI_definition"))

    # Try to overwrite with lower importance
    learner.learn("AI_definition", "AI is just math", 0.3)
    print("After low-importance overwrite:", learner.recall("AI_definition"))

    # Overwrite with higher importance
    learner.learn("AI_definition", "AI is the future of computing", 0.9)
    print("After high-importance overwrite:", learner.recall("AI_definition"))

    # Get important knowledge
    print("\nImportant knowledge:", learner.get_important_knowledge())

    # Consolidate
    for _ in range(5):
        learner.recall("ML_definition")
    consolidated = learner.consolidate()
    print(f"\nConsolidated {consolidated} items")
    print("ML_definition importance:", learner.importance.get("ML_definition"))
