"""
Meta-Learner: Learn how to learn.

Implements MAML, learning strategies, and task adaptation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np


@dataclass
class TaskDistribution:
    """Distribution of tasks for meta-learning."""

    name: str
    task_generator: Callable[[], Dict[str, Any]]
    n_samples: int = 100
    input_dim: int = 128
    output_dim: int = 10


@dataclass
class AdaptationResult:
    """Result of adapting to a new task."""

    task_id: str
    initial_loss: float
    final_loss: float
    n_adaptation_steps: int
    adapted_parameters: Dict[str, np.ndarray]
    improvement: float


@dataclass
class LearningStrategy:
    """A learned strategy for approaching tasks."""

    id: str
    name: str
    applicable_tasks: List[str]
    initialization: Dict[str, np.ndarray]
    learning_rate: float
    n_adaptation_steps: int
    success_rate: float


class SimpleNN:
    """
    Simple neural network for meta-learning demonstrations.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        output_dim: int = 10,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize parameters
        np.random.seed(42)
        self.params = {
            "w1": np.random.randn(input_dim, hidden_dim) * np.sqrt(2 / input_dim),
            "b1": np.zeros(hidden_dim),
            "w2": np.random.randn(hidden_dim, output_dim) * np.sqrt(2 / hidden_dim),
            "b2": np.zeros(output_dim),
        }

    def forward(
        self,
        x: np.ndarray,
        params: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Forward pass."""
        if params is None:
            params = self.params

        h = x @ params["w1"] + params["b1"]
        h = np.maximum(0, h)  # ReLU
        out = h @ params["w2"] + params["b2"]

        # Softmax
        exp_out = np.exp(out - out.max(axis=-1, keepdims=True))
        return exp_out / exp_out.sum(axis=-1, keepdims=True)

    def loss(
        self,
        x: np.ndarray,
        y: np.ndarray,
        params: Optional[Dict[str, np.ndarray]] = None,
    ) -> float:
        """Cross-entropy loss."""
        probs = self.forward(x, params)
        # Add small epsilon for numerical stability
        return -np.mean(np.sum(y * np.log(probs + 1e-10), axis=-1))

    def gradient(
        self,
        x: np.ndarray,
        y: np.ndarray,
        params: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute gradients via finite differences (simplified)."""
        if params is None:
            params = self.params

        grads = {}
        eps = 1e-5

        for key, param in params.items():
            grad = np.zeros_like(param)
            flat_param = param.flatten()

            for i in range(min(len(flat_param), 100)):  # Limit for efficiency
                flat_param_plus = flat_param.copy()
                flat_param_plus[i] += eps
                param_plus = flat_param_plus.reshape(param.shape)

                flat_param_minus = flat_param.copy()
                flat_param_minus[i] -= eps
                param_minus = flat_param_minus.reshape(param.shape)

                params_plus = params.copy()
                params_plus[key] = param_plus
                loss_plus = self.loss(x, y, params_plus)

                params_minus = params.copy()
                params_minus[key] = param_minus
                loss_minus = self.loss(x, y, params_minus)

                grad.flatten()[i] = (loss_plus - loss_minus) / (2 * eps)

            grads[key] = grad

        return grads

    def clone_params(self) -> Dict[str, np.ndarray]:
        """Clone current parameters."""
        return {k: v.copy() for k, v in self.params.items()}


class MAML:
    """
    Model-Agnostic Meta-Learning (MAML).

    Learns an initialization that can be quickly adapted to new tasks.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        output_dim: int = 10,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        n_inner_steps: int = 5,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps

        self.model = SimpleNN(input_dim, hidden_dim, output_dim)
        self._meta_params = self.model.clone_params()

    def inner_loop(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Inner loop: adapt to a task.
        """
        adapted_params = {k: v.copy() for k, v in params.items()}

        for _ in range(self.n_inner_steps):
            grads = self.model.gradient(support_x, support_y, adapted_params)

            for key in adapted_params:
                adapted_params[key] -= self.inner_lr * grads[key]

        return adapted_params

    def adapt(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        task_id: str = "unknown",
    ) -> AdaptationResult:
        """
        Adapt to a new task.

        Args:
            support_x: Support set inputs
            support_y: Support set labels (one-hot)
            task_id: Task identifier

        Returns:
            AdaptationResult with adapted parameters
        """
        # Initial loss
        initial_loss = self.model.loss(support_x, support_y, self._meta_params)

        # Adapt
        adapted_params = self.inner_loop(support_x, support_y, self._meta_params)

        # Final loss
        final_loss = self.model.loss(support_x, support_y, adapted_params)

        return AdaptationResult(
            task_id=task_id,
            initial_loss=float(initial_loss),
            final_loss=float(final_loss),
            n_adaptation_steps=self.n_inner_steps,
            adapted_parameters=adapted_params,
            improvement=float(initial_loss - final_loss),
        )

    def meta_train_step(
        self,
        task_batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> float:
        """
        Meta-training step.

        Args:
            task_batch: List of (support_x, support_y, query_x, query_y)

        Returns:
            Average meta-loss
        """
        meta_grads = {k: np.zeros_like(v) for k, v in self._meta_params.items()}
        total_loss = 0.0

        for support_x, support_y, query_x, query_y in task_batch:
            # Inner loop
            adapted = self.inner_loop(support_x, support_y, self._meta_params)

            # Query loss
            query_loss = self.model.loss(query_x, query_y, adapted)
            total_loss += query_loss

            # Compute meta-gradients (simplified)
            grads = self.model.gradient(query_x, query_y, adapted)
            for key in meta_grads:
                meta_grads[key] += grads[key] / len(task_batch)

        # Update meta-parameters
        for key in self._meta_params:
            self._meta_params[key] -= self.outer_lr * meta_grads[key]

        return total_loss / len(task_batch)

    def predict(
        self,
        x: np.ndarray,
        adapted_params: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Make predictions."""
        params = adapted_params or self._meta_params
        return self.model.forward(x, params)

    def get_meta_params(self) -> Dict[str, np.ndarray]:
        """Get current meta-parameters."""
        return {k: v.copy() for k, v in self._meta_params.items()}


class MetaLearner:
    """
    Complete meta-learning system.

    Manages task distributions, learning strategies, and adaptation.
    """

    def __init__(
        self,
        input_dim: int = 128,
        output_dim: int = 10,
        n_inner_steps: int = 5,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.maml = MAML(
            input_dim=input_dim,
            output_dim=output_dim,
            n_inner_steps=n_inner_steps,
        )

        # Task distributions
        self._task_distributions: Dict[str, TaskDistribution] = {}

        # Learning strategies
        self._strategies: Dict[str, LearningStrategy] = {}
        self._strategy_counter = 0

        # Adaptation history
        self._adaptation_history: List[AdaptationResult] = []

    def register_task_distribution(
        self,
        distribution: TaskDistribution,
    ) -> None:
        """Register a task distribution for meta-training."""
        self._task_distributions[distribution.name] = distribution

    def meta_train(
        self,
        distribution_name: str,
        n_epochs: int = 100,
        batch_size: int = 4,
    ) -> List[float]:
        """
        Meta-train on a task distribution.

        Returns:
            List of meta-losses per epoch
        """
        distribution = self._task_distributions.get(distribution_name)
        if not distribution:
            raise ValueError(f"Unknown distribution: {distribution_name}")

        losses = []

        for epoch in range(n_epochs):
            # Sample task batch
            task_batch = []
            for _ in range(batch_size):
                task = distribution.task_generator()
                task_batch.append(
                    (
                        task["support_x"],
                        task["support_y"],
                        task["query_x"],
                        task["query_y"],
                    )
                )

            # Meta-train step
            loss = self.maml.meta_train_step(task_batch)
            losses.append(loss)

        return losses

    def adapt(
        self,
        support_x: np.ndarray,
        support_y: np.ndarray,
        task_id: str = "unknown",
    ) -> AdaptationResult:
        """Adapt to a new task."""
        result = self.maml.adapt(support_x, support_y, task_id)
        self._adaptation_history.append(result)
        return result

    def predict(
        self,
        x: np.ndarray,
        adaptation_result: Optional[AdaptationResult] = None,
    ) -> np.ndarray:
        """Make predictions using adapted model."""
        if adaptation_result:
            return self.maml.predict(x, adaptation_result.adapted_parameters)
        return self.maml.predict(x)

    def create_strategy(
        self,
        name: str,
        applicable_tasks: List[str],
    ) -> LearningStrategy:
        """
        Create a learning strategy from current meta-parameters.
        """
        self._strategy_counter += 1

        strategy = LearningStrategy(
            id=f"strategy_{self._strategy_counter}",
            name=name,
            applicable_tasks=applicable_tasks,
            initialization=self.maml.get_meta_params(),
            learning_rate=self.maml.inner_lr,
            n_adaptation_steps=self.maml.n_inner_steps,
            success_rate=0.0,
        )

        self._strategies[strategy.id] = strategy
        return strategy

    def apply_strategy(
        self,
        strategy_id: str,
        support_x: np.ndarray,
        support_y: np.ndarray,
    ) -> AdaptationResult:
        """Apply a learning strategy to a new task."""
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            raise ValueError(f"Unknown strategy: {strategy_id}")

        # Temporarily set meta-params to strategy initialization
        original_params = self.maml.get_meta_params()
        self.maml._meta_params = {k: v.copy() for k, v in strategy.initialization.items()}

        # Adapt
        result = self.maml.adapt(support_x, support_y)

        # Restore
        self.maml._meta_params = original_params

        return result

    def evaluate_strategy(
        self,
        strategy_id: str,
        test_tasks: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    ) -> Dict[str, float]:
        """Evaluate a strategy on test tasks."""
        strategy = self._strategies.get(strategy_id)
        if not strategy:
            raise ValueError(f"Unknown strategy: {strategy_id}")

        successes = 0
        total_improvement = 0.0

        for support_x, support_y, query_x, query_y in test_tasks:
            result = self.apply_strategy(strategy_id, support_x, support_y)
            total_improvement += result.improvement

            # Check success (positive improvement)
            if result.improvement > 0:
                successes += 1

        n_tasks = len(test_tasks)
        success_rate = successes / max(n_tasks, 1)

        # Update strategy success rate
        strategy.success_rate = success_rate

        return {
            "success_rate": success_rate,
            "avg_improvement": total_improvement / max(n_tasks, 1),
            "n_tasks": n_tasks,
        }

    def get_best_strategy(
        self,
        task_type: str,
    ) -> Optional[LearningStrategy]:
        """Get the best strategy for a task type."""
        applicable = [s for s in self._strategies.values() if task_type in s.applicable_tasks]

        if not applicable:
            return None

        return max(applicable, key=lambda s: s.success_rate)

    def get_adaptation_history(
        self,
        task_id: Optional[str] = None,
    ) -> List[AdaptationResult]:
        """Get adaptation history."""
        if task_id is None:
            return self._adaptation_history
        return [r for r in self._adaptation_history if r.task_id == task_id]

    def statistics(self) -> Dict[str, Any]:
        """Get meta-learner statistics."""
        avg_improvement = 0.0
        if self._adaptation_history:
            avg_improvement = np.mean([r.improvement for r in self._adaptation_history])

        return {
            "n_distributions": len(self._task_distributions),
            "n_strategies": len(self._strategies),
            "n_adaptations": len(self._adaptation_history),
            "avg_improvement": float(avg_improvement),
        }
