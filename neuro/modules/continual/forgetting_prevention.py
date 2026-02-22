"""
Catastrophic Forgetting Prevention

Implements multiple strategies to prevent catastrophic forgetting:
- Elastic Weight Consolidation (EWC)
- PackNet for capacity allocation
- Synaptic Intelligence (SI)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum
import numpy as np


class ForgettingPreventionMethod(Enum):
    """Methods for preventing forgetting."""

    EWC = "ewc"
    PACKNET = "packnet"
    SYNAPTIC_INTELLIGENCE = "si"
    COMBINED = "combined"


@dataclass
class ForgettingPreventionConfig:
    """Configuration for forgetting prevention."""

    method: ForgettingPreventionMethod = ForgettingPreventionMethod.EWC
    ewc_lambda: float = 1000.0
    si_c: float = 0.1
    packnet_prune_ratio: float = 0.5
    fisher_samples: int = 100
    online_ewc: bool = True
    ewc_decay: float = 0.99


@dataclass
class TaskMemory:
    """Memory of parameters for a task."""

    task_id: str
    params: Dict[str, np.ndarray]
    fisher_info: Dict[str, np.ndarray]
    importance: Dict[str, np.ndarray]
    mask: Optional[Dict[str, np.ndarray]] = None


class CatastrophicForgettingPrevention:
    """
    Prevents catastrophic forgetting in continual learning.

    Implements three complementary approaches:
    1. EWC: Penalizes changes to important parameters
    2. PackNet: Allocates separate capacity for each task
    3. SI: Tracks online importance of parameters
    """

    def __init__(
        self,
        config: Optional[ForgettingPreventionConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or ForgettingPreventionConfig()
        self._rng = np.random.default_rng(random_seed)

        self._task_memories: Dict[str, TaskMemory] = {}
        self._current_task: Optional[str] = None

        self._online_fisher: Dict[str, np.ndarray] = {}
        self._si_omega: Dict[str, np.ndarray] = {}
        self._si_prev_params: Dict[str, np.ndarray] = {}
        self._si_running_sum: Dict[str, np.ndarray] = {}

        self._allocated_capacity: Dict[str, Set[Tuple[str, int]]] = {}

        self._timestep = 0

    def compute_fisher_information(
        self,
        params: Dict[str, np.ndarray],
        samples: List[Tuple[np.ndarray, np.ndarray]],
        gradient_fn: Optional[callable] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute Fisher information matrix (diagonal approximation).

        Args:
            params: Current parameter values
            samples: List of (input, target) samples
            gradient_fn: Optional function to compute gradients

        Returns:
            Dict mapping param names to Fisher information
        """
        fisher = {name: np.zeros_like(p) for name, p in params.items()}

        if not samples:
            return fisher

        for _ in range(min(len(samples), self.config.fisher_samples)):
            idx = self._rng.integers(0, len(samples))
            x, y = samples[idx]

            if gradient_fn is not None:
                grads = gradient_fn(params, x, y)
            else:
                grads = self._estimate_gradients(params, x, y)

            for name in fisher:
                if name in grads:
                    fisher[name] += grads[name] ** 2

        for name in fisher:
            fisher[name] /= self.config.fisher_samples

        return fisher

    def _estimate_gradients(
        self,
        params: Dict[str, np.ndarray],
        x: np.ndarray,
        y: np.ndarray,
        loss_fn: Optional[callable] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Estimate gradients using finite differences.

        Args:
            params: Current parameter values
            x: Input sample
            y: Target output
            loss_fn: Optional loss function(params, x, y) -> float
                     If None, uses MSE between forward pass and target.

        Returns:
            Dict mapping param names to estimated gradients
        """
        grads = {}
        eps = 1e-5

        # Default loss: MSE between predicted and target
        if loss_fn is None:

            def loss_fn(p, x, y):
                # Simple forward pass: weighted sum of params applied to input
                # This is a basic approximation for when no model is provided
                pred = sum(
                    np.sum(v * x.flatten()[: v.size].reshape(v.shape) if v.size <= x.size else v)
                    for v in p.values()
                )
                return float(np.mean((pred - y) ** 2))

        for name, p in params.items():
            grad = np.zeros_like(p)
            flat_p = p.flatten()

            # Compute gradients for ALL dimensions, not just 10
            for i in range(len(flat_p)):
                params_plus = {n: v.copy() for n, v in params.items()}
                params_minus = {n: v.copy() for n, v in params.items()}

                params_plus[name].flat[i] += eps
                params_minus[name].flat[i] -= eps

                loss_plus = loss_fn(params_plus, x, y)
                loss_minus = loss_fn(params_minus, x, y)

                grad.flat[i] = (loss_plus - loss_minus) / (2 * eps)

            grads[name] = grad

        return grads

    def compute_ewc_loss(
        self,
        current_params: Dict[str, np.ndarray],
        task_id: Optional[str] = None,
    ) -> float:
        """
        Compute EWC regularization loss.

        Args:
            current_params: Current parameter values
            task_id: Specific task ID (None = all tasks)

        Returns:
            EWC regularization loss
        """
        loss = 0.0

        if task_id is not None:
            tasks = [task_id] if task_id in self._task_memories else []
        else:
            tasks = list(self._task_memories.keys())

        for tid in tasks:
            memory = self._task_memories[tid]

            for name in current_params:
                if name in memory.params and name in memory.fisher_info:
                    diff = current_params[name] - memory.params[name]
                    fisher = memory.fisher_info[name]
                    loss += 0.5 * np.sum(fisher * diff**2)

        return float(self.config.ewc_lambda * loss)

    def allocate_capacity(
        self,
        task_id: str,
        param_shapes: Dict[str, Tuple[int, ...]],
        required_fraction: float = 0.5,
        param_values: Optional[Dict[str, np.ndarray]] = None,
        use_magnitude_pruning: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Allocate network capacity for a task (PackNet).

        Uses magnitude-based pruning to allocate capacity to most important weights.

        Args:
            task_id: Task identifier
            param_shapes: Dict mapping param names to shapes
            required_fraction: Fraction of capacity to allocate
            param_values: Optional current parameter values for magnitude-based allocation
            use_magnitude_pruning: If True, use magnitude; otherwise random

        Returns:
            Dict mapping param names to binary masks
        """
        masks = {}

        for name, shape in param_shapes.items():
            size = int(np.prod(shape))
            mask = np.zeros(size, dtype=bool)

            # Find already allocated indices from other tasks
            already_allocated = set()
            for tid in self._allocated_capacity:
                if tid != task_id:
                    for n, idx in self._allocated_capacity[tid]:
                        if n == name:
                            already_allocated.add(idx)

            available = [i for i in range(size) if i not in already_allocated]

            if available:
                num_to_allocate = int(len(available) * required_fraction)
                num_to_allocate = max(1, num_to_allocate)

                # Magnitude-based selection (real PackNet)
                if use_magnitude_pruning and param_values is not None and name in param_values:
                    flat_params = param_values[name].flatten()
                    # Get magnitudes for available indices
                    available_magnitudes = [(i, abs(flat_params[i])) for i in available]
                    # Sort by magnitude (descending) - keep highest magnitude weights
                    available_magnitudes.sort(key=lambda x: x[1], reverse=True)
                    # Select top-k by magnitude
                    selected = [idx for idx, _ in available_magnitudes[:num_to_allocate]]
                else:
                    # Fallback to random selection
                    selected = self._rng.choice(
                        available,
                        size=min(num_to_allocate, len(available)),
                        replace=False,
                    ).tolist()

                for idx in selected:
                    mask[idx] = True

                    if task_id not in self._allocated_capacity:
                        self._allocated_capacity[task_id] = set()
                    self._allocated_capacity[task_id].add((name, idx))

            masks[name] = mask.reshape(shape)

        return masks

    def prune_and_reallocate(
        self,
        task_id: str,
        param_values: Dict[str, np.ndarray],
        prune_fraction: float = 0.2,
    ) -> Dict[str, np.ndarray]:
        """
        Prune low-magnitude weights and reallocate capacity (real PackNet).

        This implements the actual pruning step from PackNet where weights
        below a threshold are pruned and their capacity freed for future tasks.

        Args:
            task_id: Task that owns the weights to prune
            param_values: Current parameter values
            prune_fraction: Fraction of weights to prune (keep 1-prune_fraction)

        Returns:
            Updated masks after pruning
        """
        if task_id not in self._allocated_capacity:
            return {}

        masks = {}
        pruned_indices = set()

        for name, params in param_values.items():
            flat_params = params.flatten()
            task_indices = [(n, idx) for n, idx in self._allocated_capacity[task_id] if n == name]

            if not task_indices:
                masks[name] = np.ones_like(params, dtype=bool)
                continue

            # Get magnitudes of task's allocated weights
            indices = [idx for _, idx in task_indices]
            magnitudes = [(idx, abs(flat_params[idx])) for idx in indices]
            magnitudes.sort(key=lambda x: x[1])

            # Prune lowest magnitude weights
            num_to_prune = int(len(magnitudes) * prune_fraction)
            to_prune = [idx for idx, _ in magnitudes[:num_to_prune]]

            # Update allocated capacity
            for idx in to_prune:
                self._allocated_capacity[task_id].discard((name, idx))
                pruned_indices.add((name, idx))

            # Create updated mask
            mask = np.zeros_like(params, dtype=bool)
            for idx in indices:
                if idx not in to_prune:
                    mask.flat[idx] = True
            masks[name] = mask

        return masks

    def get_task_mask(self, task_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Get the capacity mask for a task."""
        if task_id in self._task_memories and self._task_memories[task_id].mask:
            return self._task_memories[task_id].mask
        return None

    def compute_synaptic_importance(
        self,
        gradients: Dict[str, np.ndarray],
        params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Compute synaptic importance using Synaptic Intelligence (SI).

        Implements the path integral from the SI paper:
        ω_k = Σ_t |g_k(t)| * |Δw_k(t)|

        Where g_k is the gradient and Δw_k is the parameter change at each step.
        The importance accumulates over the training trajectory.

        Args:
            gradients: Current gradients (∂L/∂w)
            params: Current parameters

        Returns:
            Dict mapping param names to importance (ω)
        """
        importance = {}

        for name in params:
            # Initialize tracking on first call
            if name not in self._si_prev_params:
                self._si_prev_params[name] = params[name].copy()
                self._si_running_sum[name] = np.zeros_like(params[name])
                self._si_omega[name] = np.zeros_like(params[name])

            # Compute parameter change since last step
            param_change = params[name] - self._si_prev_params[name]

            # Path integral contribution: |gradient| * |param_change|
            # This accumulates the "work" done by each parameter
            if name in gradients:
                # SI formula: ω += |g| * |Δw| (element-wise)
                # Using absolute values ensures positive importance
                path_contribution = np.abs(gradients[name]) * np.abs(param_change)
                self._si_running_sum[name] += path_contribution

            # Compute regularization importance:
            # Ω = Σω / (Δ² + ε) where Δ is total change since task start
            # For ongoing training, we track cumulative change
            if name not in self._si_omega:
                self._si_omega[name] = np.zeros_like(params[name])

            # Update cumulative importance with damping
            # The denominator prevents division by zero for unchanged parameters
            total_change_sq = param_change**2 + 1e-8
            self._si_omega[name] = self._si_running_sum[name] / total_change_sq

            importance[name] = self._si_omega[name].copy()

            # Update previous params for next iteration
            self._si_prev_params[name] = params[name].copy()

        return importance

    def begin_task_training(self, task_id: str, initial_params: Dict[str, np.ndarray]) -> None:
        """
        Begin training on a new task (resets SI accumulators).

        Call this at the start of each new task to properly track SI importance.

        Args:
            task_id: Identifier for the new task
            initial_params: Parameters at start of task training
        """
        self._current_task = task_id
        self._si_running_sum.clear()
        self._si_omega.clear()

        for name, p in initial_params.items():
            self._si_prev_params[name] = p.copy()
            self._si_running_sum[name] = np.zeros_like(p)
            self._si_omega[name] = np.zeros_like(p)

    def end_task_training(
        self,
        task_id: str,
        final_params: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        End training on a task and compute final SI importance.

        Call this at the end of task training to finalize importance scores.

        Args:
            task_id: Task identifier
            final_params: Parameters at end of task training

        Returns:
            Final importance scores for the task
        """
        importance = {}

        for name in final_params:
            if name in self._si_omega:
                importance[name] = self._si_omega[name].copy()
            else:
                importance[name] = np.ones_like(final_params[name])

        return importance

    def compute_si_loss(
        self,
        current_params: Dict[str, np.ndarray],
        task_id: Optional[str] = None,
    ) -> float:
        """
        Compute SI regularization loss.

        Args:
            current_params: Current parameter values
            task_id: Specific task ID (None = all tasks)

        Returns:
            SI regularization loss
        """
        loss = 0.0

        if task_id is not None:
            tasks = [task_id] if task_id in self._task_memories else []
        else:
            tasks = list(self._task_memories.keys())

        for tid in tasks:
            memory = self._task_memories[tid]

            for name in current_params:
                if name in memory.params and name in memory.importance:
                    diff = current_params[name] - memory.params[name]
                    omega = memory.importance[name]
                    loss += np.sum(omega * diff**2)

        return float(self.config.si_c * loss)

    def compute_combined_loss(
        self,
        current_params: Dict[str, np.ndarray],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute combined forgetting prevention loss.

        Args:
            current_params: Current parameter values

        Returns:
            Tuple of (total loss, dict of individual losses)
        """
        losses = {}

        if self.config.method in [
            ForgettingPreventionMethod.EWC,
            ForgettingPreventionMethod.COMBINED,
        ]:
            losses["ewc"] = self.compute_ewc_loss(current_params)
        else:
            losses["ewc"] = 0.0

        if self.config.method in [
            ForgettingPreventionMethod.SYNAPTIC_INTELLIGENCE,
            ForgettingPreventionMethod.COMBINED,
        ]:
            losses["si"] = self.compute_si_loss(current_params)
        else:
            losses["si"] = 0.0

        total = sum(losses.values())
        return total, losses

    def register_task(
        self,
        task_id: str,
        params: Dict[str, np.ndarray],
        fisher_info: Optional[Dict[str, np.ndarray]] = None,
        importance: Optional[Dict[str, np.ndarray]] = None,
        mask: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Register task memory for forgetting prevention."""
        if fisher_info is None:
            fisher_info = {name: np.ones_like(p) for name, p in params.items()}

        if importance is None:
            importance = {name: np.ones_like(p) for name, p in params.items()}

        params_copy = {name: p.copy() for name, p in params.items()}

        self._task_memories[task_id] = TaskMemory(
            task_id=task_id,
            params=params_copy,
            fisher_info=fisher_info,
            importance=importance,
            mask=mask,
        )

        if self.config.online_ewc and fisher_info:
            for name, fi in fisher_info.items():
                if name in self._online_fisher:
                    self._online_fisher[name] = (
                        self.config.ewc_decay * self._online_fisher[name]
                        + (1 - self.config.ewc_decay) * fi
                    )
                else:
                    self._online_fisher[name] = fi.copy()

    def get_registered_tasks(self) -> List[str]:
        """Get all registered task IDs."""
        return list(self._task_memories.keys())

    def get_task_memory(self, task_id: str) -> Optional[TaskMemory]:
        """Get memory for a specific task."""
        return self._task_memories.get(task_id)

    def apply_mask(
        self,
        gradients: Dict[str, np.ndarray],
        task_id: str,
    ) -> Dict[str, np.ndarray]:
        """Apply PackNet mask to gradients."""
        mask = self.get_task_mask(task_id)
        if mask is None:
            return gradients

        masked_grads = {}
        for name, grad in gradients.items():
            if name in mask:
                masked_grads[name] = grad * mask[name]
            else:
                masked_grads[name] = grad

        return masked_grads

    def reset(self) -> None:
        """Reset all task memories."""
        self._task_memories.clear()
        self._online_fisher.clear()
        self._si_omega.clear()
        self._si_prev_params.clear()
        self._si_running_sum.clear()
        self._allocated_capacity.clear()
        self._timestep = 0

    def statistics(self) -> Dict[str, Any]:
        """Get forgetting prevention statistics."""
        return {
            "num_tasks": len(self._task_memories),
            "method": self.config.method.value,
            "ewc_lambda": self.config.ewc_lambda,
            "si_c": self.config.si_c,
            "online_ewc": self.config.online_ewc,
            "allocated_capacity": {
                tid: len(caps) for tid, caps in self._allocated_capacity.items()
            },
            "tasks": list(self._task_memories.keys()),
        }
