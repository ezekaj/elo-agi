"""
Continual Learning Controller Integration

Unified controller for continual learning combining:
- Task inference and boundaries
- Selective consolidation
- Forgetting prevention
- Experience replay
- Capability tracking
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

from .task_inference import TaskInference, TaskInferenceConfig
from .selective_consolidation import (
    SelectiveConsolidation,
    ConsolidationConfig,
    ConsolidationPlan,
)
from .forgetting_prevention import (
    CatastrophicForgettingPrevention,
    ForgettingPreventionConfig,
    ForgettingPreventionMethod,
)
from .experience_replay import (
    ImportanceWeightedReplay,
    ReplayConfig,
    Experience,
)
from .capability_tracking import (
    CapabilityTracker,
    CapabilityConfig,
    CapabilityMetric,
)


@dataclass
class ContinualLearningConfig:
    """Configuration for the continual learning controller."""

    task_change_threshold: float = 0.5
    consolidation_frequency: int = 100
    replay_batch_size: int = 32
    ewc_lambda: float = 1000.0
    si_c: float = 0.1
    forgetting_method: ForgettingPreventionMethod = ForgettingPreventionMethod.EWC
    auto_consolidate: bool = True
    auto_detect_tasks: bool = True
    random_seed: Optional[int] = None


class ContinualLearningController:
    """
    Unified controller for continual learning.

    Orchestrates:
    - Task boundary detection
    - Memory consolidation
    - Forgetting prevention
    - Experience replay
    - Capability monitoring
    """

    def __init__(
        self,
        config: Optional[ContinualLearningConfig] = None,
    ):
        self.config = config or ContinualLearningConfig()

        task_config = TaskInferenceConfig(
            change_threshold=self.config.task_change_threshold,
        )
        self._task_inference = TaskInference(
            config=task_config,
            random_seed=self.config.random_seed,
        )

        consolidation_config = ConsolidationConfig()
        self._consolidation = SelectiveConsolidation(
            config=consolidation_config,
            random_seed=self.config.random_seed,
        )

        forgetting_config = ForgettingPreventionConfig(
            method=self.config.forgetting_method,
            ewc_lambda=self.config.ewc_lambda,
            si_c=self.config.si_c,
        )
        self._forgetting = CatastrophicForgettingPrevention(
            config=forgetting_config,
            random_seed=self.config.random_seed,
        )

        replay_config = ReplayConfig()
        self._replay = ImportanceWeightedReplay(
            config=replay_config,
            random_seed=self.config.random_seed,
        )

        capability_config = CapabilityConfig()
        self._capabilities = CapabilityTracker(
            config=capability_config,
            random_seed=self.config.random_seed,
        )

        self._current_task: Optional[str] = None
        self._current_params: Dict[str, np.ndarray] = {}

        self._timestep = 0
        self._total_task_changes = 0
        self._total_consolidations = 0

    def observe(
        self,
        state: np.ndarray,
        action: Any,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        params: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        """
        Process a new observation.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
            params: Optional current parameters

        Returns:
            Dict with observation results
        """
        results = {
            "task_changed": False,
            "task_id": self._current_task,
            "consolidation_triggered": False,
            "forgetting_loss": 0.0,
        }

        if self.config.auto_detect_tasks:
            task_changed = self._task_inference.detect_task_change(np.asarray(state))

            if task_changed:
                new_task = self._task_inference.infer_task_id(np.asarray(state))
                results["task_changed"] = True
                results["previous_task"] = self._current_task
                results["task_id"] = new_task

                self._handle_task_change(new_task, params)
                self._current_task = new_task
                self._total_task_changes += 1

        if self._current_task is None:
            self._current_task = "default"

        self._replay.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            task_id=self._current_task,
            td_error=abs(reward),
        )

        if params is not None:
            self._current_params = params
            loss, _ = self._forgetting.compute_combined_loss(params)
            results["forgetting_loss"] = loss

        if (
            self.config.auto_consolidate
            and self._timestep % self.config.consolidation_frequency == 0
        ):
            plan = self.trigger_consolidation()
            if plan.task_priorities:
                results["consolidation_triggered"] = True
                results["consolidation_plan"] = plan
                self._total_consolidations += 1

        self._timestep += 1
        return results

    def _handle_task_change(
        self,
        new_task: str,
        params: Optional[Dict[str, np.ndarray]],
    ) -> None:
        """Handle transition to a new task."""
        if self._current_task is not None and params:
            self._forgetting.register_task(
                self._current_task,
                params,
            )

        self._consolidation.register_task(new_task)

    def trigger_consolidation(
        self,
        budget: Optional[int] = None,
    ) -> ConsolidationPlan:
        """
        Trigger a consolidation cycle.

        Args:
            budget: Optional consolidation budget

        Returns:
            ConsolidationPlan with priorities and allocation
        """
        return self._consolidation.create_consolidation_plan(budget)

    def sample_replay(
        self,
        batch_size: Optional[int] = None,
    ) -> List[Tuple[Experience, float]]:
        """
        Sample experiences for replay.

        Args:
            batch_size: Number of experiences to sample

        Returns:
            List of (experience, importance_weight) tuples
        """
        if batch_size is None:
            batch_size = self.config.replay_batch_size

        return self._replay.sample_batch(batch_size)

    def update_replay_priorities(
        self,
        indices: List[int],
        td_errors: List[float],
    ) -> None:
        """Update priorities after learning."""
        self._replay.update_priorities(indices, td_errors)

    def compute_forgetting_loss(
        self,
        params: Dict[str, np.ndarray],
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute forgetting prevention loss.

        Args:
            params: Current parameters

        Returns:
            Tuple of (total loss, individual losses)
        """
        return self._forgetting.compute_combined_loss(params)

    def register_task_params(
        self,
        task_id: str,
        params: Dict[str, np.ndarray],
        fisher_info: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        """Register parameters for a task."""
        self._forgetting.register_task(task_id, params, fisher_info)

    def measure_capability(
        self,
        name: str,
        test_results: Dict[str, float],
    ) -> CapabilityMetric:
        """
        Measure a capability.

        Args:
            name: Capability name
            test_results: Test results

        Returns:
            CapabilityMetric
        """
        return self._capabilities.measure_capability(name, test_results)

    def update_performance(self, task_id: str, performance: float) -> None:
        """Update performance for a task."""
        self._consolidation.update_performance(task_id, performance)
        self._task_inference.record_performance(task_id, performance)

    def get_regressing_capabilities(self) -> List[str]:
        """Get capabilities that are regressing."""
        return self._capabilities.get_regressing_capabilities()

    def get_remediation_suggestions(self) -> Dict[str, List[str]]:
        """Get remediation suggestions for regressing capabilities."""
        regressing = self._capabilities.get_regressing_capabilities()
        return self._capabilities.suggest_remediation(regressing)

    def get_current_task(self) -> Optional[str]:
        """Get current task ID."""
        return self._current_task

    def get_all_tasks(self) -> List[str]:
        """Get all known task IDs."""
        return self._task_inference.get_all_tasks()

    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a task."""
        task = self._task_inference.get_task_info(task_id)
        if task is None:
            return None

        return {
            "task_id": task.task_id,
            "sample_count": task.sample_count,
            "first_seen": task.first_seen,
            "last_seen": task.last_seen,
            "performance": task.performance_history[-1] if task.performance_history else None,
        }

    def get_replay_buffer_size(self) -> int:
        """Get current replay buffer size."""
        return len(self._replay)

    def reset(self) -> None:
        """Reset the controller."""
        self._task_inference = TaskInference(random_seed=self.config.random_seed)
        self._consolidation.reset()
        self._forgetting.reset()
        self._replay = ImportanceWeightedReplay(random_seed=self.config.random_seed)
        self._capabilities.reset()

        self._current_task = None
        self._current_params = {}
        self._timestep = 0
        self._total_task_changes = 0
        self._total_consolidations = 0

    def statistics(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            "timestep": self._timestep,
            "current_task": self._current_task,
            "total_task_changes": self._total_task_changes,
            "total_consolidations": self._total_consolidations,
            "task_inference": self._task_inference.statistics(),
            "consolidation": self._consolidation.statistics(),
            "forgetting": self._forgetting.statistics(),
            "replay": self._replay.statistics(),
            "capabilities": self._capabilities.statistics(),
        }
