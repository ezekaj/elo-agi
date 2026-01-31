"""
Dynamic Orchestrator for Meta-Reasoning

Orchestrates reasoning across modules:
- Plan creation and execution
- Checkpoint evaluation
- Dynamic module switching
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np

from .problem_classifier import ProblemAnalysis, ProblemType
from .style_selector import StyleSelection, ReasoningStyle
from .efficiency_monitor import EfficiencyMonitor, EfficiencyConfig


class PlanStatus(Enum):
    """Status of an orchestration plan."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SWITCHED = "switched"


class CheckpointAction(Enum):
    """Actions at checkpoints."""
    CONTINUE = "continue"
    SWITCH_MODULE = "switch_module"
    SWITCH_STYLE = "switch_style"
    TERMINATE = "terminate"
    RESTART = "restart"


@dataclass
class OrchestratorConfig:
    """Configuration for orchestration."""
    max_steps: int = 100
    checkpoint_frequency: int = 5
    switch_threshold: float = 0.3
    min_progress_for_continue: float = 0.05
    enable_dynamic_switching: bool = True


@dataclass
class OrchestrationStep:
    """A step in the orchestration plan."""
    step_id: int
    module: str
    style: ReasoningStyle
    expected_progress: float
    dependencies: List[int]


@dataclass
class OrchestrationPlan:
    """Plan for orchestrating reasoning."""
    plan_id: str
    problem_type: ProblemType
    primary_style: ReasoningStyle
    steps: List[OrchestrationStep]
    total_expected_progress: float
    estimated_cost: float
    status: PlanStatus


@dataclass
class ExecutionResult:
    """Result of plan execution."""
    plan_id: str
    final_progress: float
    final_quality: float
    steps_executed: int
    modules_used: List[str]
    switches_made: int
    success: bool


class DynamicOrchestrator:
    """
    Orchestrates reasoning across modules dynamically.

    Capabilities:
    - Create reasoning plans
    - Execute with checkpoints
    - Switch modules/styles as needed
    - Track and adapt execution
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or OrchestratorConfig()
        self._rng = np.random.default_rng(random_seed)

        efficiency_config = EfficiencyConfig()
        self._efficiency = EfficiencyMonitor(config=efficiency_config)

        self._plans: Dict[str, OrchestrationPlan] = {}
        self._execution_history: List[ExecutionResult] = []

        self._module_registry: Dict[str, Callable] = {}
        self._plan_counter = 0

    def register_module(self, name: str, module_fn: Callable) -> None:
        """Register a reasoning module."""
        self._module_registry[name] = module_fn

    def create_plan(
        self,
        analysis: ProblemAnalysis,
        style: StyleSelection,
    ) -> OrchestrationPlan:
        """
        Create an orchestration plan.

        Args:
            analysis: Problem analysis
            style: Selected reasoning style

        Returns:
            OrchestrationPlan
        """
        plan_id = f"plan_{self._plan_counter}"
        self._plan_counter += 1

        steps = self._generate_steps(analysis, style)

        total_progress = sum(s.expected_progress for s in steps)
        estimated_cost = len(steps) * 2.0

        plan = OrchestrationPlan(
            plan_id=plan_id,
            problem_type=analysis.problem_type,
            primary_style=style.primary_style,
            steps=steps,
            total_expected_progress=total_progress,
            estimated_cost=estimated_cost,
            status=PlanStatus.PENDING,
        )

        self._plans[plan_id] = plan
        return plan

    def _generate_steps(
        self,
        analysis: ProblemAnalysis,
        style: StyleSelection,
    ) -> List[OrchestrationStep]:
        """Generate orchestration steps."""
        steps = []

        num_steps = min(analysis.estimated_steps, self.config.max_steps)
        progress_per_step = 1.0 / num_steps if num_steps > 0 else 1.0

        module = self._select_module_for_style(style.primary_style)

        for i in range(num_steps):
            step = OrchestrationStep(
                step_id=i,
                module=module,
                style=style.primary_style,
                expected_progress=progress_per_step,
                dependencies=[i - 1] if i > 0 else [],
            )
            steps.append(step)

        return steps

    def _select_module_for_style(self, style: ReasoningStyle) -> str:
        """Select appropriate module for style."""
        style_modules = {
            ReasoningStyle.DEDUCTIVE: "logical_reasoner",
            ReasoningStyle.INDUCTIVE: "pattern_learner",
            ReasoningStyle.ABDUCTIVE: "hypothesis_generator",
            ReasoningStyle.ANALOGICAL: "analogy_mapper",
            ReasoningStyle.CAUSAL: "causal_reasoner",
            ReasoningStyle.SPATIAL: "spatial_reasoner",
            ReasoningStyle.TEMPORAL: "temporal_reasoner",
            ReasoningStyle.HEURISTIC: "heuristic_solver",
            ReasoningStyle.SYSTEMATIC: "systematic_search",
            ReasoningStyle.CREATIVE: "creative_generator",
        }

        preferred = style_modules.get(style, "default_reasoner")

        if preferred in self._module_registry:
            return preferred

        if self._module_registry:
            return list(self._module_registry.keys())[0]

        return preferred

    def execute_plan(
        self,
        plan: OrchestrationPlan,
        problem: Dict[str, Any],
    ) -> ExecutionResult:
        """
        Execute an orchestration plan.

        Args:
            plan: The plan to execute
            problem: Problem to solve

        Returns:
            ExecutionResult
        """
        plan.status = PlanStatus.IN_PROGRESS
        self._efficiency.start_monitoring(plan.plan_id, plan.primary_style)

        current_progress = 0.0
        current_quality = 0.0
        steps_executed = 0
        modules_used = []
        switches = 0

        current_module = plan.steps[0].module if plan.steps else "default"
        modules_used.append(current_module)

        for i, step in enumerate(plan.steps):
            step_result = self._execute_step(step, problem, current_progress)

            current_progress = step_result["progress"]
            current_quality = step_result["quality"]
            steps_executed += 1

            self._efficiency.update_progress(
                plan.plan_id,
                current_progress,
                current_quality,
                cost=1.0,
            )

            if i % self.config.checkpoint_frequency == 0:
                action = self.checkpoint_evaluation(plan, i, {
                    "progress": current_progress,
                    "quality": current_quality,
                })

                if action == CheckpointAction.TERMINATE:
                    break
                elif action == CheckpointAction.SWITCH_MODULE:
                    new_module = self._select_alternative_module(current_module)
                    if new_module != current_module:
                        current_module = new_module
                        modules_used.append(current_module)
                        switches += 1

            should_term, reason = self._efficiency.should_terminate_early(plan.plan_id)
            if should_term:
                break

        self._efficiency.complete_session(plan.plan_id, current_quality)

        success = current_progress >= 0.9 or current_quality >= 0.8
        plan.status = PlanStatus.COMPLETED if success else PlanStatus.FAILED

        result = ExecutionResult(
            plan_id=plan.plan_id,
            final_progress=current_progress,
            final_quality=current_quality,
            steps_executed=steps_executed,
            modules_used=modules_used,
            switches_made=switches,
            success=success,
        )

        self._execution_history.append(result)
        return result

    def _execute_step(
        self,
        step: OrchestrationStep,
        problem: Dict[str, Any],
        current_progress: float,
    ) -> Dict[str, Any]:
        """
        Execute a single step using a registered module.

        Args:
            step: The step to execute
            problem: Problem data to pass to module
            current_progress: Current execution progress

        Returns:
            Dict with 'progress' and 'quality' keys

        Raises:
            ValueError: If no module is registered for the step
        """
        if step.module in self._module_registry:
            module_fn = self._module_registry[step.module]
            result = module_fn(problem, step)
            return result

        # No random fallback - require actual module registration
        # This ensures orchestration is meaningful, not simulated
        raise ValueError(
            f"No module registered for '{step.module}'. "
            f"Register a module using orchestrator.register_module('{step.module}', fn) "
            f"before executing plans. Available modules: {list(self._module_registry.keys())}"
        )

    def checkpoint_evaluation(
        self,
        plan: OrchestrationPlan,
        step: int,
        partial_result: Dict[str, Any],
    ) -> CheckpointAction:
        """
        Evaluate progress at checkpoint.

        Args:
            plan: Current plan
            step: Current step
            partial_result: Partial results so far

        Returns:
            CheckpointAction
        """
        progress = partial_result.get("progress", 0.0)
        quality = partial_result.get("quality", 0.0)

        expected_progress = (step + 1) / len(plan.steps) if plan.steps else 0

        if progress >= 0.95 and quality >= 0.8:
            return CheckpointAction.TERMINATE

        if step > 0:
            progress_rate = progress / step
            if progress_rate < self.config.min_progress_for_continue:
                if self.config.enable_dynamic_switching:
                    return CheckpointAction.SWITCH_MODULE

        if progress < expected_progress * self.config.switch_threshold:
            if self.config.enable_dynamic_switching:
                return CheckpointAction.SWITCH_STYLE

        return CheckpointAction.CONTINUE

    def switch_module(
        self,
        plan: OrchestrationPlan,
        current: str,
        reason: str,
    ) -> str:
        """
        Switch to a different module.

        Args:
            plan: Current plan
            current: Current module
            reason: Reason for switch

        Returns:
            New module name
        """
        return self._select_alternative_module(current)

    def _select_alternative_module(self, current: str) -> str:
        """Select an alternative module."""
        available = [m for m in self._module_registry.keys() if m != current]

        if available:
            return self._rng.choice(available)

        return current

    def get_plan(self, plan_id: str) -> Optional[OrchestrationPlan]:
        """Get a plan by ID."""
        return self._plans.get(plan_id)

    def get_execution_history(
        self,
        n: Optional[int] = None,
    ) -> List[ExecutionResult]:
        """Get execution history."""
        if n is not None:
            return self._execution_history[-n:]
        return list(self._execution_history)

    def statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        if not self._execution_history:
            return {
                "total_plans": len(self._plans),
                "total_executions": 0,
                "avg_progress": 0.0,
                "success_rate": 0.0,
                "registered_modules": list(self._module_registry.keys()),
            }

        progresses = [r.final_progress for r in self._execution_history]
        successes = [r.success for r in self._execution_history]

        return {
            "total_plans": len(self._plans),
            "total_executions": len(self._execution_history),
            "avg_progress": float(np.mean(progresses)),
            "avg_quality": float(np.mean([r.final_quality for r in self._execution_history])),
            "success_rate": sum(successes) / len(successes),
            "avg_steps": float(np.mean([r.steps_executed for r in self._execution_history])),
            "avg_switches": float(np.mean([r.switches_made for r in self._execution_history])),
            "registered_modules": list(self._module_registry.keys()),
        }
