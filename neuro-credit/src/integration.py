"""
Credit Assignment Integration

Integrates all credit assignment components into a unified system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Set
import numpy as np

from .eligibility_traces import EligibilityTraceManager, TraceConfig, TraceType
from .policy_gradient import CrossModulePolicyGradient, GAEConfig, PolicyGradientResult
from .blame_assignment import BlameAssignment, Failure, FailureType, BlameResult, ModuleAction
from .surprise_modulation import SurpriseModulatedLearning, SurpriseConfig, SurpriseMetrics
from .contribution_accounting import ContributionAccountant, ShapleyConfig


@dataclass
class CreditConfig:
    """Configuration for the integrated credit assignment system."""
    trace_lambda: float = 0.9
    gamma: float = 0.99
    gae_lambda: float = 0.95
    base_learning_rate: float = 0.01
    surprise_scale: float = 1.0
    use_shapley: bool = True
    shapley_samples: int = 100
    random_seed: Optional[int] = None


class CreditAssignmentSystem:
    """
    Integrated credit assignment system.

    Combines:
    - Eligibility traces for temporal credit
    - Policy gradients for learning
    - Blame assignment for failure analysis
    - Surprise modulation for adaptive learning
    - Shapley values for fair credit distribution
    """

    def __init__(self, config: Optional[CreditConfig] = None):
        self.config = config or CreditConfig()
        self._rng = np.random.default_rng(self.config.random_seed)

        trace_config = TraceConfig(
            lambda_param=self.config.trace_lambda,
            gamma=self.config.gamma,
        )
        self._traces = EligibilityTraceManager(
            config=trace_config,
            random_seed=self.config.random_seed,
        )

        gae_config = GAEConfig(
            gamma=self.config.gamma,
            lambda_param=self.config.gae_lambda,
        )
        self._policy_gradient = CrossModulePolicyGradient(
            config=gae_config,
            random_seed=self.config.random_seed,
        )

        self._blame = BlameAssignment(random_seed=self.config.random_seed)

        surprise_config = SurpriseConfig(
            base_learning_rate=self.config.base_learning_rate,
            surprise_scale=self.config.surprise_scale,
        )
        self._surprise = SurpriseModulatedLearning(
            config=surprise_config,
            random_seed=self.config.random_seed,
        )

        shapley_config = ShapleyConfig(
            use_approximation=True,
            num_samples=self.config.shapley_samples,
        )
        self._contributions = ContributionAccountant(
            config=shapley_config,
            random_seed=self.config.random_seed,
        )

        self._active_modules: Set[str] = set()
        self._action_history: List[ModuleAction] = []
        self._timestep = 0

    def register_module(self, module_id: str) -> None:
        """Register a module for credit tracking."""
        self._active_modules.add(module_id)

    def unregister_module(self, module_id: str) -> None:
        """Unregister a module."""
        self._active_modules.discard(module_id)

    def record_action(
        self,
        module_id: str,
        state: Any,
        action: Any,
        state_after: Optional[Any] = None,
        predicted_outcome: Optional[Any] = None,
    ) -> None:
        """
        Record a module action for credit assignment.

        Args:
            module_id: Module identifier
            state: State before action
            action: Action taken
            state_after: Optional state after action
            predicted_outcome: Optional predicted outcome
        """
        self._traces.mark_eligible(state, action, module_id)

        action_record = ModuleAction(
            module_id=module_id,
            action=action,
            state_before=state,
            state_after=state_after,
            timestamp=self._timestep,
            predicted_outcome=predicted_outcome,
        )
        self._action_history.append(action_record)
        self._timestep += 1

    def receive_reward(
        self,
        reward: float,
        td_error: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Receive a reward and distribute credit.

        Args:
            reward: Reward signal
            td_error: Optional TD error

        Returns:
            Credit distribution across modules
        """
        trace_credits = self._traces.distribute_credit(reward, td_error)

        self._traces.decay_traces()

        if self.config.use_shapley and len(self._active_modules) > 1:
            def value_fn(coalition: Set[str]) -> float:
                if not coalition:
                    return 0.0
                return reward * len(coalition) / len(self._active_modules)

            shapley_credits = self._contributions.compute_shapley_values(
                reward, list(self._active_modules), value_fn
            )

            combined = {}
            for module_id in self._active_modules:
                trace_credit = trace_credits.get(module_id, 0.0)
                shapley_credit = shapley_credits.get(module_id, 0.0)
                combined[module_id] = 0.5 * trace_credit + 0.5 * shapley_credit

            return combined

        return trace_credits

    def process_trajectory(
        self,
        trajectory: Dict[str, Any],
        module_id: str,
    ) -> PolicyGradientResult:
        """
        Process a trajectory for policy gradient updates.

        Args:
            trajectory: Dict with states, actions, rewards, values, log_probs
            module_id: Module identifier

        Returns:
            PolicyGradientResult
        """
        return self._policy_gradient.compute_full_result(trajectory, module_id)

    def handle_failure(
        self,
        failure_type: FailureType,
        description: str,
        severity: float = 1.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, BlameResult]:
        """
        Handle a failure event and assign blame.

        Args:
            failure_type: Type of failure
            description: Description of failure
            severity: Severity [0, 1]
            context: Optional context

        Returns:
            Blame results per module
        """
        failure = Failure(
            failure_type=failure_type,
            description=description,
            timestamp=self._timestep,
            severity=severity,
            context=context or {},
        )

        blame_results = {}
        for module_id in self._active_modules:
            result = self._blame.compute_blame(
                failure, self._action_history, module_id
            )
            blame_results[module_id] = result

        blame_distribution = {
            module_id: result.blame_score
            for module_id, result in blame_results.items()
        }
        self._blame.propagate_blame(failure, blame_distribution)

        return blame_results

    def get_adaptive_learning_rate(
        self,
        module_id: str,
        predicted: Any,
        actual: Any,
    ) -> float:
        """
        Get adaptive learning rate based on surprise.

        Args:
            module_id: Module identifier
            predicted: Predicted value
            actual: Actual value

        Returns:
            Adapted learning rate
        """
        return self._surprise.get_adaptive_lr(module_id, predicted, actual)

    def compute_surprise(
        self,
        predicted: Any,
        actual: Any,
    ) -> SurpriseMetrics:
        """Compute surprise metrics."""
        return self._surprise.compute_surprise(predicted, actual)

    def get_module_credits(self) -> Dict[str, float]:
        """Get cumulative credits for all modules."""
        return self._contributions.get_cumulative_contributions()

    def get_underperforming_modules(
        self,
        threshold: float = 0.0,
    ) -> List[str]:
        """Get modules with low contributions."""
        return self._contributions.identify_underperforming_modules(threshold)

    def get_top_contributors(self, n: int = 5) -> List[Tuple[str, float]]:
        """Get top contributing modules."""
        return self._contributions.identify_top_contributors(n)

    def get_chronic_failure_modules(
        self,
        threshold: int = 3,
    ) -> List[Tuple[str, int]]:
        """Get modules with repeated failures."""
        return self._blame.get_chronic_offenders(threshold)

    def clear_history(self) -> None:
        """Clear action history."""
        self._action_history.clear()

    def reset_traces(self) -> None:
        """Reset eligibility traces."""
        self._traces.reset_traces()

    def set_world_model(self, model: Callable[[Any, Any], Any]) -> None:
        """Set world model for counterfactual analysis."""
        self._blame.set_world_model(model)

    def statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "active_modules": len(self._active_modules),
            "timestep": self._timestep,
            "action_history_size": len(self._action_history),
            "trace_stats": self._traces.statistics(),
            "policy_gradient_stats": self._policy_gradient.statistics(),
            "blame_stats": self._blame.statistics(),
            "surprise_stats": self._surprise.statistics(),
            "contribution_stats": self._contributions.statistics(),
        }
