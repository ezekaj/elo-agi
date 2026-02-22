"""
Darwin Gödel Machine: Recursive self-improvement system.

The Darwin Gödel Machine is the central self-improvement engine that
coordinates all components to enable autonomous recursive self-modification.

Based on:
- Gödel Machine (Schmidhuber, 2003)
- Darwin Gödel Machine (arXiv:2505.22954)
- Recursive self-improvement in AI
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import time

from .generator import ModificationGenerator, GeneratorParams, Modification
from .verifier import ChangeVerifier, VerifierParams, VerificationMethod
from .updater import SystemUpdater, UpdaterParams, UpdateStatus
from .meta_learner import MetaLearner, MetaParams


class ImprovementPhase(Enum):
    """Phases of the improvement cycle."""

    IDLE = "idle"  # Waiting for trigger
    GENERATING = "generating"  # Generating candidates
    VERIFYING = "verifying"  # Verifying candidates
    APPLYING = "applying"  # Applying modifications
    EVALUATING = "evaluating"  # Evaluating results
    LEARNING = "learning"  # Meta-learning


@dataclass
class DGMParams:
    """Parameters for the Darwin Gödel Machine."""

    auto_improve: bool = True  # Automatically run improvement cycles
    improvement_interval: float = 60.0  # Seconds between improvement attempts
    min_improvement_threshold: float = 0.01
    max_failed_attempts: int = 10  # Stop after N consecutive failures
    safety_mode: bool = True  # Enable safety constraints
    log_all_attempts: bool = True


@dataclass
class ImprovementCycle:
    """Record of an improvement cycle."""

    cycle_id: int
    phase: ImprovementPhase
    start_time: float
    end_time: Optional[float]
    candidates_generated: int
    candidates_verified: int
    candidates_applied: int
    initial_performance: float
    final_performance: float
    improvement: float
    best_modification: Optional[Modification]
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class DarwinGodelMachine:
    """
    The Darwin Gödel Machine: A self-improving AI system.

    The DGM combines:
    1. **Generator**: Proposes modifications
    2. **Verifier**: Validates improvements
    3. **Updater**: Applies changes safely
    4. **Meta-learner**: Learns how to improve

    Key principle (from Gödel Machine):
    "Only apply a modification if you can prove it improves
    expected future performance."

    The Darwin variant adds evolutionary search over modification
    strategies, enabling open-ended self-improvement.

    Usage:
        dgm = DarwinGodelMachine()
        dgm.set_target_system(system, performance_fn)
        dgm.run_improvement_cycle()
    """

    def __init__(self, params: Optional[DGMParams] = None):
        self.params = params or DGMParams()

        # Core components
        self.generator = ModificationGenerator(GeneratorParams(n_candidates=10))
        self.verifier = ChangeVerifier(VerifierParams(min_improvement=0.01))
        self.updater = SystemUpdater(UpdaterParams(gradual_application=True))
        self.meta_learner = MetaLearner(MetaParams(strategy_adaptation=True))

        # State
        self._phase = ImprovementPhase.IDLE
        self._cycle_count = 0
        self._consecutive_failures = 0
        self._last_improvement_time = 0.0

        # Target system
        self._target_system: Optional[Any] = None
        self._performance_fn: Optional[Callable[[], float]] = None
        self._apply_fn: Optional[Callable[[Modification], None]] = None
        self._rollback_fn: Optional[Callable[[], None]] = None

        # History
        self._cycle_history: List[ImprovementCycle] = []
        self._total_improvement = 0.0

    def set_target_system(
        self,
        system: Any,
        performance_fn: Callable[[], float],
        apply_fn: Optional[Callable[[Modification], None]] = None,
        rollback_fn: Optional[Callable[[], None]] = None,
        state_accessor: Optional[Tuple[Callable, Callable]] = None,
    ) -> None:
        """
        Set the target system to improve.

        Args:
            system: The system to improve
            performance_fn: Function that returns current performance
            apply_fn: Function to apply modifications
            rollback_fn: Function to rollback modifications
            state_accessor: Tuple of (get_state, set_state) functions
        """
        self._target_system = system
        self._performance_fn = performance_fn

        # Default apply/rollback if not provided
        if apply_fn is not None:
            self._apply_fn = apply_fn
        else:
            self._apply_fn = lambda m: None  # No-op

        if rollback_fn is not None:
            self._rollback_fn = rollback_fn
        else:
            self._rollback_fn = lambda: None  # No-op

        # Set up verifier
        self.verifier.register_test_suite("main", performance_fn)
        self.verifier.update_baseline(performance_fn())

        # Set up updater
        self.updater.set_performance_monitor(performance_fn)
        if state_accessor:
            self.updater.set_state_accessors(*state_accessor)

        # Register components with generator
        self._register_components()

    def _register_components(self) -> None:
        """Register system components with the generator."""
        if self._target_system is None:
            return

        # Try to discover components
        if hasattr(self._target_system, "__dict__"):
            for name, value in self._target_system.__dict__.items():
                if not name.startswith("_"):
                    self.generator.register_component(
                        name,
                        {
                            "type": type(value).__name__,
                            "adjustable_params": ["weights", "bias"],
                        },
                    )

    def run_improvement_cycle(self) -> ImprovementCycle:
        """
        Run a single improvement cycle.

        Returns:
            ImprovementCycle record
        """
        self._cycle_count += 1
        cycle = ImprovementCycle(
            cycle_id=self._cycle_count,
            phase=ImprovementPhase.GENERATING,
            start_time=time.time(),
            end_time=None,
            candidates_generated=0,
            candidates_verified=0,
            candidates_applied=0,
            initial_performance=self._performance_fn() if self._performance_fn else 0.0,
            final_performance=0.0,
            improvement=0.0,
            best_modification=None,
        )

        try:
            # Phase 1: Generate candidates
            self._phase = ImprovementPhase.GENERATING
            strategy = self.meta_learner.select_strategy()

            candidates = self.generator.generate_candidates(
                cycle.initial_performance, context={"strategy": strategy.strategy_id}
            )
            cycle.candidates_generated = len(candidates)
            cycle.details["strategy"] = strategy.strategy_id

            if not candidates:
                cycle.phase = ImprovementPhase.IDLE
                cycle.end_time = time.time()
                self._cycle_history.append(cycle)
                return cycle

            # Phase 2: Verify candidates
            self._phase = ImprovementPhase.VERIFYING
            verified = []

            for candidate in candidates:
                result = self.verifier.verify(
                    candidate,
                    self._apply_fn,
                    self._rollback_fn,
                    method=VerificationMethod.ROLLBACK_TEST,
                )

                if result.verified:
                    verified.append((candidate, result))

            cycle.candidates_verified = len(verified)

            if not verified:
                self._consecutive_failures += 1
                cycle.phase = ImprovementPhase.IDLE
                cycle.end_time = time.time()
                self._cycle_history.append(cycle)
                return cycle

            # Phase 3: Apply best modification
            self._phase = ImprovementPhase.APPLYING

            # Sort by expected improvement
            verified.sort(key=lambda x: x[1].measured_improvement, reverse=True)
            best_candidate, best_verification = verified[0]

            update_result = self.updater.apply(best_candidate, best_verification)

            if update_result.status == UpdateStatus.APPLIED:
                cycle.candidates_applied = 1
                cycle.best_modification = best_candidate

                # Re-apply since verifier rolled back
                self._apply_fn(best_candidate)

            # Phase 4: Evaluate
            self._phase = ImprovementPhase.EVALUATING
            cycle.final_performance = self._performance_fn() if self._performance_fn else 0.0
            cycle.improvement = cycle.final_performance - cycle.initial_performance

            # Phase 5: Learn
            self._phase = ImprovementPhase.LEARNING

            # Record with meta-learner
            self.meta_learner.record_experience(best_candidate, update_result)
            self.meta_learner.record_performance(cycle.final_performance)

            # Record with generator
            self.generator.record_outcome(best_candidate, cycle.improvement)

            # Update statistics
            if cycle.improvement > 0:
                self._consecutive_failures = 0
                self._total_improvement += cycle.improvement
                self._last_improvement_time = time.time()
            else:
                self._consecutive_failures += 1

                # Rollback if no improvement and safety mode
                if self.params.safety_mode and cycle.improvement < 0:
                    self._rollback_fn()

        except Exception as e:
            cycle.details["error"] = str(e)
            self._consecutive_failures += 1

        finally:
            self._phase = ImprovementPhase.IDLE
            cycle.phase = ImprovementPhase.IDLE
            cycle.end_time = time.time()
            self._cycle_history.append(cycle)

        return cycle

    def should_improve(self) -> Tuple[bool, str]:
        """Check if improvement should be attempted."""
        # Check failure limit
        if self._consecutive_failures >= self.params.max_failed_attempts:
            return False, "max_failures_reached"

        # Check interval
        if time.time() - self._last_improvement_time < self.params.improvement_interval:
            return False, "interval_not_reached"

        # Consult meta-learner
        should, reason = self.meta_learner.should_improve()
        return should, reason

    def run_auto_improve(
        self,
        max_cycles: int = 100,
        target_performance: Optional[float] = None,
    ) -> List[ImprovementCycle]:
        """
        Run automatic improvement loop.

        Args:
            max_cycles: Maximum improvement cycles
            target_performance: Stop when this performance is reached

        Returns:
            List of improvement cycles
        """
        cycles = []

        for i in range(max_cycles):
            # Check stopping conditions
            if target_performance is not None and self._performance_fn:
                if self._performance_fn() >= target_performance:
                    break

            should, reason = self.should_improve()
            if not should and reason == "max_failures_reached":
                break

            # Run cycle
            cycle = self.run_improvement_cycle()
            cycles.append(cycle)

            # Brief pause
            time.sleep(0.01)

        return cycles

    def add_safety_constraint(
        self,
        constraint: Callable[[], bool],
    ) -> None:
        """Add a safety constraint that must always hold."""
        self.verifier.register_constraint(constraint)

    def get_improvement_summary(self) -> Dict[str, Any]:
        """Get summary of improvement progress."""
        if not self._cycle_history:
            return {
                "total_cycles": 0,
                "successful_cycles": 0,
                "total_improvement": 0.0,
            }

        successful = [c for c in self._cycle_history if c.improvement > 0]

        return {
            "total_cycles": len(self._cycle_history),
            "successful_cycles": len(successful),
            "success_rate": len(successful) / len(self._cycle_history),
            "total_improvement": self._total_improvement,
            "avg_improvement_per_cycle": self._total_improvement / len(self._cycle_history),
            "consecutive_failures": self._consecutive_failures,
            "current_performance": self._performance_fn() if self._performance_fn else 0.0,
            "best_cycle": max(self._cycle_history, key=lambda c: c.improvement).cycle_id
            if self._cycle_history
            else None,
        }

    def get_component_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics from all components."""
        return {
            "generator": self.generator.get_statistics(),
            "verifier": self.verifier.get_statistics(),
            "updater": self.updater.get_statistics(),
            "meta_learner": self.meta_learner.get_statistics(),
        }

    def get_recent_cycles(self, n: int = 10) -> List[ImprovementCycle]:
        """Get the n most recent cycles."""
        return self._cycle_history[-n:]

    def reset(self) -> None:
        """Reset the entire system."""
        self.generator.reset()
        self.verifier.reset()
        self.updater.reset()
        self.meta_learner.reset()

        self._phase = ImprovementPhase.IDLE
        self._cycle_count = 0
        self._consecutive_failures = 0
        self._last_improvement_time = 0.0
        self._cycle_history = []
        self._total_improvement = 0.0

    def save_state(self) -> Dict[str, Any]:
        """Save current state for persistence."""
        return {
            "cycle_count": self._cycle_count,
            "total_improvement": self._total_improvement,
            "consecutive_failures": self._consecutive_failures,
            "generator_stats": self.generator.get_statistics(),
            "meta_learner_stats": self.meta_learner.get_statistics(),
            "cycle_history_summary": [
                {
                    "cycle_id": c.cycle_id,
                    "improvement": c.improvement,
                    "duration": c.duration,
                }
                for c in self._cycle_history[-100:]
            ],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall DGM statistics."""
        return {
            "phase": self._phase.value,
            "cycle_count": self._cycle_count,
            "total_improvement": self._total_improvement,
            "consecutive_failures": self._consecutive_failures,
            "auto_improve": self.params.auto_improve,
            "safety_mode": self.params.safety_mode,
            **self.get_improvement_summary(),
        }
