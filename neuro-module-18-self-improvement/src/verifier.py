"""
Change Verifier: Evaluates proposed modifications before application.

The verifier ensures that proposed changes actually improve the system
and don't introduce regressions. It uses multiple validation strategies.

Based on:
- Gödel Machine proof verification
- Regression testing
- Safety constraints for self-modifying systems
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import Enum
import numpy as np
import time
import copy

from .generator import Modification, ModificationType


class VerificationMethod(Enum):
    """Methods for verifying modifications."""
    SIMULATION = "simulation"     # Simulate change effects
    ROLLBACK_TEST = "rollback"    # Apply and test with rollback option
    PROOF = "proof"               # Formal verification (limited)
    STATISTICAL = "statistical"   # Statistical significance testing
    ENSEMBLE = "ensemble"         # Multiple verification methods


@dataclass
class VerifierParams:
    """Parameters for the verifier."""
    min_improvement: float = 0.01      # Minimum improvement to accept
    confidence_threshold: float = 0.7  # Minimum confidence required
    max_regression_risk: float = 0.1   # Maximum acceptable regression risk
    n_simulations: int = 10            # Simulations for statistical tests
    rollback_on_failure: bool = True   # Auto-rollback if fails
    require_reversibility: bool = True # Only accept reversible changes


@dataclass
class VerificationResult:
    """Result of verification."""
    modification: Modification
    verified: bool
    method_used: VerificationMethod
    measured_improvement: float
    confidence: float
    regression_risk: float
    validation_details: Dict[str, Any]
    warnings: List[str]
    timestamp: float = field(default_factory=time.time)

    @property
    def should_apply(self) -> bool:
        """Whether the modification should be applied."""
        return self.verified and len(self.warnings) == 0


class ChangeVerifier:
    """
    Verifier that validates proposed modifications.

    The verifier is a critical safety component that prevents harmful
    self-modifications. It uses multiple strategies:

    1. **Simulation**: Test changes in a sandboxed environment
    2. **Statistical testing**: Ensure improvements are significant
    3. **Regression detection**: Check for unintended side effects
    4. **Safety constraints**: Enforce invariants

    The Gödel Machine concept requires proving that changes are
    improvements before applying them.
    """

    def __init__(self, params: Optional[VerifierParams] = None):
        self.params = params or VerifierParams()

        # Registered test suites
        self._test_suites: Dict[str, Callable[[], float]] = {}

        # Safety constraints (invariants that must hold)
        self._constraints: List[Callable[[], bool]] = []

        # Baseline performance (cached)
        self._baseline_performance: Optional[float] = None
        self._baseline_timestamp: float = 0.0

        # History
        self._verification_history: List[VerificationResult] = []

    def register_test_suite(
        self,
        name: str,
        test_fn: Callable[[], float],
    ) -> None:
        """Register a test suite that returns a performance score."""
        self._test_suites[name] = test_fn

    def register_constraint(
        self,
        constraint_fn: Callable[[], bool],
    ) -> None:
        """Register a safety constraint that must always hold."""
        self._constraints.append(constraint_fn)

    def update_baseline(self, performance: float) -> None:
        """Update the baseline performance."""
        self._baseline_performance = performance
        self._baseline_timestamp = time.time()

    def verify(
        self,
        modification: Modification,
        apply_fn: Callable[[Modification], None],
        rollback_fn: Callable[[], None],
        method: VerificationMethod = VerificationMethod.ROLLBACK_TEST,
    ) -> VerificationResult:
        """
        Verify a proposed modification.

        Args:
            modification: The modification to verify
            apply_fn: Function to apply the modification
            rollback_fn: Function to rollback the modification
            method: Verification method to use

        Returns:
            VerificationResult
        """
        warnings = []

        # Check reversibility requirement
        if self.params.require_reversibility and not modification.reversible:
            return VerificationResult(
                modification=modification,
                verified=False,
                method_used=method,
                measured_improvement=0.0,
                confidence=0.0,
                regression_risk=1.0,
                validation_details={'reason': 'not_reversible'},
                warnings=['Modification is not reversible'],
            )

        # Get baseline if not cached
        if self._baseline_performance is None:
            self._baseline_performance = self._run_tests()
            self._baseline_timestamp = time.time()

        # Perform verification based on method
        if method == VerificationMethod.SIMULATION:
            result = self._verify_simulation(modification, apply_fn, rollback_fn)
        elif method == VerificationMethod.ROLLBACK_TEST:
            result = self._verify_rollback(modification, apply_fn, rollback_fn)
        elif method == VerificationMethod.STATISTICAL:
            result = self._verify_statistical(modification, apply_fn, rollback_fn)
        elif method == VerificationMethod.ENSEMBLE:
            result = self._verify_ensemble(modification, apply_fn, rollback_fn)
        else:
            result = self._verify_rollback(modification, apply_fn, rollback_fn)

        # Check safety constraints
        constraint_violations = self._check_constraints()
        if constraint_violations:
            result.verified = False
            result.warnings.extend(constraint_violations)
            result.regression_risk = 1.0

        self._verification_history.append(result)

        return result

    def _verify_simulation(
        self,
        modification: Modification,
        apply_fn: Callable,
        rollback_fn: Callable,
    ) -> VerificationResult:
        """Verify by simulating effects without actual application."""
        # Estimate improvement based on modification type and history
        similar_mods = [
            r for r in self._verification_history
            if r.modification.mod_type == modification.mod_type
        ]

        if similar_mods:
            avg_improvement = np.mean([r.measured_improvement for r in similar_mods])
            confidence = min(0.8, len(similar_mods) / 10)
        else:
            avg_improvement = modification.expected_improvement
            confidence = 0.3

        # Simulation is less confident but doesn't risk actual changes
        return VerificationResult(
            modification=modification,
            verified=avg_improvement > self.params.min_improvement,
            method_used=VerificationMethod.SIMULATION,
            measured_improvement=float(avg_improvement),
            confidence=confidence,
            regression_risk=0.1,  # Low risk since no actual change
            validation_details={'simulated': True, 'similar_mods': len(similar_mods)},
            warnings=['Verification by simulation only'],
        )

    def _verify_rollback(
        self,
        modification: Modification,
        apply_fn: Callable,
        rollback_fn: Callable,
    ) -> VerificationResult:
        """Verify by applying, testing, and rolling back."""
        warnings = []
        baseline = self._baseline_performance or 0.0

        # Apply the modification
        try:
            apply_fn(modification)
        except Exception as e:
            return VerificationResult(
                modification=modification,
                verified=False,
                method_used=VerificationMethod.ROLLBACK_TEST,
                measured_improvement=0.0,
                confidence=0.0,
                regression_risk=1.0,
                validation_details={'apply_error': str(e)},
                warnings=[f'Failed to apply: {e}'],
            )

        # Run tests
        try:
            new_performance = self._run_tests()
        except Exception as e:
            rollback_fn()
            return VerificationResult(
                modification=modification,
                verified=False,
                method_used=VerificationMethod.ROLLBACK_TEST,
                measured_improvement=0.0,
                confidence=0.0,
                regression_risk=1.0,
                validation_details={'test_error': str(e)},
                warnings=[f'Tests failed: {e}'],
            )

        improvement = new_performance - baseline

        # Check for regression
        if improvement < -self.params.max_regression_risk:
            warnings.append('Significant regression detected')

        # Rollback to assess if reversible
        try:
            rollback_fn()
            rollback_performance = self._run_tests()
            if abs(rollback_performance - baseline) > 0.01:
                warnings.append('Rollback did not restore baseline')
        except Exception as e:
            warnings.append(f'Rollback failed: {e}')

        # Re-apply if verified (caller will decide)
        verified = (
            improvement >= self.params.min_improvement
            and len(warnings) == 0
        )

        return VerificationResult(
            modification=modification,
            verified=verified,
            method_used=VerificationMethod.ROLLBACK_TEST,
            measured_improvement=float(improvement),
            confidence=0.9,  # High confidence from actual testing
            regression_risk=max(0, -improvement) if improvement < 0 else 0.0,
            validation_details={
                'baseline': baseline,
                'new_performance': new_performance,
            },
            warnings=warnings,
        )

    def _verify_statistical(
        self,
        modification: Modification,
        apply_fn: Callable,
        rollback_fn: Callable,
    ) -> VerificationResult:
        """Verify with statistical significance testing."""
        baseline = self._baseline_performance or 0.0
        n_trials = self.params.n_simulations

        # Collect baseline samples
        baseline_samples = []
        for _ in range(n_trials):
            baseline_samples.append(self._run_tests())

        # Apply and collect modified samples
        try:
            apply_fn(modification)
        except Exception as e:
            return VerificationResult(
                modification=modification,
                verified=False,
                method_used=VerificationMethod.STATISTICAL,
                measured_improvement=0.0,
                confidence=0.0,
                regression_risk=1.0,
                validation_details={'apply_error': str(e)},
                warnings=[f'Failed to apply: {e}'],
            )

        modified_samples = []
        for _ in range(n_trials):
            modified_samples.append(self._run_tests())

        # Rollback
        rollback_fn()

        # Statistical test (simple t-test approximation)
        baseline_mean = np.mean(baseline_samples)
        modified_mean = np.mean(modified_samples)
        baseline_std = np.std(baseline_samples) + 1e-8
        modified_std = np.std(modified_samples) + 1e-8

        improvement = modified_mean - baseline_mean

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((baseline_std**2 + modified_std**2) / 2)
        effect_size = improvement / pooled_std

        # Confidence based on effect size and sample size
        confidence = min(0.95, abs(effect_size) * np.sqrt(n_trials) / 10)

        verified = (
            improvement >= self.params.min_improvement
            and effect_size > 0.2  # Small effect size threshold
        )

        return VerificationResult(
            modification=modification,
            verified=verified,
            method_used=VerificationMethod.STATISTICAL,
            measured_improvement=float(improvement),
            confidence=float(confidence),
            regression_risk=max(0, -improvement),
            validation_details={
                'baseline_mean': float(baseline_mean),
                'modified_mean': float(modified_mean),
                'effect_size': float(effect_size),
                'n_samples': n_trials,
            },
            warnings=[],
        )

    def _verify_ensemble(
        self,
        modification: Modification,
        apply_fn: Callable,
        rollback_fn: Callable,
    ) -> VerificationResult:
        """Verify using multiple methods and aggregate results."""
        results = []

        # Simulation
        results.append(self._verify_simulation(modification, apply_fn, rollback_fn))

        # Rollback test
        results.append(self._verify_rollback(modification, apply_fn, rollback_fn))

        # Aggregate
        verified_count = sum(1 for r in results if r.verified)
        avg_improvement = np.mean([r.measured_improvement for r in results])
        avg_confidence = np.mean([r.confidence for r in results])
        max_risk = max(r.regression_risk for r in results)

        all_warnings = []
        for r in results:
            all_warnings.extend(r.warnings)

        return VerificationResult(
            modification=modification,
            verified=verified_count >= len(results) / 2,
            method_used=VerificationMethod.ENSEMBLE,
            measured_improvement=float(avg_improvement),
            confidence=float(avg_confidence),
            regression_risk=float(max_risk),
            validation_details={
                'methods_used': [r.method_used.value for r in results],
                'votes': verified_count,
            },
            warnings=list(set(all_warnings)),
        )

    def _run_tests(self) -> float:
        """Run all registered test suites and return aggregate score."""
        if not self._test_suites:
            return 0.5  # Default neutral score

        scores = []
        for name, test_fn in self._test_suites.items():
            try:
                score = test_fn()
                scores.append(score)
            except Exception:
                scores.append(0.0)

        return float(np.mean(scores))

    def _check_constraints(self) -> List[str]:
        """Check all safety constraints."""
        violations = []
        for i, constraint in enumerate(self._constraints):
            try:
                if not constraint():
                    violations.append(f'Constraint {i} violated')
            except Exception as e:
                violations.append(f'Constraint {i} error: {e}')

        return violations

    def get_statistics(self) -> Dict[str, Any]:
        """Get verifier statistics."""
        if not self._verification_history:
            return {
                'n_verifications': 0,
                'approval_rate': 0.0,
                'avg_improvement': 0.0,
            }

        verified = [r for r in self._verification_history if r.verified]

        return {
            'n_verifications': len(self._verification_history),
            'approval_rate': len(verified) / len(self._verification_history),
            'avg_improvement': float(np.mean([
                r.measured_improvement for r in self._verification_history
            ])),
            'avg_confidence': float(np.mean([
                r.confidence for r in self._verification_history
            ])),
            'n_test_suites': len(self._test_suites),
            'n_constraints': len(self._constraints),
        }

    def reset(self) -> None:
        """Reset verifier state."""
        self._verification_history = []
        self._baseline_performance = None
        self._baseline_timestamp = 0.0
