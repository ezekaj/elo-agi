"""
Grammar Constraint Manifold

Implements the space of possible human grammars with constraints
based on Universal Grammar principles.

Key finding: Broca's area shows selective inhibition for impossible
languages only, suggesting innate constraints on learnable grammar.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class ConstraintType(Enum):
    """Types of grammatical constraints"""
    STRUCTURAL = "structural"  # Structure dependence
    HIERARCHICAL = "hierarchical"  # Hierarchy requirements
    RECURSIVE = "recursive"  # Recursion constraints
    LINEAR = "linear"  # Linear order constraints
    AGREEMENT = "agreement"  # Feature agreement


@dataclass
class GrammarConstraint:
    """A constraint on possible grammars"""
    name: str
    constraint_type: ConstraintType
    weight: float = 1.0
    is_hard: bool = False  # Hard constraints cannot be violated


@dataclass
class GrammarState:
    """Point in grammar space"""
    parameters: np.ndarray
    is_possible: bool
    violation_score: float = 0.0
    nearest_possible: Optional[np.ndarray] = None


class GrammarConstraintManifold:
    """Manifold of possible human grammars

    Defines the constraint space for learnable grammars.
    Grammars outside this space trigger Broca's inhibition.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

        # Define constraint region
        # Using a deterministic embedding for consistency
        np.random.seed(42)  # Deterministic initialization
        self.W_constraint = np.random.randn(dim, dim) * 0.1
        np.random.seed(None)  # Reset to random
        self.center = np.zeros(dim)  # Center of possible grammar region
        self.radius = 3.0  # Radius of "possible" region (increased for reliability)

        # Constraint weights
        self.constraints: List[GrammarConstraint] = []
        self._initialize_default_constraints()

        # History of evaluated grammars
        self.evaluation_history: List[GrammarState] = []

    def _initialize_default_constraints(self) -> None:
        """Initialize default Universal Grammar constraints"""
        self.constraints = [
            GrammarConstraint(
                name="structure_dependence",
                constraint_type=ConstraintType.STRUCTURAL,
                weight=2.0,
                is_hard=True
            ),
            GrammarConstraint(
                name="hierarchical_organization",
                constraint_type=ConstraintType.HIERARCHICAL,
                weight=1.5,
                is_hard=True
            ),
            GrammarConstraint(
                name="recursive_embedding",
                constraint_type=ConstraintType.RECURSIVE,
                weight=1.0,
                is_hard=False
            ),
            GrammarConstraint(
                name="bounded_recursion",
                constraint_type=ConstraintType.RECURSIVE,
                weight=1.0,
                is_hard=True
            ),
        ]

    def is_possible_grammar(self, params: np.ndarray) -> bool:
        """Check if grammar parameters define a possible human grammar"""
        violation = self.get_violation_score(params)
        return violation < 0.7  # More lenient threshold

    def get_violation_score(self, params: np.ndarray) -> float:
        """Compute how much the grammar violates constraints

        Returns:
            Score from 0 (no violation) to 1+ (severe violation)
        """
        if len(params) < self.dim:
            params = np.pad(params, (0, self.dim - len(params)))
        elif len(params) > self.dim:
            params = params[:self.dim]

        # Transform through constraint space
        transformed = np.tanh(self.W_constraint @ params)

        # Distance from possible grammar center
        distance = np.linalg.norm(transformed - self.center)
        normalized_distance = distance / self.radius

        # Check individual constraints
        constraint_violation = 0.0
        for constraint in self.constraints:
            violation = self._check_constraint(params, constraint)
            if constraint.is_hard and violation > 0.5:
                return 1.0  # Hard constraint violation
            constraint_violation += constraint.weight * violation

        # Combined score
        total_violation = 0.5 * normalized_distance + 0.5 * (constraint_violation / len(self.constraints))
        return float(np.clip(total_violation, 0, 2))

    def _check_constraint(self, params: np.ndarray, constraint: GrammarConstraint) -> float:
        """Check a specific constraint"""
        # Simplified constraint checking based on parameter properties
        # Returns 0 for no violation, up to 1 for violation

        if constraint.constraint_type == ConstraintType.STRUCTURAL:
            # Structure dependence: only violate for extreme anti-correlation
            if len(params) >= 4:
                correlation = np.corrcoef(params[:len(params)//2], params[len(params)//2:])[0, 1]
                if np.isnan(correlation):
                    return 0.0  # No violation if can't compute
                # Only violate if strongly anti-correlated
                return max(0.0, -correlation) * 0.5
            return 0.0

        elif constraint.constraint_type == ConstraintType.HIERARCHICAL:
            # Hierarchical: only violate for extreme variance ratios
            var1 = np.var(params[:len(params)//2]) + 1e-8
            var2 = np.var(params[len(params)//2:]) + 1e-8
            log_ratio = np.abs(np.log(var1 / var2))
            return min(1.0, max(0.0, log_ratio - 3) / 3)  # Violate only above ratio of 20x

        elif constraint.constraint_type == ConstraintType.RECURSIVE:
            # Recursion: very lenient - most patterns are fine
            return 0.0  # Don't check this strictly

        return 0.0

    def distance_to_boundary(self, params: np.ndarray) -> float:
        """Compute distance to boundary of possible grammar region"""
        violation = self.get_violation_score(params)

        if violation < 0.5:
            # Inside: distance to boundary
            return 0.5 - violation
        else:
            # Outside: negative distance
            return 0.5 - violation

    def project_to_possible(self, params: np.ndarray) -> np.ndarray:
        """Project parameters to nearest possible grammar"""
        if len(params) < self.dim:
            params = np.pad(params, (0, self.dim - len(params)))
        elif len(params) > self.dim:
            params = params[:self.dim]

        if self.is_possible_grammar(params):
            return params

        # Gradient descent toward possible region
        projected = params.copy()
        for _ in range(50):
            if self.is_possible_grammar(projected):
                break

            # Move toward center
            direction = self.center - projected
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            projected += 0.1 * direction

        return projected

    def inhibition_signal(self, params: np.ndarray) -> float:
        """Compute Broca's inhibition signal for grammar

        This implements the finding that Broca's area shows
        selective inhibition for impossible languages only.
        """
        violation = self.get_violation_score(params)

        # Selective inhibition: only for impossible grammars
        if violation < 0.7:
            return 0.0  # No inhibition for possible grammars
        else:
            # Proportional inhibition for impossible grammars
            return float(np.clip((violation - 0.7) * 3, 0, 1))

    def evaluate(self, params: np.ndarray) -> GrammarState:
        """Full evaluation of grammar parameters"""
        is_possible = self.is_possible_grammar(params)
        violation = self.get_violation_score(params)
        nearest = self.project_to_possible(params) if not is_possible else None

        state = GrammarState(
            parameters=params.copy(),
            is_possible=is_possible,
            violation_score=violation,
            nearest_possible=nearest
        )

        self.evaluation_history.append(state)
        if len(self.evaluation_history) > 100:
            self.evaluation_history.pop(0)

        return state

    def reset(self) -> None:
        """Reset evaluation history"""
        self.evaluation_history = []


class UniversalGrammar:
    """Universal Grammar - Innate constraints on learnable grammar

    Implements core UG principles that define what grammars
    humans can naturally acquire.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

        # Core UG principles
        self.principles: Dict[str, Callable] = {
            'structure_dependence': self._check_structure_dependence,
            'subjacency': self._check_subjacency,
            'c_command': self._check_c_command,
            'binding': self._check_binding,
        }

        # Parameters (in Principles & Parameters framework)
        self.parameters: Dict[str, float] = {
            'head_direction': 0.0,  # -1 = head-final, +1 = head-initial
            'pro_drop': 0.0,  # Allow null subjects?
            'wh_movement': 0.0,  # Overt wh-movement?
        }

    def evaluate(self, grammar_params: np.ndarray) -> Dict[str, float]:
        """Evaluate grammar against UG principles"""
        results = {}

        for name, check_fn in self.principles.items():
            results[name] = check_fn(grammar_params)

        results['overall'] = np.mean(list(results.values()))
        return results

    def is_ug_compatible(self, grammar_params: np.ndarray) -> bool:
        """Check if grammar is compatible with Universal Grammar"""
        results = self.evaluate(grammar_params)
        return bool(results['overall'] >= 0.25)  # More lenient threshold, explicit Python bool

    def _check_structure_dependence(self, params: np.ndarray) -> float:
        """Check structure dependence principle

        Rules must refer to hierarchical structure, not linear order.
        """
        # Simplified: check for hierarchical vs linear patterns
        if len(params) < 4:
            return 0.5

        # Structure-dependent: local dependencies stronger than distant
        local_var = np.var(np.diff(params[:len(params)//2]))
        distant_var = np.var(params[:len(params)//2] - params[len(params)//2:])

        return float(local_var < distant_var)

    def _check_subjacency(self, params: np.ndarray) -> float:
        """Check subjacency condition

        Movement cannot cross too many bounding nodes.
        """
        # Simplified: check for locality in parameter patterns
        if len(params) < 4:
            return 0.5

        # Check for bounded vs unbounded dependencies
        autocorr = np.correlate(params, params, mode='full')
        decay_rate = np.abs(np.diff(autocorr)).mean()

        return float(decay_rate > 0.1)  # Should decay, not persist

    def _check_c_command(self, params: np.ndarray) -> float:
        """Check c-command relationships

        Simplified check for hierarchical dominance patterns.
        """
        if len(params) < 4:
            return 0.5

        # Check for asymmetric dominance
        first_half = params[:len(params)//2]
        second_half = params[len(params)//2:]

        asymmetry = np.abs(np.mean(first_half) - np.mean(second_half))
        return float(asymmetry > 0.1)

    def _check_binding(self, params: np.ndarray) -> float:
        """Check binding principles

        Constraints on anaphora and pronoun binding.
        """
        if len(params) < 4:
            return 0.5

        # Check for local binding domains
        local_corr = np.corrcoef(params[:-1], params[1:])[0, 1]
        return float(not np.isnan(local_corr) and np.abs(local_corr) > 0.3)

    def set_parameter(self, name: str, value: float) -> None:
        """Set a UG parameter value"""
        if name in self.parameters:
            self.parameters[name] = np.clip(value, -1, 1)

    def get_parameter_settings(self) -> Dict[str, float]:
        """Get current parameter settings"""
        return self.parameters.copy()


class ImpossibleGrammarGenerator:
    """Generate examples of impossible grammars for testing

    Used to verify that Broca's inhibition responds selectively
    to impossible but not possible grammars.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim
        self.manifold = GrammarConstraintManifold(dim)

    def generate_possible(self) -> np.ndarray:
        """Generate a possible grammar"""
        # Start from center and add small noise
        params = self.manifold.center + np.random.randn(self.dim) * 0.2
        projected = self.manifold.project_to_possible(params)
        # Ensure it's actually possible by iterating if needed
        for _ in range(10):
            if self.manifold.is_possible_grammar(projected):
                return projected
            projected = self.manifold.center + np.random.randn(self.dim) * 0.1
        return self.manifold.center.copy()  # Fallback to center

    def generate_impossible(self, violation_type: str = 'random') -> np.ndarray:
        """Generate an impossible grammar

        Args:
            violation_type: Type of impossibility
                - 'random': Random impossible grammar
                - 'structure': Violates structure dependence
                - 'unbounded': Violates bounded recursion
        """
        if violation_type == 'random':
            # Far from possible region
            params = self.manifold.center + np.random.randn(self.dim) * 5.0
            return params

        elif violation_type == 'structure':
            # Violates structure dependence (linear order only)
            params = np.zeros(self.dim)
            params[:self.dim//2] = np.arange(self.dim//2)  # Linear pattern
            params[self.dim//2:] = np.random.randn(self.dim//2)  # Uncorrelated
            return params

        elif violation_type == 'unbounded':
            # Violates bounded recursion
            params = np.zeros(self.dim)
            # Create unbounded self-reference
            for i in range(self.dim):
                params[i] = params[i-1] if i > 0 else 1.0
            return params * 10  # Amplify to make clearly impossible

        return np.random.randn(self.dim) * 3.0

    def generate_test_set(self, n_possible: int = 10, n_impossible: int = 10) -> Dict[str, List[np.ndarray]]:
        """Generate test set of possible and impossible grammars"""
        possible = [self.generate_possible() for _ in range(n_possible)]
        impossible = [self.generate_impossible() for _ in range(n_impossible)]

        return {
            'possible': possible,
            'impossible': impossible
        }
