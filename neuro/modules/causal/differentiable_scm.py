"""
Differentiable Structural Causal Models.

Implements gradient-based learning of causal mechanisms using
a numpy-based approach that mirrors PyTorch semantics for portability.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class ActivationType(Enum):
    """Activation functions for causal mechanisms."""

    LINEAR = "linear"
    TANH = "tanh"
    RELU = "relu"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"


def apply_activation(x: np.ndarray, activation: ActivationType) -> np.ndarray:
    """Apply activation function."""
    if activation == ActivationType.LINEAR:
        return x
    elif activation == ActivationType.TANH:
        return np.tanh(x)
    elif activation == ActivationType.RELU:
        return np.maximum(0, x)
    elif activation == ActivationType.SIGMOID:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    elif activation == ActivationType.SOFTPLUS:
        return np.log1p(np.exp(np.clip(x, -500, 500)))
    return x


def activation_gradient(x: np.ndarray, activation: ActivationType) -> np.ndarray:
    """Compute gradient of activation function."""
    if activation == ActivationType.LINEAR:
        return np.ones_like(x)
    elif activation == ActivationType.TANH:
        return 1 - np.tanh(x) ** 2
    elif activation == ActivationType.RELU:
        return (x > 0).astype(float)
    elif activation == ActivationType.SIGMOID:
        s = apply_activation(x, ActivationType.SIGMOID)
        return s * (1 - s)
    elif activation == ActivationType.SOFTPLUS:
        return apply_activation(x, ActivationType.SIGMOID)
    return np.ones_like(x)


@dataclass
class MLPLayer:
    """Single layer of an MLP."""

    weights: np.ndarray
    bias: np.ndarray
    activation: ActivationType = ActivationType.TANH

    # Gradients (for learning)
    grad_weights: Optional[np.ndarray] = None
    grad_bias: Optional[np.ndarray] = None

    # Cached values for backprop
    _input: Optional[np.ndarray] = None
    _pre_activation: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self._input = x
        self._pre_activation = x @ self.weights + self.bias
        return apply_activation(self._pre_activation, self.activation)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass, returns gradient w.r.t. input."""
        if self._pre_activation is None or self._input is None:
            raise RuntimeError("Must call forward before backward")

        # Gradient through activation
        grad_pre = grad_output * activation_gradient(self._pre_activation, self.activation)

        # Gradients for parameters
        self.grad_weights = np.outer(self._input, grad_pre)
        self.grad_bias = grad_pre

        # Gradient w.r.t. input
        grad_input = grad_pre @ self.weights.T
        return grad_input

    def update(self, learning_rate: float) -> None:
        """Update parameters using computed gradients."""
        if self.grad_weights is not None:
            self.weights -= learning_rate * self.grad_weights
        if self.grad_bias is not None:
            self.bias -= learning_rate * self.grad_bias


class NeuralNetwork:
    """Simple MLP for causal mechanisms."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: ActivationType = ActivationType.TANH,
        random_seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or [32]
        self.activation = activation

        rng = np.random.default_rng(random_seed)

        # Build layers
        self.layers: List[MLPLayer] = []
        dims = [input_dim] + self.hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            weights = rng.normal(0, scale, (dims[i], dims[i + 1]))
            bias = np.zeros(dims[i + 1])

            # Last layer is linear, others use activation
            act = ActivationType.LINEAR if i == len(dims) - 2 else activation
            self.layers.append(MLPLayer(weights, bias, act))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through network."""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def update(self, learning_rate: float) -> None:
        """Update all layer parameters."""
        for layer in self.layers:
            layer.update(learning_rate)

    def parameters(self) -> List[np.ndarray]:
        """Get all parameters as a list."""
        params = []
        for layer in self.layers:
            params.extend([layer.weights, layer.bias])
        return params


@dataclass
class CausalMechanism:
    """
    A learnable causal mechanism: V_i = f_i(PA_i, U_i)

    Represents how a variable is determined by its parents
    and an exogenous noise term, with a learnable neural network.
    """

    variable: str
    parents: List[str]
    noise_dim: int = 1

    # Neural network f_i
    network: Optional[NeuralNetwork] = None

    # For analytical mechanisms (optional)
    analytical_fn: Optional[Callable[[Dict[str, float], float], float]] = None

    # Noise distribution parameters
    noise_mean: float = 0.0
    noise_std: float = 1.0

    # Statistics
    n_forward_calls: int = 0
    n_backward_calls: int = 0

    def __post_init__(self):
        if self.network is None and self.analytical_fn is None:
            # Create default network
            input_dim = len(self.parents) + self.noise_dim
            self.network = NeuralNetwork(input_dim, 1, hidden_dims=[16, 16])

    def forward(
        self,
        parent_values: Dict[str, float],
        noise: Optional[float] = None,
    ) -> float:
        """Evaluate the causal mechanism."""
        self.n_forward_calls += 1

        if noise is None:
            noise = np.random.normal(self.noise_mean, self.noise_std)

        if self.analytical_fn is not None:
            return self.analytical_fn(parent_values, noise)

        # Construct input vector: [parent_values..., noise]
        input_vec = np.array([parent_values.get(p, 0.0) for p in self.parents] + [noise])
        output = self.network.forward(input_vec)
        return float(output[0])

    def backward(self, grad_output: float) -> Dict[str, float]:
        """
        Backward pass through mechanism.

        Returns gradients w.r.t. parent values.
        """
        self.n_backward_calls += 1

        if self.network is None:
            return {p: 0.0 for p in self.parents}

        grad = self.network.backward(np.array([grad_output]))

        # Split gradient into parent gradients
        parent_grads = {}
        for i, p in enumerate(self.parents):
            parent_grads[p] = float(grad[i])

        return parent_grads

    def sample_noise(self, n_samples: int = 1) -> np.ndarray:
        """Sample noise values."""
        return np.random.normal(self.noise_mean, self.noise_std, n_samples)


@dataclass
class InterventionSpec:
    """Specification of a causal intervention."""

    variable: str
    value: float
    soft: bool = False  # If True, use soft intervention (shift instead of fix)
    shift: float = 0.0  # For soft interventions


class DifferentiableSCM:
    """
    Differentiable Structural Causal Model.

    A PyTorch-style differentiable implementation of Pearl's SCM
    framework, supporting:
    - Forward sampling
    - Interventions (do-calculus)
    - Counterfactual inference
    - Gradient-based learning of mechanisms
    - Causal effect estimation via gradients
    """

    def __init__(
        self,
        name: str = "differentiable_scm",
        random_seed: Optional[int] = None,
    ):
        self.name = name
        self._rng = np.random.default_rng(random_seed)

        # Variables and mechanisms
        self._variables: Set[str] = set()
        self._mechanisms: Dict[str, CausalMechanism] = {}
        self._exogenous: Dict[str, Tuple[float, float]] = {}  # (mean, std)

        # Graph structure
        self._parents: Dict[str, List[str]] = {}
        self._children: Dict[str, List[str]] = {}

        # Cached topological order
        self._topo_order: Optional[List[str]] = None

        # Learning rate
        self.learning_rate = 0.01

        # Statistics
        self._n_samples = 0
        self._n_interventions = 0
        self._n_counterfactuals = 0

    def add_variable(
        self,
        name: str,
        parents: Optional[List[str]] = None,
        mechanism: Optional[CausalMechanism] = None,
        noise_mean: float = 0.0,
        noise_std: float = 1.0,
    ) -> None:
        """Add a variable to the SCM."""
        parents = parents or []
        self._variables.add(name)
        self._parents[name] = parents
        self._exogenous[name] = (noise_mean, noise_std)

        # Update children
        for p in parents:
            if p not in self._children:
                self._children[p] = []
            self._children[p].append(name)

        # Create mechanism if not provided
        if mechanism is None:
            mechanism = CausalMechanism(
                variable=name,
                parents=parents,
                noise_mean=noise_mean,
                noise_std=noise_std,
            )
        self._mechanisms[name] = mechanism

        # Invalidate cached topological order
        self._topo_order = None

    def add_linear_mechanism(
        self,
        name: str,
        parents: List[str],
        coefficients: Dict[str, float],
        intercept: float = 0.0,
        noise_std: float = 1.0,
    ) -> None:
        """Add a variable with linear mechanism."""

        def linear_fn(parent_values: Dict[str, float], noise: float) -> float:
            result = intercept
            for p, coef in coefficients.items():
                result += coef * parent_values.get(p, 0.0)
            return result + noise

        mechanism = CausalMechanism(
            variable=name,
            parents=parents,
            analytical_fn=linear_fn,
            noise_std=noise_std,
        )
        self.add_variable(name, parents, mechanism, noise_std=noise_std)

    def _topological_order(self) -> List[str]:
        """Get variables in topological order."""
        if self._topo_order is not None:
            return self._topo_order

        in_degree = {v: len(self._parents.get(v, [])) for v in self._variables}
        queue = [v for v, d in in_degree.items() if d == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for child in self._children.get(node, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        self._topo_order = order
        return order

    def forward(
        self,
        noise: Optional[Dict[str, float]] = None,
        interventions: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Forward pass through SCM.

        Computes all endogenous variable values given noise
        and optional interventions.
        """
        self._n_samples += 1
        interventions = interventions or {}

        # Sample noise if not provided
        if noise is None:
            noise = {}
            for var in self._variables:
                mean, std = self._exogenous.get(var, (0.0, 1.0))
                noise[var] = float(self._rng.normal(mean, std))

        # Compute values in topological order
        values = {}
        for var in self._topological_order():
            if var in interventions:
                # Intervention: override mechanism
                values[var] = interventions[var]
                self._n_interventions += 1
            else:
                # Apply mechanism
                mechanism = self._mechanisms.get(var)
                if mechanism:
                    parent_values = {p: values.get(p, 0.0) for p in mechanism.parents}
                    values[var] = mechanism.forward(parent_values, noise.get(var))
                else:
                    values[var] = noise.get(var, 0.0)

        return values

    def sample(
        self,
        n_samples: int = 1,
        interventions: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, float]]:
        """Sample from the (possibly intervened) model."""
        return [self.forward(interventions=interventions) for _ in range(n_samples)]

    def intervene(
        self,
        interventions: Dict[str, float],
    ) -> "DifferentiableSCM":
        """
        Create an intervened model do(X=x).

        Returns a new SCM with the intervention applied.
        """
        # Create copy
        intervened = DifferentiableSCM(f"{self.name}_intervened")
        intervened._variables = self._variables.copy()
        intervened._exogenous = self._exogenous.copy()

        for var in self._variables:
            if var in interventions:
                # Replace mechanism with constant
                def const_fn(pv, n, val=interventions[var]):
                    return val

                mechanism = CausalMechanism(
                    variable=var,
                    parents=[],
                    analytical_fn=const_fn,
                )
                intervened._mechanisms[var] = mechanism
                intervened._parents[var] = []
            else:
                intervened._mechanisms[var] = self._mechanisms[var]
                intervened._parents[var] = self._parents.get(var, [])

        # Rebuild children
        intervened._children = {}
        for var, parents in intervened._parents.items():
            for p in parents:
                if p not in intervened._children:
                    intervened._children[p] = []
                intervened._children[p].append(var)

        return intervened

    def counterfactual(
        self,
        evidence: Dict[str, float],
        intervention: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Compute counterfactual: What would Y be if we had done do(X=x),
        given that we observed evidence.

        Steps:
        1. Abduction: Infer noise values consistent with evidence
        2. Action: Apply intervention
        3. Prediction: Compute counterfactual values
        """
        self._n_counterfactuals += 1

        # Step 1: Abduction - infer noise values
        inferred_noise = self._abduct(evidence)

        # Step 2 & 3: Action and Prediction
        return self.forward(noise=inferred_noise, interventions=intervention)

    def _abduct(
        self,
        evidence: Dict[str, float],
        n_iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Infer noise values consistent with observed evidence.

        Uses iterative refinement for non-linear mechanisms.
        """
        # Start with default noise
        noise = {var: 0.0 for var in self._variables}

        for _ in range(n_iterations):
            # Forward pass
            values = self.forward(noise=noise, interventions=None)

            # Update noise to match evidence
            for var, obs in evidence.items():
                if var in values:
                    error = obs - values[var]
                    # Adjust noise for this variable
                    noise[var] = noise.get(var, 0.0) + 0.5 * error

            # Check convergence
            max_error = max(abs(evidence.get(v, values[v]) - values[v]) for v in evidence.keys())
            if max_error < 1e-6:
                break

        return noise

    def causal_effect(
        self,
        treatment: str,
        outcome: str,
        treatment_value: float = 1.0,
        baseline_value: float = 0.0,
    ) -> float:
        """
        Estimate causal effect of treatment on outcome.

        Returns E[Y | do(X=treatment_value)] - E[Y | do(X=baseline_value)]
        """
        # Sample under both interventions
        n_samples = 100
        y_treatment = []
        y_baseline = []

        for _ in range(n_samples):
            noise = {v: float(self._rng.normal()) for v in self._variables}

            val_t = self.forward(noise, {treatment: treatment_value})
            val_b = self.forward(noise, {treatment: baseline_value})

            y_treatment.append(val_t.get(outcome, 0.0))
            y_baseline.append(val_b.get(outcome, 0.0))

        return float(np.mean(y_treatment) - np.mean(y_baseline))

    def causal_gradient(
        self,
        treatment: str,
        outcome: str,
        values: Dict[str, float],
    ) -> float:
        """
        Compute gradient d(outcome)/d(treatment) at given values.

        Uses finite differences for simplicity.
        """
        eps = 1e-5
        noise = {v: 0.0 for v in self._variables}

        # Forward pass at current value
        v0 = self.forward(noise, values)

        # Forward pass with perturbed treatment
        values_perturbed = values.copy()
        values_perturbed[treatment] = values.get(treatment, 0.0) + eps
        v1 = self.forward(noise, values_perturbed)

        # Finite difference gradient
        return (v1.get(outcome, 0.0) - v0.get(outcome, 0.0)) / eps

    def fit(
        self,
        data: List[Dict[str, float]],
        n_epochs: int = 100,
    ) -> Dict[str, float]:
        """
        Fit SCM parameters to observational data.

        Returns dictionary of final losses per variable.
        """
        losses = {v: 0.0 for v in self._variables}

        for epoch in range(n_epochs):
            epoch_loss = {v: 0.0 for v in self._variables}

            for sample in data:
                # Compute predictions
                noise = {v: 0.0 for v in self._variables}
                predictions = {}

                for var in self._topological_order():
                    mechanism = self._mechanisms.get(var)
                    if mechanism and mechanism.network is not None:
                        parent_values = {p: sample.get(p, 0.0) for p in mechanism.parents}
                        pred = mechanism.forward(parent_values, noise.get(var, 0.0))
                        predictions[var] = pred

                        # Compute loss and gradient
                        target = sample.get(var, 0.0)
                        error = pred - target
                        epoch_loss[var] += error**2

                        # Backward pass
                        mechanism.backward(error)

                        # Update parameters
                        if mechanism.network is not None:
                            mechanism.network.update(self.learning_rate)

            # Average losses
            for v in self._variables:
                if len(data) > 0:
                    epoch_loss[v] /= len(data)
                losses[v] = epoch_loss[v]

        return losses

    def get_parents(self, variable: str) -> List[str]:
        """Get parents of a variable."""
        return self._parents.get(variable, [])

    def get_children(self, variable: str) -> List[str]:
        """Get children of a variable."""
        return self._children.get(variable, [])

    def get_ancestors(self, variable: str) -> Set[str]:
        """Get all ancestors of a variable."""
        ancestors = set()
        queue = list(self._parents.get(variable, []))

        while queue:
            current = queue.pop(0)
            if current not in ancestors:
                ancestors.add(current)
                queue.extend(self._parents.get(current, []))

        return ancestors

    def get_descendants(self, variable: str) -> Set[str]:
        """Get all descendants of a variable."""
        descendants = set()
        queue = list(self._children.get(variable, []))

        while queue:
            current = queue.pop(0)
            if current not in descendants:
                descendants.add(current)
                queue.extend(self._children.get(current, []))

        return descendants

    def is_d_separated(
        self,
        x: str,
        y: str,
        conditioning: Optional[Set[str]] = None,
    ) -> bool:
        """
        Check if X and Y are d-separated given conditioning set.

        Implements the Bayes-Ball algorithm correctly:
        - Chains (A→B→C): blocked if B is conditioned
        - Forks (A←B→C): blocked if B is conditioned
        - Colliders (A→B←C): blocked UNLESS B (or descendant) is conditioned
        """
        conditioning = conditioning or set()

        # Check if any descendant of a collider is conditioned
        def has_conditioned_descendant(node: str) -> bool:
            if node in conditioning:
                return True
            for child in self._children.get(node, []):
                if has_conditioned_descendant(child):
                    return True
            return False

        visited = set()
        # Start in BOTH directions from the source node
        queue = [(x, "up"), (x, "down")]

        while queue:
            current, direction = queue.pop(0)
            if current == y:
                return False

            if (current, direction) in visited:
                continue
            visited.add((current, direction))

            is_conditioned = current in conditioning

            if direction == "down":
                # Coming from parent (going down through chain/fork/collider)
                is_collider = len(self._parents.get(current, [])) > 1

                if is_conditioned and is_collider:
                    # Collider is conditioned - path is UNBLOCKED to other parents
                    for parent in self._parents.get(current, []):
                        queue.append((parent, "up"))
                elif not is_conditioned:
                    # Can continue to children (chain/fork)
                    for child in self._children.get(current, []):
                        queue.append((child, "down"))
                # If conditioned and NOT collider: blocked (chain/fork blocked)
            else:
                # Coming from child (going up) or starting node
                # Check if this is a collider being activated
                is_collider = len(self._parents.get(current, [])) > 1
                collider_active = is_collider and has_conditioned_descendant(current)

                if is_conditioned or collider_active:
                    # Collider is activated - can go to OTHER parents
                    for parent in self._parents.get(current, []):
                        queue.append((parent, "up"))

                if not is_conditioned:
                    # Chain/fork: can continue up to parents
                    for parent in self._parents.get(current, []):
                        queue.append((parent, "up"))
                    # Can also go down to children (fork behavior)
                    for child in self._children.get(current, []):
                        queue.append((child, "down"))

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize SCM to dictionary."""
        return {
            "name": self.name,
            "variables": list(self._variables),
            "parents": self._parents,
            "children": self._children,
            "exogenous": self._exogenous,
        }

    def statistics(self) -> Dict[str, Any]:
        """Get SCM statistics."""
        return {
            "name": self.name,
            "n_variables": len(self._variables),
            "n_mechanisms": len(self._mechanisms),
            "n_samples": self._n_samples,
            "n_interventions": self._n_interventions,
            "n_counterfactuals": self._n_counterfactuals,
            "variables": list(self._variables),
            "graph": {v: self._parents.get(v, []) for v in self._variables},
        }
