"""
Hierarchical Predictive Coding Engine

Implements the Free Energy formulation:
- y = g(x,v,θ) + z (observations from hidden states)
- ẋ = f(x,v,θ) + w (hidden state dynamics)

Higher layers generate predictions for lower layers.
Lower layers send prediction errors upward.
"""

import numpy as np
from typing import List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class LayerState:
    """Container for layer state information"""
    hidden: np.ndarray
    prediction: Optional[np.ndarray] = None
    error: Optional[np.ndarray] = None
    precision: float = 1.0


class PredictiveLayer:
    """Single layer in the predictive hierarchy.

    Each layer maintains beliefs about hidden states and generates
    predictions for the layer below while receiving errors from it.
    """

    def __init__(
        self,
        state_dim: int,
        output_dim: int,
        learning_rate: float = 0.1,
        timescale: float = 1.0,
        nonlinearity: Callable = np.tanh
    ):
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.timescale = timescale
        self.nonlinearity = nonlinearity

        # Hidden state x - what this layer "believes"
        self.hidden_state = np.zeros(state_dim)

        # Generative model weights: maps hidden state to predictions
        # g(x) = W_g @ nonlinearity(x) + b_g
        self.W_g = np.random.randn(output_dim, state_dim) * 0.1
        self.b_g = np.zeros(output_dim)

        # Dynamics model: f(x) for state evolution
        # ẋ = W_f @ nonlinearity(x) + b_f
        self.W_f = np.random.randn(state_dim, state_dim) * 0.1
        self.b_f = np.zeros(state_dim)

        # Error from layer below
        self.prediction_error = np.zeros(output_dim)

        # Precision (inverse variance) of predictions
        self.precision = 1.0

        # History for precision estimation
        self.error_history: List[np.ndarray] = []
        self.max_history = 100

    def generate_prediction(self) -> np.ndarray:
        """Generate top-down prediction for the layer below.

        Implements: ŷ = g(x) = W_g @ σ(x) + b_g
        """
        activated = self.nonlinearity(self.hidden_state)
        prediction = self.W_g @ activated + self.b_g
        return prediction

    def receive_error(self, error: np.ndarray, precision: float = 1.0) -> None:
        """Receive bottom-up prediction error from layer below.

        Args:
            error: Prediction error ε = actual - predicted
            precision: Precision weight Π for this error
        """
        self.prediction_error = error
        self.precision = precision

        # Track error history for precision estimation
        self.error_history.append(error.copy())
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

    def compute_state_gradient(self) -> np.ndarray:
        """Compute gradient of free energy with respect to hidden state.

        The gradient drives state updates to minimize prediction error.
        ∂F/∂x = -Π * ε * ∂g/∂x
        """
        # Jacobian of nonlinearity
        activated = self.nonlinearity(self.hidden_state)
        if self.nonlinearity == np.tanh:
            d_nonlin = 1 - activated ** 2
        else:
            d_nonlin = np.ones_like(self.hidden_state)

        # Gradient: backprop error through generative weights
        # ∂g/∂x = W_g @ diag(σ'(x))
        weighted_error = self.precision * self.prediction_error
        gradient = self.W_g.T @ weighted_error * d_nonlin

        return gradient

    def update_state(self, dt: float = 0.1) -> None:
        """Update hidden state to minimize prediction error.

        Implements gradient descent on free energy:
        ẋ = -∂F/∂x + f(x)

        Args:
            dt: Time step (scaled by layer's timescale)
        """
        effective_dt = dt / self.timescale

        # Prediction error minimization
        error_gradient = self.compute_state_gradient()

        # Clip gradient for stability
        grad_norm = np.linalg.norm(error_gradient)
        if grad_norm > 10.0:
            error_gradient = error_gradient * 10.0 / grad_norm

        # Intrinsic dynamics
        activated = self.nonlinearity(self.hidden_state)
        dynamics = self.W_f @ activated + self.b_f

        # Combined update with clipping
        update = effective_dt * (error_gradient + dynamics)
        update = np.clip(update, -1.0, 1.0)
        self.hidden_state += update

        # Keep state bounded
        self.hidden_state = np.clip(self.hidden_state, -10.0, 10.0)

    def update_weights(self, dt: float = 0.1) -> None:
        """Update generative model weights via gradient descent.

        ∂F/∂θ for the generative weights.
        """
        activated = self.nonlinearity(self.hidden_state)

        # Clip error for stability
        clipped_error = np.clip(self.prediction_error, -5.0, 5.0)

        # Weight gradient: outer product of error and activated state
        dW = np.outer(clipped_error, activated)
        db = clipped_error

        # Clip precision
        effective_precision = min(self.precision, 10.0)

        # Update weights with clipping
        weight_update = self.learning_rate * dt * effective_precision * dW
        weight_update = np.clip(weight_update, -0.1, 0.1)
        self.W_g += weight_update

        bias_update = self.learning_rate * dt * effective_precision * db
        bias_update = np.clip(bias_update, -0.1, 0.1)
        self.b_g += bias_update

        # Keep weights bounded
        self.W_g = np.clip(self.W_g, -5.0, 5.0)
        self.b_g = np.clip(self.b_g, -5.0, 5.0)

    def estimate_precision(self) -> float:
        """Estimate precision from error history.

        Precision = 1 / variance of prediction errors
        """
        if len(self.error_history) < 2:
            return 1.0

        errors = np.array(self.error_history)
        variance = np.mean(np.var(errors, axis=0)) + 1e-8
        return 1.0 / variance

    def reset(self) -> None:
        """Reset layer state"""
        self.hidden_state = np.zeros(self.state_dim)
        self.prediction_error = np.zeros(self.output_dim)
        self.error_history = []


class PredictiveHierarchy:
    """Full hierarchical predictive coding stack.

    Implements bidirectional message passing:
    - Top-down: predictions flow from higher to lower layers
    - Bottom-up: prediction errors flow from lower to higher layers
    """

    def __init__(
        self,
        layer_dims: List[int],
        learning_rate: float = 0.1,
        timescale_factor: float = 2.0
    ):
        """Initialize the hierarchy.

        Args:
            layer_dims: Dimensions of each layer [input, hidden1, hidden2, ...]
            learning_rate: Base learning rate
            timescale_factor: How much slower each higher layer is
        """
        self.n_layers = len(layer_dims) - 1
        self.layer_dims = layer_dims

        # Create layers with increasing timescales
        self.layers: List[PredictiveLayer] = []
        for i in range(self.n_layers):
            timescale = timescale_factor ** i
            layer = PredictiveLayer(
                state_dim=layer_dims[i + 1],
                output_dim=layer_dims[i],
                learning_rate=learning_rate,
                timescale=timescale
            )
            self.layers.append(layer)

        # Current input
        self.current_input: Optional[np.ndarray] = None

        # Track prediction errors at each level
        self.layer_errors: List[np.ndarray] = []

    def forward(self, observation: np.ndarray) -> List[np.ndarray]:
        """Bottom-up pass: propagate prediction errors upward.

        Args:
            observation: Sensory input at the lowest level

        Returns:
            List of prediction errors at each layer
        """
        self.current_input = observation
        self.layer_errors = []

        current_signal = observation

        for i, layer in enumerate(self.layers):
            # Generate prediction for this level
            prediction = layer.generate_prediction()

            # Compute prediction error
            error = current_signal - prediction
            self.layer_errors.append(error)

            # Send error to this layer
            precision = layer.estimate_precision()
            layer.receive_error(error, precision)

            # Signal for next layer is this layer's hidden state
            current_signal = layer.hidden_state

        return self.layer_errors

    def backward(self) -> List[np.ndarray]:
        """Top-down pass: generate predictions from high to low.

        Returns:
            List of predictions at each layer
        """
        predictions = []

        for layer in reversed(self.layers):
            prediction = layer.generate_prediction()
            predictions.insert(0, prediction)

        return predictions

    def step(
        self,
        observation: np.ndarray,
        dt: float = 0.1,
        update_weights: bool = True
    ) -> dict:
        """One complete cycle of prediction-error-update.

        Args:
            observation: Current sensory input
            dt: Time step
            update_weights: Whether to update generative model weights

        Returns:
            Dictionary with errors, predictions, and states
        """
        # Forward pass: compute prediction errors
        errors = self.forward(observation)

        # Update states to minimize errors
        for layer in self.layers:
            layer.update_state(dt)

        # Optionally update generative models
        if update_weights:
            for layer in self.layers:
                layer.update_weights(dt)

        # Backward pass: generate updated predictions
        predictions = self.backward()

        # Collect states
        states = [layer.hidden_state.copy() for layer in self.layers]

        return {
            "errors": errors,
            "predictions": predictions,
            "states": states,
            "total_error": sum(np.sum(e ** 2) for e in errors)
        }

    def predict_next(self) -> np.ndarray:
        """Generate prediction for the next sensory input.

        Uses the lowest layer's prediction.
        """
        if self.layers:
            return self.layers[0].generate_prediction()
        return np.array([])

    def get_beliefs(self) -> List[np.ndarray]:
        """Get current beliefs (hidden states) at all levels"""
        return [layer.hidden_state.copy() for layer in self.layers]

    def get_precisions(self) -> List[float]:
        """Get current precision estimates at all levels"""
        return [layer.estimate_precision() for layer in self.layers]

    def reset(self) -> None:
        """Reset all layers"""
        for layer in self.layers:
            layer.reset()
        self.current_input = None
        self.layer_errors = []
