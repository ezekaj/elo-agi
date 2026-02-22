"""
Hebbian Learning

"Neurons that fire together, wire together"
Mathematical form: Δw = η × pre × post
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class LearningParams:
    """Parameters for Hebbian learning"""

    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    weight_max: float = 1.0
    weight_min: float = 0.0


class HebbianLearning:
    """Basic Hebbian learning rule

    Δw = η × pre × post

    Where:
    - η = learning rate
    - pre = presynaptic activity
    - post = postsynaptic activity
    """

    def __init__(self, params: Optional[LearningParams] = None):
        self.params = params or LearningParams()

    def compute_weight_change(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """Compute Hebbian weight change

        Args:
            pre: Presynaptic activity (n_pre,)
            post: Postsynaptic activity (n_post,)

        Returns:
            Weight change matrix (n_post, n_pre)
        """
        # Outer product gives correlation
        dW = self.params.learning_rate * np.outer(post, pre)
        return dW

    def update_weights(self, weights: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """Update weights with Hebbian rule

        Args:
            weights: Current weight matrix (n_post, n_pre)
            pre: Presynaptic activity
            post: Postsynaptic activity

        Returns:
            Updated weight matrix
        """
        dW = self.compute_weight_change(pre, post)

        # Apply weight decay
        weights = weights * (1 - self.params.weight_decay)

        # Update
        weights = weights + dW

        # Clip to bounds
        weights = np.clip(weights, self.params.weight_min, self.params.weight_max)

        return weights


class OjaRule(HebbianLearning):
    """Oja's rule - normalized Hebbian learning

    Δw = η × post × (pre - post × w)

    Prevents unbounded weight growth by subtracting decay term.
    """

    def compute_weight_change(
        self, pre: np.ndarray, post: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute Oja weight change"""
        if weights is None:
            return super().compute_weight_change(pre, post)

        # Oja's rule: η × post × (pre - post × w)
        dW = np.zeros_like(weights)
        for i in range(len(post)):
            decay = post[i] * weights[i, :]
            dW[i, :] = self.params.learning_rate * post[i] * (pre - decay)

        return dW

    def update_weights(self, weights: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """Update with Oja's rule"""
        dW = self.compute_weight_change(pre, post, weights)
        weights = weights + dW
        weights = np.clip(weights, self.params.weight_min, self.params.weight_max)
        return weights


class BCMRule(HebbianLearning):
    """BCM (Bienenstock-Cooper-Munro) rule

    Includes sliding modification threshold for bidirectional plasticity.
    """

    def __init__(self, params: Optional[LearningParams] = None):
        super().__init__(params)
        self.threshold = 0.5  # Modification threshold
        self.threshold_tau = 0.1  # Threshold adaptation rate
        self.activity_history: List[float] = []

    def update_threshold(self, post_mean: float) -> None:
        """Update sliding threshold based on activity history"""
        self.activity_history.append(post_mean)
        if len(self.activity_history) > 100:
            self.activity_history.pop(0)

        # Threshold tracks recent activity
        self.threshold = np.mean(self.activity_history) ** 2

    def compute_weight_change(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """Compute BCM weight change

        LTP when post > threshold, LTD when post < threshold
        """
        self.update_threshold(np.mean(post))

        # BCM rule: post * (post - threshold)
        modulation = post * (post - self.threshold)
        dW = self.params.learning_rate * np.outer(modulation, pre)

        return dW


class HebbianNetwork:
    """Network with Hebbian learning"""

    def __init__(
        self, layer_sizes: List[int], learning_rule: str = "basic", learning_rate: float = 0.01
    ):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes)

        # Initialize weights
        self.weights = []
        for i in range(self.n_layers - 1):
            w = np.random.randn(layer_sizes[i + 1], layer_sizes[i]) * 0.1
            self.weights.append(w)

        # Initialize learning rule
        params = LearningParams(learning_rate=learning_rate)
        if learning_rule == "basic":
            self.rule = HebbianLearning(params)
        elif learning_rule == "oja":
            self.rule = OjaRule(params)
        elif learning_rule == "bcm":
            self.rule = BCMRule(params)
        else:
            raise ValueError(f"Unknown rule: {learning_rule}")

        # Activity storage
        self.activations: List[np.ndarray] = []

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        self.activations = [x]

        for w in self.weights:
            x = np.tanh(w @ x)
            self.activations.append(x)

        return x

    def learn(self) -> None:
        """Apply Hebbian learning to all layers"""
        for i, w in enumerate(self.weights):
            pre = self.activations[i]
            post = self.activations[i + 1]
            self.weights[i] = self.rule.update_weights(w, pre, post)

    def train_step(self, x: np.ndarray) -> np.ndarray:
        """Forward pass + learning"""
        output = self.forward(x)
        self.learn()
        return output

    def train(self, data: np.ndarray, n_epochs: int = 10) -> List[float]:
        """Train on dataset"""
        errors = []

        for epoch in range(n_epochs):
            epoch_error = 0.0
            for x in data:
                output = self.train_step(x)
                # Reconstruction error as proxy for learning
                reconstructed = self.weights[0].T @ self.activations[1]
                error = np.mean((x - reconstructed) ** 2)
                epoch_error += error
            errors.append(epoch_error / len(data))

        return errors
