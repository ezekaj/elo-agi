"""
Transition Model: Predicts next states from current state and action.

The transition model is the core of the world model - it learns the
dynamics of the environment and can predict future states. This enables
planning, imagination, and counterfactual reasoning.

Based on:
- Forward models in motor control (Wolpert & Kawato, 1998)
- Recurrent State Space Models (Hafner et al., 2019)
- arXiv:2510.16732 - World models for embodied AI
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import time


class ActionType(Enum):
    """Types of actions the model can predict effects for."""

    MOTOR = "motor"  # Physical movement
    COGNITIVE = "cognitive"  # Internal cognitive action
    PERCEPTUAL = "perceptual"  # Attention shift
    SOCIAL = "social"  # Social action
    VERBAL = "verbal"  # Language production


@dataclass
class TransitionParams:
    """Parameters for the transition model."""

    n_latent: int = 128  # Latent state dimensionality
    n_action: int = 32  # Action vector dimensionality
    n_hidden: int = 256  # Hidden layer size
    n_ensemble: int = 5  # Number of ensemble models
    uncertainty_threshold: float = 0.5  # Above this, predictions are uncertain
    learning_rate: float = 0.001
    momentum: float = 0.9
    regularization: float = 0.001


@dataclass
class Transition:
    """A predicted state transition."""

    current_state: np.ndarray
    action: np.ndarray
    predicted_state: np.ndarray
    predicted_reward: float
    uncertainty: float  # Epistemic uncertainty
    ensemble_variance: float  # Variance across ensemble
    timestamp: float = field(default_factory=time.time)

    def is_reliable(self, threshold: float = 0.5) -> bool:
        """Check if prediction is reliable enough."""
        return self.uncertainty < threshold


class TransitionModel:
    """
    Model that predicts state transitions given actions.

    The transition model learns P(s'|s, a) - the probability distribution
    over next states given current state and action. Key features:

    1. **Ensemble**: Multiple models for uncertainty estimation
    2. **Stochastic**: Can model stochastic dynamics
    3. **Reward prediction**: Also predicts immediate reward
    4. **Online learning**: Continuously improves from experience

    The model is trained via gradient descent on prediction errors,
    implementing the core prediction error minimization of predictive processing.
    """

    def __init__(self, params: Optional[TransitionParams] = None):
        self.params = params or TransitionParams()

        # Ensemble of transition models
        self._ensemble_weights: List[Dict[str, np.ndarray]] = []
        self._ensemble_biases: List[Dict[str, np.ndarray]] = []

        for _ in range(self.params.n_ensemble):
            weights = {
                "input_hidden": np.random.randn(
                    self.params.n_hidden, self.params.n_latent + self.params.n_action
                )
                * 0.01,
                "hidden_hidden": np.random.randn(self.params.n_hidden, self.params.n_hidden) * 0.01,
                "hidden_output": np.random.randn(self.params.n_latent, self.params.n_hidden) * 0.01,
                "hidden_reward": np.random.randn(1, self.params.n_hidden) * 0.01,
            }
            biases = {
                "input_hidden": np.zeros(self.params.n_hidden),
                "hidden_hidden": np.zeros(self.params.n_hidden),
                "hidden_output": np.zeros(self.params.n_latent),
                "hidden_reward": np.zeros(1),
            }
            self._ensemble_weights.append(weights)
            self._ensemble_biases.append(biases)

        # Momentum terms for SGD
        self._momentum_weights: List[Dict[str, np.ndarray]] = [
            {k: np.zeros_like(v) for k, v in w.items()} for w in self._ensemble_weights
        ]

        # History
        self._transition_history: List[Transition] = []
        self._prediction_errors: List[float] = []

    def predict(
        self,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Transition:
        """
        Predict next state given current state and action.

        Uses ensemble to estimate epistemic uncertainty.
        """
        # Ensure correct dimensions
        if len(state) != self.params.n_latent:
            state = np.resize(state, self.params.n_latent)
        if len(action) != self.params.n_action:
            action = np.resize(action, self.params.n_action)

        # Get predictions from all ensemble members
        predictions = []
        rewards = []

        for i in range(self.params.n_ensemble):
            pred, reward = self._forward(i, state, action)
            predictions.append(pred)
            rewards.append(reward)

        predictions = np.array(predictions)
        rewards = np.array(rewards)

        # Ensemble mean and variance
        mean_prediction = np.mean(predictions, axis=0)
        ensemble_variance = float(np.mean(np.var(predictions, axis=0)))
        mean_reward = float(np.mean(rewards))

        # Uncertainty from variance
        uncertainty = min(1.0, ensemble_variance * 10)  # Scale to [0, 1]

        transition = Transition(
            current_state=state,
            action=action,
            predicted_state=mean_prediction,
            predicted_reward=mean_reward,
            uncertainty=uncertainty,
            ensemble_variance=ensemble_variance,
        )

        self._transition_history.append(transition)
        if len(self._transition_history) > 10000:
            self._transition_history.pop(0)

        return transition

    def _forward(
        self,
        ensemble_idx: int,
        state: np.ndarray,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Forward pass through one ensemble member."""
        W = self._ensemble_weights[ensemble_idx]
        b = self._ensemble_biases[ensemble_idx]

        # Concatenate state and action
        x = np.concatenate([state, action])

        # Hidden layer 1
        h1 = np.tanh(W["input_hidden"] @ x + b["input_hidden"])

        # Hidden layer 2
        h2 = np.tanh(W["hidden_hidden"] @ h1 + b["hidden_hidden"])

        # Output (next state delta)
        delta = W["hidden_output"] @ h2 + b["hidden_output"]
        next_state = state + delta  # Residual prediction

        # Reward prediction
        reward_raw = W["hidden_reward"] @ h2 + b["hidden_reward"]
        reward = float(np.tanh(reward_raw[0]))  # Extract scalar and bound

        return next_state, reward

    def train_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
    ) -> float:
        """
        Train the model on a single transition.

        Updates all ensemble members with slightly different data
        (bootstrap) for diversity.
        """
        total_loss = 0.0

        # Ensure correct dimensions
        if len(state) != self.params.n_latent:
            state = np.resize(state, self.params.n_latent)
        if len(action) != self.params.n_action:
            action = np.resize(action, self.params.n_action)
        if len(next_state) != self.params.n_latent:
            next_state = np.resize(next_state, self.params.n_latent)

        for i in range(self.params.n_ensemble):
            # Bootstrap: randomly skip some training samples
            if np.random.rand() < 0.3:
                continue

            loss = self._train_ensemble_member(i, state, action, next_state, reward)
            total_loss += loss

        avg_loss = total_loss / self.params.n_ensemble
        self._prediction_errors.append(avg_loss)
        if len(self._prediction_errors) > 10000:
            self._prediction_errors.pop(0)

        return avg_loss

    def _train_ensemble_member(
        self,
        idx: int,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
    ) -> float:
        """Train one ensemble member via backprop."""
        W = self._ensemble_weights[idx]
        b = self._ensemble_biases[idx]
        M = self._momentum_weights[idx]

        # Forward pass (with intermediate values)
        x = np.concatenate([state, action])
        h1_pre = W["input_hidden"] @ x + b["input_hidden"]
        h1 = np.tanh(h1_pre)
        h2_pre = W["hidden_hidden"] @ h1 + b["hidden_hidden"]
        h2 = np.tanh(h2_pre)
        delta = W["hidden_output"] @ h2 + b["hidden_output"]
        pred_next = state + delta
        pred_reward = float((W["hidden_reward"] @ h2 + b["hidden_reward"])[0])

        # Compute errors
        state_error = next_state - pred_next
        reward_error = reward - pred_reward
        state_loss = 0.5 * np.sum(state_error**2)
        reward_loss = 0.5 * reward_error**2
        total_loss = state_loss + reward_loss

        # Backprop
        # Output layer gradients
        d_delta = -state_error
        d_reward = -reward_error

        # Hidden layer 2 gradients
        d_h2 = W["hidden_output"].T @ d_delta + W["hidden_reward"].T.flatten() * d_reward
        d_h2 *= 1 - h2**2  # tanh derivative

        # Hidden layer 1 gradients
        d_h1 = W["hidden_hidden"].T @ d_h2
        d_h1 *= 1 - h1**2

        # Weight gradients
        dW = {
            "input_hidden": np.outer(d_h1, x),
            "hidden_hidden": np.outer(d_h2, h1),
            "hidden_output": np.outer(d_delta, h2),
            "hidden_reward": np.outer([d_reward], h2),
        }
        db = {
            "input_hidden": d_h1,
            "hidden_hidden": d_h2,
            "hidden_output": d_delta,
            "hidden_reward": np.array([d_reward]),
        }

        # Update with momentum
        lr = self.params.learning_rate
        mom = self.params.momentum
        reg = self.params.regularization

        for key in W:
            M[key] = mom * M[key] - lr * (dW[key] + reg * W[key])
            W[key] += M[key]
            b[key] -= lr * db[key]

        return float(total_loss)

    def predict_trajectory(
        self,
        initial_state: np.ndarray,
        action_sequence: List[np.ndarray],
    ) -> List[Transition]:
        """
        Predict a sequence of states from action sequence.

        Useful for planning and imagination.
        """
        trajectory = []
        current_state = initial_state.copy()

        for action in action_sequence:
            transition = self.predict(current_state, action)
            trajectory.append(transition)
            current_state = transition.predicted_state

        return trajectory

    def compute_trajectory_uncertainty(
        self,
        trajectory: List[Transition],
    ) -> float:
        """Compute cumulative uncertainty along a trajectory."""
        if not trajectory:
            return 0.0

        # Uncertainty compounds over time
        cumulative = 0.0
        for i, t in enumerate(trajectory):
            # Later predictions are more uncertain
            weight = 1.0 + 0.1 * i
            cumulative += t.uncertainty * weight

        return cumulative / len(trajectory)

    def get_prediction_error_stats(self) -> Dict[str, float]:
        """Get statistics about prediction errors."""
        if not self._prediction_errors:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        recent = self._prediction_errors[-1000:]
        return {
            "mean": float(np.mean(recent)),
            "std": float(np.std(recent)),
            "min": float(np.min(recent)),
            "max": float(np.max(recent)),
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        if not self._transition_history:
            return {
                "n_predictions": 0,
                "avg_uncertainty": 0.0,
                "prediction_errors": self.get_prediction_error_stats(),
            }

        recent = self._transition_history[-1000:]
        return {
            "n_predictions": len(self._transition_history),
            "avg_uncertainty": float(np.mean([t.uncertainty for t in recent])),
            "avg_reward": float(np.mean([t.predicted_reward for t in recent])),
            "prediction_errors": self.get_prediction_error_stats(),
            "n_ensemble": self.params.n_ensemble,
        }

    def reset(self) -> None:
        """Reset history (but keep learned weights)."""
        self._transition_history = []
        self._prediction_errors = []
