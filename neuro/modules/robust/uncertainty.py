"""
Uncertainty Quantification.

Implements:
- Epistemic uncertainty (model uncertainty)
- Aleatoric uncertainty (data uncertainty)
- Monte Carlo Dropout
- Deep Ensembles
- Evidential Deep Learning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np


class UncertaintyType(Enum):
    """Types of uncertainty."""
    EPISTEMIC = "epistemic"      # Model uncertainty (reducible with more data)
    ALEATORIC = "aleatoric"      # Data uncertainty (inherent noise)
    TOTAL = "total"              # Combined uncertainty


@dataclass
class UncertaintyEstimate:
    """Result of uncertainty quantification."""
    prediction: np.ndarray
    epistemic: float
    aleatoric: float
    total: float
    confidence: float
    samples: Optional[np.ndarray] = None


class SimpleDropoutNN:
    """Neural network with dropout for uncertainty estimation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.2,
        random_seed: Optional[int] = None,
    ):
        self.dropout_rate = dropout_rate
        self.rng = np.random.default_rng(random_seed)

        # Xavier initialization
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))

        self.W1 = self.rng.normal(0, scale1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = self.rng.normal(0, scale2, (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

        self.training = True

    def forward(self, x: np.ndarray, apply_dropout: bool = True) -> np.ndarray:
        """Forward pass with optional dropout."""
        h = np.maximum(0, x @ self.W1 + self.b1)

        if apply_dropout and self.dropout_rate > 0:
            mask = self.rng.random(h.shape) > self.dropout_rate
            h = h * mask / (1 - self.dropout_rate)

        logits = h @ self.W2 + self.b2
        return logits

    def predict_proba(self, x: np.ndarray, apply_dropout: bool = False) -> np.ndarray:
        """Get class probabilities."""
        logits = self.forward(x, apply_dropout)
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)


class UncertaintyQuantifier:
    """
    Quantify epistemic and aleatoric uncertainty.

    Methods:
    - Monte Carlo Dropout
    - Deep Ensembles
    - Evidential Deep Learning
    """

    def __init__(
        self,
        model: SimpleDropoutNN,
        n_samples: int = 50,
    ):
        self.model = model
        self.n_samples = n_samples

        # Statistics
        self._n_estimates = 0

    def monte_carlo_dropout(
        self,
        x: np.ndarray,
        n_samples: Optional[int] = None,
    ) -> UncertaintyEstimate:
        """
        Estimate uncertainty using Monte Carlo Dropout.

        Run multiple forward passes with dropout enabled to
        approximate Bayesian inference.

        Args:
            x: Input
            n_samples: Number of stochastic forward passes

        Returns:
            UncertaintyEstimate with epistemic uncertainty
        """
        self._n_estimates += 1
        n_samples = n_samples or self.n_samples

        # Collect predictions
        predictions = []
        for _ in range(n_samples):
            probs = self.model.predict_proba(x, apply_dropout=True)
            predictions.append(probs)

        predictions = np.array(predictions)

        # Mean prediction
        mean_pred = np.mean(predictions, axis=0)

        # Epistemic uncertainty: variance of predictions
        epistemic = float(np.mean(np.var(predictions, axis=0)))

        # Aleatoric uncertainty: mean of predictive entropy
        entropies = -np.sum(predictions * np.log(predictions + 1e-8), axis=1)
        aleatoric = float(np.mean(entropies))

        # Total uncertainty: entropy of mean prediction
        total = float(-np.sum(mean_pred * np.log(mean_pred + 1e-8)))

        # Confidence: max probability
        confidence = float(np.max(mean_pred))

        return UncertaintyEstimate(
            prediction=mean_pred,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total=total,
            confidence=confidence,
            samples=predictions,
        )

    def epistemic_uncertainty(self, x: np.ndarray) -> float:
        """Get epistemic uncertainty only."""
        estimate = self.monte_carlo_dropout(x)
        return estimate.epistemic

    def aleatoric_uncertainty(self, x: np.ndarray) -> float:
        """Get aleatoric uncertainty only."""
        estimate = self.monte_carlo_dropout(x)
        return estimate.aleatoric

    def total_uncertainty(self, x: np.ndarray) -> float:
        """Get total uncertainty."""
        estimate = self.monte_carlo_dropout(x)
        return estimate.total

    def predictive_entropy(self, x: np.ndarray) -> float:
        """Compute predictive entropy."""
        probs = self.model.predict_proba(x, apply_dropout=False)
        return float(-np.sum(probs * np.log(probs + 1e-8)))

    def mutual_information(self, x: np.ndarray) -> float:
        """
        Compute mutual information between prediction and model parameters.

        This measures epistemic uncertainty directly.
        """
        estimate = self.monte_carlo_dropout(x)

        # Mutual information = H[y|x] - E[H[y|x,w]]
        # = Total entropy - Mean sample entropy
        mean_entropy = float(np.mean(
            -np.sum(estimate.samples * np.log(estimate.samples + 1e-8), axis=1)
        ))

        return estimate.total - mean_entropy

    def statistics(self) -> Dict[str, Any]:
        """Get quantifier statistics."""
        return {
            "n_estimates": self._n_estimates,
            "n_samples": self.n_samples,
            "dropout_rate": self.model.dropout_rate,
        }


class EnsembleUncertainty:
    """
    Uncertainty estimation using deep ensembles.

    Train multiple models and use disagreement as uncertainty.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_models: int = 5,
        random_seed: Optional[int] = None,
    ):
        self.n_models = n_models

        # Create ensemble
        self.models: List[SimpleDropoutNN] = []
        for i in range(n_models):
            seed = None if random_seed is None else random_seed + i
            model = SimpleDropoutNN(
                input_dim, hidden_dim, output_dim,
                dropout_rate=0.0,  # No dropout, use ensemble instead
                random_seed=seed,
            )
            self.models.append(model)

        self._n_estimates = 0

    def train(
        self,
        data: List[Tuple[np.ndarray, int]],
        n_epochs: int = 50,
        learning_rate: float = 0.01,
        bootstrap: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train ensemble models.

        Args:
            data: Training data as (input, label) pairs
            n_epochs: Training epochs per model
            learning_rate: Learning rate
            bootstrap: If True, use bootstrap sampling for each model

        Returns:
            Training history
        """
        history = {"losses": []}

        for model_idx, model in enumerate(self.models):
            # Bootstrap sample
            if bootstrap:
                indices = np.random.choice(len(data), len(data), replace=True)
                train_data = [data[i] for i in indices]
            else:
                train_data = data

            model_losses = []
            for epoch in range(n_epochs):
                epoch_loss = 0
                for x, y in train_data:
                    # Forward
                    logits = model.forward(x, apply_dropout=False)
                    probs = np.exp(logits - np.max(logits))
                    probs = probs / np.sum(probs)
                    loss = -np.log(probs[y] + 1e-8)
                    epoch_loss += loss

                    # Backward (simplified)
                    grad_logits = probs.copy()
                    grad_logits[y] -= 1

                    h = np.maximum(0, x @ model.W1 + model.b1)
                    grad_W2 = np.outer(h, grad_logits)
                    grad_b2 = grad_logits
                    grad_h = grad_logits @ model.W2.T
                    grad_pre_h = grad_h * (h > 0).astype(float)
                    grad_W1 = np.outer(x, grad_pre_h)
                    grad_b1 = grad_pre_h

                    model.W1 -= learning_rate * grad_W1
                    model.b1 -= learning_rate * grad_b1
                    model.W2 -= learning_rate * grad_W2
                    model.b2 -= learning_rate * grad_b2

                model_losses.append(epoch_loss / len(train_data))

            history["losses"].append(model_losses)

        return history

    def predict(self, x: np.ndarray) -> UncertaintyEstimate:
        """
        Get prediction with uncertainty from ensemble.

        Args:
            x: Input

        Returns:
            UncertaintyEstimate with ensemble uncertainty
        """
        self._n_estimates += 1

        # Get predictions from all models
        predictions = []
        for model in self.models:
            probs = model.predict_proba(x, apply_dropout=False)
            predictions.append(probs)

        predictions = np.array(predictions)

        # Mean prediction
        mean_pred = np.mean(predictions, axis=0)

        # Epistemic uncertainty: disagreement between models
        epistemic = float(np.mean(np.var(predictions, axis=0)))

        # Aleatoric: average entropy of individual predictions
        entropies = -np.sum(predictions * np.log(predictions + 1e-8), axis=1)
        aleatoric = float(np.mean(entropies))

        # Total uncertainty
        total = float(-np.sum(mean_pred * np.log(mean_pred + 1e-8)))

        # Confidence
        confidence = float(np.max(mean_pred))

        return UncertaintyEstimate(
            prediction=mean_pred,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total=total,
            confidence=confidence,
            samples=predictions,
        )

    def epistemic_uncertainty(self, x: np.ndarray) -> float:
        """Get epistemic uncertainty from ensemble disagreement."""
        return self.predict(x).epistemic

    def statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics."""
        return {
            "n_models": self.n_models,
            "n_estimates": self._n_estimates,
        }


class EvidentialUncertainty:
    """
    Evidential Deep Learning for uncertainty.

    Predict Dirichlet distribution parameters instead of class probabilities.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        n_classes: int,
        random_seed: Optional[int] = None,
    ):
        self.n_classes = n_classes
        rng = np.random.default_rng(random_seed)

        # Network outputs Dirichlet concentrations (evidence)
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + n_classes))

        self.W1 = rng.normal(0, scale1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, scale2, (hidden_dim, n_classes))
        self.b2 = np.zeros(n_classes)

        self._n_estimates = 0

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Get Dirichlet evidence."""
        h = np.maximum(0, x @ self.W1 + self.b1)
        evidence = np.exp(h @ self.W2 + self.b2)  # Ensure positive
        return evidence

    def predict(self, x: np.ndarray) -> UncertaintyEstimate:
        """
        Get prediction with uncertainty from evidential output.

        The Dirichlet parameters encode both the prediction and uncertainty.
        """
        self._n_estimates += 1

        evidence = self.forward(x)
        alpha = evidence + 1  # Dirichlet concentration parameters

        # Dirichlet strength
        S = np.sum(alpha)

        # Expected probabilities
        probs = alpha / S

        # Uncertainty from Dirichlet
        # Epistemic: uncertainty due to lack of evidence
        epistemic = float(self.n_classes / S)

        # Aleatoric: expected entropy under Dirichlet
        from scipy.special import digamma
        aleatoric = float(
            -np.sum(probs * (digamma(alpha + 1) - digamma(S + 1)))
        )

        # Total uncertainty
        total = epistemic + aleatoric

        # Confidence
        confidence = float(np.max(probs))

        return UncertaintyEstimate(
            prediction=probs,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total=total,
            confidence=confidence,
        )

    def train_step(
        self,
        x: np.ndarray,
        y: int,
        learning_rate: float = 0.01,
        kl_weight: float = 0.1,
    ) -> float:
        """
        Single training step with evidential loss.

        Loss = Expected MSE under Dirichlet + KL to uniform prior
        """
        evidence = self.forward(x)
        alpha = evidence + 1
        S = np.sum(alpha)

        # Expected probabilities
        probs = alpha / S

        # One-hot target
        target = np.zeros(self.n_classes)
        target[y] = 1

        # MSE loss (cross-entropy can also be used)
        mse_loss = np.sum((probs - target) ** 2)

        # KL divergence to uniform Dirichlet(1, 1, ..., 1)
        # Regularizes to be uncertain when lacking evidence
        alpha_tilde = alpha.copy()
        alpha_tilde[y] = 1  # Remove correct class from KL
        kl_loss = self._kl_dirichlet(alpha_tilde, np.ones(self.n_classes))

        total_loss = mse_loss + kl_weight * kl_loss

        # Simplified gradient (numerical)
        eps = 1e-5
        grad_W2 = np.zeros_like(self.W2)
        grad_b2 = np.zeros_like(self.b2)

        h = np.maximum(0, x @ self.W1 + self.b1)

        for i in range(self.n_classes):
            self.b2[i] += eps
            evidence_plus = self.forward(x)
            alpha_plus = evidence_plus + 1
            S_plus = np.sum(alpha_plus)
            probs_plus = alpha_plus / S_plus
            mse_plus = np.sum((probs_plus - target) ** 2)
            loss_plus = mse_plus
            self.b2[i] -= eps

            grad_b2[i] = (loss_plus - mse_loss) / eps

        grad_W2 = np.outer(h, grad_b2)

        # Update
        self.W2 -= learning_rate * grad_W2
        self.b2 -= learning_rate * grad_b2

        return total_loss

    def _kl_dirichlet(self, alpha: np.ndarray, beta: np.ndarray) -> float:
        """KL divergence between two Dirichlet distributions."""
        from scipy.special import gammaln, digamma

        alpha0 = np.sum(alpha)
        beta0 = np.sum(beta)

        kl = (
            gammaln(alpha0) - gammaln(beta0)
            - np.sum(gammaln(alpha)) + np.sum(gammaln(beta))
            + np.sum((alpha - beta) * (digamma(alpha) - digamma(alpha0)))
        )
        return float(kl)

    def statistics(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "n_classes": self.n_classes,
            "n_estimates": self._n_estimates,
        }
