"""
Adversarial Attacks and Defenses.

Implements:
- FGSM (Fast Gradient Sign Method)
- PGD (Projected Gradient Descent)
- C&W (Carlini-Wagner) attack
- Adversarial training
- Input preprocessing defenses
- Certified robustness
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class AttackType(Enum):
    """Types of adversarial attacks."""
    FGSM = "fgsm"                    # Fast Gradient Sign Method
    PGD = "pgd"                      # Projected Gradient Descent
    CARLINI_WAGNER = "carlini_wagner"  # C&W attack
    DEEPFOOL = "deepfool"            # DeepFool
    BOUNDARY = "boundary"            # Decision boundary attack


class DefenseType(Enum):
    """Types of adversarial defenses."""
    ADVERSARIAL_TRAINING = "adversarial_training"
    INPUT_DENOISING = "input_denoising"
    RANDOMIZATION = "randomization"
    CERTIFIED = "certified"
    DETECTION = "detection"


@dataclass
class AttackResult:
    """Result of an adversarial attack."""
    original: np.ndarray
    adversarial: np.ndarray
    perturbation: np.ndarray
    success: bool
    iterations: int
    l2_distance: float
    linf_distance: float
    original_prediction: int
    adversarial_prediction: int


@dataclass
class DefenseResult:
    """Result of applying a defense."""
    defended_input: np.ndarray
    detected_attack: bool
    confidence: float
    defense_type: DefenseType


class SimpleNN:
    """Simple neural network for demonstration."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        random_seed: Optional[int] = None,
    ):
        rng = np.random.default_rng(random_seed)

        # Xavier initialization
        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + output_dim))

        self.W1 = rng.normal(0, scale1, (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, scale2, (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim)

        # Cached activations for gradient computation
        self._x = None
        self._h = None
        self._logits = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        self._x = x
        self._h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        self._logits = self._h @ self.W2 + self.b2
        return self._logits

    def predict(self, x: np.ndarray) -> int:
        """Get predicted class."""
        logits = self.forward(x)
        return int(np.argmax(logits))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Get class probabilities."""
        logits = self.forward(x)
        exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
        return exp_logits / np.sum(exp_logits)

    def gradient_wrt_input(self, x: np.ndarray, target_class: int) -> np.ndarray:
        """Compute gradient of loss w.r.t. input."""
        # Forward pass
        logits = self.forward(x)

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Gradient of cross-entropy loss w.r.t. logits
        grad_logits = probs.copy()
        grad_logits[target_class] -= 1

        # Gradient w.r.t. hidden layer
        grad_h = grad_logits @ self.W2.T

        # Gradient w.r.t. pre-activation (ReLU)
        grad_pre_h = grad_h * (self._h > 0).astype(float)

        # Gradient w.r.t. input
        grad_x = grad_pre_h @ self.W1.T

        return grad_x

    def update(self, grad_W1, grad_b1, grad_W2, grad_b2, lr: float):
        """Update weights."""
        self.W1 -= lr * grad_W1
        self.b1 -= lr * grad_b1
        self.W2 -= lr * grad_W2
        self.b2 -= lr * grad_b2


class AdversarialAttack:
    """
    Generate adversarial examples using various attack methods.

    Supports:
    - FGSM: Single-step gradient attack
    - PGD: Iterative gradient attack
    - C&W: Optimization-based attack
    """

    def __init__(
        self,
        model: SimpleNN,
        epsilon: float = 0.1,
        attack_type: AttackType = AttackType.FGSM,
    ):
        self.model = model
        self.epsilon = epsilon
        self.attack_type = attack_type

        # Statistics
        self._n_attacks = 0
        self._n_successful = 0

    def fgsm(
        self,
        x: np.ndarray,
        y: int,
        epsilon: Optional[float] = None,
        targeted: bool = False,
    ) -> AttackResult:
        """
        Fast Gradient Sign Method.

        Args:
            x: Input to attack
            y: True label (or target label if targeted)
            epsilon: Perturbation budget (L-infinity)
            targeted: If True, try to classify as y instead of away from y

        Returns:
            AttackResult with adversarial example
        """
        self._n_attacks += 1
        epsilon = epsilon or self.epsilon

        # Get gradient
        grad = self.model.gradient_wrt_input(x, y)

        # Generate perturbation
        if targeted:
            perturbation = -epsilon * np.sign(grad)
        else:
            perturbation = epsilon * np.sign(grad)

        # Create adversarial example
        x_adv = np.clip(x + perturbation, 0, 1)
        actual_perturbation = x_adv - x

        # Check success
        original_pred = self.model.predict(x)
        adv_pred = self.model.predict(x_adv)

        if targeted:
            success = adv_pred == y
        else:
            success = adv_pred != y

        if success:
            self._n_successful += 1

        return AttackResult(
            original=x,
            adversarial=x_adv,
            perturbation=actual_perturbation,
            success=success,
            iterations=1,
            l2_distance=float(np.linalg.norm(actual_perturbation)),
            linf_distance=float(np.max(np.abs(actual_perturbation))),
            original_prediction=original_pred,
            adversarial_prediction=adv_pred,
        )

    def pgd(
        self,
        x: np.ndarray,
        y: int,
        epsilon: Optional[float] = None,
        n_steps: int = 10,
        step_size: Optional[float] = None,
        targeted: bool = False,
        random_start: bool = True,
    ) -> AttackResult:
        """
        Projected Gradient Descent attack.

        Args:
            x: Input to attack
            y: True label (or target if targeted)
            epsilon: L-infinity perturbation budget
            n_steps: Number of attack iterations
            step_size: Step size per iteration
            targeted: If True, minimize loss to target class
            random_start: If True, start from random point in epsilon ball

        Returns:
            AttackResult with adversarial example
        """
        self._n_attacks += 1
        epsilon = epsilon or self.epsilon
        step_size = step_size or epsilon / 4

        # Random start
        if random_start:
            x_adv = x + np.random.uniform(-epsilon, epsilon, x.shape)
            x_adv = np.clip(x_adv, 0, 1)
        else:
            x_adv = x.copy()

        original_pred = self.model.predict(x)

        for i in range(n_steps):
            # Get gradient
            grad = self.model.gradient_wrt_input(x_adv, y)

            # Update
            if targeted:
                x_adv = x_adv - step_size * np.sign(grad)
            else:
                x_adv = x_adv + step_size * np.sign(grad)

            # Project back to epsilon ball
            x_adv = np.clip(x_adv, x - epsilon, x + epsilon)
            x_adv = np.clip(x_adv, 0, 1)

            # Check if attack succeeded
            adv_pred = self.model.predict(x_adv)
            if targeted:
                if adv_pred == y:
                    break
            else:
                if adv_pred != y:
                    break

        perturbation = x_adv - x
        adv_pred = self.model.predict(x_adv)

        if targeted:
            success = adv_pred == y
        else:
            success = adv_pred != y

        if success:
            self._n_successful += 1

        return AttackResult(
            original=x,
            adversarial=x_adv,
            perturbation=perturbation,
            success=success,
            iterations=i + 1,
            l2_distance=float(np.linalg.norm(perturbation)),
            linf_distance=float(np.max(np.abs(perturbation))),
            original_prediction=original_pred,
            adversarial_prediction=adv_pred,
        )

    def carlini_wagner(
        self,
        x: np.ndarray,
        y: int,
        c: float = 1.0,
        n_steps: int = 100,
        learning_rate: float = 0.01,
        targeted: bool = True,
    ) -> AttackResult:
        """
        Carlini-Wagner L2 attack.

        Minimizes ||delta||_2 + c * f(x + delta)
        where f is a loss function that's negative when attack succeeds.

        Args:
            x: Input to attack
            y: Target class (C&W is typically targeted)
            c: Constant balancing perturbation size and attack success
            n_steps: Optimization steps
            learning_rate: Learning rate
            targeted: If True, classify as y; else, misclassify

        Returns:
            AttackResult with adversarial example
        """
        self._n_attacks += 1
        original_pred = self.model.predict(x)

        # Initialize perturbation in tanh space for box constraints
        # x_adv = 0.5 * (tanh(w) + 1) to ensure [0, 1]
        w = np.arctanh(2 * np.clip(x, 0.001, 0.999) - 1)

        best_x_adv = None
        best_l2 = float('inf')

        for step in range(n_steps):
            # Convert to input space
            x_adv = 0.5 * (np.tanh(w) + 1)

            # Get logits
            logits = self.model.forward(x_adv)

            # C&W loss function
            if targeted:
                # Want logits[y] to be highest
                target_logit = logits[y]
                other_logits = np.concatenate([logits[:y], logits[y + 1:]])
                max_other = np.max(other_logits)
                f_loss = max(max_other - target_logit, -1)
            else:
                # Want any class except y
                target_logit = logits[y]
                other_logits = np.concatenate([logits[:y], logits[y + 1:]])
                max_other = np.max(other_logits)
                f_loss = max(target_logit - max_other, -1)

            # L2 loss
            perturbation = x_adv - x
            l2_loss = np.sum(perturbation ** 2)

            # Total loss
            total_loss = l2_loss + c * f_loss

            # Check if attack succeeded
            pred = self.model.predict(x_adv)
            if targeted:
                success = pred == y
            else:
                success = pred != y

            if success and l2_loss < best_l2:
                best_x_adv = x_adv.copy()
                best_l2 = l2_loss

            # Gradient (simplified - compute numerically)
            eps = 1e-5
            grad_w = np.zeros_like(w)
            for i in range(len(w)):
                w_plus = w.copy()
                w_plus[i] += eps
                x_plus = 0.5 * (np.tanh(w_plus) + 1)
                logits_plus = self.model.forward(x_plus)

                if targeted:
                    f_plus = max(np.max(np.concatenate([logits_plus[:y], logits_plus[y + 1:]])) - logits_plus[y], -1)
                else:
                    f_plus = max(logits_plus[y] - np.max(np.concatenate([logits_plus[:y], logits_plus[y + 1:]])), -1)

                l2_plus = np.sum((x_plus - x) ** 2)
                loss_plus = l2_plus + c * f_plus

                grad_w[i] = (loss_plus - total_loss) / eps

            # Update
            w = w - learning_rate * grad_w

        # Use best adversarial example found
        if best_x_adv is None:
            best_x_adv = 0.5 * (np.tanh(w) + 1)

        perturbation = best_x_adv - x
        adv_pred = self.model.predict(best_x_adv)

        if targeted:
            success = adv_pred == y
        else:
            success = adv_pred != y

        if success:
            self._n_successful += 1

        return AttackResult(
            original=x,
            adversarial=best_x_adv,
            perturbation=perturbation,
            success=success,
            iterations=n_steps,
            l2_distance=float(np.linalg.norm(perturbation)),
            linf_distance=float(np.max(np.abs(perturbation))),
            original_prediction=original_pred,
            adversarial_prediction=adv_pred,
        )

    def attack(
        self,
        x: np.ndarray,
        y: int,
        **kwargs,
    ) -> AttackResult:
        """Run attack based on configured attack type."""
        if self.attack_type == AttackType.FGSM:
            return self.fgsm(x, y, **kwargs)
        elif self.attack_type == AttackType.PGD:
            return self.pgd(x, y, **kwargs)
        elif self.attack_type == AttackType.CARLINI_WAGNER:
            return self.carlini_wagner(x, y, **kwargs)
        else:
            return self.fgsm(x, y, **kwargs)

    def success_rate(self) -> float:
        """Get attack success rate."""
        if self._n_attacks == 0:
            return 0.0
        return self._n_successful / self._n_attacks

    def statistics(self) -> Dict[str, Any]:
        """Get attack statistics."""
        return {
            "n_attacks": self._n_attacks,
            "n_successful": self._n_successful,
            "success_rate": self.success_rate(),
            "attack_type": self.attack_type.value,
            "epsilon": self.epsilon,
        }


class AdversarialDefense:
    """
    Defend against adversarial attacks.

    Supports:
    - Adversarial training
    - Input preprocessing (denoising, quantization)
    - Randomization
    - Detection
    - Certified defenses
    """

    def __init__(
        self,
        model: SimpleNN,
        defense_type: DefenseType = DefenseType.INPUT_DENOISING,
    ):
        self.model = model
        self.defense_type = defense_type

        # Statistics
        self._n_defended = 0
        self._n_detected = 0

    def adversarial_training(
        self,
        data: List[Tuple[np.ndarray, int]],
        epsilon: float = 0.1,
        n_epochs: int = 10,
        learning_rate: float = 0.01,
        attack_ratio: float = 0.5,
    ) -> Dict[str, List[float]]:
        """
        Train model on mixture of clean and adversarial examples.

        Args:
            data: List of (input, label) pairs
            epsilon: Perturbation budget for generating adversarial examples
            n_epochs: Training epochs
            learning_rate: Learning rate
            attack_ratio: Fraction of examples to attack

        Returns:
            Training history
        """
        attacker = AdversarialAttack(self.model, epsilon=epsilon)
        history = {"clean_loss": [], "adv_loss": [], "accuracy": []}

        for epoch in range(n_epochs):
            np.random.shuffle(data)
            clean_losses = []
            adv_losses = []
            correct = 0

            for x, y in data:
                # Decide whether to use adversarial example
                use_adversarial = np.random.random() < attack_ratio

                if use_adversarial:
                    # Generate adversarial example using PGD
                    result = attacker.pgd(x, y, epsilon=epsilon, n_steps=5)
                    x_train = result.adversarial
                else:
                    x_train = x

                # Forward pass
                logits = self.model.forward(x_train)
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)

                # Loss
                loss = -np.log(probs[y] + 1e-8)
                if use_adversarial:
                    adv_losses.append(loss)
                else:
                    clean_losses.append(loss)

                # Accuracy
                if np.argmax(logits) == y:
                    correct += 1

                # Backward pass (simplified)
                grad_logits = probs.copy()
                grad_logits[y] -= 1

                grad_W2 = np.outer(self.model._h, grad_logits)
                grad_b2 = grad_logits
                grad_h = grad_logits @ self.model.W2.T
                grad_pre_h = grad_h * (self.model._h > 0).astype(float)
                grad_W1 = np.outer(x_train, grad_pre_h)
                grad_b1 = grad_pre_h

                # Update
                self.model.update(grad_W1, grad_b1, grad_W2, grad_b2, learning_rate)

            history["clean_loss"].append(np.mean(clean_losses) if clean_losses else 0)
            history["adv_loss"].append(np.mean(adv_losses) if adv_losses else 0)
            history["accuracy"].append(correct / len(data))

        return history

    def input_denoising(
        self,
        x: np.ndarray,
        method: str = "median",
        **kwargs,
    ) -> DefenseResult:
        """
        Apply input denoising defense.

        Args:
            x: Input to defend
            method: Denoising method ("median", "gaussian", "quantize")

        Returns:
            DefenseResult with denoised input
        """
        self._n_defended += 1

        if method == "median":
            # Median filter (simplified for 1D)
            kernel_size = kwargs.get("kernel_size", 3)
            defended = self._median_filter(x, kernel_size)
        elif method == "gaussian":
            # Gaussian smoothing
            sigma = kwargs.get("sigma", 0.5)
            defended = self._gaussian_smooth(x, sigma)
        elif method == "quantize":
            # Bit-depth reduction
            levels = kwargs.get("levels", 8)
            defended = np.round(x * levels) / levels
        else:
            defended = x.copy()

        return DefenseResult(
            defended_input=defended,
            detected_attack=False,  # Denoising doesn't detect
            confidence=1.0,
            defense_type=DefenseType.INPUT_DENOISING,
        )

    def _median_filter(self, x: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply median filter."""
        result = x.copy()
        pad = kernel_size // 2
        for i in range(len(x)):
            start = max(0, i - pad)
            end = min(len(x), i + pad + 1)
            result[i] = np.median(x[start:end])
        return result

    def _gaussian_smooth(self, x: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian smoothing."""
        kernel_size = int(6 * sigma) | 1  # Ensure odd
        kernel = np.exp(-np.arange(-kernel_size // 2, kernel_size // 2 + 1) ** 2 / (2 * sigma ** 2))
        kernel = kernel / np.sum(kernel)

        # Convolve
        result = np.convolve(x, kernel, mode='same')
        return result

    def randomization(
        self,
        x: np.ndarray,
        noise_std: float = 0.05,
        n_samples: int = 10,
    ) -> DefenseResult:
        """
        Apply randomization defense.

        Add random noise and take majority vote.

        Args:
            x: Input to defend
            noise_std: Standard deviation of noise
            n_samples: Number of random samples

        Returns:
            DefenseResult
        """
        self._n_defended += 1

        predictions = []
        for _ in range(n_samples):
            x_noisy = x + np.random.normal(0, noise_std, x.shape)
            x_noisy = np.clip(x_noisy, 0, 1)
            pred = self.model.predict(x_noisy)
            predictions.append(pred)

        # Majority vote
        unique, counts = np.unique(predictions, return_counts=True)
        majority_pred = unique[np.argmax(counts)]
        confidence = np.max(counts) / n_samples

        # Detection: high variance in predictions suggests adversarial
        detected = len(unique) > n_samples / 2

        if detected:
            self._n_detected += 1

        return DefenseResult(
            defended_input=x,  # Return original, but prediction is robust
            detected_attack=detected,
            confidence=confidence,
            defense_type=DefenseType.RANDOMIZATION,
        )

    def detect_adversarial(
        self,
        x: np.ndarray,
        threshold: float = 0.5,
    ) -> DefenseResult:
        """
        Detect whether input is adversarial.

        Uses multiple heuristics:
        - Input statistics
        - Prediction confidence
        - Local smoothness

        Args:
            x: Input to check
            threshold: Detection threshold

        Returns:
            DefenseResult with detection result
        """
        self._n_defended += 1

        # Heuristic 1: Check input statistics
        stat_score = self._check_input_stats(x)

        # Heuristic 2: Check prediction confidence
        conf_score = self._check_confidence(x)

        # Heuristic 3: Check local smoothness
        smooth_score = self._check_smoothness(x)

        # Combine scores
        detection_score = 0.4 * stat_score + 0.3 * conf_score + 0.3 * smooth_score
        detected = detection_score > threshold

        if detected:
            self._n_detected += 1

        return DefenseResult(
            defended_input=x,
            detected_attack=detected,
            confidence=detection_score,
            defense_type=DefenseType.DETECTION,
        )

    def _check_input_stats(self, x: np.ndarray) -> float:
        """Check if input statistics are anomalous."""
        # Simple check: unusual variance or range
        std = np.std(x)
        mean = np.mean(x)

        # Normal inputs should have reasonable stats
        if std < 0.1 or std > 0.5:
            return 0.7
        if mean < 0.2 or mean > 0.8:
            return 0.6
        return 0.3

    def _check_confidence(self, x: np.ndarray) -> float:
        """Check if prediction confidence is anomalous."""
        probs = self.model.predict_proba(x)
        max_prob = np.max(probs)
        entropy = -np.sum(probs * np.log(probs + 1e-8))

        # High entropy or low confidence suggests adversarial
        if max_prob < 0.5:
            return 0.7
        if entropy > 1.5:
            return 0.6
        return 0.3

    def _check_smoothness(self, x: np.ndarray) -> float:
        """Check local smoothness of predictions."""
        n_neighbors = 5
        neighbors_same = 0

        original_pred = self.model.predict(x)

        for _ in range(n_neighbors):
            x_neighbor = x + np.random.normal(0, 0.01, x.shape)
            x_neighbor = np.clip(x_neighbor, 0, 1)
            neighbor_pred = self.model.predict(x_neighbor)
            if neighbor_pred == original_pred:
                neighbors_same += 1

        # Low consistency suggests adversarial boundary
        consistency = neighbors_same / n_neighbors
        return 1.0 - consistency

    def certified_defense(
        self,
        x: np.ndarray,
        sigma: float = 0.25,
        n_samples: int = 100,
        alpha: float = 0.001,
    ) -> Tuple[int, float]:
        """
        Certified robust prediction using randomized smoothing.

        Provides provable robustness guarantee: the prediction is guaranteed
        to be the same for any perturbation with L2 norm less than radius.

        Args:
            x: Input to classify
            sigma: Noise standard deviation
            n_samples: Number of samples for certification
            alpha: Failure probability

        Returns:
            (predicted_class, certified_radius)
        """
        self._n_defended += 1

        # Sample noisy predictions
        predictions = []
        for _ in range(n_samples):
            x_noisy = x + np.random.normal(0, sigma, x.shape)
            pred = self.model.predict(x_noisy)
            predictions.append(pred)

        # Count predictions
        unique, counts = np.unique(predictions, return_counts=True)
        top_class = unique[np.argmax(counts)]
        top_count = np.max(counts)

        # Compute certified radius using Cohen et al. (2019)
        p_lower = self._lower_confidence_bound(top_count, n_samples, alpha)

        if p_lower > 0.5:
            # Can certify
            from scipy.stats import norm
            radius = sigma * norm.ppf(p_lower)
            return top_class, max(0, radius)
        else:
            # Cannot certify
            return top_class, 0.0

    def _lower_confidence_bound(
        self,
        k: int,
        n: int,
        alpha: float,
    ) -> float:
        """Compute lower confidence bound for binomial proportion."""
        from scipy.stats import beta
        return beta.ppf(alpha, k, n - k + 1)

    def defend(
        self,
        x: np.ndarray,
        **kwargs,
    ) -> DefenseResult:
        """Apply configured defense."""
        if self.defense_type == DefenseType.INPUT_DENOISING:
            return self.input_denoising(x, **kwargs)
        elif self.defense_type == DefenseType.RANDOMIZATION:
            return self.randomization(x, **kwargs)
        elif self.defense_type == DefenseType.DETECTION:
            return self.detect_adversarial(x, **kwargs)
        else:
            return DefenseResult(
                defended_input=x,
                detected_attack=False,
                confidence=1.0,
                defense_type=self.defense_type,
            )

    def detection_rate(self) -> float:
        """Get detection rate."""
        if self._n_defended == 0:
            return 0.0
        return self._n_detected / self._n_defended

    def statistics(self) -> Dict[str, Any]:
        """Get defense statistics."""
        return {
            "n_defended": self._n_defended,
            "n_detected": self._n_detected,
            "detection_rate": self.detection_rate(),
            "defense_type": self.defense_type.value,
        }
