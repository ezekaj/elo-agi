"""
State Encoder: Compresses multimodal observations into unified latent state.

The state encoder transforms raw sensory input from multiple modalities
(visual, auditory, proprioceptive, etc.) into a compact latent representation
suitable for prediction and planning.

Based on:
- Variational autoencoders for world models (Ha & Schmidhuber, 2018)
- PlaNet and Dreamer architectures (Hafner et al., 2019)
- arXiv:2509.20021 - Embodied AI world models
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import time


class Modality(Enum):
    """Sensory modalities for encoding."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    PROPRIOCEPTIVE = "proprioceptive"
    TACTILE = "tactile"
    INTEROCEPTIVE = "interoceptive"
    SEMANTIC = "semantic"
    LINGUISTIC = "linguistic"


@dataclass
class EncoderParams:
    """Parameters for the state encoder."""

    n_latent: int = 128  # Latent state dimensionality
    n_features_per_modality: int = 64  # Features per modality
    compression_ratio: float = 0.5  # How much to compress
    noise_std: float = 0.01  # Noise for regularization
    sparsity_target: float = 0.1  # Target activation sparsity
    use_variational: bool = True  # Use VAE-style encoding
    kl_weight: float = 0.001  # KL divergence weight


@dataclass
class EncodedState:
    """An encoded latent state representation."""

    mean: np.ndarray  # Mean of latent distribution
    log_var: np.ndarray  # Log variance (for variational)
    sample: np.ndarray  # Sampled latent state
    modality_contributions: Dict[Modality, float]  # Per-modality contribution
    uncertainty: float  # Overall encoding uncertainty
    timestamp: float = field(default_factory=time.time)

    @property
    def std(self) -> np.ndarray:
        """Get standard deviation from log variance."""
        return np.exp(0.5 * self.log_var)

    def kl_divergence(self) -> float:
        """Compute KL divergence from standard normal."""
        return float(-0.5 * np.mean(1 + self.log_var - self.mean**2 - np.exp(self.log_var)))


class StateEncoder:
    """
    Encoder that compresses multimodal observations into latent states.

    The encoder learns to represent observations in a compact format
    that preserves information relevant for prediction while discarding
    irrelevant details. Key features:

    1. **Multimodal**: Handles multiple sensory modalities
    2. **Variational**: Produces distributions, not point estimates
    3. **Sparse**: Encourages sparse activations for efficiency
    4. **Hierarchical**: Can encode at multiple abstraction levels

    Based on predictive processing: the encoder learns representations
    that minimize prediction error in the transition model.
    """

    def __init__(self, params: Optional[EncoderParams] = None):
        self.params = params or EncoderParams()

        # Encoding weights for each modality
        self._encoder_weights: Dict[Modality, np.ndarray] = {}
        self._encoder_biases: Dict[Modality, np.ndarray] = {}

        # Latent projection weights
        self._mean_proj = (
            np.random.randn(
                self.params.n_latent, len(Modality) * self.params.n_features_per_modality
            )
            * 0.01
        )
        self._logvar_proj = (
            np.random.randn(
                self.params.n_latent, len(Modality) * self.params.n_features_per_modality
            )
            * 0.01
        )

        # Initialize modality encoders
        for modality in Modality:
            n_hidden = int(self.params.n_features_per_modality * self.params.compression_ratio)
            self._encoder_weights[modality] = (
                np.random.randn(n_hidden, self.params.n_features_per_modality) * 0.1
            )
            self._encoder_biases[modality] = np.zeros(n_hidden)

        # Reconstruction weights (decoder)
        self._decoder_weights: Dict[Modality, np.ndarray] = {}
        for modality in Modality:
            n_hidden = int(self.params.n_features_per_modality * self.params.compression_ratio)
            self._decoder_weights[modality] = (
                np.random.randn(self.params.n_features_per_modality, n_hidden) * 0.1
            )

        # Statistics
        self._encoding_history: List[EncodedState] = []
        self._reconstruction_errors: List[float] = []

    def encode(
        self,
        observations: Dict[Modality, np.ndarray],
        deterministic: bool = False,
    ) -> EncodedState:
        """
        Encode multimodal observations into a latent state.

        Args:
            observations: Dict mapping modalities to observation vectors
            deterministic: If True, return mean instead of sample

        Returns:
            EncodedState with latent representation
        """
        # Encode each modality
        modality_encodings = []
        modality_contributions = {}

        for modality in Modality:
            if modality in observations:
                obs = observations[modality]
                # Ensure correct size
                if len(obs) != self.params.n_features_per_modality:
                    obs = np.resize(obs, self.params.n_features_per_modality)

                # Forward pass through modality encoder
                encoded = self._encode_modality(modality, obs)
                modality_encodings.append(encoded)

                # Compute contribution
                contribution = np.mean(np.abs(encoded))
                modality_contributions[modality] = float(contribution)
            else:
                # Zero encoding for missing modality
                n_hidden = int(self.params.n_features_per_modality * self.params.compression_ratio)
                modality_encodings.append(np.zeros(n_hidden))
                modality_contributions[modality] = 0.0

        # Concatenate all modality encodings
        concatenated = np.concatenate(modality_encodings)

        # Pad or truncate to match projection size
        expected_size = len(Modality) * self.params.n_features_per_modality
        if len(concatenated) < expected_size:
            concatenated = np.pad(concatenated, (0, expected_size - len(concatenated)))
        elif len(concatenated) > expected_size:
            concatenated = concatenated[:expected_size]

        # Project to latent space
        mean = np.tanh(self._mean_proj @ concatenated)
        log_var = np.clip(self._logvar_proj @ concatenated, -10, 2)

        # Sample from latent distribution
        if deterministic or not self.params.use_variational:
            sample = mean
        else:
            std = np.exp(0.5 * log_var)
            sample = mean + std * np.random.randn(*mean.shape)

        # Apply sparsity (soft thresholding)
        sample = self._apply_sparsity(sample)

        # Compute uncertainty
        uncertainty = float(np.mean(np.exp(log_var)))

        # Create encoded state
        encoded = EncodedState(
            mean=mean,
            log_var=log_var,
            sample=sample,
            modality_contributions=modality_contributions,
            uncertainty=uncertainty,
        )

        # Record history
        self._encoding_history.append(encoded)
        if len(self._encoding_history) > 1000:
            self._encoding_history.pop(0)

        return encoded

    def _encode_modality(self, modality: Modality, obs: np.ndarray) -> np.ndarray:
        """Encode a single modality observation."""
        W = self._encoder_weights[modality]
        b = self._encoder_biases[modality]

        # Linear + ReLU
        z = W @ obs + b
        z = np.maximum(0, z)  # ReLU

        return z

    def _apply_sparsity(self, z: np.ndarray) -> np.ndarray:
        """Apply soft sparsity constraint."""
        # Soft thresholding
        threshold = np.percentile(np.abs(z), 100 * (1 - self.params.sparsity_target))
        sparse_z = np.sign(z) * np.maximum(0, np.abs(z) - threshold * 0.5)
        return sparse_z

    def decode(self, state: EncodedState, modality: Modality) -> np.ndarray:
        """
        Decode latent state back to observation space.

        Used for computing reconstruction error and generating imagined observations.
        """
        # Simple linear decoder
        W = self._decoder_weights[modality]

        # Get modality slice from latent state
        modality_idx = list(Modality).index(modality)
        n_hidden = int(self.params.n_features_per_modality * self.params.compression_ratio)
        start = modality_idx * n_hidden
        end = start + n_hidden

        # Ensure we don't exceed sample size
        if end > len(state.sample):
            z_slice = np.resize(state.sample, n_hidden)
        else:
            z_slice = state.sample[start:end]

        # Pad if necessary
        if len(z_slice) < n_hidden:
            z_slice = np.pad(z_slice, (0, n_hidden - len(z_slice)))

        reconstruction = W @ z_slice
        return reconstruction

    def compute_reconstruction_error(
        self,
        observations: Dict[Modality, np.ndarray],
        state: EncodedState,
    ) -> float:
        """Compute reconstruction error for given observations."""
        total_error = 0.0
        n_modalities = 0

        for modality, obs in observations.items():
            if len(obs) != self.params.n_features_per_modality:
                obs = np.resize(obs, self.params.n_features_per_modality)

            reconstruction = self.decode(state, modality)
            error = np.mean((obs - reconstruction) ** 2)
            total_error += error
            n_modalities += 1

        avg_error = total_error / max(1, n_modalities)
        self._reconstruction_errors.append(avg_error)
        if len(self._reconstruction_errors) > 1000:
            self._reconstruction_errors.pop(0)

        return avg_error

    def update_weights(
        self,
        observations: Dict[Modality, np.ndarray],
        state: EncodedState,
        learning_rate: float = 0.001,
    ) -> float:
        """
        Update encoder weights to minimize reconstruction error.

        Simple gradient descent on reconstruction loss.
        """
        total_loss = 0.0

        for modality, obs in observations.items():
            if len(obs) != self.params.n_features_per_modality:
                obs = np.resize(obs, self.params.n_features_per_modality)

            reconstruction = self.decode(state, modality)
            error = obs - reconstruction

            # Update decoder weights (gradient descent)
            modality_idx = list(Modality).index(modality)
            n_hidden = int(self.params.n_features_per_modality * self.params.compression_ratio)
            start = modality_idx * n_hidden
            end = start + min(n_hidden, len(state.sample) - start)

            if end > start:
                z_slice = state.sample[start:end]
                if len(z_slice) < n_hidden:
                    z_slice = np.pad(z_slice, (0, n_hidden - len(z_slice)))

                # Outer product gradient
                dW = np.outer(error, z_slice)
                self._decoder_weights[modality] += learning_rate * dW

            total_loss += np.mean(error**2)

        # Add KL loss if variational
        if self.params.use_variational:
            kl_loss = state.kl_divergence() * self.params.kl_weight
            total_loss += kl_loss

        return total_loss

    def blend_states(
        self,
        state1: EncodedState,
        state2: EncodedState,
        alpha: float = 0.5,
    ) -> EncodedState:
        """Blend two encoded states (for interpolation/imagination)."""
        mean = alpha * state1.mean + (1 - alpha) * state2.mean
        log_var = alpha * state1.log_var + (1 - alpha) * state2.log_var
        sample = alpha * state1.sample + (1 - alpha) * state2.sample

        # Blend contributions
        contributions = {}
        for modality in Modality:
            c1 = state1.modality_contributions.get(modality, 0)
            c2 = state2.modality_contributions.get(modality, 0)
            contributions[modality] = alpha * c1 + (1 - alpha) * c2

        uncertainty = alpha * state1.uncertainty + (1 - alpha) * state2.uncertainty

        return EncodedState(
            mean=mean,
            log_var=log_var,
            sample=sample,
            modality_contributions=contributions,
            uncertainty=uncertainty,
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get encoder statistics."""
        if not self._reconstruction_errors:
            return {
                "n_encodings": 0,
                "avg_reconstruction_error": 0.0,
                "avg_uncertainty": 0.0,
            }

        recent_states = self._encoding_history[-100:]

        return {
            "n_encodings": len(self._encoding_history),
            "avg_reconstruction_error": float(np.mean(self._reconstruction_errors[-100:])),
            "avg_uncertainty": float(np.mean([s.uncertainty for s in recent_states])),
            "avg_kl_divergence": float(np.mean([s.kl_divergence() for s in recent_states])),
        }

    def reset(self) -> None:
        """Reset encoder statistics."""
        self._encoding_history = []
        self._reconstruction_errors = []
