"""
Causal Representation Learning.

Learns disentangled causal factors from observations:
- Causal encoder: x -> z (causal latent factors)
- Causal decoder: z -> x (reconstruction)
- Intervention-based disentanglement
- Causal mechanism learning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np

from .differentiable_scm import NeuralNetwork, ActivationType


class DisentanglementObjective(Enum):
    """Objectives for disentanglement."""
    BETA_VAE = "beta_vae"           # KL divergence regularization
    FACTOR_VAE = "factor_vae"       # Total correlation
    TC_VAE = "tc_vae"               # Total correlation decomposition
    INTERVENTION = "intervention"   # Intervention-based


@dataclass
class CausalFactor:
    """A learned causal factor."""
    index: int
    name: str
    mean: float
    std: float
    interventional_sensitivity: float  # How much interventions affect this factor
    sparsity: float  # Sparsity of causal connections


@dataclass
class CausalEncoderConfig:
    """Configuration for causal encoder."""
    input_dim: int
    latent_dim: int
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    activation: ActivationType = ActivationType.RELU
    variational: bool = True
    beta: float = 1.0  # KL weight for beta-VAE
    random_seed: Optional[int] = None


class CausalEncoder:
    """
    Encoder that maps observations to causal latent factors.

    Supports:
    - Deterministic encoding
    - Variational encoding (for uncertainty)
    - Intervention-aware encoding
    """

    def __init__(self, config: CausalEncoderConfig):
        self.config = config

        # Main encoder network
        self.encoder = NeuralNetwork(
            input_dim=config.input_dim,
            output_dim=config.latent_dim * (2 if config.variational else 1),
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            random_seed=config.random_seed,
        )

        # Running statistics for latent factors
        self._z_mean = np.zeros(config.latent_dim)
        self._z_var = np.ones(config.latent_dim)
        self._n_samples = 0

    def encode(
        self,
        x: np.ndarray,
        return_params: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """
        Encode observation to latent space.

        Args:
            x: Input observation
            return_params: If True, also return mean and log_var

        Returns:
            z: Latent representation (sampled if variational)
            params: (mean, log_var) if return_params and variational
        """
        h = self.encoder.forward(x)

        if self.config.variational:
            mean = h[:self.config.latent_dim]
            log_var = h[self.config.latent_dim:]

            # Reparameterization trick (clamp log_var for numerical stability)
            log_var = np.clip(log_var, -20, 20)
            std = np.exp(0.5 * log_var)
            eps = np.random.normal(0, 1, size=mean.shape)
            z = mean + std * eps

            # Update running statistics
            self._update_stats(z)

            if return_params:
                return z, (mean, log_var)
            return z, None
        else:
            z = h
            self._update_stats(z)
            return z, None

    def encode_deterministic(self, x: np.ndarray) -> np.ndarray:
        """Encode without sampling (use mean for variational)."""
        h = self.encoder.forward(x)
        if self.config.variational:
            return h[:self.config.latent_dim]
        return h

    def _update_stats(self, z: np.ndarray) -> None:
        """Update running statistics."""
        self._n_samples += 1
        delta = z - self._z_mean
        self._z_mean += delta / self._n_samples
        self._z_var += (z - self._z_mean) * delta - self._z_var / self._n_samples


class CausalDecoder:
    """
    Decoder that reconstructs observations from causal factors.

    Also learns the causal mechanisms between latent factors.
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None,
        activation: ActivationType = ActivationType.RELU,
        random_seed: Optional[int] = None,
    ):
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Decoder network
        self.decoder = NeuralNetwork(
            input_dim=latent_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims or [64, 128],
            activation=activation,
            random_seed=random_seed,
        )

        # Causal structure between latent factors (adjacency matrix)
        self.causal_adjacency = np.zeros((latent_dim, latent_dim))

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent factors to observation."""
        return self.decoder.forward(z)

    def intervene(
        self,
        z: np.ndarray,
        factor_idx: int,
        value: float,
    ) -> np.ndarray:
        """Intervene on a latent factor and propagate effects."""
        z_intervened = z.copy()
        z_intervened[factor_idx] = value

        # Propagate through causal structure
        for i in range(self.latent_dim):
            if i != factor_idx:
                # Apply learned causal mechanism
                effect = 0.0
                for j in range(self.latent_dim):
                    effect += self.causal_adjacency[j, i] * z_intervened[j]
                if effect != 0:
                    z_intervened[i] = effect

        return z_intervened


class CausalRepresentationLearner:
    """
    Complete causal representation learning system.

    Learns:
    1. Disentangled latent factors from observations
    2. Causal relationships between factors
    3. Mechanisms that generate observations from factors

    Key insight: Interventions reveal causal structure.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] = None,
        objective: DisentanglementObjective = DisentanglementObjective.BETA_VAE,
        beta: float = 4.0,
        learning_rate: float = 0.001,
        random_seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.objective = objective
        self.beta = beta
        self.learning_rate = learning_rate

        hidden_dims = hidden_dims or [128, 64]

        # Encoder
        encoder_config = CausalEncoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            variational=True,
            beta=beta,
            random_seed=random_seed,
        )
        self.encoder = CausalEncoder(encoder_config)

        # Decoder
        self.decoder = CausalDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=list(reversed(hidden_dims)),
            random_seed=random_seed,
        )

        # Learned causal factors
        self._factors: List[CausalFactor] = []

        # Training history
        self._recon_losses: List[float] = []
        self._kl_losses: List[float] = []
        self._total_losses: List[float] = []

        # Statistics
        self._n_train_steps = 0
        self._n_interventions = 0

    def forward(
        self,
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Forward pass: encode and decode.

        Returns:
            z: Latent representation
            x_recon: Reconstructed observation
            mean: Latent mean (variational)
            log_var: Latent log variance (variational)
        """
        z, params = self.encoder.encode(x, return_params=True)
        x_recon = self.decoder.decode(z)

        if params is not None:
            mean, log_var = params
        else:
            mean = z
            log_var = np.zeros_like(z)

        return z, x_recon, mean, log_var

    def compute_loss(
        self,
        x: np.ndarray,
        x_recon: np.ndarray,
        mean: np.ndarray,
        log_var: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Compute training loss.

        Returns:
            total_loss, reconstruction_loss, kl_loss
        """
        # Reconstruction loss (MSE)
        recon_loss = np.mean((x - x_recon) ** 2)

        # KL divergence loss (clamp for numerical stability)
        log_var_clamp = np.clip(log_var, -20, 20)
        kl_loss = -0.5 * np.mean(1 + log_var_clamp - mean ** 2 - np.exp(log_var_clamp))

        # Total loss
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def train_step(
        self,
        x: np.ndarray,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            x: Input observation

        Returns:
            Dictionary of losses
        """
        self._n_train_steps += 1

        # Forward pass
        z, x_recon, mean, log_var = self.forward(x)

        # Clip reconstruction for numerical stability
        x_recon = np.clip(x_recon, -100, 100)

        # Compute loss
        total_loss, recon_loss, kl_loss = self.compute_loss(x, x_recon, mean, log_var)

        # Handle NaN
        if np.isnan(total_loss):
            return {"total_loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0}

        # Backward pass (simplified gradient descent)
        recon_error = np.clip(x - x_recon, -10, 10)  # Clip gradients

        # Update decoder
        z_grad = self.decoder.decoder.backward(recon_error)
        z_grad = np.clip(z_grad, -5, 5)  # Gradient clipping
        self.decoder.decoder.update(self.learning_rate)

        # Update encoder (through reparameterization)
        encoder_grad = np.clip(z_grad, -5, 5)
        # Add KL gradient (clamp for numerical stability)
        kl_grad_mean = np.clip(self.beta * mean, -5, 5)
        log_var_clamp = np.clip(log_var, -20, 20)
        kl_grad_log_var = np.clip(self.beta * 0.5 * (np.exp(log_var_clamp) - 1), -5, 5)

        full_grad = np.concatenate([encoder_grad + kl_grad_mean, kl_grad_log_var])
        full_grad = np.clip(full_grad, -5, 5)  # Final gradient clipping
        self.encoder.encoder.backward(full_grad)
        self.encoder.encoder.update(self.learning_rate)

        # Record losses
        self._recon_losses.append(recon_loss)
        self._kl_losses.append(kl_loss)
        self._total_losses.append(total_loss)

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def train(
        self,
        data: np.ndarray,
        n_epochs: int = 100,
        batch_size: int = 32,
    ) -> List[Dict[str, float]]:
        """
        Train on dataset.

        Args:
            data: (n_samples, input_dim) training data
            n_epochs: Number of epochs
            batch_size: Batch size

        Returns:
            List of loss dictionaries per epoch
        """
        n_samples = data.shape[0]
        epoch_losses = []

        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)

            epoch_loss = {"total_loss": 0, "reconstruction_loss": 0, "kl_loss": 0}
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = data[batch_indices]

                # Average loss over batch
                batch_loss = {"total_loss": 0, "reconstruction_loss": 0, "kl_loss": 0}
                for x in batch:
                    losses = self.train_step(x)
                    for key in batch_loss:
                        batch_loss[key] += losses[key]

                for key in batch_loss:
                    batch_loss[key] /= len(batch)
                    epoch_loss[key] += batch_loss[key]

                n_batches += 1

            # Average over batches
            for key in epoch_loss:
                epoch_loss[key] /= n_batches

            epoch_losses.append(epoch_loss)

        return epoch_losses

    def loss_disentanglement(
        self,
        x: np.ndarray,
        intervention_idx: int,
        intervention_value: float,
    ) -> float:
        """
        Compute disentanglement loss using intervention.

        A well-disentangled representation should change only
        the intervened factor when we intervene on a single
        generative factor.
        """
        self._n_interventions += 1

        # Encode original
        z_orig = self.encoder.encode_deterministic(x)

        # Create intervened observation (this would come from environment)
        z_intervened = z_orig.copy()
        z_intervened[intervention_idx] = intervention_value

        x_intervened = self.decoder.decode(z_intervened)

        # Re-encode intervened observation
        z_reencoded = self.encoder.encode_deterministic(x_intervened)

        # Disentanglement loss: other factors should not change
        other_indices = [i for i in range(self.latent_dim) if i != intervention_idx]
        disentanglement_loss = np.mean((z_reencoded[other_indices] - z_orig[other_indices]) ** 2)

        return disentanglement_loss

    def learn_causal_mechanisms(
        self,
        data: np.ndarray,
        n_samples: int = 100,
    ) -> Dict[str, Callable]:
        """
        Learn causal mechanisms between latent factors.

        Uses interventions to identify which factors affect which.
        """
        mechanisms = {}

        # Encode all data
        latents = np.array([self.encoder.encode_deterministic(x) for x in data[:n_samples]])

        # For each pair of factors, estimate causal effect
        for i in range(self.latent_dim):
            for j in range(self.latent_dim):
                if i == j:
                    continue

                # Interventional approach: vary factor i, measure effect on j
                effects = []
                for z in latents:
                    for delta in [-1, 0, 1]:
                        z_int = z.copy()
                        z_int[i] = z[i] + delta

                        # Propagate through decoder and re-encode
                        x_int = self.decoder.decode(z_int)
                        z_out = self.encoder.encode_deterministic(x_int)

                        effects.append((delta, z_out[j] - z[j]))

                # Estimate linear effect
                if effects:
                    deltas = np.array([e[0] for e in effects])
                    changes = np.array([e[1] for e in effects])
                    coef = np.dot(deltas, changes) / (np.dot(deltas, deltas) + 1e-8)

                    self.decoder.causal_adjacency[i, j] = coef

                    if abs(coef) > 0.1:
                        mechanisms[f"z{i}->z{j}"] = lambda z_i, c=coef: c * z_i

        return mechanisms

    def extract_factors(self) -> List[CausalFactor]:
        """
        Extract information about learned causal factors.
        """
        self._factors = []

        for i in range(self.latent_dim):
            # Compute interventional sensitivity
            sensitivity = np.sum(np.abs(self.decoder.causal_adjacency[i, :]))

            # Compute sparsity (how many other factors it affects)
            sparsity = np.mean(np.abs(self.decoder.causal_adjacency[i, :]) > 0.1)

            factor = CausalFactor(
                index=i,
                name=f"z_{i}",
                mean=float(self.encoder._z_mean[i]),
                std=float(np.sqrt(self.encoder._z_var[i])),
                interventional_sensitivity=float(sensitivity),
                sparsity=float(sparsity),
            )
            self._factors.append(factor)

        return self._factors

    def interventional_consistency(
        self,
        data: np.ndarray,
        n_tests: int = 50,
    ) -> float:
        """
        Measure interventional consistency of learned representation.

        A good causal representation should be consistent under interventions:
        encode(decode(intervene(z))) ~= intervene(z)
        """
        consistencies = []

        for _ in range(n_tests):
            x = data[np.random.randint(len(data))]
            z = self.encoder.encode_deterministic(x)

            # Random intervention
            factor_idx = np.random.randint(self.latent_dim)
            value = np.random.normal(0, 1)

            z_int = z.copy()
            z_int[factor_idx] = value

            # Reconstruct and re-encode
            x_int = self.decoder.decode(z_int)
            z_reencoded = self.encoder.encode_deterministic(x_int)

            # Measure consistency
            consistency = 1 - np.mean((z_int - z_reencoded) ** 2)
            consistencies.append(max(0, consistency))

        return float(np.mean(consistencies))

    def statistics(self) -> Dict[str, Any]:
        """Get learner statistics."""
        return {
            "input_dim": self.input_dim,
            "latent_dim": self.latent_dim,
            "objective": self.objective.value,
            "beta": self.beta,
            "n_train_steps": self._n_train_steps,
            "n_interventions": self._n_interventions,
            "avg_recon_loss": float(np.mean(self._recon_losses[-100:])) if self._recon_losses else 0.0,
            "avg_kl_loss": float(np.mean(self._kl_losses[-100:])) if self._kl_losses else 0.0,
        }
