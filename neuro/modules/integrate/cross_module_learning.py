"""
Cross-Module Learning: Enables modules to learn from each other.

Implements gradient routing and learning signal propagation across
the cognitive architecture, enabling transfer and mutual improvement.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np

from .shared_space import SemanticEmbedding, ModalityType


class SignalType(Enum):
    """Types of learning signals."""
    ERROR = "error"           # Prediction error
    REWARD = "reward"         # Reward signal
    SURPRISE = "surprise"     # Information-theoretic surprise
    RELEVANCE = "relevance"   # Task relevance
    CONFLICT = "conflict"     # Conflict between modules


@dataclass
class LearningSignal:
    """A learning signal that propagates between modules."""
    signal_type: SignalType
    source_module: str
    target_modules: List[str]
    value: np.ndarray
    magnitude: float
    context: Optional[SemanticEmbedding] = None
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def strength(self) -> float:
        """Overall strength of the signal."""
        return float(self.magnitude * np.linalg.norm(self.value))


@dataclass
class ModuleSynapse:
    """
    Synaptic connection between two modules for learning.

    Enables credit assignment and gradient flow across modules.
    """
    source: str
    target: str
    weights: np.ndarray
    learning_rate: float = 0.01
    eligibility_trace: Optional[np.ndarray] = None
    plasticity: float = 1.0  # Modulates learning rate

    def __post_init__(self):
        if self.eligibility_trace is None:
            self.eligibility_trace = np.zeros_like(self.weights)

    def transmit(self, signal: np.ndarray) -> np.ndarray:
        """Transmit signal through synapse."""
        return signal @ self.weights

    def update(
        self,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        reward: float = 0.0,
        trace_decay: float = 0.9,
    ) -> None:
        """Update synaptic weights via Hebbian + reward modulation."""
        # Update eligibility trace
        self.eligibility_trace = (
            trace_decay * self.eligibility_trace +
            np.outer(pre_activity, post_activity)
        )

        # Apply learning with reward modulation
        effective_lr = self.learning_rate * self.plasticity
        if reward != 0:
            # Reward-modulated learning
            self.weights += effective_lr * reward * self.eligibility_trace
        else:
            # Standard Hebbian
            self.weights += effective_lr * np.outer(pre_activity, post_activity)

        # Weight normalization
        norm = np.linalg.norm(self.weights)
        if norm > 10.0:
            self.weights = self.weights * (10.0 / norm)


class GradientRouter:
    """
    Routes learning gradients between modules.

    Implements credit assignment by tracking which modules contributed
    to successful (or unsuccessful) outcomes.
    """

    def __init__(
        self,
        n_modules: int,
        embedding_dim: int = 512,
        random_seed: Optional[int] = None,
    ):
        self.n_modules = n_modules
        self.embedding_dim = embedding_dim
        self._rng = np.random.default_rng(random_seed)

        # Module influence matrix
        self._influence = np.eye(n_modules) * 0.5 + 0.5 / n_modules

        # Activity history for credit assignment
        self._activity_history: List[Dict[str, np.ndarray]] = []
        self._max_history = 100

        # Module name to index mapping
        self._module_indices: Dict[str, int] = {}

    def register_module(self, module_name: str) -> int:
        """Register a module and return its index."""
        if module_name not in self._module_indices:
            idx = len(self._module_indices)
            self._module_indices[module_name] = idx
        return self._module_indices[module_name]

    def get_module_index(self, module_name: str) -> Optional[int]:
        """Get index for a module name."""
        return self._module_indices.get(module_name)

    def record_activity(self, activities: Dict[str, np.ndarray]) -> None:
        """Record module activities for credit assignment."""
        self._activity_history.append(activities.copy())
        if len(self._activity_history) > self._max_history:
            self._activity_history.pop(0)

    def route_gradient(
        self,
        source_module: str,
        gradient: np.ndarray,
        decay: float = 0.9,
    ) -> Dict[str, np.ndarray]:
        """Route gradient from source to all connected modules."""
        source_idx = self.get_module_index(source_module)
        if source_idx is None:
            return {}

        routed = {}
        for module_name, idx in self._module_indices.items():
            if module_name == source_module:
                continue

            # Scale gradient by influence
            influence = self._influence[source_idx, idx]
            scaled_gradient = influence * decay * gradient

            routed[module_name] = scaled_gradient

        return routed

    def assign_credit(
        self,
        reward: float,
        current_activities: Dict[str, np.ndarray],
        lookback: int = 10,
    ) -> Dict[str, float]:
        """Assign credit to modules based on temporal activity."""
        credits = {name: 0.0 for name in self._module_indices}

        # Get relevant history
        history = self._activity_history[-lookback:]
        if not history:
            # Equal credit if no history
            for name in credits:
                credits[name] = reward / len(credits)
            return credits

        # Compute activity correlations with current
        for past_activities in history:
            for name in self._module_indices:
                if name in past_activities and name in current_activities:
                    # Correlation-based credit
                    past = past_activities[name]
                    curr = current_activities[name]
                    corr = np.corrcoef(past.flatten(), curr.flatten())[0, 1]
                    if not np.isnan(corr):
                        credits[name] += corr

        # Normalize and scale by reward
        total = sum(abs(c) for c in credits.values())
        if total > 1e-8:
            credits = {k: reward * (v / total) for k, v in credits.items()}

        return credits

    def update_influence(
        self,
        source: str,
        target: str,
        success: bool,
        learning_rate: float = 0.01,
    ) -> None:
        """Update influence matrix based on outcome."""
        src_idx = self.get_module_index(source)
        tgt_idx = self.get_module_index(target)

        if src_idx is None or tgt_idx is None:
            return

        delta = learning_rate if success else -learning_rate
        self._influence[src_idx, tgt_idx] = np.clip(
            self._influence[src_idx, tgt_idx] + delta,
            0.0, 1.0
        )

    def get_influence_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Get the influence matrix and module names."""
        names = sorted(self._module_indices.keys(), key=lambda x: self._module_indices[x])
        return self._influence.copy(), names


class CrossModuleLearner:
    """
    Central coordinator for cross-module learning.

    Manages:
    - Learning signal propagation
    - Synaptic connections between modules
    - Credit assignment
    - Plasticity modulation
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        n_modules: int = 20,
        random_seed: Optional[int] = None,
    ):
        self.embedding_dim = embedding_dim
        self._rng = np.random.default_rng(random_seed)

        # Gradient router
        self.router = GradientRouter(
            n_modules=n_modules,
            embedding_dim=embedding_dim,
            random_seed=random_seed,
        )

        # Synapses between modules
        self._synapses: Dict[Tuple[str, str], ModuleSynapse] = {}

        # Pending learning signals
        self._signal_queue: List[LearningSignal] = []

        # Global plasticity (neuromodulation)
        self._global_plasticity = 1.0
        self._dopamine = 0.0  # Reward signal
        self._norepinephrine = 0.0  # Arousal/attention

        # Statistics
        self._total_signals = 0
        self._total_updates = 0

    def register_module(self, module_name: str) -> None:
        """Register a module for cross-module learning."""
        self.router.register_module(module_name)

    def create_synapse(
        self,
        source: str,
        target: str,
        weight_dim: Tuple[int, int] = (512, 512),
    ) -> ModuleSynapse:
        """Create a synaptic connection between modules."""
        key = (source, target)
        if key not in self._synapses:
            weights = self._rng.normal(0, 0.1, weight_dim)
            self._synapses[key] = ModuleSynapse(
                source=source,
                target=target,
                weights=weights,
            )
        return self._synapses[key]

    def get_synapse(self, source: str, target: str) -> Optional[ModuleSynapse]:
        """Get synapse between two modules."""
        return self._synapses.get((source, target))

    def emit_signal(
        self,
        signal_type: SignalType,
        source: str,
        targets: List[str],
        value: np.ndarray,
        magnitude: float = 1.0,
        context: Optional[SemanticEmbedding] = None,
    ) -> LearningSignal:
        """Emit a learning signal from source to targets."""
        signal = LearningSignal(
            signal_type=signal_type,
            source_module=source,
            target_modules=targets,
            value=value,
            magnitude=magnitude * self._global_plasticity,
            context=context,
            timestamp=float(self._total_signals),
        )
        self._signal_queue.append(signal)
        self._total_signals += 1
        return signal

    def broadcast_signal(
        self,
        signal_type: SignalType,
        source: str,
        value: np.ndarray,
        magnitude: float = 1.0,
    ) -> LearningSignal:
        """Broadcast signal to all registered modules."""
        targets = list(self.router._module_indices.keys())
        targets = [t for t in targets if t != source]
        return self.emit_signal(signal_type, source, targets, value, magnitude)

    def process_signals(self) -> Dict[str, List[np.ndarray]]:
        """Process all pending signals and route to targets."""
        module_gradients: Dict[str, List[np.ndarray]] = {}

        for signal in self._signal_queue:
            # Route gradient
            routed = self.router.route_gradient(
                signal.source_module,
                signal.value,
                decay=signal.magnitude,
            )

            for target, gradient in routed.items():
                if target not in module_gradients:
                    module_gradients[target] = []
                module_gradients[target].append(gradient)

        # Clear queue
        self._signal_queue = []

        return module_gradients

    def apply_reward(self, reward: float, activities: Dict[str, np.ndarray]) -> None:
        """Apply reward signal for credit assignment."""
        self._dopamine = reward

        # Assign credit
        credits = self.router.assign_credit(reward, activities)

        # Update synapses with reward modulation
        for (src, tgt), synapse in self._synapses.items():
            if src in activities and tgt in activities:
                synapse.update(
                    pre_activity=activities[src],
                    post_activity=activities[tgt],
                    reward=credits.get(src, 0) + credits.get(tgt, 0),
                )
                self._total_updates += 1

    def modulate_plasticity(
        self,
        dopamine: Optional[float] = None,
        norepinephrine: Optional[float] = None,
    ) -> None:
        """Set neuromodulatory signals affecting plasticity."""
        if dopamine is not None:
            self._dopamine = np.clip(dopamine, -1.0, 1.0)

        if norepinephrine is not None:
            self._norepinephrine = np.clip(norepinephrine, 0.0, 1.0)

        # Compute global plasticity
        # High dopamine increases plasticity for rewarded actions
        # High norepinephrine increases general plasticity (arousal)
        self._global_plasticity = 1.0 + 0.5 * self._dopamine + 0.3 * self._norepinephrine
        self._global_plasticity = np.clip(self._global_plasticity, 0.1, 2.0)

        # Update synapse plasticity
        for synapse in self._synapses.values():
            synapse.plasticity = self._global_plasticity

    def transfer_knowledge(
        self,
        source: str,
        target: str,
        knowledge: np.ndarray,
        transfer_rate: float = 0.1,
    ) -> bool:
        """Transfer learned knowledge from source to target module."""
        synapse = self.get_synapse(source, target)
        if synapse is None:
            synapse = self.create_synapse(source, target)

        # Blend transferred knowledge with existing weights
        if knowledge.shape == synapse.weights.shape:
            synapse.weights = (
                (1 - transfer_rate) * synapse.weights +
                transfer_rate * knowledge
            )
            return True

        return False

    def statistics(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            "n_registered_modules": len(self.router._module_indices),
            "n_synapses": len(self._synapses),
            "total_signals": self._total_signals,
            "total_updates": self._total_updates,
            "pending_signals": len(self._signal_queue),
            "global_plasticity": self._global_plasticity,
            "dopamine": self._dopamine,
            "norepinephrine": self._norepinephrine,
        }
