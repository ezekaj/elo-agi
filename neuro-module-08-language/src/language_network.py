"""
Distributed Language Network

Implements the updated (2025) understanding of language processing:
- Broca's area: Syntactic processing, grammar, phonology
- Wernicke's area: Semantic processing, word meaning
- Arcuate fasciculus: Bidirectional information flow

Key insight: Damage to Broca's area alone does NOT necessarily cause
speech/language deficits. Language is distributed, not localized.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum


class RegionType(Enum):
    """Types of language regions"""
    BROCA = "broca"
    WERNICKE = "wernicke"
    ARCUATE = "arcuate"


@dataclass
class RegionState:
    """State of a language region"""
    activation: np.ndarray
    output: np.ndarray
    is_active: bool = True
    damage_level: float = 0.0  # 0 = intact, 1 = fully lesioned


class BrocaRegion:
    """Broca's area - Syntactic processing, grammar, phonology

    Current understanding (2025): Not just speech production,
    but primarily syntactic and grammatical processing.

    Selective inhibition for impossible languages suggests
    innate constraints on learnable grammar.
    """

    def __init__(self, dim: int = 64, learning_rate: float = 0.1):
        self.dim = dim
        self.learning_rate = learning_rate

        # Processing state
        self.state = np.zeros(dim)
        self.syntactic_buffer = np.zeros(dim)

        # Weights for syntactic processing
        self.W_syntax = np.random.randn(dim, dim) * 0.1
        self.W_grammar = np.random.randn(dim, dim) * 0.1

        # Inhibition for impossible structures
        self.inhibition_threshold = 0.5
        self.current_inhibition = 0.0

        # Damage simulation
        self.is_active = True
        self.damage_level = 0.0

    def process_syntax(self, input_signal: np.ndarray) -> np.ndarray:
        """Extract grammatical structure from input"""
        if not self.is_active:
            return np.zeros(self.dim)

        # Apply damage
        effective_signal = input_signal * (1.0 - self.damage_level)

        # Syntactic processing
        syntax_activation = np.tanh(self.W_syntax @ effective_signal)

        # Update state
        self.state = 0.8 * self.state + 0.2 * syntax_activation
        self.syntactic_buffer = syntax_activation

        return self.state * (1.0 - self.current_inhibition)

    def apply_grammar_constraints(self, structure: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply grammatical constraints to structure

        Returns:
            Tuple of (constrained structure, violation score)
        """
        # Transform through grammar weights
        grammar_output = np.tanh(self.W_grammar @ structure)

        # Check for constraint violations
        violation_score = self._compute_violation(grammar_output)

        # Apply inhibition if violation detected
        if violation_score > self.inhibition_threshold:
            self.current_inhibition = min(1.0, violation_score)
        else:
            self.current_inhibition = max(0.0, self.current_inhibition - 0.1)

        return grammar_output, violation_score

    def inhibit_impossible(self, grammar_params: np.ndarray) -> float:
        """Generate inhibition signal for impossible grammars

        This implements the finding that Broca's area shows selective
        inhibition for impossible languages only.
        """
        # Compute "impossibility" score
        impossibility = self._compute_violation(grammar_params)

        # Selective inhibition
        if impossibility > self.inhibition_threshold:
            inhibition = impossibility
        else:
            inhibition = 0.0

        self.current_inhibition = inhibition
        return inhibition

    def _compute_violation(self, params: np.ndarray) -> float:
        """Compute grammatical constraint violation score"""
        # Violation based on deviation from learned patterns
        deviation = np.linalg.norm(params - self.state)
        normalized = deviation / (np.linalg.norm(self.state) + 1e-8)
        return float(np.clip(normalized, 0, 1))

    def lesion(self, damage_level: float = 1.0) -> None:
        """Simulate damage to Broca's area"""
        self.damage_level = np.clip(damage_level, 0, 1)
        if damage_level >= 1.0:
            self.is_active = False

    def restore(self) -> None:
        """Restore from lesion"""
        self.damage_level = 0.0
        self.is_active = True

    def reset(self) -> None:
        """Reset state"""
        self.state = np.zeros(self.dim)
        self.syntactic_buffer = np.zeros(self.dim)
        self.current_inhibition = 0.0


class WernickeRegion:
    """Wernicke's area - Semantic processing, word meaning

    Handles lexical access, semantic composition, and
    context-dependent meaning integration.
    """

    def __init__(self, dim: int = 64, learning_rate: float = 0.1):
        self.dim = dim
        self.learning_rate = learning_rate

        # Processing state
        self.state = np.zeros(dim)
        self.semantic_buffer = np.zeros(dim)

        # Weights for semantic processing
        self.W_semantic = np.random.randn(dim, dim) * 0.1
        self.W_context = np.random.randn(dim, dim) * 0.1

        # Lexicon (word -> meaning mapping)
        self.lexicon: Dict[str, np.ndarray] = {}

        # Damage simulation
        self.is_active = True
        self.damage_level = 0.0

    def process_semantics(self, input_signal: np.ndarray) -> np.ndarray:
        """Extract meaning from input"""
        if not self.is_active:
            return np.zeros(self.dim)

        # Apply damage
        effective_signal = input_signal * (1.0 - self.damage_level)

        # Semantic processing
        semantic_activation = np.tanh(self.W_semantic @ effective_signal)

        # Update state
        self.state = 0.8 * self.state + 0.2 * semantic_activation
        self.semantic_buffer = semantic_activation

        return self.state

    def lexical_access(self, phoneme_repr: np.ndarray) -> Optional[np.ndarray]:
        """Map phonological representation to meaning

        In a full implementation, this would use learned associations.
        Here we use similarity matching to the lexicon.
        """
        if not self.lexicon:
            return self.process_semantics(phoneme_repr)

        # Find closest match in lexicon
        best_match = None
        best_score = -np.inf

        for word, meaning in self.lexicon.items():
            score = np.dot(phoneme_repr, meaning) / (
                np.linalg.norm(phoneme_repr) * np.linalg.norm(meaning) + 1e-8
            )
            if score > best_score:
                best_score = score
                best_match = meaning

        return best_match if best_match is not None else self.process_semantics(phoneme_repr)

    def integrate_context(self, meaning: np.ndarray, context: np.ndarray) -> np.ndarray:
        """Integrate meaning with context"""
        if not self.is_active:
            return meaning * (1.0 - self.damage_level)

        # Context modulation
        context_transform = np.tanh(self.W_context @ context)
        integrated = meaning + 0.5 * context_transform

        return np.tanh(integrated)

    def register_word(self, word: str, meaning: np.ndarray) -> None:
        """Add word to lexicon"""
        self.lexicon[word] = meaning.copy()

    def lesion(self, damage_level: float = 1.0) -> None:
        """Simulate damage to Wernicke's area"""
        self.damage_level = np.clip(damage_level, 0, 1)
        if damage_level >= 1.0:
            self.is_active = False

    def restore(self) -> None:
        """Restore from lesion"""
        self.damage_level = 0.0
        self.is_active = True

    def reset(self) -> None:
        """Reset state"""
        self.state = np.zeros(self.dim)
        self.semantic_buffer = np.zeros(self.dim)


class ArcuateFasciculus:
    """Arcuate Fasciculus - Bidirectional information flow

    White matter tract connecting Broca's and Wernicke's areas.
    Critical for language processing coordination.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Bidirectional connection weights
        self.W_broca_to_wernicke = np.random.randn(dim, dim) * 0.1
        self.W_wernicke_to_broca = np.random.randn(dim, dim) * 0.1

        # Transmission state
        self.forward_buffer = np.zeros(dim)
        self.backward_buffer = np.zeros(dim)

        # Damage simulation
        self.is_active = True
        self.damage_level = 0.0

    def forward_flow(self, broca_output: np.ndarray) -> np.ndarray:
        """Transmit from Broca's to Wernicke's area"""
        if not self.is_active:
            return np.zeros(self.dim)

        transmission = self.W_broca_to_wernicke @ broca_output
        transmission *= (1.0 - self.damage_level)
        self.forward_buffer = transmission

        return transmission

    def backward_flow(self, wernicke_output: np.ndarray) -> np.ndarray:
        """Transmit from Wernicke's to Broca's area"""
        if not self.is_active:
            return np.zeros(self.dim)

        transmission = self.W_wernicke_to_broca @ wernicke_output
        transmission *= (1.0 - self.damage_level)
        self.backward_buffer = transmission

        return transmission

    def synchronize(self, broca_state: np.ndarray, wernicke_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Coordinate processing between regions

        Returns:
            Tuple of (broca_update, wernicke_update)
        """
        if not self.is_active:
            return np.zeros(self.dim), np.zeros(self.dim)

        # Bidirectional flow
        to_wernicke = self.forward_flow(broca_state)
        to_broca = self.backward_flow(wernicke_state)

        return to_broca, to_wernicke

    def lesion(self, damage_level: float = 1.0) -> None:
        """Simulate damage to arcuate fasciculus"""
        self.damage_level = np.clip(damage_level, 0, 1)
        if damage_level >= 1.0:
            self.is_active = False

    def restore(self) -> None:
        """Restore from lesion"""
        self.damage_level = 0.0
        self.is_active = True


class DistributedLanguageNetwork:
    """Full distributed language network

    Key insight from 2025 research: Language processing is
    distributed across multiple regions. Damage to Broca's
    area alone does NOT necessarily cause deficits due to
    redundancy and compensation from other regions.
    """

    def __init__(self, dim: int = 64, learning_rate: float = 0.1):
        self.dim = dim
        self.learning_rate = learning_rate

        # Create regions
        self.broca = BrocaRegion(dim, learning_rate)
        self.wernicke = WernickeRegion(dim, learning_rate)
        self.arcuate = ArcuateFasciculus(dim)

        # Network state
        self.combined_state = np.zeros(dim)

        # Processing history
        self.processing_history: List[Dict[str, np.ndarray]] = []

    def process(self, input_signal: np.ndarray, n_iterations: int = 3) -> Dict[str, np.ndarray]:
        """Coordinated distributed processing

        Multiple iterations allow for recurrent processing
        between regions via the arcuate fasciculus.
        """
        broca_state = np.zeros(self.dim)
        wernicke_state = np.zeros(self.dim)

        # Ensure input is right dimension
        if len(input_signal) < self.dim:
            input_signal = np.pad(input_signal, (0, self.dim - len(input_signal)))
        elif len(input_signal) > self.dim:
            input_signal = input_signal[:self.dim]

        for _ in range(n_iterations):
            # Broca processes syntax
            broca_input = input_signal + 0.5 * wernicke_state
            broca_out = self.broca.process_syntax(broca_input)

            # Wernicke processes semantics
            wernicke_input = input_signal + 0.5 * broca_state
            wernicke_out = self.wernicke.process_semantics(wernicke_input)

            # Arcuate coordinates
            broca_update, wernicke_update = self.arcuate.synchronize(broca_out, wernicke_out)

            # Update states
            broca_state = broca_out + 0.3 * broca_update
            wernicke_state = wernicke_out + 0.3 * wernicke_update

        # Combined output
        self.combined_state = 0.5 * broca_state + 0.5 * wernicke_state

        result = {
            'broca': broca_state,
            'wernicke': wernicke_state,
            'combined': self.combined_state,
            'broca_inhibition': self.broca.current_inhibition
        }

        self.processing_history.append(result)
        if len(self.processing_history) > 100:
            self.processing_history.pop(0)

        return result

    def lesion(self, region: str, damage_level: float = 1.0) -> None:
        """Simulate damage to a specific region

        Args:
            region: 'broca', 'wernicke', or 'arcuate'
            damage_level: 0.0 (intact) to 1.0 (fully lesioned)
        """
        if region == 'broca':
            self.broca.lesion(damage_level)
        elif region == 'wernicke':
            self.wernicke.lesion(damage_level)
        elif region == 'arcuate':
            self.arcuate.lesion(damage_level)
        else:
            raise ValueError(f"Unknown region: {region}")

    def restore(self, region: Optional[str] = None) -> None:
        """Restore from lesion

        Args:
            region: Specific region to restore, or None for all
        """
        if region is None or region == 'broca':
            self.broca.restore()
        if region is None or region == 'wernicke':
            self.wernicke.restore()
        if region is None or region == 'arcuate':
            self.arcuate.restore()

    def get_region_states(self) -> Dict[str, np.ndarray]:
        """Get current state of all regions"""
        return {
            'broca': self.broca.state.copy(),
            'wernicke': self.wernicke.state.copy(),
            'combined': self.combined_state.copy()
        }

    def get_damage_levels(self) -> Dict[str, float]:
        """Get damage level of each region"""
        return {
            'broca': self.broca.damage_level,
            'wernicke': self.wernicke.damage_level,
            'arcuate': self.arcuate.damage_level
        }

    def is_functional(self) -> bool:
        """Check if network can still process language

        Key finding: Broca's damage alone doesn't prevent function
        due to distributed redundancy.
        """
        # Both regions lesioned = non-functional
        if not self.broca.is_active and not self.wernicke.is_active:
            return False

        # Severe damage to both = impaired but may still function
        combined_damage = self.broca.damage_level + self.wernicke.damage_level
        return combined_damage < 1.8  # Some redundancy

    def reset(self) -> None:
        """Reset all regions"""
        self.broca.reset()
        self.wernicke.reset()
        self.combined_state = np.zeros(self.dim)
        self.processing_history = []
