"""
Language Processing Hierarchy

Four-level processing hierarchy with different timescales:
- Phonological (sounds) - fastest (~10-50ms)
- Syntactic (grammar) - medium (~100-200ms)
- Semantic (meaning) - slower (~200-500ms)
- Pragmatic (context) - slowest (~500ms+)

Based on the hierarchical nature of language processing in the brain.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Callable
from dataclasses import dataclass


@dataclass
class LayerActivation:
    """Activation state of a language layer"""

    state: np.ndarray
    prediction: np.ndarray
    error: np.ndarray
    confidence: float


class LanguageLayer:
    """Base class for language processing layers"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        timescale: float,
        learning_rate: float = 0.1,
        name: str = "layer",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.timescale = timescale
        self.learning_rate = learning_rate
        self.name = name

        # Hidden state
        self.state = np.zeros(hidden_dim)

        # Weights for processing
        self.W_in = np.random.randn(hidden_dim, input_dim) * 0.1
        self.W_out = np.random.randn(output_dim, hidden_dim) * 0.1
        self.W_pred = np.random.randn(input_dim, hidden_dim) * 0.1  # Top-down prediction

        # Biases
        self.b_hidden = np.zeros(hidden_dim)
        self.b_out = np.zeros(output_dim)

        # Error tracking
        self.prediction_error = np.zeros(input_dim)
        self.error_history: List[np.ndarray] = []

    def process(self, input_signal: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """Process input and update state"""
        effective_dt = dt / self.timescale

        # Compute new activation
        activation = np.tanh(self.W_in @ input_signal + self.b_hidden)

        # Leaky integration
        self.state += effective_dt * (activation - self.state)
        self.state = np.clip(self.state, -5.0, 5.0)

        # Output
        output = np.tanh(self.W_out @ self.state + self.b_out)
        return output

    def predict_input(self) -> np.ndarray:
        """Generate top-down prediction for layer below"""
        return np.tanh(self.W_pred @ self.state)

    def receive_error(self, error: np.ndarray) -> None:
        """Receive prediction error from layer below"""
        # Ensure error matches input_dim
        if len(error) < self.input_dim:
            error = np.pad(error, (0, self.input_dim - len(error)))
        elif len(error) > self.input_dim:
            error = error[: self.input_dim]
        self.prediction_error = error
        self.error_history.append(error.copy())
        if len(self.error_history) > 100:
            self.error_history.pop(0)

    def update_weights(self, input_signal: np.ndarray, dt: float = 0.1) -> None:
        """Update weights based on prediction error"""
        # Ensure prediction_error has correct dimension
        if len(self.prediction_error) != self.input_dim:
            return  # Skip if dimensions don't match

        clipped_error = np.clip(self.prediction_error, -2.0, 2.0)

        # Update prediction weights (input_dim x hidden_dim)
        dW = np.outer(clipped_error, self.state)
        self.W_pred += self.learning_rate * dt * np.clip(dW, -0.1, 0.1)
        self.W_pred = np.clip(self.W_pred, -3.0, 3.0)

    def get_confidence(self) -> float:
        """Estimate confidence from error variance"""
        if len(self.error_history) < 2:
            return 0.5
        errors = np.array(self.error_history[-20:])
        variance = np.mean(np.var(errors, axis=0)) + 1e-8
        return 1.0 / (1.0 + variance)

    def reset(self) -> None:
        """Reset layer state"""
        self.state = np.zeros(self.hidden_dim)
        self.prediction_error = np.zeros(self.input_dim)
        self.error_history = []


class PhonologicalLayer(LanguageLayer):
    """Sound processing - fastest layer (~10-50ms)

    Processes acoustic/phonetic features and extracts phonemes.
    Predicts upcoming sounds based on phonotactic constraints.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, learning_rate: float = 0.1
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            timescale=0.01,  # 10ms
            learning_rate=learning_rate,
            name="phonological",
        )

        # Phoneme representations
        self.phoneme_inventory: Dict[str, np.ndarray] = {}
        self.current_phoneme: Optional[str] = None

    def extract_phonemes(self, acoustic_input: np.ndarray) -> np.ndarray:
        """Extract phoneme representation from acoustic input"""
        return self.process(acoustic_input)

    def predict_next_phoneme(self) -> np.ndarray:
        """Predict the next phoneme based on current state"""
        return self.predict_input()

    def register_phoneme(self, name: str, features: np.ndarray) -> None:
        """Register a phoneme in the inventory"""
        self.phoneme_inventory[name] = features.copy()


class SyntacticLayer(LanguageLayer):
    """Grammar processing - medium timescale (~100-200ms)

    Builds phrase structure from phonological input.
    Tracks grammatical dependencies and structure.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, learning_rate: float = 0.1
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            timescale=0.1,  # 100ms
            learning_rate=learning_rate,
            name="syntactic",
        )

        # Syntactic state tracking
        self.open_dependencies: List[np.ndarray] = []
        self.structure_stack: List[np.ndarray] = []

    def parse_structure(self, phonological_output: np.ndarray) -> np.ndarray:
        """Build syntactic structure from phonological input"""
        output = self.process(phonological_output)

        # Track structure
        self.structure_stack.append(self.state.copy())
        if len(self.structure_stack) > 20:
            self.structure_stack.pop(0)

        return output

    def check_grammaticality(self) -> float:
        """Return grammaticality score (0 to 1)"""
        # Based on state coherence and error history
        if len(self.error_history) < 2:
            return 0.5

        recent_errors = np.array(self.error_history[-10:])
        mean_error = np.mean(np.abs(recent_errors))
        return float(np.exp(-mean_error))

    def get_current_structure(self) -> np.ndarray:
        """Return current syntactic representation"""
        return self.state.copy()


class SemanticLayer(LanguageLayer):
    """Meaning processing - slower timescale (~200-500ms)

    Extracts compositional meaning from syntactic structure.
    Handles lexical semantics and composition.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, learning_rate: float = 0.1
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            timescale=0.3,  # 300ms
            learning_rate=learning_rate,
            name="semantic",
        )

        # Semantic memory
        self.lexicon: Dict[str, np.ndarray] = {}
        self.context_buffer: List[np.ndarray] = []

    def extract_meaning(self, syntactic_output: np.ndarray) -> np.ndarray:
        """Extract semantic representation from syntax"""
        output = self.process(syntactic_output)

        # Update context
        self.context_buffer.append(self.state.copy())
        if len(self.context_buffer) > 10:
            self.context_buffer.pop(0)

        return output

    def resolve_ambiguity(self, meanings: List[np.ndarray]) -> np.ndarray:
        """Select best meaning given context"""
        if not meanings:
            return self.state

        # Use context to disambiguate
        context = (
            np.mean(self.context_buffer, axis=0)
            if self.context_buffer
            else np.zeros(self.hidden_dim)
        )

        best_idx = 0
        best_score = -np.inf
        for i, meaning in enumerate(meanings):
            score = np.dot(meaning, context)
            if score > best_score:
                best_score = score
                best_idx = i

        return meanings[best_idx]

    def register_word(self, word: str, meaning: np.ndarray) -> None:
        """Register word meaning in lexicon"""
        self.lexicon[word] = meaning.copy()

    def lookup_word(self, word: str) -> Optional[np.ndarray]:
        """Look up word meaning"""
        return self.lexicon.get(word)


class PragmaticLayer(LanguageLayer):
    """Context processing - slowest layer (~500ms+)

    Interprets meaning in situational context.
    Infers speaker intent and handles discourse.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, learning_rate: float = 0.1
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            timescale=1.0,  # 1000ms
            learning_rate=learning_rate,
            name="pragmatic",
        )

        # Discourse state
        self.discourse_state = np.zeros(hidden_dim)
        self.speaker_model = np.zeros(hidden_dim)

    def interpret_context(
        self, semantic_output: np.ndarray, situation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Interpret meaning in context"""
        if situation is not None:
            combined = np.concatenate([semantic_output, situation[: len(semantic_output)]])[
                : self.input_dim
            ]
        else:
            combined = semantic_output

        output = self.process(combined)

        # Update discourse
        self.discourse_state = 0.9 * self.discourse_state + 0.1 * self.state

        return output

    def infer_intent(self, utterance_repr: np.ndarray) -> np.ndarray:
        """Infer speaker's communicative intent"""
        # Combine utterance with speaker model
        intent = np.tanh(
            0.5 * utterance_repr + 0.3 * self.speaker_model + 0.2 * self.discourse_state
        )
        return intent

    def update_speaker_model(self, speaker_features: np.ndarray) -> None:
        """Update model of current speaker"""
        self.speaker_model = 0.8 * self.speaker_model + 0.2 * speaker_features


class LanguageProcessingHierarchy:
    """Full 4-layer language processing hierarchy

    Implements bidirectional processing:
    - Bottom-up: acoustic → phonological → syntactic → semantic → pragmatic
    - Top-down: predictions flow downward to constrain processing
    """

    def __init__(
        self,
        input_dim: int = 64,
        phonological_dim: int = 32,
        syntactic_dim: int = 48,
        semantic_dim: int = 64,
        pragmatic_dim: int = 32,
        learning_rate: float = 0.1,
    ):
        self.input_dim = input_dim

        # Create layers
        self.phonological = PhonologicalLayer(
            input_dim=input_dim,
            hidden_dim=phonological_dim,
            output_dim=syntactic_dim,
            learning_rate=learning_rate,
        )

        self.syntactic = SyntacticLayer(
            input_dim=syntactic_dim,
            hidden_dim=syntactic_dim,
            output_dim=semantic_dim,
            learning_rate=learning_rate,
        )

        self.semantic = SemanticLayer(
            input_dim=semantic_dim,
            hidden_dim=semantic_dim,
            output_dim=pragmatic_dim,
            learning_rate=learning_rate,
        )

        self.pragmatic = PragmaticLayer(
            input_dim=pragmatic_dim,
            hidden_dim=pragmatic_dim,
            output_dim=pragmatic_dim,
            learning_rate=learning_rate,
        )

        self.layers = [self.phonological, self.syntactic, self.semantic, self.pragmatic]

        # Error tracking
        self.layer_errors: List[np.ndarray] = []
        self.total_error = 0.0

    def forward(self, acoustic_input: np.ndarray, dt: float = 0.1) -> Dict[str, np.ndarray]:
        """Bottom-up processing through hierarchy"""
        self.layer_errors = []

        # Phonological processing
        phon_out = self.phonological.extract_phonemes(acoustic_input)

        # Compute error from syntactic prediction
        syn_pred = self.syntactic.predict_input()
        phon_error = (
            phon_out - syn_pred[: len(phon_out)]
            if len(syn_pred) >= len(phon_out)
            else phon_out - np.pad(syn_pred, (0, len(phon_out) - len(syn_pred)))
        )
        self.phonological.receive_error(phon_error[: self.phonological.input_dim])
        self.layer_errors.append(phon_error)

        # Syntactic processing
        syn_out = self.syntactic.parse_structure(phon_out)

        # Compute error from semantic prediction
        sem_pred = self.semantic.predict_input()
        syn_error = (
            syn_out - sem_pred[: len(syn_out)]
            if len(sem_pred) >= len(syn_out)
            else syn_out - np.pad(sem_pred, (0, len(syn_out) - len(sem_pred)))
        )
        self.syntactic.receive_error(syn_error[: self.syntactic.input_dim])
        self.layer_errors.append(syn_error)

        # Semantic processing
        sem_out = self.semantic.extract_meaning(syn_out)

        # Compute error from pragmatic prediction
        prag_pred = self.pragmatic.predict_input()
        sem_error = (
            sem_out - prag_pred[: len(sem_out)]
            if len(prag_pred) >= len(sem_out)
            else sem_out - np.pad(prag_pred, (0, len(sem_out) - len(prag_pred)))
        )
        self.semantic.receive_error(sem_error[: self.semantic.input_dim])
        self.layer_errors.append(sem_error)

        # Pragmatic processing
        prag_out = self.pragmatic.interpret_context(sem_out)
        self.layer_errors.append(
            np.zeros_like(prag_out)
        )  # Top layer has no prediction error from above

        # Total error
        self.total_error = sum(np.sum(e**2) for e in self.layer_errors)

        return {
            "phonological": phon_out,
            "syntactic": syn_out,
            "semantic": sem_out,
            "pragmatic": prag_out,
            "errors": self.layer_errors,
            "total_error": self.total_error,
        }

    def backward(self) -> List[np.ndarray]:
        """Generate top-down predictions"""
        predictions = []
        for layer in reversed(self.layers):
            pred = layer.predict_input()
            predictions.insert(0, pred)
        return predictions

    def step(
        self, acoustic_input: np.ndarray, dt: float = 0.1, update_weights: bool = True
    ) -> Dict[str, np.ndarray]:
        """Complete processing cycle"""
        result = self.forward(acoustic_input, dt)

        if update_weights:
            # Update layers with proper inputs
            # Only update prediction weights based on actual prediction errors
            for i, layer in enumerate(self.layers):
                if i < len(self.layer_errors):
                    # Use a very small learning rate to ensure stability
                    layer.learning_rate = min(layer.learning_rate, 0.01)
                    layer.update_weights(layer.state, dt * 0.1)

        return result

    def get_timescales(self) -> List[float]:
        """Get timescales of all layers"""
        return [layer.timescale for layer in self.layers]

    def get_confidences(self) -> Dict[str, float]:
        """Get confidence at each layer"""
        return {layer.name: layer.get_confidence() for layer in self.layers}

    def reset(self) -> None:
        """Reset all layers"""
        for layer in self.layers:
            layer.reset()
        self.layer_errors = []
        self.total_error = 0.0
