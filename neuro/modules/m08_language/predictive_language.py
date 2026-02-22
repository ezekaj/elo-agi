"""
Predictive Language Processor

Integrates all language processing components into a unified
predictive coding framework.

Combines:
- Hierarchical processing (phonological → pragmatic)
- Distributed network (Broca's, Wernicke's, Arcuate)
- Grammar constraints (possible vs impossible)
- Recursive structure parsing
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .language_hierarchy import LanguageProcessingHierarchy
from .language_network import DistributedLanguageNetwork
from .grammar_manifold import GrammarConstraintManifold, UniversalGrammar
from .recursive_parser import RecursiveGrammar, ConstituentParser, Constituent


@dataclass
class ProcessingResult:
    """Result of language processing"""

    phonological: np.ndarray
    syntactic: np.ndarray
    semantic: np.ndarray
    pragmatic: np.ndarray
    network_state: Dict[str, np.ndarray]
    grammar_evaluation: Dict[str, float]
    parse_tree: Optional[Constituent]
    total_error: float
    broca_inhibition: float
    is_grammatical: bool


class PredictiveLanguageProcessor:
    """Unified predictive language processing system

    Implements language processing using predictive coding principles:
    - Top-down predictions constrain bottom-up processing
    - Prediction errors drive learning
    - Grammar constraints (UG) limit learnable patterns
    - Broca's inhibition signals impossible structures
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 48, learning_rate: float = 0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate

        # Core components
        self.hierarchy = LanguageProcessingHierarchy(
            input_dim=input_dim,
            phonological_dim=hidden_dim,
            syntactic_dim=hidden_dim,
            semantic_dim=hidden_dim,
            pragmatic_dim=hidden_dim // 2,
            learning_rate=learning_rate,
        )

        self.network = DistributedLanguageNetwork(dim=hidden_dim, learning_rate=learning_rate)

        self.grammar_manifold = GrammarConstraintManifold(dim=hidden_dim)
        self.universal_grammar = UniversalGrammar(dim=hidden_dim)

        # Parsing components
        self.grammar = RecursiveGrammar()
        self.parser = ConstituentParser(self.grammar)

        # Processing state
        self.current_state: Optional[ProcessingResult] = None
        self.processing_history: List[ProcessingResult] = []

        # Token embeddings (simple random initialization)
        self.embeddings: Dict[str, np.ndarray] = {}

    def _get_embedding(self, token: str) -> np.ndarray:
        """Get or create embedding for token"""
        if token not in self.embeddings:
            self.embeddings[token] = np.random.randn(self.input_dim) * 0.3
        return self.embeddings[token]

    def process_token(
        self, token: str, dt: float = 0.1, update_weights: bool = True
    ) -> ProcessingResult:
        """Process a single token through the full system"""
        # Get embedding
        embedding = self._get_embedding(token)

        # Hierarchical processing
        hier_result = self.hierarchy.step(embedding, dt, update_weights)

        # Network processing (Broca's + Wernicke's)
        network_result = self.network.process(embedding)

        # Grammar evaluation
        syntactic_state = hier_result["syntactic"]
        grammar_eval = self.universal_grammar.evaluate(syntactic_state)

        # Check grammar constraints
        violation = self.grammar_manifold.get_violation_score(syntactic_state)
        broca_inhibition = self.grammar_manifold.inhibition_signal(syntactic_state)

        # Determine grammaticality
        is_grammatical = violation < 0.5 and broca_inhibition < 0.3

        result = ProcessingResult(
            phonological=hier_result["phonological"],
            syntactic=hier_result["syntactic"],
            semantic=hier_result["semantic"],
            pragmatic=hier_result["pragmatic"],
            network_state=network_result,
            grammar_evaluation=grammar_eval,
            parse_tree=None,
            total_error=hier_result["total_error"],
            broca_inhibition=broca_inhibition,
            is_grammatical=is_grammatical,
        )

        self.current_state = result
        self.processing_history.append(result)
        if len(self.processing_history) > 100:
            self.processing_history.pop(0)

        return result

    def process_utterance(
        self, tokens: List[str], dt: float = 0.1, update_weights: bool = True
    ) -> Dict[str, any]:
        """Process a complete utterance (multiple tokens)"""
        results = []

        for token in tokens:
            result = self.process_token(token, dt, update_weights)
            results.append(result)

        # Parse the utterance
        try:
            parse_tree = self.parser.parse(tokens)
        except:
            parse_tree = None

        # Aggregate results
        return {
            "tokens": tokens,
            "token_results": results,
            "parse_tree": parse_tree,
            "final_state": {
                "phonological": results[-1].phonological if results else np.zeros(self.hidden_dim),
                "syntactic": results[-1].syntactic if results else np.zeros(self.hidden_dim),
                "semantic": results[-1].semantic if results else np.zeros(self.hidden_dim),
                "pragmatic": results[-1].pragmatic if results else np.zeros(self.hidden_dim // 2),
            },
            "total_error": sum(r.total_error for r in results),
            "mean_broca_inhibition": np.mean([r.broca_inhibition for r in results])
            if results
            else 0,
            "grammaticality": all(r.is_grammatical for r in results),
            "layer_timescales": self.hierarchy.get_timescales(),
        }

    def process_sequence(
        self, utterances: List[List[str]], dt: float = 0.1, update_weights: bool = True
    ) -> List[Dict[str, any]]:
        """Process a sequence of utterances"""
        return [self.process_utterance(utt, dt, update_weights) for utt in utterances]

    def test_grammar(self, grammar_params: np.ndarray) -> Dict[str, any]:
        """Test if grammar parameters represent a possible grammar

        Returns detailed analysis of grammar constraints.
        """
        # Evaluate against constraint manifold
        violation = self.grammar_manifold.get_violation_score(grammar_params)
        inhibition = self.grammar_manifold.inhibition_signal(grammar_params)
        is_possible = self.grammar_manifold.is_possible_grammar(grammar_params)

        # Evaluate against Universal Grammar
        ug_eval = self.universal_grammar.evaluate(grammar_params)

        # Project to nearest possible if impossible
        if not is_possible:
            nearest = self.grammar_manifold.project_to_possible(grammar_params)
        else:
            nearest = grammar_params

        return {
            "is_possible": is_possible,
            "violation_score": violation,
            "broca_inhibition": inhibition,
            "ug_compatibility": ug_eval,
            "nearest_possible": nearest,
            "distance_to_boundary": self.grammar_manifold.distance_to_boundary(grammar_params),
        }

    def lesion_experiment(
        self, region: str, test_utterances: List[List[str]], damage_level: float = 1.0
    ) -> Dict[str, any]:
        """Run lesion experiment

        Args:
            region: 'broca', 'wernicke', or 'arcuate'
            test_utterances: Utterances to test
            damage_level: 0.0 to 1.0

        Returns:
            Comparison of performance before and after lesion
        """
        # Baseline (intact)
        self.network.restore()
        baseline_results = self.process_sequence(test_utterances, update_weights=False)

        # Lesioned
        self.network.lesion(region, damage_level)
        lesioned_results = self.process_sequence(test_utterances, update_weights=False)

        # Restore
        self.network.restore()

        # Compare
        baseline_errors = [r["total_error"] for r in baseline_results]
        lesioned_errors = [r["total_error"] for r in lesioned_results]

        baseline_grammatical = sum(1 for r in baseline_results if r["grammaticality"])
        lesioned_grammatical = sum(1 for r in lesioned_results if r["grammaticality"])

        return {
            "region": region,
            "damage_level": damage_level,
            "baseline": {
                "mean_error": np.mean(baseline_errors),
                "grammatical_count": baseline_grammatical,
                "network_functional": True,
            },
            "lesioned": {
                "mean_error": np.mean(lesioned_errors),
                "grammatical_count": lesioned_grammatical,
                "network_functional": self.network.is_functional(),
            },
            "error_increase": np.mean(lesioned_errors) - np.mean(baseline_errors),
            "grammaticality_drop": baseline_grammatical - lesioned_grammatical,
            # Key finding: Broca's alone doesn't necessarily cause deficits
            "broca_alone_functional": region == "broca" and self.network.is_functional(),
        }

    def get_layer_activations(self) -> Dict[str, np.ndarray]:
        """Get current activations at each hierarchy level"""
        return {
            "phonological": self.hierarchy.phonological.state.copy(),
            "syntactic": self.hierarchy.syntactic.state.copy(),
            "semantic": self.hierarchy.semantic.state.copy(),
            "pragmatic": self.hierarchy.pragmatic.state.copy(),
        }

    def get_network_states(self) -> Dict[str, np.ndarray]:
        """Get current states of network regions"""
        return self.network.get_region_states()

    def get_confidences(self) -> Dict[str, float]:
        """Get confidence levels at each processing stage"""
        return self.hierarchy.get_confidences()

    def reset(self) -> None:
        """Reset all components"""
        self.hierarchy.reset()
        self.network.reset()
        self.grammar_manifold.reset()
        self.current_state = None
        self.processing_history = []


class LanguageAcquisitionSimulator:
    """Simulate language acquisition with grammar constraints

    Demonstrates how:
    - Possible grammars can be learned
    - Impossible grammars trigger Broca's inhibition
    """

    def __init__(self, processor: PredictiveLanguageProcessor):
        self.processor = processor
        self.learning_history: List[Dict[str, any]] = []

    def attempt_grammar_learning(
        self, grammar_samples: List[np.ndarray], n_epochs: int = 10
    ) -> Dict[str, any]:
        """Attempt to learn a grammar from samples

        Returns learning success/failure and inhibition patterns.
        """
        inhibitions = []
        errors = []
        learning_success = True

        for epoch in range(n_epochs):
            epoch_inhibitions = []
            epoch_errors = []

            for sample in grammar_samples:
                # Test grammar
                result = self.processor.test_grammar(sample)

                epoch_inhibitions.append(result["broca_inhibition"])
                epoch_errors.append(result["violation_score"])

                # High inhibition = learning blocked
                if result["broca_inhibition"] > 0.7:
                    learning_success = False

            inhibitions.append(np.mean(epoch_inhibitions))
            errors.append(np.mean(epoch_errors))

        result = {
            "success": learning_success,
            "final_inhibition": inhibitions[-1] if inhibitions else 0,
            "final_error": errors[-1] if errors else 0,
            "inhibition_history": inhibitions,
            "error_history": errors,
            "epochs": n_epochs,
        }

        self.learning_history.append(result)
        return result

    def compare_possible_vs_impossible(
        self, possible_samples: List[np.ndarray], impossible_samples: List[np.ndarray]
    ) -> Dict[str, any]:
        """Compare learning outcomes for possible vs impossible grammars"""
        possible_result = self.attempt_grammar_learning(possible_samples)
        impossible_result = self.attempt_grammar_learning(impossible_samples)

        return {
            "possible": {
                "success": possible_result["success"],
                "final_inhibition": possible_result["final_inhibition"],
            },
            "impossible": {
                "success": impossible_result["success"],
                "final_inhibition": impossible_result["final_inhibition"],
            },
            "selective_inhibition": (
                impossible_result["final_inhibition"] > possible_result["final_inhibition"]
            ),
            "key_finding": "Broca's shows selective inhibition for impossible grammars only",
        }
