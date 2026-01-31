"""Integration tests for full language processing system"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.predictive_language import (
    PredictiveLanguageProcessor,
    LanguageAcquisitionSimulator
)
from src.recursive_parser import (
    RecursiveGrammar,
    ConstituentParser,
    LinearPredictor,
    Constituent,
    compare_human_vs_llm
)
from src.grammar_manifold import ImpossibleGrammarGenerator


class TestPredictiveLanguageProcessor:
    """Tests for integrated language processor"""

    def test_initialization(self):
        """Test processor initialization"""
        processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)

        assert processor.hierarchy is not None
        assert processor.network is not None
        assert processor.grammar_manifold is not None

    def test_process_token(self):
        """Test single token processing"""
        processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)

        result = processor.process_token("hello")

        assert result.phonological.shape[0] > 0
        assert result.syntactic.shape[0] > 0
        assert result.semantic.shape[0] > 0
        assert np.isfinite(result.total_error)
        assert 0 <= result.broca_inhibition <= 1

    def test_process_utterance(self):
        """Test utterance processing"""
        processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)

        tokens = ["the", "cat", "sat"]
        result = processor.process_utterance(tokens)

        assert result['tokens'] == tokens
        assert len(result['token_results']) == 3
        assert 'final_state' in result
        assert 'total_error' in result

    def test_grammar_testing(self):
        """Test grammar evaluation"""
        processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)

        grammar_params = np.random.randn(32)
        result = processor.test_grammar(grammar_params)

        assert 'is_possible' in result
        assert 'violation_score' in result
        assert 'broca_inhibition' in result
        assert 'ug_compatibility' in result

    def test_lesion_experiment(self):
        """Test lesion experiment"""
        processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)

        test_utterances = [
            ["the", "cat"],
            ["a", "dog", "runs"]
        ]

        result = processor.lesion_experiment('broca', test_utterances, damage_level=1.0)

        assert result['region'] == 'broca'
        assert 'baseline' in result
        assert 'lesioned' in result
        # Key finding: Broca's alone should still be functional
        assert result['broca_alone_functional']

    def test_layer_activations(self):
        """Test getting layer activations"""
        processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)

        processor.process_token("test")
        activations = processor.get_layer_activations()

        assert 'phonological' in activations
        assert 'syntactic' in activations
        assert 'semantic' in activations
        assert 'pragmatic' in activations


class TestRecursiveParser:
    """Tests for recursive parser"""

    def test_grammar_generation(self):
        """Test recursive grammar generation"""
        grammar = RecursiveGrammar()

        tree = grammar.generate('S', max_depth=3)

        assert isinstance(tree, Constituent)
        assert tree.label == 'S'

    def test_constituent_structure(self):
        """Test constituent tree structure"""
        grammar = RecursiveGrammar()
        tree = grammar.generate('S', max_depth=4)

        # Has depth
        assert tree.depth() >= 1

        # Has size
        assert tree.size() >= 1

        # Can get terminals
        terminals = tree.get_terminals()
        assert len(terminals) >= 1

    def test_parser(self):
        """Test constituent parser"""
        grammar = RecursiveGrammar()
        parser = ConstituentParser(grammar)

        # Parse known structure
        tokens = ["the", "cat", "saw", "the", "dog"]
        tree = parser.parse(tokens)

        assert tree is not None
        assert isinstance(tree, Constituent)

    def test_linear_predictor(self):
        """Test linear predictor (LLM-style)"""
        predictor = LinearPredictor(vocab_size=100, context_size=5)

        # Predict next
        context = ["the", "cat", "sat"]
        predicted, probs = predictor.predict_next(context)

        assert isinstance(predicted, str)
        assert probs.shape[0] == 100
        assert np.allclose(probs.sum(), 1.0, atol=0.01)

        # No hierarchical structure
        assert not predictor.has_hierarchical_structure()

    def test_human_vs_llm_comparison(self):
        """Test human vs LLM comparison"""
        result = compare_human_vs_llm("the cat sat on the mat")

        assert result['human_has_hierarchy']
        assert not result['llm_has_hierarchy']
        assert result['human_depth'] >= 1


class TestLanguageAcquisition:
    """Tests for language acquisition simulation"""

    def test_acquisition_simulator(self):
        """Test acquisition simulator"""
        processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)
        simulator = LanguageAcquisitionSimulator(processor)

        samples = [np.random.randn(32) * 0.3 for _ in range(5)]
        result = simulator.attempt_grammar_learning(samples, n_epochs=3)

        assert 'success' in result
        assert 'final_inhibition' in result
        assert 'inhibition_history' in result

    def test_possible_vs_impossible(self):
        """Test selective inhibition in acquisition"""
        processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)
        simulator = LanguageAcquisitionSimulator(processor)
        gen = ImpossibleGrammarGenerator(dim=32)

        possible = [gen.generate_possible() for _ in range(5)]
        impossible = [gen.generate_impossible() for _ in range(5)]

        result = simulator.compare_possible_vs_impossible(possible, impossible)

        assert 'possible' in result
        assert 'impossible' in result
        assert 'selective_inhibition' in result


class TestFullPipeline:
    """End-to-end integration tests"""

    def test_full_sentence_processing(self):
        """Test processing a full sentence"""
        processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)

        sentence = ["the", "big", "cat", "saw", "a", "small", "dog"]
        result = processor.process_utterance(sentence, update_weights=True)

        # Should complete without error
        assert result['tokens'] == sentence
        assert np.isfinite(result['total_error'])

        # Timescales should be ordered
        timescales = result['layer_timescales']
        assert timescales[0] < timescales[1] < timescales[2] < timescales[3]

    def test_multiple_sentences(self):
        """Test processing multiple sentences"""
        processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)

        sentences = [
            ["the", "cat", "ran"],
            ["a", "dog", "barked"],
            ["the", "bird", "flew"]
        ]

        results = processor.process_sequence(sentences)

        assert len(results) == 3
        for result in results:
            assert np.isfinite(result['total_error'])

    def test_lesion_comparison(self):
        """Test comparing different lesion conditions"""
        processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)

        test_sentences = [
            ["the", "cat"],
            ["a", "dog"]
        ]

        # Broca lesion
        broca_result = processor.lesion_experiment('broca', test_sentences)

        # Wernicke lesion
        wernicke_result = processor.lesion_experiment('wernicke', test_sentences)

        # Both should have results
        assert broca_result['region'] == 'broca'
        assert wernicke_result['region'] == 'wernicke'

    def test_reset(self):
        """Test system reset"""
        processor = PredictiveLanguageProcessor(input_dim=64, hidden_dim=32)

        # Process some input
        processor.process_utterance(["test", "sentence"])

        # Reset
        processor.reset()

        # State should be cleared
        assert processor.current_state is None
        assert len(processor.processing_history) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
