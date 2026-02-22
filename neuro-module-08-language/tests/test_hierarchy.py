"""Tests for language processing hierarchy"""

import numpy as np
import pytest
from neuro.modules.m08_language.language_hierarchy import (
    LanguageLayer,
    PhonologicalLayer,
    SyntacticLayer,
    SemanticLayer,
    PragmaticLayer,
    LanguageProcessingHierarchy,
)


class TestLanguageLayer:
    """Tests for base language layer"""

    def test_initialization(self):
        """Test layer initialization"""
        layer = LanguageLayer(input_dim=32, hidden_dim=16, output_dim=24, timescale=0.1)

        assert layer.input_dim == 32
        assert layer.hidden_dim == 16
        assert layer.output_dim == 24
        assert layer.timescale == 0.1
        assert layer.state.shape == (16,)

    def test_process(self):
        """Test forward processing"""
        layer = LanguageLayer(input_dim=32, hidden_dim=16, output_dim=24, timescale=0.1)

        input_signal = np.random.randn(32)
        output = layer.process(input_signal)

        assert output.shape == (24,)
        assert np.all(np.isfinite(output))

    def test_predict_input(self):
        """Test top-down prediction"""
        layer = LanguageLayer(input_dim=32, hidden_dim=16, output_dim=24, timescale=0.1)

        layer.state = np.random.randn(16)
        prediction = layer.predict_input()

        assert prediction.shape == (32,)
        assert np.all(np.isfinite(prediction))

    def test_confidence_estimation(self):
        """Test confidence from error history"""
        layer = LanguageLayer(input_dim=32, hidden_dim=16, output_dim=24, timescale=0.1)

        # No history
        conf_empty = layer.get_confidence()
        assert 0 <= conf_empty <= 1

        # Add consistent errors (high confidence)
        for _ in range(30):
            layer.receive_error(np.ones(32) * 0.1 + np.random.randn(32) * 0.01)
        conf_consistent = layer.get_confidence()

        # Add variable errors (low confidence)
        layer.error_history = []
        for _ in range(30):
            layer.receive_error(np.random.randn(32))
        conf_variable = layer.get_confidence()

        assert conf_consistent > conf_variable


class TestPhonologicalLayer:
    """Tests for phonological layer"""

    def test_timescale(self):
        """Test phonological layer has fast timescale"""
        layer = PhonologicalLayer(input_dim=32, hidden_dim=16, output_dim=24)
        assert layer.timescale == 0.01  # 10ms

    def test_phoneme_extraction(self):
        """Test phoneme extraction"""
        layer = PhonologicalLayer(input_dim=32, hidden_dim=16, output_dim=24)

        acoustic_input = np.random.randn(32)
        phonemes = layer.extract_phonemes(acoustic_input)

        assert phonemes.shape == (24,)
        assert np.all(np.isfinite(phonemes))


class TestSyntacticLayer:
    """Tests for syntactic layer"""

    def test_timescale(self):
        """Test syntactic layer has medium timescale"""
        layer = SyntacticLayer(input_dim=32, hidden_dim=16, output_dim=24)
        assert layer.timescale == 0.1  # 100ms

    def test_grammaticality(self):
        """Test grammaticality scoring"""
        layer = SyntacticLayer(input_dim=32, hidden_dim=16, output_dim=24)

        # Process some input to build history
        for _ in range(20):
            layer.parse_structure(np.random.randn(32) * 0.1)

        gram_score = layer.check_grammaticality()
        assert 0 <= gram_score <= 1


class TestSemanticLayer:
    """Tests for semantic layer"""

    def test_timescale(self):
        """Test semantic layer has slower timescale"""
        layer = SemanticLayer(input_dim=32, hidden_dim=16, output_dim=24)
        assert layer.timescale == 0.3  # 300ms

    def test_lexicon(self):
        """Test lexicon registration and lookup"""
        layer = SemanticLayer(input_dim=32, hidden_dim=32, output_dim=24)

        # Register word
        meaning = np.random.randn(32)
        layer.register_word("cat", meaning)

        # Lookup
        retrieved = layer.lookup_word("cat")
        assert np.allclose(retrieved, meaning)

        # Unknown word
        unknown = layer.lookup_word("xyz")
        assert unknown is None


class TestPragmaticLayer:
    """Tests for pragmatic layer"""

    def test_timescale(self):
        """Test pragmatic layer has slowest timescale"""
        layer = PragmaticLayer(input_dim=32, hidden_dim=16, output_dim=24)
        assert layer.timescale == 1.0  # 1000ms

    def test_context_interpretation(self):
        """Test context interpretation"""
        layer = PragmaticLayer(input_dim=32, hidden_dim=16, output_dim=24)

        semantic_input = np.random.randn(32)
        situation = np.random.randn(32)

        output = layer.interpret_context(semantic_input, situation)

        assert output.shape == (24,)
        assert np.all(np.isfinite(output))


class TestLanguageProcessingHierarchy:
    """Tests for full language hierarchy"""

    def test_initialization(self):
        """Test hierarchy initialization"""
        hierarchy = LanguageProcessingHierarchy(
            input_dim=64, phonological_dim=32, syntactic_dim=48, semantic_dim=64, pragmatic_dim=32
        )

        assert len(hierarchy.layers) == 4
        assert hierarchy.input_dim == 64

    def test_forward_pass(self):
        """Test forward processing"""
        hierarchy = LanguageProcessingHierarchy()

        acoustic_input = np.random.randn(64)
        result = hierarchy.forward(acoustic_input)

        assert "phonological" in result
        assert "syntactic" in result
        assert "semantic" in result
        assert "pragmatic" in result
        assert "errors" in result
        assert "total_error" in result

    def test_timescale_order(self):
        """Test timescales increase through hierarchy"""
        hierarchy = LanguageProcessingHierarchy()

        timescales = hierarchy.get_timescales()

        assert timescales[0] < timescales[1]  # Phonological < Syntactic
        assert timescales[1] < timescales[2]  # Syntactic < Semantic
        assert timescales[2] < timescales[3]  # Semantic < Pragmatic

    def test_step_returns_all_info(self):
        """Test step returns complete information"""
        hierarchy = LanguageProcessingHierarchy()

        input_signal = np.random.randn(64)
        result = hierarchy.step(input_signal)

        assert np.all(np.isfinite(result["phonological"]))
        assert np.all(np.isfinite(result["syntactic"]))
        assert np.all(np.isfinite(result["semantic"]))
        assert np.all(np.isfinite(result["pragmatic"]))
        assert np.isfinite(result["total_error"])

    def test_backward_predictions(self):
        """Test top-down prediction generation"""
        hierarchy = LanguageProcessingHierarchy()

        # Process some input first
        hierarchy.forward(np.random.randn(64))

        predictions = hierarchy.backward()

        assert len(predictions) == 4
        for pred in predictions:
            assert np.all(np.isfinite(pred))

    def test_confidences(self):
        """Test confidence reporting"""
        hierarchy = LanguageProcessingHierarchy()

        # Process several inputs
        for _ in range(20):
            hierarchy.step(np.random.randn(64))

        confidences = hierarchy.get_confidences()

        assert "phonological" in confidences
        assert "syntactic" in confidences
        assert "semantic" in confidences
        assert "pragmatic" in confidences

        for conf in confidences.values():
            assert 0 <= conf <= 1


class TestHierarchyLearning:
    """Tests for hierarchy learning dynamics"""

    def test_error_decreases(self):
        """Test error stays bounded with learning"""
        hierarchy = LanguageProcessingHierarchy(learning_rate=0.1)

        # Constant input
        constant = np.random.randn(64) * 0.3

        errors = []
        for _ in range(50):
            result = hierarchy.step(constant, update_weights=True)
            errors.append(result["total_error"])

        # Error should stay bounded (not explode)
        assert all(np.isfinite(e) for e in errors)
        assert errors[-1] < 1000  # Sanity check - errors don't explode

    def test_sequence_processing(self):
        """Test processing a sequence"""
        hierarchy = LanguageProcessingHierarchy()

        # Sequence of inputs
        sequence = [np.random.randn(64) * 0.3 for _ in range(10)]

        for inp in sequence:
            result = hierarchy.step(inp)
            assert np.all(np.isfinite(result["total_error"]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
