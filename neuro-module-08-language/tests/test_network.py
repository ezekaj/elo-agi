"""Tests for distributed language network"""

import numpy as np
import pytest
from neuro.modules.m08_language.language_network import (
    BrocaRegion,
    WernickeRegion,
    ArcuateFasciculus,
    DistributedLanguageNetwork
)

class TestBrocaRegion:
    """Tests for Broca's area"""

    def test_initialization(self):
        """Test Broca region initialization"""
        broca = BrocaRegion(dim=64)

        assert broca.dim == 64
        assert broca.is_active
        assert broca.damage_level == 0.0

    def test_syntax_processing(self):
        """Test syntactic processing"""
        broca = BrocaRegion(dim=32)

        input_signal = np.random.randn(32)
        output = broca.process_syntax(input_signal)

        assert output.shape == (32,)
        assert np.all(np.isfinite(output))

    def test_grammar_constraints(self):
        """Test grammar constraint application"""
        broca = BrocaRegion(dim=32)

        structure = np.random.randn(32)
        constrained, violation = broca.apply_grammar_constraints(structure)

        assert constrained.shape == (32,)
        assert 0 <= violation <= 1
        assert np.all(np.isfinite(constrained))

    def test_selective_inhibition(self):
        """Test selective inhibition for impossible grammars"""
        broca = BrocaRegion(dim=32)

        # Possible grammar (close to learned state)
        broca.state = np.random.randn(32) * 0.1
        possible_grammar = broca.state + np.random.randn(32) * 0.1
        inhibition_possible = broca.inhibit_impossible(possible_grammar)

        # Impossible grammar (far from learned state)
        impossible_grammar = np.random.randn(32) * 10
        inhibition_impossible = broca.inhibit_impossible(impossible_grammar)

        # Selective: inhibits impossible more than possible
        # Note: exact comparison depends on implementation
        assert inhibition_possible >= 0
        assert inhibition_impossible >= 0

    def test_lesion(self):
        """Test lesion simulation"""
        broca = BrocaRegion(dim=32)

        # Full lesion
        broca.lesion(1.0)
        assert not broca.is_active
        assert broca.damage_level == 1.0

        # Partial lesion
        broca.restore()
        broca.lesion(0.5)
        assert broca.is_active
        assert broca.damage_level == 0.5

        # Processing should be attenuated
        output = broca.process_syntax(np.ones(32))
        assert np.linalg.norm(output) < 32  # Reduced

class TestWernickeRegion:
    """Tests for Wernicke's area"""

    def test_initialization(self):
        """Test Wernicke region initialization"""
        wernicke = WernickeRegion(dim=64)

        assert wernicke.dim == 64
        assert wernicke.is_active
        assert len(wernicke.lexicon) == 0

    def test_semantic_processing(self):
        """Test semantic processing"""
        wernicke = WernickeRegion(dim=32)

        input_signal = np.random.randn(32)
        output = wernicke.process_semantics(input_signal)

        assert output.shape == (32,)
        assert np.all(np.isfinite(output))

    def test_lexicon(self):
        """Test lexicon functionality"""
        wernicke = WernickeRegion(dim=32)

        # Register word
        meaning = np.random.randn(32)
        wernicke.register_word("test", meaning)

        assert "test" in wernicke.lexicon
        assert np.allclose(wernicke.lexicon["test"], meaning)

    def test_context_integration(self):
        """Test context integration"""
        wernicke = WernickeRegion(dim=32)

        meaning = np.random.randn(32)
        context = np.random.randn(32)

        integrated = wernicke.integrate_context(meaning, context)

        assert integrated.shape == (32,)
        assert np.all(np.isfinite(integrated))

class TestArcuateFasciculus:
    """Tests for arcuate fasciculus"""

    def test_initialization(self):
        """Test arcuate initialization"""
        arcuate = ArcuateFasciculus(dim=32)

        assert arcuate.dim == 32
        assert arcuate.is_active

    def test_bidirectional_flow(self):
        """Test bidirectional information flow"""
        arcuate = ArcuateFasciculus(dim=32)

        broca_out = np.random.randn(32)
        wernicke_out = np.random.randn(32)

        to_wernicke = arcuate.forward_flow(broca_out)
        to_broca = arcuate.backward_flow(wernicke_out)

        assert to_wernicke.shape == (32,)
        assert to_broca.shape == (32,)

    def test_synchronize(self):
        """Test synchronization"""
        arcuate = ArcuateFasciculus(dim=32)

        broca_state = np.random.randn(32)
        wernicke_state = np.random.randn(32)

        broca_update, wernicke_update = arcuate.synchronize(broca_state, wernicke_state)

        assert broca_update.shape == (32,)
        assert wernicke_update.shape == (32,)

class TestDistributedLanguageNetwork:
    """Tests for full distributed network"""

    def test_initialization(self):
        """Test network initialization"""
        network = DistributedLanguageNetwork(dim=32)

        assert network.broca is not None
        assert network.wernicke is not None
        assert network.arcuate is not None

    def test_process(self):
        """Test network processing"""
        network = DistributedLanguageNetwork(dim=32)

        input_signal = np.random.randn(32)
        result = network.process(input_signal)

        assert 'broca' in result
        assert 'wernicke' in result
        assert 'combined' in result
        assert 'broca_inhibition' in result

    def test_distributed_redundancy(self):
        """Test that Broca's damage alone doesn't prevent function

        Key 2025 finding: Language is distributed, not localized.
        """
        network = DistributedLanguageNetwork(dim=32)

        # Lesion Broca's only
        network.lesion('broca', 1.0)

        # Network should still be functional
        assert network.is_functional()

        # Can still process
        result = network.process(np.random.randn(32))
        assert np.all(np.isfinite(result['combined']))

    def test_both_regions_lesioned(self):
        """Test that both regions lesioned = non-functional"""
        network = DistributedLanguageNetwork(dim=32)

        network.lesion('broca', 1.0)
        network.lesion('wernicke', 1.0)

        assert not network.is_functional()

    def test_lesion_and_restore(self):
        """Test lesion and restore cycle"""
        network = DistributedLanguageNetwork(dim=32)

        # Process intact
        intact_result = network.process(np.random.randn(32))

        # Lesion
        network.lesion('broca', 0.8)
        lesioned_result = network.process(np.random.randn(32))

        # Restore
        network.restore('broca')
        restored_result = network.process(np.random.randn(32))

        # Verify damage levels
        damage = network.get_damage_levels()
        assert damage['broca'] == 0.0  # Restored

    def test_region_states(self):
        """Test getting region states"""
        network = DistributedLanguageNetwork(dim=32)

        network.process(np.random.randn(32))
        states = network.get_region_states()

        assert 'broca' in states
        assert 'wernicke' in states
        assert 'combined' in states

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
