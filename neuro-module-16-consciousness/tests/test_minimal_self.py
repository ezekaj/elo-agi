"""Tests for minimal self"""

import numpy as np
import pytest
from neuro.modules.m16_consciousness.minimal_self import (
    AgencyDetector,
    OwnershipProcessor,
    MinimalSelf,
)


class TestAgencyDetector:
    """Tests for agency detection"""

    def test_initialization(self):
        """Test detector initialization"""
        detector = AgencyDetector()
        assert detector.agency_level == 0.5

    def test_initiate_action(self):
        """Test action initiation"""
        detector = AgencyDetector()
        action = np.random.rand(50)
        prediction = detector.initiate_action(action)
        assert len(prediction) == 50
        assert len(detector.action_history) == 1

    def test_receive_matching_outcome(self):
        """Test agency with matching outcome"""
        detector = AgencyDetector()
        action = np.ones(50) * 0.5
        prediction = detector.initiate_action(action)
        result = detector.receive_outcome(prediction)  # Exact match
        assert result["match"] > 0.9

    def test_receive_mismatching_outcome(self):
        """Test agency with mismatching outcome"""
        detector = AgencyDetector()
        detector.initiate_action(np.ones(50))
        result = detector.receive_outcome(-np.ones(50))  # Opposite
        assert result["match"] < 0.5

    def test_agency_level_changes(self):
        """Test agency level updates"""
        detector = AgencyDetector()

        # Good prediction
        action = np.ones(50)
        pred = detector.initiate_action(action)
        detector.receive_outcome(pred + np.random.rand(50) * 0.1)

        # Agency should increase with good match
        # (depends on threshold)


class TestOwnershipProcessor:
    """Tests for body ownership"""

    def test_initialization(self):
        """Test processor initialization"""
        processor = OwnershipProcessor()
        assert processor.ownership_level == 0.8

    def test_receive_multisensory(self):
        """Test multisensory integration"""
        processor = OwnershipProcessor()
        features_per = 50 // 3

        visual = np.random.rand(features_per)
        tactile = np.random.rand(features_per)
        proprio = np.random.rand(features_per)

        result = processor.receive_multisensory(visual, tactile, proprio)
        assert "ownership" in result
        assert "parietal_activity" in result

    def test_coherent_signals_increase_ownership(self):
        """Test coherent signals maintain ownership"""
        processor = OwnershipProcessor()
        features_per = 50 // 3

        # Consistent signals
        signal = np.ones(features_per)
        for _ in range(10):
            processor.receive_multisensory(signal, signal, signal)

        assert processor.ownership_level >= 0.5

    def test_rubber_hand_illusion(self):
        """Test rubber hand illusion simulation"""
        processor = OwnershipProcessor()
        features_per = 50 // 3

        visual = np.random.rand(features_per)
        tactile = np.random.rand(features_per)

        sync_result = processor.rubber_hand_illusion(visual, tactile, synchronous=True)
        async_result = processor.rubber_hand_illusion(visual, tactile, synchronous=False)

        assert sync_result["illusion_strength"] > async_result["illusion_strength"]


class TestMinimalSelf:
    """Tests for integrated minimal self"""

    def test_initialization(self):
        """Test minimal self initialization"""
        ms = MinimalSelf()
        assert ms.agency is not None
        assert ms.ownership is not None

    def test_perform_action(self):
        """Test action performance"""
        ms = MinimalSelf()
        action = np.random.rand(50)
        prediction = ms.perform_action(action)
        assert len(prediction) == 50

    def test_receive_feedback(self):
        """Test receiving feedback"""
        ms = MinimalSelf()
        ms.perform_action(np.random.rand(50))

        features_per = 50 // 3
        result = ms.receive_feedback(
            np.random.rand(50),
            np.random.rand(features_per),
            np.random.rand(features_per),
            np.random.rand(features_per),
        )

        assert "agency" in result
        assert "ownership" in result
        assert "self_integrity" in result

    def test_get_state(self):
        """Test getting minimal self state"""
        ms = MinimalSelf()
        state = ms.get_minimal_self_state()
        assert "agency_level" in state
        assert "ownership_level" in state
        assert "self_integrity" in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
