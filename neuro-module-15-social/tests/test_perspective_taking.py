"""Tests for perspective taking"""

import numpy as np
import pytest
from neuro.modules.m15_social.perspective_taking import (
    SelfOtherDistinction,
    TPJNetwork,
    PerspectiveTaking,
)


class TestSelfOtherDistinction:
    """Tests for self-other distinction"""

    def test_initialization(self):
        """Test distinction initialization"""
        sod = SelfOtherDistinction()
        assert sod.boundary_strength > 0

    def test_update_self(self):
        """Test updating self representation"""
        sod = SelfOtherDistinction()
        initial = sod.self_representation.copy()
        sod.update_self(np.ones(50))
        assert not np.allclose(sod.self_representation, initial)

    def test_model_other(self):
        """Test modeling another agent"""
        sod = SelfOtherDistinction()
        sod.model_other("other1", np.random.rand(50))
        assert "other1" in sod.other_representations

    def test_distinguish(self):
        """Test distinction calculation"""
        sod = SelfOtherDistinction()
        sod.self_representation = np.ones(50)
        sod.model_other("other1", -np.ones(50))  # Opposite
        distinction = sod.distinguish("other1")
        assert distinction > 0.5

    def test_self_influence(self):
        """Test self influence on other"""
        sod = SelfOtherDistinction()
        sod.self_representation = np.ones(50)
        sod.model_other("similar", np.ones(50) * 0.9)
        influence = sod.get_self_influence("similar")
        assert influence > 0


class TestTPJNetwork:
    """Tests for TPJ network"""

    def test_initialization(self):
        """Test TPJ initialization"""
        tpj = TPJNetwork()
        assert tpj.current_perspective == "self"
        assert "self" in tpj.perspectives

    def test_add_perspective(self):
        """Test adding perspective"""
        tpj = TPJNetwork()
        tpj.add_perspective("other1", np.random.rand(50))
        assert "other1" in tpj.perspectives

    def test_switch_perspective(self):
        """Test switching perspective"""
        tpj = TPJNetwork()
        tpj.add_perspective("other1", np.random.rand(50))
        result = tpj.switch_perspective("other1")
        assert result["success"]
        assert tpj.current_perspective == "other1"

    def test_switch_cost(self):
        """Test switching has cost"""
        tpj = TPJNetwork()
        tpj.add_perspective("other1", np.random.rand(50))
        result = tpj.switch_perspective("other1")
        assert result["switch_cost"] > 0

    def test_view_from_perspective(self):
        """Test viewing from perspective"""
        tpj = TPJNetwork()
        tpj.add_perspective("other1", np.ones(50))
        tpj.switch_perspective("other1")
        viewed = tpj.view_from_perspective(np.zeros(50))
        assert np.sum(np.abs(viewed)) > 0  # Modified by perspective


class TestPerspectiveTaking:
    """Tests for integrated perspective taking"""

    def test_initialization(self):
        """Test PT initialization"""
        pt = PerspectiveTaking()
        assert pt.self_other is not None
        assert pt.tpj is not None

    def test_visual_perspective_level1(self):
        """Test level 1 visual perspective"""
        pt = PerspectiveTaking()
        pt.update_model("other1", np.random.rand(50))
        result = pt.take_visual_perspective("other1", np.random.rand(50), level=1)
        assert "their_view" in result
        assert result["level"] == 1

    def test_visual_perspective_level2(self):
        """Test level 2 visual perspective"""
        pt = PerspectiveTaking()
        pt.update_model("other1", np.random.rand(50))
        result = pt.take_visual_perspective("other1", np.random.rand(50), level=2)
        assert result["level"] == 2

    def test_affective_perspective(self):
        """Test affective perspective taking"""
        pt = PerspectiveTaking()
        pt.update_model("other1", np.random.rand(50))
        result = pt.take_affective_perspective("other1", np.random.rand(50))
        assert "inferred_feeling" in result
        assert "egocentric_bias" in result

    def test_return_to_self(self):
        """Test returning to self perspective"""
        pt = PerspectiveTaking()
        pt.update_model("other1", np.random.rand(50))
        pt.tpj.switch_perspective("other1")
        pt.return_to_self()
        assert pt.tpj.get_current_perspective() == "self"

    def test_update_model(self):
        """Test updating agent model"""
        pt = PerspectiveTaking()
        pt.update_model("other1", np.random.rand(50))
        assert "other1" in pt.tpj.perspectives
        assert "other1" in pt.self_other.other_representations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
