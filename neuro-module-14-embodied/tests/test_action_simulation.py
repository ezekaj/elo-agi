"""Tests for action simulation"""

import numpy as np
import pytest
from neuro.modules.m14_embodied.action_simulation import (
    MotorSimulator,
    MirrorSystem,
    ActionUnderstanding,
    SimulationParams,
)


class TestMotorSimulator:
    """Tests for motor simulation"""

    def test_initialization(self):
        """Test simulator initialization"""
        simulator = MotorSimulator()

        assert not simulator.simulating
        assert simulator.inhibition_level == 0.0

    def test_learn_action(self):
        """Test learning an action"""
        simulator = MotorSimulator()

        pattern = np.random.rand(50)
        simulator.learn_action("reach", pattern)

        assert "reach" in simulator.action_representations

    def test_simulate_action(self):
        """Test simulating an action"""
        simulator = MotorSimulator()
        simulator.learn_action("reach", np.random.rand(50))

        result = simulator.simulate_action("reach")

        assert result["success"]
        assert simulator.simulating
        assert simulator.inhibition_level > 0  # Inhibited during simulation

    def test_simulation_reduces_activation(self):
        """Test simulation has reduced motor activation"""
        params = SimulationParams(simulation_strength=0.5)
        simulator = MotorSimulator(params)

        pattern = np.ones(50)
        simulator.learn_action("reach", pattern)

        result = simulator.simulate_action("reach")

        # Activation should be scaled down
        assert np.mean(result["activation"]) < np.mean(pattern)

    def test_predict_outcome(self):
        """Test predicting action outcome"""
        simulator = MotorSimulator()
        simulator.learn_action("reach", np.random.rand(50))

        outcome = simulator.predict_outcome("reach")

        assert len(outcome) == 50

    def test_stop_simulation(self):
        """Test stopping simulation"""
        simulator = MotorSimulator()
        simulator.learn_action("reach", np.random.rand(50))
        simulator.simulate_action("reach")

        simulator.stop_simulation()

        assert not simulator.simulating
        assert simulator.inhibition_level == 0.0


class TestMirrorSystem:
    """Tests for mirror neuron system"""

    def test_initialization(self):
        """Test mirror system initialization"""
        mirror = MirrorSystem()

        assert mirror.mode == "idle"
        assert len(mirror.action_mirrors) == 0

    def test_learn_action_mirror(self):
        """Test learning action mirror"""
        mirror = MirrorSystem()

        pattern = np.random.rand(50)
        mirror.learn_action_mirror("grasp", pattern)

        assert "grasp" in mirror.action_mirrors

    def test_observe_action(self):
        """Test observing an action"""
        mirror = MirrorSystem()
        pattern = np.random.rand(50)
        mirror.learn_action_mirror("grasp", pattern)

        result = mirror.observe_action(pattern)  # Same pattern

        assert result["recognized_action"] == "grasp"
        assert result["confidence"] > 0.5
        assert mirror.mode == "observing"

    def test_observe_unknown_action(self):
        """Test observing unknown action"""
        mirror = MirrorSystem()

        result = mirror.observe_action(np.random.rand(50))

        # Low confidence for unknown
        assert result["confidence"] < 0.5 or result["recognized_action"] is None

    def test_execute_action(self):
        """Test executing an action"""
        mirror = MirrorSystem()
        mirror.learn_action_mirror("grasp", np.random.rand(50))

        result = mirror.execute_action("grasp")

        assert result["success"]
        assert mirror.mode == "executing"

    def test_resonance(self):
        """Test motor resonance"""
        mirror = MirrorSystem()
        pattern = np.random.rand(50)
        mirror.learn_action_mirror("grasp", pattern)
        mirror.observe_action(pattern)

        resonance = mirror.get_resonance()

        assert resonance > 0

    def test_decay(self):
        """Test activation decay"""
        mirror = MirrorSystem()
        pattern = np.random.rand(50)
        mirror.learn_action_mirror("grasp", pattern)
        mirror.observe_action(pattern)

        initial = mirror.get_resonance()
        mirror.decay(dt=10.0)

        assert mirror.get_resonance() < initial


class TestActionUnderstanding:
    """Tests for action understanding"""

    def test_initialization(self):
        """Test understanding initialization"""
        understanding = ActionUnderstanding()

        assert understanding.simulator is not None
        assert understanding.mirror_system is not None

    def test_learn_action(self):
        """Test learning action"""
        understanding = ActionUnderstanding()

        pattern = np.random.rand(50)
        understanding.learn_action("wave", pattern)

        assert "wave" in understanding.simulator.action_representations
        assert "wave" in understanding.mirror_system.action_mirrors

    def test_understand_observed_action(self):
        """Test understanding observed action"""
        understanding = ActionUnderstanding()
        pattern = np.random.rand(50)
        understanding.learn_action("wave", pattern)

        result = understanding.understand_observed_action(pattern)

        assert "action" in result
        assert "confidence" in result
        assert "motor_resonance" in result

    def test_predict_intention(self):
        """Test intention prediction from sequence"""
        understanding = ActionUnderstanding()
        pattern = np.random.rand(50)
        understanding.learn_action("wave", pattern)

        # Sequence of the same action
        sequence = [pattern] * 3
        intention = understanding.predict_intention(sequence)

        assert intention == "wave"

    def test_update(self):
        """Test system update"""
        understanding = ActionUnderstanding()
        pattern = np.random.rand(50)
        understanding.learn_action("wave", pattern)
        understanding.understand_observed_action(pattern)

        initial_conf = understanding.understanding_confidence
        understanding.update(dt=10.0)

        assert understanding.understanding_confidence <= initial_conf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
