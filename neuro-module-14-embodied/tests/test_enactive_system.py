"""Tests for enactive cognitive system"""

import numpy as np
import pytest
from neuro.modules.m14_embodied.enactive_system import (
    AutopoieticSystem,
    SensoriMotorEnaction,
    EnactiveCognitiveSystem,
    EnactiveParams,
)


class TestAutopoieticSystem:
    """Tests for autopoietic system"""

    def test_initialization(self):
        """Test system initialization"""
        system = AutopoieticSystem()

        assert system.energy == 1.0
        assert system.boundary_integrity == 1.0
        assert system.is_viable()

    def test_maintain_organization(self):
        """Test self-maintenance"""
        system = AutopoieticSystem()

        result = system.maintain_organization(dt=1.0)

        assert "energy" in result
        assert "boundary_integrity" in result

    def test_energy_consumption(self):
        """Test energy is consumed"""
        system = AutopoieticSystem()

        for _ in range(100):
            system.maintain_organization(dt=1.0)

        assert system.energy < 1.0

    def test_environmental_interaction(self):
        """Test interaction with environment"""
        system = AutopoieticSystem()

        env_input = np.random.rand(system.params.n_features)
        result = system.interact_with_environment(env_input)

        assert "perturbation" in result
        assert "boundary_stress" in result

    def test_replenish_energy(self):
        """Test energy replenishment"""
        system = AutopoieticSystem()
        system.energy = 0.5

        system.replenish_energy(0.3)

        assert system.energy == 0.8

    def test_viability_check(self):
        """Test viability checking"""
        system = AutopoieticSystem()
        system.energy = 0.05  # Very low

        assert not system.is_viable()


class TestSensoriMotorEnaction:
    """Tests for sensorimotor enaction"""

    def test_initialization(self):
        """Test enaction initialization"""
        enaction = SensoriMotorEnaction()

        assert len(enaction.contingencies) == 0
        assert len(enaction.interaction_history) == 0

    def test_learn_contingency(self):
        """Test learning sensorimotor contingency"""
        enaction = SensoriMotorEnaction()

        motor = np.random.rand(50)
        sensory = np.random.rand(50)
        enaction.learn_contingency("grasp", motor, sensory)

        assert "grasp" in enaction.contingencies

    def test_enact_contingency(self):
        """Test enacting a contingency"""
        enaction = SensoriMotorEnaction()
        enaction.learn_contingency("grasp", np.random.rand(50), np.random.rand(50))

        result = enaction.enact("grasp")

        assert result["success"]
        assert "motor_state" in result
        assert "sensory_state" in result

    def test_perceive_through_action(self):
        """Test perception through appropriate action"""
        enaction = SensoriMotorEnaction()

        sensory_consequence = np.random.rand(50)
        enaction.learn_contingency("look", np.random.rand(50), sensory_consequence)

        result = enaction.perceive_through_action(sensory_consequence)

        assert result["success"]
        assert result["action"] == "look"

    def test_interaction_history(self):
        """Test interaction history is recorded"""
        enaction = SensoriMotorEnaction()
        enaction.learn_contingency("grasp", np.random.rand(50), np.random.rand(50))

        enaction.enact("grasp")
        enaction.enact("grasp")

        assert len(enaction.interaction_history) == 2


class TestEnactiveCognitiveSystem:
    """Tests for full enactive system"""

    def test_initialization(self):
        """Test system initialization"""
        system = EnactiveCognitiveSystem()

        assert system.sensorimotor_loop is not None
        assert system.concept_grounding is not None
        assert system.action_understanding is not None
        assert system.autopoietic is not None
        assert system.viability

    def test_sense_act_loop(self):
        """Test sense-act cycle"""
        system = EnactiveCognitiveSystem()

        result = system.sense_act_loop()

        assert result["viable"]
        assert "sensorimotor" in result
        assert "energy" in result

    def test_sense_act_with_input(self):
        """Test sense-act with external input"""
        system = EnactiveCognitiveSystem()

        external = np.random.rand(system.params.n_features)
        result = system.sense_act_loop(external)

        assert result["viable"]

    def test_learn_action(self):
        """Test learning action"""
        system = EnactiveCognitiveSystem()

        motor = np.random.rand(system.params.n_features)
        sensory = np.random.rand(system.params.n_features)
        system.learn_action("wave", motor, sensory)

        # Check it's in both subsystems
        assert "wave" in system.action_understanding.simulator.action_representations

    def test_understand_action(self):
        """Test action understanding"""
        system = EnactiveCognitiveSystem()

        motor = np.random.rand(system.params.n_features)
        system.learn_action("wave", motor, np.random.rand(system.params.n_features))

        result = system.understand_action(motor)

        assert "action" in result
        assert "confidence" in result

    def test_reason_about(self):
        """Test situated reasoning"""
        system = EnactiveCognitiveSystem()

        problem = np.random.rand(system.params.n_features)
        result = system.reason_about(problem)

        assert "solution" in result

    def test_set_goal(self):
        """Test goal setting"""
        system = EnactiveCognitiveSystem()

        goal = np.random.rand(system.params.n_features)
        system.set_goal(goal)

        assert np.allclose(system.sensorimotor_loop.goal_state, goal)

    def test_replenish(self):
        """Test energy replenishment"""
        system = EnactiveCognitiveSystem()

        for _ in range(50):
            system.sense_act_loop()

        energy_before = system.autopoietic.energy
        system.replenish(0.5)

        assert system.autopoietic.energy > energy_before

    def test_get_state(self):
        """Test state retrieval"""
        system = EnactiveCognitiveSystem()
        system.sense_act_loop()

        state = system.get_state()

        assert "viable" in state
        assert "autopoietic" in state
        assert "sensorimotor" in state
        assert "concepts" in state

    def test_viability_tracking(self):
        """Test viability is tracked"""
        system = EnactiveCognitiveSystem()

        # Drain energy
        system.autopoietic.energy = 0.05

        result = system.sense_act_loop()

        assert not result["viable"]
        assert not system.viability


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
