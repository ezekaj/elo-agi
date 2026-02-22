"""Tests for sensorimotor processing"""

import numpy as np
import pytest
from neuro.modules.m14_embodied.sensorimotor import (
    MotorSensoryCoupling,
    PredictiveProcessor,
    SensorimotorLoop,
)


class TestMotorSensoryCoupling:
    """Tests for motor-sensory coupling"""

    def test_initialization(self):
        """Test coupling initialization"""
        coupling = MotorSensoryCoupling(n_sensory=50, n_motor=30)

        assert coupling.n_sensory == 50
        assert coupling.n_motor == 30
        assert coupling.motor_to_sensory.shape == (50, 30)

    def test_predict_sensory(self):
        """Test sensory prediction from motor command"""
        coupling = MotorSensoryCoupling(n_sensory=50, n_motor=30)

        motor_command = np.random.rand(30)
        prediction = coupling.predict_sensory(motor_command)

        assert len(prediction) == 50

    def test_compute_motor(self):
        """Test motor command computation from desired sensory"""
        coupling = MotorSensoryCoupling(n_sensory=50, n_motor=30)

        desired = np.random.rand(50)
        motor = coupling.compute_motor(desired)

        assert len(motor) == 30

    def test_coupling_update(self):
        """Test coupling weights update"""
        coupling = MotorSensoryCoupling(n_sensory=50, n_motor=30)
        initial_weights = coupling.motor_to_sensory.copy()

        motor = np.random.rand(30)
        predicted = coupling.predict_sensory(motor)
        actual = predicted + np.random.rand(50) * 0.1

        coupling.update_coupling(motor, predicted, actual, learning_rate=0.1)

        assert not np.allclose(coupling.motor_to_sensory, initial_weights)


class TestPredictiveProcessor:
    """Tests for predictive processing"""

    def test_initialization(self):
        """Test processor initialization"""
        processor = PredictiveProcessor()

        assert processor.prediction.shape == (processor.n_sensory,)
        assert len(processor.error_history) == 0

    def test_motor_command_generates_prediction(self):
        """Test motor command creates sensory prediction"""
        processor = PredictiveProcessor()

        command = np.random.rand(processor.n_motor)
        prediction = processor.send_motor_command(command)

        assert len(prediction) == processor.n_sensory
        assert np.allclose(processor.efference_copy, command)

    def test_sensory_computes_error(self):
        """Test sensory input computes prediction error"""
        processor = PredictiveProcessor()

        processor.send_motor_command(np.random.rand(processor.n_motor))
        sensory = np.random.rand(processor.n_sensory)
        error = processor.receive_sensory(sensory)

        assert len(error) == processor.n_sensory
        assert len(processor.error_history) == 1

    def test_surprise_metric(self):
        """Test surprise calculation"""
        processor = PredictiveProcessor()

        processor.send_motor_command(np.random.rand(processor.n_motor))
        processor.receive_sensory(np.random.rand(processor.n_sensory))

        surprise = processor.get_surprise()

        assert surprise >= 0

    def test_error_learning(self):
        """Test prediction error drives learning"""
        processor = PredictiveProcessor()
        initial_model = processor.forward_model.copy()

        # Multiple prediction cycles
        for _ in range(10):
            processor.send_motor_command(np.random.rand(processor.n_motor))
            processor.receive_sensory(np.random.rand(processor.n_sensory))

        assert not np.allclose(processor.forward_model, initial_model)


class TestSensorimotorLoop:
    """Tests for complete sensorimotor loop"""

    def test_initialization(self):
        """Test loop initialization"""
        loop = SensorimotorLoop()

        assert loop.coupling is not None
        assert loop.predictor is not None
        assert loop.t == 0

    def test_set_goal(self):
        """Test goal setting"""
        loop = SensorimotorLoop()

        goal = np.random.rand(loop.params.n_sensory)
        loop.set_goal(goal)

        assert np.allclose(loop.goal_state, goal)

    def test_step_execution(self):
        """Test single step execution"""
        loop = SensorimotorLoop()
        loop.set_goal(np.random.rand(loop.params.n_sensory))

        result = loop.step()

        assert "motor_output" in result
        assert "prediction" in result
        assert "sensory_input" in result
        assert "prediction_error" in result
        assert loop.t == 1

    def test_run_to_goal(self):
        """Test running loop to goal"""
        loop = SensorimotorLoop()

        goal = np.zeros(loop.params.n_sensory)
        history = loop.run_to_goal(goal, max_steps=50, tolerance=1.0)

        assert len(history) > 0
        assert len(history) <= 50

    def test_state_getter(self):
        """Test state retrieval"""
        loop = SensorimotorLoop()
        loop.step()

        state = loop.get_state()

        assert "motor_output" in state
        assert "sensory_input" in state
        assert "goal_state" in state
        assert "coupling_strength" in state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
