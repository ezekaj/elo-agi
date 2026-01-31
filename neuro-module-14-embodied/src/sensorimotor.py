"""Sensorimotor Processing - Perception-action loops

Core principle: Perception and action are tightly coupled in continuous loops
Key features: Forward models, efference copies, sensorimotor predictions
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict


@dataclass
class SensorimotorParams:
    """Parameters for sensorimotor processing"""
    n_sensory: int = 50
    n_motor: int = 30
    prediction_gain: float = 0.8
    learning_rate: float = 0.1
    noise_level: float = 0.05
    delay_steps: int = 2


class MotorSensoryCoupling:
    """Bidirectional coupling between motor and sensory systems

    Models how motor commands influence sensory predictions
    and sensory input guides motor actions
    """

    def __init__(self, n_sensory: int = 50, n_motor: int = 30):
        self.n_sensory = n_sensory
        self.n_motor = n_motor

        # Motor to sensory mapping (forward model)
        self.motor_to_sensory = np.random.randn(n_sensory, n_motor) * 0.1
        # Sensory to motor mapping (inverse model)
        self.sensory_to_motor = np.random.randn(n_motor, n_sensory) * 0.1

        # Current states
        self.motor_state = np.zeros(n_motor)
        self.sensory_state = np.zeros(n_sensory)

    def predict_sensory(self, motor_command: np.ndarray) -> np.ndarray:
        """Predict sensory consequences of motor command (forward model)"""
        if len(motor_command) != self.n_motor:
            motor_command = np.resize(motor_command, self.n_motor)

        prediction = np.tanh(np.dot(self.motor_to_sensory, motor_command))
        return prediction

    def compute_motor(self, desired_sensory: np.ndarray) -> np.ndarray:
        """Compute motor command to achieve desired sensory state (inverse model)"""
        if len(desired_sensory) != self.n_sensory:
            desired_sensory = np.resize(desired_sensory, self.n_sensory)

        motor_command = np.tanh(np.dot(self.sensory_to_motor, desired_sensory))
        return motor_command

    def update_coupling(self, motor: np.ndarray, predicted: np.ndarray,
                       actual: np.ndarray, learning_rate: float = 0.01):
        """Update forward model based on prediction error"""
        error = actual - predicted

        # Delta rule update
        delta = learning_rate * np.outer(error, motor)
        self.motor_to_sensory += delta

    def get_coupling_strength(self) -> float:
        """Get measure of motor-sensory coupling strength"""
        return np.mean(np.abs(self.motor_to_sensory))


class PredictiveProcessor:
    """Predictive processing for sensorimotor integration

    Generates predictions about sensory consequences of actions
    and computes prediction errors
    """

    def __init__(self, params: Optional[SensorimotorParams] = None):
        self.params = params or SensorimotorParams()

        # State dimensions
        self.n_sensory = self.params.n_sensory
        self.n_motor = self.params.n_motor

        # Forward model weights
        self.forward_model = np.random.randn(self.n_sensory, self.n_motor) * 0.1

        # Prediction state
        self.prediction = np.zeros(self.n_sensory)
        self.prediction_error = np.zeros(self.n_sensory)

        # Efference copy (copy of motor command)
        self.efference_copy = np.zeros(self.n_motor)

        # History
        self.error_history = []

    def send_motor_command(self, command: np.ndarray) -> np.ndarray:
        """Process outgoing motor command

        Creates efference copy and generates sensory prediction
        """
        if len(command) != self.n_motor:
            command = np.resize(command, self.n_motor)

        # Store efference copy
        self.efference_copy = command.copy()

        # Generate prediction
        self.prediction = np.tanh(
            np.dot(self.forward_model, command) * self.params.prediction_gain
        )

        return self.prediction

    def receive_sensory(self, sensory_input: np.ndarray) -> np.ndarray:
        """Process incoming sensory input

        Computes prediction error and updates forward model
        """
        if len(sensory_input) != self.n_sensory:
            sensory_input = np.resize(sensory_input, self.n_sensory)

        # Compute prediction error
        self.prediction_error = sensory_input - self.prediction

        # Add noise
        self.prediction_error += np.random.randn(self.n_sensory) * self.params.noise_level

        self.error_history.append(np.mean(np.abs(self.prediction_error)))

        # Update forward model (error-driven learning)
        delta = self.params.learning_rate * np.outer(
            self.prediction_error, self.efference_copy
        )
        self.forward_model += delta

        return self.prediction_error

    def get_surprise(self) -> float:
        """Get current surprise (magnitude of prediction error)"""
        return np.mean(np.abs(self.prediction_error))

    def get_mean_error(self, window: int = 10) -> float:
        """Get mean prediction error over recent history"""
        if len(self.error_history) == 0:
            return 0.0
        return np.mean(self.error_history[-window:])


class SensorimotorLoop:
    """Complete sensorimotor loop

    Implements closed-loop action-perception cycle
    """

    def __init__(self, params: Optional[SensorimotorParams] = None):
        self.params = params or SensorimotorParams()

        # Components
        self.coupling = MotorSensoryCoupling(
            self.params.n_sensory, self.params.n_motor
        )
        self.predictor = PredictiveProcessor(self.params)

        # State
        self.motor_output = np.zeros(self.params.n_motor)
        self.sensory_input = np.zeros(self.params.n_sensory)
        self.goal_state = np.zeros(self.params.n_sensory)

        # Time step
        self.t = 0

    def set_goal(self, goal: np.ndarray):
        """Set desired sensory goal state"""
        if len(goal) != self.params.n_sensory:
            goal = np.resize(goal, self.params.n_sensory)
        self.goal_state = goal

    def step(self, external_sensory: Optional[np.ndarray] = None) -> Dict:
        """Execute one step of sensorimotor loop

        Args:
            external_sensory: External sensory input (if any)

        Returns:
            Step results
        """
        # 1. Compute motor command based on goal
        self.motor_output = self.coupling.compute_motor(self.goal_state)

        # 2. Predict sensory consequences
        prediction = self.predictor.send_motor_command(self.motor_output)

        # 3. Receive actual sensory input
        if external_sensory is not None:
            self.sensory_input = external_sensory
        else:
            # Simulate sensory consequences of action
            self.sensory_input = self.coupling.predict_sensory(self.motor_output)
            self.sensory_input += np.random.randn(self.params.n_sensory) * self.params.noise_level

        # 4. Compute prediction error
        error = self.predictor.receive_sensory(self.sensory_input)

        # 5. Update coupling based on error
        self.coupling.update_coupling(
            self.motor_output, prediction, self.sensory_input,
            self.params.learning_rate
        )

        self.t += 1

        return {
            "motor_output": self.motor_output.copy(),
            "prediction": prediction.copy(),
            "sensory_input": self.sensory_input.copy(),
            "prediction_error": error.copy(),
            "surprise": self.predictor.get_surprise(),
            "goal_error": np.mean(np.abs(self.sensory_input - self.goal_state))
        }

    def run_to_goal(self, goal: np.ndarray, max_steps: int = 100,
                   tolerance: float = 0.1) -> List[Dict]:
        """Run loop until goal is reached

        Args:
            goal: Target sensory state
            max_steps: Maximum steps
            tolerance: Goal achievement tolerance

        Returns:
            History of steps
        """
        self.set_goal(goal)
        history = []

        for _ in range(max_steps):
            result = self.step()
            history.append(result)

            if result["goal_error"] < tolerance:
                break

        return history

    def get_state(self) -> Dict:
        """Get current loop state"""
        return {
            "motor_output": self.motor_output.copy(),
            "sensory_input": self.sensory_input.copy(),
            "goal_state": self.goal_state.copy(),
            "prediction": self.predictor.prediction.copy(),
            "coupling_strength": self.coupling.get_coupling_strength(),
            "mean_error": self.predictor.get_mean_error()
        }
