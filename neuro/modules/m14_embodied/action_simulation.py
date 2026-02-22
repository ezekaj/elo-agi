"""Action Simulation - Motor simulation for understanding actions

Core principle: Understanding actions involves simulating them motorically
Key features: Mirror neurons, motor resonance, action prediction
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class SimulationParams:
    """Parameters for action simulation"""

    n_motor_units: int = 50
    n_action_types: int = 10
    simulation_strength: float = 0.5
    resonance_threshold: float = 0.3
    decay_rate: float = 0.1


class MotorSimulator:
    """Simulates actions without execution

    Models covert motor activation during action understanding/planning
    """

    def __init__(self, params: Optional[SimulationParams] = None):
        self.params = params or SimulationParams()

        # Motor repertoire (learned actions)
        self.action_representations = {}

        # Current simulation state
        self.simulation_activation = np.zeros(self.params.n_motor_units)
        self.simulating = False

        # Inhibition (prevents overt execution during simulation)
        self.inhibition_level = 0.0

    def learn_action(self, action_name: str, motor_pattern: np.ndarray):
        """Learn a new action's motor pattern"""
        if len(motor_pattern) != self.params.n_motor_units:
            motor_pattern = np.resize(motor_pattern, self.params.n_motor_units)

        self.action_representations[action_name] = motor_pattern.copy()

    def simulate_action(self, action_name: str) -> Dict:
        """Simulate an action without executing"""
        if action_name not in self.action_representations:
            return {"success": False, "activation": np.zeros(self.params.n_motor_units)}

        self.simulating = True
        self.inhibition_level = 0.8  # High inhibition during simulation

        # Activate motor pattern at reduced strength
        pattern = self.action_representations[action_name]
        self.simulation_activation = pattern * self.params.simulation_strength

        # Add noise
        self.simulation_activation += np.random.randn(self.params.n_motor_units) * 0.05

        return {
            "success": True,
            "action": action_name,
            "activation": self.simulation_activation.copy(),
            "inhibited": True,
        }

    def predict_outcome(self, action_name: str) -> np.ndarray:
        """Predict sensory outcome of simulated action"""
        result = self.simulate_action(action_name)
        if not result["success"]:
            return np.zeros(self.params.n_motor_units)

        # Simple outcome prediction (would be learned in full model)
        predicted_outcome = np.tanh(self.simulation_activation * 2)
        return predicted_outcome

    def stop_simulation(self):
        """Stop current simulation"""
        self.simulating = False
        self.inhibition_level = 0.0
        self.simulation_activation = np.zeros(self.params.n_motor_units)

    def get_simulation_state(self) -> Dict:
        """Get current simulation state"""
        return {
            "simulating": self.simulating,
            "activation": self.simulation_activation.copy(),
            "inhibition": self.inhibition_level,
        }


class MirrorSystem:
    """Mirror neuron system for action understanding

    Activates when both executing and observing actions
    """

    def __init__(self, params: Optional[SimulationParams] = None):
        self.params = params or SimulationParams()

        # Mirror neurons (fire for both execution and observation)
        self.mirror_activation = np.zeros(self.params.n_motor_units)

        # Action-specific mirror responses
        self.action_mirrors: Dict[str, np.ndarray] = {}

        # Observation vs execution mode
        self.mode = "idle"

    def learn_action_mirror(self, action_name: str, motor_pattern: np.ndarray):
        """Associate action with mirror response"""
        if len(motor_pattern) != self.params.n_motor_units:
            motor_pattern = np.resize(motor_pattern, self.params.n_motor_units)

        self.action_mirrors[action_name] = motor_pattern.copy()

    def observe_action(self, visual_input: np.ndarray) -> Dict:
        """Process observed action

        Activates mirror neurons based on visual input matching known actions
        """
        self.mode = "observing"

        # Match visual input to known actions
        best_match = None
        best_similarity = 0.0

        for action_name, pattern in self.action_mirrors.items():
            # Simplified: compare visual input to motor pattern
            if len(visual_input) != self.params.n_motor_units:
                visual_input = np.resize(visual_input, self.params.n_motor_units)

            similarity = np.dot(visual_input, pattern) / (
                np.linalg.norm(visual_input) * np.linalg.norm(pattern) + 1e-8
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = action_name

        # Activate mirror neurons if match found
        if best_match and best_similarity > self.params.resonance_threshold:
            self.mirror_activation = self.action_mirrors[best_match] * best_similarity
        else:
            self.mirror_activation = np.zeros(self.params.n_motor_units)

        return {
            "recognized_action": best_match,
            "confidence": best_similarity,
            "mirror_activation": self.mirror_activation.copy(),
        }

    def execute_action(self, action_name: str) -> Dict:
        """Execute an action (activates mirrors as well)"""
        self.mode = "executing"

        if action_name not in self.action_mirrors:
            return {"success": False, "activation": np.zeros(self.params.n_motor_units)}

        # Full activation during execution
        self.mirror_activation = self.action_mirrors[action_name].copy()

        return {"success": True, "action": action_name, "activation": self.mirror_activation.copy()}

    def get_resonance(self) -> float:
        """Get current motor resonance level"""
        return np.mean(np.abs(self.mirror_activation))

    def decay(self, dt: float = 1.0):
        """Decay mirror activation"""
        self.mirror_activation *= 1 - self.params.decay_rate * dt
        if np.max(np.abs(self.mirror_activation)) < 0.01:
            self.mode = "idle"


class ActionUnderstanding:
    """Integrated action understanding system

    Combines simulation and mirror systems for understanding observed actions
    """

    def __init__(self, params: Optional[SimulationParams] = None):
        self.params = params or SimulationParams()

        self.simulator = MotorSimulator(params)
        self.mirror_system = MirrorSystem(params)

        # Understanding state
        self.understood_action = None
        self.understanding_confidence = 0.0

    def learn_action(self, action_name: str, motor_pattern: np.ndarray):
        """Learn a new action"""
        self.simulator.learn_action(action_name, motor_pattern)
        self.mirror_system.learn_action_mirror(action_name, motor_pattern)

    def understand_observed_action(self, visual_input: np.ndarray) -> Dict:
        """Understand an observed action through simulation and mirroring"""
        # Mirror response
        mirror_result = self.mirror_system.observe_action(visual_input)

        # If recognized, simulate to predict outcome
        predicted_outcome = None
        if mirror_result["recognized_action"]:
            predicted_outcome = self.simulator.predict_outcome(mirror_result["recognized_action"])

        self.understood_action = mirror_result["recognized_action"]
        self.understanding_confidence = mirror_result["confidence"]

        return {
            "action": mirror_result["recognized_action"],
            "confidence": mirror_result["confidence"],
            "motor_resonance": self.mirror_system.get_resonance(),
            "predicted_outcome": predicted_outcome,
        }

    def predict_intention(self, action_sequence: List[np.ndarray]) -> str:
        """Predict intention from action sequence"""
        # Understand each action
        understood = []
        for visual_input in action_sequence:
            result = self.understand_observed_action(visual_input)
            if result["action"]:
                understood.append(result["action"])

        # Simple intention inference (would be more complex in real system)
        if understood:
            # Return most common action as "intention"
            from collections import Counter

            counter = Counter(understood)
            return counter.most_common(1)[0][0]

        return "unknown"

    def update(self, dt: float = 1.0):
        """Update system state"""
        self.mirror_system.decay(dt)
        if not self.mirror_system.mode == "observing":
            self.understanding_confidence *= 1 - self.params.decay_rate * dt

    def get_state(self) -> Dict:
        """Get understanding state"""
        return {
            "understood_action": self.understood_action,
            "confidence": self.understanding_confidence,
            "mirror_resonance": self.mirror_system.get_resonance(),
            "simulation_state": self.simulator.get_simulation_state(),
        }
