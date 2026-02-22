"""Enactive Cognitive System - Full embodied integration

Three types of enactivism:
- Autopoietic: Living system self-organization
- Sensorimotor: Perception constituted through action
- Radical: Mind-body-environment coupled system
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict

from .sensorimotor import SensorimotorLoop, SensorimotorParams
from .grounded_concepts import ConceptGrounding, GroundingParams
from .action_simulation import ActionUnderstanding, SimulationParams
from .situated_cognition import ContextualReasoner, SituatedParams


@dataclass
class EnactiveParams:
    """Parameters for enactive system"""

    n_features: int = 50
    adaptation_rate: float = 0.1
    coupling_strength: float = 0.5
    energy_decay: float = 0.01


class AutopoieticSystem:
    """Self-maintaining living system dynamics

    Models the self-organizing nature of cognitive systems
    """

    def __init__(self, params: Optional[EnactiveParams] = None):
        self.params = params or EnactiveParams()

        # Internal state (metabolic-like)
        self.internal_state = np.random.rand(self.params.n_features) * 0.5
        self.energy = 1.0

        # Boundary (self-other distinction)
        self.boundary_integrity = 1.0

        # Homeostatic setpoints
        self.setpoints = np.ones(self.params.n_features) * 0.5

    def maintain_organization(self, dt: float = 1.0) -> Dict:
        """Self-maintenance process"""
        # Drift toward setpoints (homeostasis)
        error = self.setpoints - self.internal_state
        self.internal_state += error * self.params.adaptation_rate * dt

        # Energy consumption
        maintenance_cost = np.mean(np.abs(error)) * 0.1
        self.energy = max(0, self.energy - maintenance_cost * dt)

        # Boundary maintenance
        self.boundary_integrity = min(1.0, self.boundary_integrity + 0.01 * dt)

        return {
            "internal_state": self.internal_state.copy(),
            "energy": self.energy,
            "boundary_integrity": self.boundary_integrity,
            "homeostatic_error": np.mean(np.abs(error)),
        }

    def interact_with_environment(self, environmental_input: np.ndarray) -> Dict:
        """Interact with environment while maintaining organization"""
        if len(environmental_input) != self.params.n_features:
            environmental_input = np.resize(environmental_input, self.params.n_features)

        # Environment perturbs internal state
        perturbation = environmental_input * (1 - self.boundary_integrity)
        self.internal_state += perturbation * 0.1

        # Boundary is stressed by strong perturbations
        perturbation_strength = np.mean(np.abs(perturbation))
        self.boundary_integrity = max(0.1, self.boundary_integrity - perturbation_strength * 0.1)

        return {
            "perturbation": perturbation,
            "new_state": self.internal_state.copy(),
            "boundary_stress": perturbation_strength,
        }

    def replenish_energy(self, amount: float):
        """Replenish energy (metabolic intake)"""
        self.energy = min(1.0, self.energy + amount)

    def is_viable(self) -> bool:
        """Check if system is still viable"""
        return self.energy > 0.1 and self.boundary_integrity > 0.1


class SensoriMotorEnaction:
    """Sensorimotor enactivism - perception through action

    Perception is not passive reception but active exploration
    """

    def __init__(self, params: Optional[EnactiveParams] = None):
        self.params = params or EnactiveParams()

        # Sensorimotor contingencies (action-perception relationships)
        self.contingencies: Dict[str, Dict] = {}

        # Current sensorimotor state
        self.motor_state = np.zeros(self.params.n_features)
        self.sensory_state = np.zeros(self.params.n_features)

        # History of sensorimotor interactions
        self.interaction_history = []

    def learn_contingency(
        self, action_name: str, motor_pattern: np.ndarray, sensory_consequence: np.ndarray
    ):
        """Learn sensorimotor contingency"""
        if len(motor_pattern) != self.params.n_features:
            motor_pattern = np.resize(motor_pattern, self.params.n_features)
        if len(sensory_consequence) != self.params.n_features:
            sensory_consequence = np.resize(sensory_consequence, self.params.n_features)

        self.contingencies[action_name] = {
            "motor": motor_pattern.copy(),
            "sensory": sensory_consequence.copy(),
        }

    def enact(self, action_name: str) -> Dict:
        """Enact a sensorimotor contingency"""
        if action_name not in self.contingencies:
            return {"success": False}

        contingency = self.contingencies[action_name]

        # Motor action
        self.motor_state = contingency["motor"].copy()

        # Sensory consequence (with noise)
        self.sensory_state = contingency["sensory"] + np.random.randn(self.params.n_features) * 0.1

        self.interaction_history.append(
            {
                "action": action_name,
                "motor": self.motor_state.copy(),
                "sensory": self.sensory_state.copy(),
            }
        )

        return {
            "success": True,
            "action": action_name,
            "motor_state": self.motor_state.copy(),
            "sensory_state": self.sensory_state.copy(),
        }

    def perceive_through_action(self, target_sensory: np.ndarray) -> Dict:
        """Achieve perception through appropriate action"""
        if len(target_sensory) != self.params.n_features:
            target_sensory = np.resize(target_sensory, self.params.n_features)

        # Find best matching contingency
        best_action = None
        best_similarity = -1

        for action_name, contingency in self.contingencies.items():
            similarity = np.dot(contingency["sensory"], target_sensory) / (
                np.linalg.norm(contingency["sensory"]) * np.linalg.norm(target_sensory) + 1e-8
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_action = action_name

        if best_action:
            return self.enact(best_action)

        return {"success": False, "reason": "no matching contingency"}

    def get_sensorimotor_knowledge(self) -> Dict:
        """Get knowledge about sensorimotor contingencies"""
        return {
            "n_contingencies": len(self.contingencies),
            "contingency_names": list(self.contingencies.keys()),
            "n_interactions": len(self.interaction_history),
        }


class EnactiveCognitiveSystem:
    """Full enactive cognitive system

    Integrates all embodied and enactive components
    """

    def __init__(self, params: Optional[EnactiveParams] = None):
        self.params = params or EnactiveParams()

        # Core components
        sm_params = SensorimotorParams(
            n_sensory=self.params.n_features, n_motor=self.params.n_features
        )
        self.sensorimotor_loop = SensorimotorLoop(sm_params)

        grounding_params = GroundingParams(n_features=self.params.n_features)
        self.concept_grounding = ConceptGrounding(grounding_params)

        sim_params = SimulationParams(n_motor_units=self.params.n_features)
        self.action_understanding = ActionUnderstanding(sim_params)

        sit_params = SituatedParams(n_features=self.params.n_features)
        self.situated_reasoning = ContextualReasoner(sit_params)

        # Enactive components
        self.autopoietic = AutopoieticSystem(params)
        self.sensorimotor_enaction = SensoriMotorEnaction(params)

        # Global state
        self.time = 0
        self.viability = True

    def sense_act_loop(self, external_input: Optional[np.ndarray] = None) -> Dict:
        """Execute one sense-act cycle"""
        # Maintain autopoietic organization
        maintenance = self.autopoietic.maintain_organization()

        if not self.autopoietic.is_viable():
            self.viability = False
            return {"viable": False, "maintenance": maintenance}

        # Sensorimotor loop step
        sm_result = self.sensorimotor_loop.step(external_input)

        # Ground experience in concepts
        if sm_result["surprise"] < 0.3:  # Low surprise -> stable enough to learn
            concept_name = f"experience_{self.time}"
            self.concept_grounding.ground_concept(
                concept_name, sm_result["sensory_input"], "proprioceptive"
            )

        # Update situated context
        self.situated_reasoning.context.set_physical_context(
            sm_result["sensory_input"][: self.params.n_features // 3]
        )

        self.time += 1

        return {
            "viable": True,
            "maintenance": maintenance,
            "sensorimotor": sm_result,
            "energy": self.autopoietic.energy,
        }

    def learn_action(
        self, action_name: str, motor_pattern: np.ndarray, sensory_consequence: np.ndarray
    ):
        """Learn an action across all systems"""
        self.action_understanding.learn_action(action_name, motor_pattern)
        self.sensorimotor_enaction.learn_contingency(
            action_name, motor_pattern, sensory_consequence
        )

    def understand_action(self, visual_input: np.ndarray) -> Dict:
        """Understand observed action through embodied simulation"""
        return self.action_understanding.understand_observed_action(visual_input)

    def reason_about(self, problem: np.ndarray) -> Dict:
        """Reason about problem in situated context"""
        return self.situated_reasoning.reason_in_context(problem)

    def set_goal(self, goal: np.ndarray):
        """Set sensorimotor goal"""
        self.sensorimotor_loop.set_goal(goal)

    def replenish(self, amount: float = 0.1):
        """Replenish system energy"""
        self.autopoietic.replenish_energy(amount)

    def update(self, dt: float = 1.0):
        """Update all subsystems"""
        self.concept_grounding.update(dt)
        self.action_understanding.update(dt)
        self.situated_reasoning.update(dt)

    def get_state(self) -> Dict:
        """Get comprehensive system state"""
        return {
            "time": self.time,
            "viable": self.viability,
            "autopoietic": {
                "energy": self.autopoietic.energy,
                "boundary_integrity": self.autopoietic.boundary_integrity,
                "viable": self.autopoietic.is_viable(),
            },
            "sensorimotor": self.sensorimotor_loop.get_state(),
            "concepts": len(self.concept_grounding.concepts),
            "situated": self.situated_reasoning.get_state(),
            "actions_learned": len(self.action_understanding.simulator.action_representations),
        }
