"""Social Cognition Network - Integrated social brain

Combines Theory of Mind, mentalizing, empathy, and perspective taking
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List

from .theory_of_mind import TheoryOfMind, ToMParams
from .mentalizing import MentalizingNetwork, MentalizingParams
from .empathy import EmpathySystem, EmpathyParams
from .perspective_taking import PerspectiveTaking, PerspectiveParams


class SocialBrain:
    """Hierarchical social brain model

    Cognitive level: mPFC, TPJ (mentalizing, ToM)
    Affective level: Insula, ACC (empathy)
    """

    def __init__(self, n_features: int = 50):
        self.n_features = n_features

        # Cognitive level
        self.cognitive_activation = np.zeros(n_features)

        # Affective level
        self.affective_activation = np.zeros(n_features)

        # Integration
        self.integrated_activation = np.zeros(n_features)

    def process_cognitive(self, social_input: np.ndarray) -> np.ndarray:
        """Process at cognitive level (mentalizing)"""
        if len(social_input) != self.n_features:
            social_input = np.resize(social_input, self.n_features)

        self.cognitive_activation = np.tanh(
            self.cognitive_activation * 0.3 + social_input * 0.7
        )
        return self.cognitive_activation.copy()

    def process_affective(self, emotional_input: np.ndarray) -> np.ndarray:
        """Process at affective level (empathy)"""
        if len(emotional_input) != self.n_features:
            emotional_input = np.resize(emotional_input, self.n_features)

        self.affective_activation = np.tanh(
            self.affective_activation * 0.3 + emotional_input * 0.7
        )
        return self.affective_activation.copy()

    def integrate(self) -> np.ndarray:
        """Integrate cognitive and affective processing"""
        self.integrated_activation = np.tanh(
            self.cognitive_activation * 0.5 + self.affective_activation * 0.5
        )
        return self.integrated_activation.copy()

    def get_activations(self) -> Dict:
        """Get all activation levels"""
        return {
            "cognitive": self.cognitive_activation.copy(),
            "affective": self.affective_activation.copy(),
            "integrated": self.integrated_activation.copy()
        }


class SocialCognitionNetwork:
    """Full social cognition network

    Integrates all social cognitive components
    """

    def __init__(self, n_features: int = 50):
        self.n_features = n_features

        # Core components
        tom_params = ToMParams(n_features=n_features)
        self.theory_of_mind = TheoryOfMind(tom_params)

        ment_params = MentalizingParams(n_features=n_features)
        self.mentalizing = MentalizingNetwork(ment_params)

        emp_params = EmpathyParams(n_features=n_features)
        self.empathy = EmpathySystem(emp_params)

        persp_params = PerspectiveParams(n_features=n_features)
        self.perspective_taking = PerspectiveTaking(persp_params)

        # Social brain
        self.social_brain = SocialBrain(n_features)

        # Social context
        self.current_interaction: Dict = {}

    def process_social_stimulus(self, agent: str, behavior: np.ndarray,
                               emotional_display: Optional[np.ndarray] = None,
                               is_distress: bool = False) -> Dict:
        """Full social processing of another agent"""
        if len(behavior) != self.n_features:
            behavior = np.resize(behavior, self.n_features)

        results = {"agent": agent}

        # Mentalizing (infer mental states)
        mental_states = self.mentalizing.mentalize(agent, behavior)
        results["mental_states"] = mental_states

        # Theory of Mind (track beliefs)
        tom_result = self.theory_of_mind.process_other(agent, behavior)
        results["tom"] = tom_result

        # Empathy (if emotional content)
        if emotional_display is not None:
            empathy_result = self.empathy.empathize(
                agent, emotional_display, is_distress=is_distress
            )
            results["empathy"] = empathy_result

        # Update perspective model
        self.perspective_taking.update_model(agent, behavior)

        # Social brain processing
        self.social_brain.process_cognitive(behavior)
        if emotional_display is not None:
            self.social_brain.process_affective(emotional_display)
        self.social_brain.integrate()

        results["social_brain"] = self.social_brain.get_activations()

        self.current_interaction = results
        return results

    def take_perspective(self, agent: str, context: np.ndarray,
                        perspective_type: str = "cognitive") -> Dict:
        """Take another's perspective"""
        if perspective_type == "visual":
            return self.perspective_taking.take_visual_perspective(agent, context, level=2)
        elif perspective_type == "affective":
            return self.perspective_taking.take_affective_perspective(agent, context)
        else:
            # Cognitive perspective (what do they think/believe)
            self.perspective_taking.tpj.switch_perspective(agent)
            viewed = self.perspective_taking.tpj.view_from_perspective(context)
            return {
                "agent": agent,
                "type": "cognitive",
                "their_view": viewed
            }

    def predict_behavior(self, agent: str) -> Dict:
        """Predict agent's behavior"""
        return self.mentalizing.predict_behavior(agent)

    def check_false_belief(self, agent: str, reality: np.ndarray,
                          their_belief: np.ndarray) -> Dict:
        """Check for false belief understanding"""
        return self.theory_of_mind.run_false_belief_task(agent, reality, their_belief)

    def get_empathic_response(self, agent: str) -> Dict:
        """Get empathic response to agent"""
        return {
            "empathy_level": self.empathy.empathy_level,
            "concern": self.empathy.empathic_concern.concern_levels.get(agent, 0),
            "helping_motivation": self.empathy.empathic_concern.get_helping_motivation(agent),
            "resonance": self.empathy.affective_sharing.get_emotional_resonance(
                self.perspective_taking.self_other.other_representations.get(
                    agent, np.zeros(self.n_features)
                )
            )
        }

    def process_self(self, own_state: np.ndarray):
        """Process own mental/emotional state"""
        if len(own_state) != self.n_features:
            own_state = np.resize(own_state, self.n_features)

        self.theory_of_mind.process_self(own_state)
        self.perspective_taking.self_other.update_self(own_state)

    def update(self, dt: float = 1.0):
        """Update all social cognition systems"""
        self.theory_of_mind.update(dt)
        self.mentalizing.update(dt)
        self.empathy.update(dt)
        self.perspective_taking.update(dt)

    def get_state(self) -> Dict:
        """Get comprehensive social cognition state"""
        return {
            "tom": self.theory_of_mind.get_state(),
            "mentalizing": self.mentalizing.get_state(),
            "empathy": self.empathy.get_state(),
            "perspective": self.perspective_taking.get_state(),
            "social_brain": {
                "cognitive_mean": np.mean(self.social_brain.cognitive_activation),
                "affective_mean": np.mean(self.social_brain.affective_activation),
                "integrated_mean": np.mean(self.social_brain.integrated_activation)
            }
        }
