"""Empathy - Experiencing others' affective states

Core function: Share and understand others' emotional experiences
Neural basis: Anterior insula (affective sharing), ACC (empathy for pain)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class EmpathyParams:
    """Parameters for empathy"""
    n_features: int = 50
    contagion_strength: float = 0.5
    regulation_strength: float = 0.3
    perspective_weight: float = 0.4


class AffectiveSharing:
    """Automatic sharing of emotional states

    Models emotional contagion and resonance
    """

    def __init__(self, params: Optional[EmpathyParams] = None):
        self.params = params or EmpathyParams()

        # Anterior insula activation
        self.insula_activation = np.zeros(self.params.n_features)

        # Own emotional state
        self.own_affect = np.zeros(self.params.n_features)

        # Shared affect from others
        self.shared_affect = np.zeros(self.params.n_features)

    def observe_emotion(self, other_emotion: np.ndarray) -> Dict:
        """Observe another's emotional display"""
        if len(other_emotion) != self.params.n_features:
            other_emotion = np.resize(other_emotion, self.params.n_features)

        # Anterior insula processes emotional observation
        self.insula_activation = np.tanh(
            self.insula_activation * 0.3 + other_emotion * 0.7
        )

        # Emotional contagion - automatic sharing
        contagion = other_emotion * self.params.contagion_strength
        self.shared_affect = np.tanh(self.shared_affect * 0.5 + contagion * 0.5)

        # Update own affect
        self.own_affect = np.tanh(
            self.own_affect * (1 - self.params.contagion_strength) +
            self.shared_affect * self.params.contagion_strength
        )

        return {
            "observed_emotion": other_emotion,
            "shared_affect": self.shared_affect.copy(),
            "own_affect_change": contagion,
            "insula_activity": np.mean(self.insula_activation)
        }

    def get_emotional_resonance(self, other_emotion: np.ndarray) -> float:
        """Calculate how much own state resonates with other's"""
        if len(other_emotion) != self.params.n_features:
            other_emotion = np.resize(other_emotion, self.params.n_features)

        similarity = np.dot(self.own_affect, other_emotion) / (
            np.linalg.norm(self.own_affect) * np.linalg.norm(other_emotion) + 1e-8
        )

        return float(np.clip(similarity, -1, 1))

    def regulate_affect(self, target_affect: np.ndarray) -> np.ndarray:
        """Regulate own affect toward target"""
        if len(target_affect) != self.params.n_features:
            target_affect = np.resize(target_affect, self.params.n_features)

        regulation = (target_affect - self.own_affect) * self.params.regulation_strength
        self.own_affect = np.tanh(self.own_affect + regulation)

        return regulation


class EmpathicConcern:
    """Other-oriented emotional response

    Motivates prosocial behavior toward others in distress
    """

    def __init__(self, params: Optional[EmpathyParams] = None):
        self.params = params or EmpathyParams()

        # ACC activation (empathy for pain)
        self.acc_activation = np.zeros(self.params.n_features)

        # Concern level for different agents
        self.concern_levels: Dict[str, float] = {}

        # Prosocial motivation
        self.prosocial_motivation = 0.0

    def observe_distress(self, agent: str, distress_signal: np.ndarray) -> Dict:
        """Observe another in distress"""
        if len(distress_signal) != self.params.n_features:
            distress_signal = np.resize(distress_signal, self.params.n_features)

        # ACC processes pain/distress
        self.acc_activation = np.tanh(
            self.acc_activation * 0.3 + distress_signal * 0.7
        )

        # Calculate distress intensity
        distress_intensity = np.mean(np.abs(distress_signal))

        # Update concern for this agent
        current_concern = self.concern_levels.get(agent, 0.0)
        self.concern_levels[agent] = min(1.0, current_concern + distress_intensity * 0.3)

        # Update prosocial motivation
        self.prosocial_motivation = min(1.0, self.prosocial_motivation + distress_intensity * 0.2)

        return {
            "agent": agent,
            "distress_intensity": distress_intensity,
            "concern_level": self.concern_levels[agent],
            "prosocial_motivation": self.prosocial_motivation,
            "acc_activity": np.mean(self.acc_activation)
        }

    def get_helping_motivation(self, agent: str) -> float:
        """Get motivation to help specific agent"""
        concern = self.concern_levels.get(agent, 0.0)
        return concern * self.prosocial_motivation

    def relief_response(self, agent: str, relief_signal: np.ndarray):
        """Respond to other's relief/improvement"""
        if len(relief_signal) != self.params.n_features:
            relief_signal = np.resize(relief_signal, self.params.n_features)

        # Reduce concern for agent
        if agent in self.concern_levels:
            self.concern_levels[agent] = max(0, self.concern_levels[agent] - 0.3)

        # Positive feeling from helping
        return {"relief_shared": True, "new_concern": self.concern_levels.get(agent, 0)}


class EmpathySystem:
    """Integrated empathy system

    Combines affective sharing, perspective taking, and empathic concern
    """

    def __init__(self, params: Optional[EmpathyParams] = None):
        self.params = params or EmpathyParams()

        self.affective_sharing = AffectiveSharing(params)
        self.empathic_concern = EmpathicConcern(params)

        # Overall empathy state
        self.empathy_level = 0.5

        # Perspective-taking modulation
        self.perspective_active = False

    def empathize(self, agent: str, emotional_state: np.ndarray,
                 is_distress: bool = False) -> Dict:
        """Full empathic response to another's emotional state"""
        if len(emotional_state) != self.params.n_features:
            emotional_state = np.resize(emotional_state, self.params.n_features)

        # Affective sharing (automatic)
        sharing_result = self.affective_sharing.observe_emotion(emotional_state)

        # Empathic concern (if distress)
        concern_result = None
        if is_distress:
            concern_result = self.empathic_concern.observe_distress(agent, emotional_state)

        # Calculate overall empathy
        resonance = self.affective_sharing.get_emotional_resonance(emotional_state)

        # Perspective modulates empathy
        if self.perspective_active:
            resonance *= (1 + self.params.perspective_weight)

        self.empathy_level = np.clip(0.5 + resonance * 0.5, 0, 1)

        return {
            "agent": agent,
            "sharing": sharing_result,
            "concern": concern_result,
            "resonance": resonance,
            "empathy_level": self.empathy_level,
            "helping_motivation": self.empathic_concern.get_helping_motivation(agent)
        }

    def take_perspective(self, active: bool = True):
        """Activate perspective-taking to enhance empathy"""
        self.perspective_active = active

    def regulate_empathy(self, target_level: float = 0.5):
        """Regulate empathy level (prevent burnout)"""
        target_affect = np.ones(self.params.n_features) * target_level
        self.affective_sharing.regulate_affect(target_affect)
        self.empathy_level = target_level

    def update(self, dt: float = 1.0):
        """Update empathy system"""
        # Decay activations
        self.affective_sharing.insula_activation *= (1 - 0.1 * dt)
        self.empathic_concern.acc_activation *= (1 - 0.1 * dt)

        # Decay prosocial motivation
        self.empathic_concern.prosocial_motivation *= (1 - 0.05 * dt)

    def get_state(self) -> Dict:
        """Get empathy state"""
        return {
            "empathy_level": self.empathy_level,
            "own_affect": self.affective_sharing.own_affect.copy(),
            "insula_activity": np.mean(self.affective_sharing.insula_activation),
            "acc_activity": np.mean(self.empathic_concern.acc_activation),
            "prosocial_motivation": self.empathic_concern.prosocial_motivation,
            "perspective_active": self.perspective_active,
            "concern_levels": self.empathic_concern.concern_levels.copy()
        }
