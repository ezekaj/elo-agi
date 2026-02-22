"""Perspective Taking - TPJ-based perspective shifts

Core function: Shift from own perspective to another's
Neural basis: Temporoparietal Junction (TPJ)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class PerspectiveParams:
    """Parameters for perspective taking"""

    n_features: int = 50
    self_other_separation: float = 0.7
    switching_cost: float = 0.2
    default_self_weight: float = 0.6


class SelfOtherDistinction:
    """Maintain distinction between self and other representations

    Crucial for accurate perspective taking without confusion
    """

    def __init__(self, params: Optional[PerspectiveParams] = None):
        self.params = params or PerspectiveParams()

        # Self representation
        self.self_representation = np.random.randn(self.params.n_features) * 0.1

        # Other representations
        self.other_representations: Dict[str, np.ndarray] = {}

        # Self-other boundary strength
        self.boundary_strength = self.params.self_other_separation

    def update_self(self, self_state: np.ndarray):
        """Update self representation"""
        if len(self_state) != self.params.n_features:
            self_state = np.resize(self_state, self.params.n_features)

        self.self_representation = 0.7 * self.self_representation + 0.3 * self_state

    def model_other(self, agent: str, state: np.ndarray):
        """Create/update model of another agent"""
        if len(state) != self.params.n_features:
            state = np.resize(state, self.params.n_features)

        if agent in self.other_representations:
            self.other_representations[agent] = (
                0.7 * self.other_representations[agent] + 0.3 * state
            )
        else:
            self.other_representations[agent] = state.copy()

    def distinguish(self, agent: str) -> float:
        """Get distinction score between self and other"""
        if agent not in self.other_representations:
            return 1.0  # Maximum distinction for unknown

        other = self.other_representations[agent]

        # Calculate similarity (low similarity = high distinction)
        similarity = np.dot(self.self_representation, other) / (
            np.linalg.norm(self.self_representation) * np.linalg.norm(other) + 1e-8
        )

        distinction = (1 - similarity) / 2
        return distinction * self.boundary_strength

    def get_self_influence(self, agent: str) -> float:
        """Get how much self bleeds into other representation"""
        distinction = self.distinguish(agent)
        return 1 - distinction  # Higher distinction = less self influence


class TPJNetwork:
    """Temporoparietal Junction network for perspective shifts"""

    def __init__(self, params: Optional[PerspectiveParams] = None):
        self.params = params or PerspectiveParams()

        # TPJ activation
        self.tpj_activation = np.zeros(self.params.n_features)

        # Current perspective (self vs agent name)
        self.current_perspective = "self"

        # Perspective representations
        self.perspectives: Dict[str, np.ndarray] = {
            "self": np.random.randn(self.params.n_features) * 0.1
        }

    def add_perspective(self, agent: str, perspective: np.ndarray):
        """Add another agent's perspective"""
        if len(perspective) != self.params.n_features:
            perspective = np.resize(perspective, self.params.n_features)

        self.perspectives[agent] = perspective.copy()

    def switch_perspective(self, to_agent: str) -> Dict:
        """Switch to another's perspective"""
        if to_agent not in self.perspectives and to_agent != "self":
            return {"success": False, "reason": "unknown agent"}

        # TPJ activation during switch
        old_perspective = self.perspectives.get(
            self.current_perspective, np.zeros(self.params.n_features)
        )
        new_perspective = self.perspectives.get(to_agent, np.zeros(self.params.n_features))

        # Switch cost (cognitive effort)
        switch_cost = self.params.switching_cost
        if self.current_perspective == to_agent:
            switch_cost = 0

        # Update TPJ
        self.tpj_activation = np.tanh(
            self.tpj_activation * 0.3 + (new_perspective - old_perspective) * 0.7
        )

        self.current_perspective = to_agent

        return {
            "success": True,
            "from": self.current_perspective,
            "to": to_agent,
            "switch_cost": switch_cost,
            "tpj_activity": np.mean(self.tpj_activation),
        }

    def view_from_perspective(self, stimulus: np.ndarray) -> np.ndarray:
        """View stimulus from current perspective"""
        if len(stimulus) != self.params.n_features:
            stimulus = np.resize(stimulus, self.params.n_features)

        current = self.perspectives.get(self.current_perspective, np.zeros(self.params.n_features))

        # Stimulus filtered through current perspective
        viewed = stimulus * 0.5 + current * 0.5
        return np.tanh(viewed)

    def get_current_perspective(self) -> str:
        """Get whose perspective is currently active"""
        return self.current_perspective


class PerspectiveTaking:
    """Integrated perspective taking system"""

    def __init__(self, params: Optional[PerspectiveParams] = None):
        self.params = params or PerspectiveParams()

        self.self_other = SelfOtherDistinction(params)
        self.tpj = TPJNetwork(params)

        # Visual perspective level (1 = see what they see, 2 = see how they see)
        self.visual_level = 1

        # Affective perspective (feel what they feel)
        self.affective_active = False

    def take_visual_perspective(self, agent: str, scene: np.ndarray, level: int = 1) -> Dict:
        """Take visual perspective of another

        Level 1: What do they see?
        Level 2: How do they see it?
        """
        if len(scene) != self.params.n_features:
            scene = np.resize(scene, self.params.n_features)

        self.visual_level = level

        # Switch to agent's perspective
        self.tpj.switch_perspective(agent)

        if level == 1:
            # Level 1: Simple visibility (what's in their view)
            # Assume agent sees similar scene with occlusions
            occluded = scene * (0.7 + np.random.rand(self.params.n_features) * 0.3)
            their_view = occluded
        else:
            # Level 2: How they interpret what they see
            self.tpj.perspectives.get(agent, np.zeros(self.params.n_features))
            their_view = self.tpj.view_from_perspective(scene)

        return {
            "agent": agent,
            "level": level,
            "their_view": their_view,
            "self_other_distinction": self.self_other.distinguish(agent),
            "tpj_activity": np.mean(self.tpj.tpj_activation),
        }

    def take_affective_perspective(self, agent: str, their_situation: np.ndarray) -> Dict:
        """Take affective perspective - how would they feel?"""
        if len(their_situation) != self.params.n_features:
            their_situation = np.resize(their_situation, self.params.n_features)

        self.affective_active = True

        # Switch perspective
        self.tpj.switch_perspective(agent)

        # Infer their emotional response to situation
        their_perspective = self.tpj.perspectives.get(agent, np.zeros(self.params.n_features))

        # Their feeling = situation × their perspective
        inferred_feeling = np.tanh(their_situation * 0.5 + their_perspective * 0.5)

        # Self-other distinction affects accuracy
        distinction = self.self_other.distinguish(agent)
        self_influence = self.self_other.get_self_influence(agent)

        # Some self-projection (egocentric bias)
        own_response = np.tanh(their_situation * 0.5 + self.self_other.self_representation * 0.5)

        blended = inferred_feeling * distinction + own_response * self_influence

        return {
            "agent": agent,
            "inferred_feeling": blended,
            "pure_inference": inferred_feeling,
            "self_projection": own_response,
            "distinction": distinction,
            "egocentric_bias": self_influence,
        }

    def return_to_self(self):
        """Return to own perspective"""
        self.tpj.switch_perspective("self")
        self.affective_active = False

    def update_model(self, agent: str, observed_state: np.ndarray):
        """Update model of another agent"""
        self.self_other.model_other(agent, observed_state)
        self.tpj.add_perspective(agent, observed_state)

    def update(self, dt: float = 1.0):
        """Update perspective taking system"""
        self.tpj.tpj_activation *= 1 - 0.1 * dt

    def get_state(self) -> Dict:
        """Get perspective taking state"""
        return {
            "current_perspective": self.tpj.get_current_perspective(),
            "visual_level": self.visual_level,
            "affective_active": self.affective_active,
            "tpj_activity": np.mean(self.tpj.tpj_activation),
            "known_perspectives": list(self.tpj.perspectives.keys()),
            "boundary_strength": self.self_other.boundary_strength,
        }
