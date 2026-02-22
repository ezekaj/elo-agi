"""Minimal Self - Basic sense of agency and ownership

Neural basis: Premotor cortex, parietal cortex
Core experiences: "I am the one acting" (agency), "This is my body" (ownership)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class MinimalSelfParams:
    """Parameters for minimal self"""

    n_features: int = 50
    agency_threshold: float = 0.7
    ownership_threshold: float = 0.6
    prediction_window: int = 10
    integration_rate: float = 0.3


class AgencyDetector:
    """Detect sense of agency through action-outcome matching

    Agency = feeling that "I caused this outcome"
    Based on comparing predicted vs actual outcomes
    """

    def __init__(self, params: Optional[MinimalSelfParams] = None):
        self.params = params or MinimalSelfParams()

        # Premotor cortex activation
        self.premotor_activation = np.zeros(self.params.n_features)

        # Forward model for outcome prediction
        self.forward_model = np.random.randn(self.params.n_features, self.params.n_features) * 0.1

        # Recent actions and outcomes
        self.action_history: List[np.ndarray] = []
        self.prediction_history: List[np.ndarray] = []
        self.outcome_history: List[np.ndarray] = []

        # Current agency level
        self.agency_level = 0.5

    def initiate_action(self, action: np.ndarray) -> np.ndarray:
        """Initiate an action and predict outcome"""
        if len(action) != self.params.n_features:
            action = np.resize(action, self.params.n_features)

        # Premotor processes action
        self.premotor_activation = np.tanh(self.premotor_activation * 0.3 + action * 0.7)

        # Predict outcome using forward model
        predicted_outcome = np.tanh(np.dot(self.forward_model, action))

        # Store for comparison
        self.action_history.append(action.copy())
        self.prediction_history.append(predicted_outcome.copy())

        # Limit history
        if len(self.action_history) > self.params.prediction_window:
            self.action_history.pop(0)
            self.prediction_history.pop(0)
            self.outcome_history.pop(0) if self.outcome_history else None

        return predicted_outcome

    def receive_outcome(self, outcome: np.ndarray) -> Dict:
        """Receive actual outcome and compute agency"""
        if len(outcome) != self.params.n_features:
            outcome = np.resize(outcome, self.params.n_features)

        self.outcome_history.append(outcome.copy())

        # Get most recent prediction
        if not self.prediction_history:
            return {"agency": 0.5, "match": 0.0}

        predicted = self.prediction_history[-1]

        # Compute prediction error
        error = np.linalg.norm(predicted - outcome)
        max_error = np.sqrt(2 * self.params.n_features)  # Max possible error

        # Agency inversely related to error
        match_quality = 1 - (error / max_error)

        # Update agency level
        if match_quality > self.params.agency_threshold:
            self.agency_level = min(1.0, self.agency_level + 0.1)
        else:
            self.agency_level = max(0.0, self.agency_level - 0.1)

        # Update forward model
        if self.action_history:
            action = self.action_history[-1]
            error_signal = outcome - predicted
            self.forward_model += 0.01 * np.outer(error_signal, action)

        return {
            "agency": self.agency_level,
            "match": match_quality,
            "prediction_error": error,
            "premotor_activity": np.mean(self.premotor_activation),
        }

    def get_agency_level(self) -> float:
        """Get current sense of agency"""
        return self.agency_level


class OwnershipProcessor:
    """Process body ownership through multisensory integration

    Ownership = feeling that "this body is mine"
    Based on integrating visual, tactile, proprioceptive signals
    """

    def __init__(self, params: Optional[MinimalSelfParams] = None):
        self.params = params or MinimalSelfParams()

        # Parietal cortex activation
        self.parietal_activation = np.zeros(self.params.n_features)

        # Body representation
        self.body_schema = np.random.randn(self.params.n_features) * 0.1

        # Sensory channels
        features_per = self.params.n_features // 3
        self.visual_channel = np.zeros(features_per)
        self.tactile_channel = np.zeros(features_per)
        self.proprioceptive_channel = np.zeros(features_per)

        # Ownership level
        self.ownership_level = 0.8

    def receive_multisensory(
        self, visual: np.ndarray, tactile: np.ndarray, proprioceptive: np.ndarray
    ) -> Dict:
        """Integrate multisensory signals about body"""
        features_per = self.params.n_features // 3

        # Resize inputs
        if len(visual) != features_per:
            visual = np.resize(visual, features_per)
        if len(tactile) != features_per:
            tactile = np.resize(tactile, features_per)
        if len(proprioceptive) != features_per:
            proprioceptive = np.resize(proprioceptive, features_per)

        # Update channels
        self.visual_channel = 0.5 * self.visual_channel + 0.5 * visual
        self.tactile_channel = 0.5 * self.tactile_channel + 0.5 * tactile
        self.proprioceptive_channel = 0.5 * self.proprioceptive_channel + 0.5 * proprioceptive

        # Parietal integration
        raw = np.concatenate(
            [self.visual_channel, self.tactile_channel, self.proprioceptive_channel]
        )
        combined = np.zeros(self.params.n_features)
        n = min(len(raw), self.params.n_features)
        combined[:n] = raw[:n]

        self.parietal_activation = np.tanh(self.parietal_activation * 0.3 + combined * 0.7)

        # Check coherence with body schema
        coherence = np.dot(self.parietal_activation, self.body_schema) / (
            np.linalg.norm(self.parietal_activation) * np.linalg.norm(self.body_schema) + 1e-8
        )

        # Ownership based on coherence
        if coherence > self.params.ownership_threshold:
            self.ownership_level = min(1.0, self.ownership_level + 0.05)
        else:
            self.ownership_level = max(0.3, self.ownership_level - 0.05)

        # Update body schema
        self.body_schema = (
            1 - self.params.integration_rate
        ) * self.body_schema + self.params.integration_rate * self.parietal_activation

        return {
            "ownership": self.ownership_level,
            "coherence": coherence,
            "parietal_activity": np.mean(self.parietal_activation),
        }

    def rubber_hand_illusion(
        self, visual_hand: np.ndarray, tactile_hand: np.ndarray, synchronous: bool = True
    ) -> Dict:
        """Simulate rubber hand illusion paradigm"""
        features_per = self.params.n_features // 3

        # Synchronous stimulation increases ownership of visual hand
        if synchronous:
            # Visual and tactile match -> ownership transfer
            coherence = 0.8
            self.ownership_level = min(1.0, self.ownership_level + 0.1)
        else:
            # Asynchronous -> no transfer
            coherence = 0.3

        return {
            "illusion_strength": coherence if synchronous else 0.2,
            "ownership_of_rubber_hand": self.ownership_level if synchronous else 0.1,
            "synchronous": synchronous,
        }

    def get_ownership_level(self) -> float:
        """Get current sense of ownership"""
        return self.ownership_level


class MinimalSelf:
    """Integrated minimal self system

    Combines agency and ownership for basic self-awareness
    """

    def __init__(self, params: Optional[MinimalSelfParams] = None):
        self.params = params or MinimalSelfParams()

        self.agency = AgencyDetector(params)
        self.ownership = OwnershipProcessor(params)

        # Overall minimal self integrity
        self.self_integrity = 0.7

    def perform_action(self, action: np.ndarray) -> np.ndarray:
        """Initiate an action"""
        return self.agency.initiate_action(action)

    def receive_feedback(
        self,
        outcome: np.ndarray,
        visual: np.ndarray,
        tactile: np.ndarray,
        proprioceptive: np.ndarray,
    ) -> Dict:
        """Receive sensory feedback from action"""
        agency_result = self.agency.receive_outcome(outcome)
        ownership_result = self.ownership.receive_multisensory(visual, tactile, proprioceptive)

        # Update self integrity
        self.self_integrity = 0.5 * agency_result["agency"] + 0.5 * ownership_result["ownership"]

        return {
            "agency": agency_result,
            "ownership": ownership_result,
            "self_integrity": self.self_integrity,
        }

    def get_minimal_self_state(self) -> Dict:
        """Get minimal self state"""
        return {
            "agency_level": self.agency.get_agency_level(),
            "ownership_level": self.ownership.get_ownership_level(),
            "self_integrity": self.self_integrity,
            "premotor_activity": np.mean(self.agency.premotor_activation),
            "parietal_activity": np.mean(self.ownership.parietal_activation),
        }
