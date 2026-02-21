"""Consciousness Network - Integrated conscious awareness

Combines minimal self, narrative self, metacognition, and introspection
Implements global workspace theory aspects
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List

from .minimal_self import MinimalSelf, MinimalSelfParams
from .narrative_self import NarrativeSelf, NarrativeParams
from .metacognition import MetacognitiveSystem, MetaParams
from .introspection import IntrospectionSystem, IntrospectionParams


class GlobalWorkspace:
    """Global workspace for conscious access

    Information in the workspace is globally available
    """

    def __init__(self, n_features: int = 50, capacity: int = 7):
        self.n_features = n_features
        self.capacity = capacity  # Limited conscious capacity

        # Workspace contents (globally broadcast)
        self.workspace: Dict[str, np.ndarray] = {}

        # Activation levels
        self.activations: Dict[str, float] = {}

        # Broadcasting state
        self.is_broadcasting = False

    def submit_for_broadcast(self, name: str, content: np.ndarray,
                            activation: float) -> bool:
        """Submit content to compete for global broadcast"""
        if len(content) != self.n_features:
            content = np.resize(content, self.n_features)

        # If workspace not full, add directly
        if len(self.workspace) < self.capacity:
            self.workspace[name] = content.copy()
            self.activations[name] = activation
            return True

        # Otherwise, compete with existing contents
        min_activation = min(self.activations.values())
        if activation > min_activation:
            # Replace weakest
            min_name = min(self.activations.items(), key=lambda x: x[1])[0]
            del self.workspace[min_name]
            del self.activations[min_name]

            self.workspace[name] = content.copy()
            self.activations[name] = activation
            return True

        return False

    def broadcast(self) -> Dict[str, np.ndarray]:
        """Broadcast workspace contents globally"""
        self.is_broadcasting = True
        return {k: v.copy() for k, v in self.workspace.items()}

    def get_workspace_contents(self) -> List[str]:
        """Get names of contents in workspace"""
        return list(self.workspace.keys())

    def clear_content(self, name: str):
        """Remove content from workspace"""
        if name in self.workspace:
            del self.workspace[name]
            del self.activations[name]

    def decay(self, rate: float = 0.1):
        """Decay all activations"""
        to_remove = []
        for name in self.activations:
            self.activations[name] *= (1 - rate)
            if self.activations[name] < 0.1:
                to_remove.append(name)

        for name in to_remove:
            self.clear_content(name)


class ConsciousnessNetwork:
    """Full consciousness network

    Integrates all components of conscious experience
    Key insight: Metacognition constitutive of consciousness
    """

    def __init__(self, n_features: int = 50):
        self.n_features = n_features

        # Self-awareness hierarchy
        minimal_params = MinimalSelfParams(n_features=n_features)
        self.minimal_self = MinimalSelf(minimal_params)

        narrative_params = NarrativeParams(n_features=n_features)
        self.narrative_self = NarrativeSelf(narrative_params)

        # Metacognition (constitutive of consciousness)
        meta_params = MetaParams(n_features=n_features)
        self.metacognition = MetacognitiveSystem(meta_params)

        # Introspection
        intro_params = IntrospectionParams(n_features=n_features)
        self.introspection = IntrospectionSystem(intro_params)

        # Global workspace
        self.workspace = GlobalWorkspace(n_features)

        # Consciousness level
        self.consciousness_level = 0.5

        # Time tracking
        self.time = 0

    def process_experience(self, sensory_input: np.ndarray,
                          action: Optional[np.ndarray] = None,
                          context: Optional[Dict] = None) -> Dict:
        """Process a conscious experience"""
        if len(sensory_input) != self.n_features:
            sensory_input = np.resize(sensory_input, self.n_features)

        context = context or {}
        results = {"time": self.time}

        # Minimal self processing
        if action is not None:
            predicted = self.minimal_self.perform_action(action)

            # Create mock sensory channels
            features_per = self.n_features // 3
            visual = sensory_input[:features_per]
            tactile = sensory_input[features_per:2*features_per]
            proprio = sensory_input[2*features_per:3*features_per]

            minimal_result = self.minimal_self.receive_feedback(
                sensory_input, visual, tactile, proprio
            )
            results["minimal_self"] = minimal_result

            # Submit to workspace
            self.workspace.submit_for_broadcast(
                "agency", predicted, minimal_result["agency"]["agency"]
            )

        # Narrative self processing
        narrative_result = self.narrative_self.experience_event(
            sensory_input, context, emotional_impact=0.5
        )
        results["narrative_self"] = narrative_result

        # Submit relevant experiences to workspace
        self.workspace.submit_for_broadcast(
            "experience", sensory_input, narrative_result["self_relevance"]["self_relevance"]
        )

        # Metacognitive monitoring
        expected = np.zeros(self.n_features)  # Simplified
        meta_result = self.metacognition.monitor_task(sensory_input, expected)
        results["metacognition"] = meta_result

        # Register states for introspection
        self.introspection.register_mental_state("current_experience", sensory_input, "sensory")

        # Update consciousness level (metacognition is key)
        self._update_consciousness_level(meta_result)

        # Global broadcast
        broadcast = self.workspace.broadcast()
        results["workspace_contents"] = list(broadcast.keys())

        self.time += 1
        return results

    def _update_consciousness_level(self, meta_result: Dict):
        """Update overall consciousness level

        Metacognition is constitutive - consciousness depends on it
        """
        metacognitive_contribution = 1 - meta_result["error_signal"]
        self_integrity = self.minimal_self.self_integrity
        narrative_coherence = self.narrative_self.narrative_coherence

        # Weighted combination (metacognition weighted heavily)
        self.consciousness_level = (
            0.5 * metacognitive_contribution +
            0.25 * self_integrity +
            0.25 * narrative_coherence
        )

    def introspect_current_state(self) -> Dict:
        """Introspect on current conscious state"""
        # Access conscious contents
        contents = self.introspection.get_conscious_contents()

        # Reflect on consciousness
        reflection = self.introspection.reflect_on("current awareness")

        # Metacognitive assessment
        meta_state = self.metacognition.introspect_confidence()

        return {
            "conscious_contents": contents,
            "reflection": reflection,
            "metacognitive_assessment": meta_state,
            "consciousness_level": self.consciousness_level
        }

    def recall_autobiographical(self, cue: np.ndarray) -> Dict:
        """Recall autobiographical memories"""
        return self.narrative_self.recall_life_period(cue)

    def make_decision(self, options: List[np.ndarray],
                     evidence: np.ndarray) -> Dict:
        """Make a conscious decision with metacognitive monitoring"""
        if len(evidence) != self.n_features:
            evidence = np.resize(evidence, self.n_features)

        # Evaluate each option
        evaluations = []
        for i, option in enumerate(options):
            if len(option) != self.n_features:
                option = np.resize(option, self.n_features)

            eval_result = self.metacognition.evaluate_decision(evidence, option)
            evaluations.append({
                "option": i,
                "confidence": eval_result["confidence"]["confidence"]
            })

        # Select highest confidence option
        best = max(evaluations, key=lambda x: x["confidence"])

        # Submit decision to workspace
        self.workspace.submit_for_broadcast(
            "decision", options[best["option"]], best["confidence"]
        )

        return {
            "selected_option": best["option"],
            "confidence": best["confidence"],
            "all_evaluations": evaluations,
            "recommend_more_deliberation": best["confidence"] < 0.6
        }

    def update(self, dt: float = 1.0):
        """Update consciousness network"""
        self.workspace.decay(0.1 * dt)

    def get_consciousness_state(self) -> Dict:
        """Get comprehensive consciousness state"""
        return {
            "consciousness_level": self.consciousness_level,
            "minimal_self": self.minimal_self.get_minimal_self_state(),
            "narrative_self": self.narrative_self.get_narrative_self_state(),
            "metacognition": self.metacognition.get_metacognitive_state(),
            "introspection": self.introspection.get_introspection_state(),
            "workspace_contents": self.workspace.get_workspace_contents(),
            "time": self.time
        }
