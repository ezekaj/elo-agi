"""Grounded Concepts - Embodied concept representations

Core principle: Abstract concepts are grounded in sensorimotor experience
Key features: Modal simulations, perceptual symbols, embodied semantics
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class GroundingParams:
    """Parameters for concept grounding"""

    n_features: int = 50
    n_modalities: int = 5  # visual, auditory, motor, tactile, proprioceptive
    similarity_threshold: float = 0.5
    activation_decay: float = 0.1
    learning_rate: float = 0.1


class ModalityBindings:
    """Bindings between concepts and sensory modalities

    Each concept has representations in multiple modalities
    """

    def __init__(self, n_modalities: int = 5, features_per_modality: int = 20):
        self.n_modalities = n_modalities
        self.features_per = features_per_modality

        # Modality names
        self.modality_names = ["visual", "auditory", "motor", "tactile", "proprioceptive"]

        # Bindings: concept_name -> {modality -> representation}
        self.bindings: Dict[str, Dict[str, np.ndarray]] = {}

    def add_binding(self, concept: str, modality: str, representation: np.ndarray):
        """Add modality-specific representation for concept"""
        if concept not in self.bindings:
            self.bindings[concept] = {}

        if len(representation) != self.features_per:
            representation = np.resize(representation, self.features_per)

        self.bindings[concept][modality] = representation

    def get_binding(self, concept: str, modality: str) -> Optional[np.ndarray]:
        """Get modality-specific representation"""
        if concept in self.bindings and modality in self.bindings[concept]:
            return self.bindings[concept][modality].copy()
        return None

    def get_all_modalities(self, concept: str) -> Dict[str, np.ndarray]:
        """Get all modality representations for concept"""
        if concept in self.bindings:
            return {k: v.copy() for k, v in self.bindings[concept].items()}
        return {}

    def compute_similarity(self, concept1: str, concept2: str) -> float:
        """Compute similarity between concepts across modalities"""
        if concept1 not in self.bindings or concept2 not in self.bindings:
            return 0.0

        similarities = []
        for modality in self.modality_names:
            if modality in self.bindings[concept1] and modality in self.bindings[concept2]:
                v1 = self.bindings[concept1][modality]
                v2 = self.bindings[concept2][modality]
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0


class GroundedConcept:
    """A single grounded concept

    Represented through modal simulations of sensorimotor experience
    """

    def __init__(self, name: str, params: Optional[GroundingParams] = None):
        self.name = name
        self.params = params or GroundingParams()

        # Feature vector (abstract representation)
        self.features = np.random.randn(self.params.n_features) * 0.1

        # Modal representations
        features_per = self.params.n_features // self.params.n_modalities
        self.modal_features = {
            "visual": np.zeros(features_per),
            "auditory": np.zeros(features_per),
            "motor": np.zeros(features_per),
            "tactile": np.zeros(features_per),
            "proprioceptive": np.zeros(features_per),
        }

        # Activation level
        self.activation = 0.0

        # Associated experiences
        self.experiences: List[np.ndarray] = []

    def ground_in_experience(self, experience: np.ndarray, modality: str):
        """Ground concept in sensorimotor experience"""
        features_per = self.params.n_features // self.params.n_modalities

        if len(experience) != features_per:
            experience = np.resize(experience, features_per)

        # Update modal representation
        if modality in self.modal_features:
            self.modal_features[modality] = 0.7 * self.modal_features[modality] + 0.3 * experience

        self.experiences.append(experience.copy())

        # Update abstract features
        self._update_abstract_features()

    def _update_abstract_features(self):
        """Update abstract features from modal representations"""
        all_modal = np.concatenate(list(self.modal_features.values()))
        self.features = all_modal[: self.params.n_features]

    def activate(self, strength: float = 1.0):
        """Activate concept"""
        self.activation = min(1.0, self.activation + strength)

    def simulate_modality(self, modality: str) -> np.ndarray:
        """Simulate modal experience of concept"""
        if modality in self.modal_features:
            simulation = self.modal_features[modality] * self.activation
            simulation += np.random.randn(len(simulation)) * 0.05
            return simulation
        return np.array([])

    def decay(self, dt: float = 1.0):
        """Decay activation"""
        self.activation *= 1 - self.params.activation_decay * dt

    def get_similarity(self, other: "GroundedConcept") -> float:
        """Compute similarity to another concept"""
        return np.dot(self.features, other.features) / (
            np.linalg.norm(self.features) * np.linalg.norm(other.features) + 1e-8
        )


class ConceptGrounding:
    """System for grounding concepts in sensorimotor experience"""

    def __init__(self, params: Optional[GroundingParams] = None):
        self.params = params or GroundingParams()

        # Concept storage
        self.concepts: Dict[str, GroundedConcept] = {}
        self.bindings = ModalityBindings(
            self.params.n_modalities, self.params.n_features // self.params.n_modalities
        )

        # Current simulation state
        self.active_concepts: List[str] = []

    def create_concept(self, name: str) -> GroundedConcept:
        """Create a new grounded concept"""
        concept = GroundedConcept(name, self.params)
        self.concepts[name] = concept
        return concept

    def ground_concept(self, name: str, experience: np.ndarray, modality: str):
        """Ground concept in experience"""
        if name not in self.concepts:
            self.create_concept(name)

        self.concepts[name].ground_in_experience(experience, modality)
        self.bindings.add_binding(name, modality, experience)

    def activate_concept(self, name: str, strength: float = 1.0) -> Dict[str, np.ndarray]:
        """Activate concept and get modal simulations"""
        if name not in self.concepts:
            return {}

        self.concepts[name].activate(strength)
        if name not in self.active_concepts:
            self.active_concepts.append(name)

        # Generate modal simulations
        simulations = {}
        for modality in ["visual", "auditory", "motor", "tactile", "proprioceptive"]:
            simulations[modality] = self.concepts[name].simulate_modality(modality)

        return simulations

    def find_similar(self, name: str, n: int = 5) -> List[Tuple[str, float]]:
        """Find similar concepts"""
        if name not in self.concepts:
            return []

        target = self.concepts[name]
        similarities = []

        for other_name, other_concept in self.concepts.items():
            if other_name != name:
                sim = target.get_similarity(other_concept)
                similarities.append((other_name, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

    def update(self, dt: float = 1.0):
        """Update all concepts"""
        for concept in self.concepts.values():
            concept.decay(dt)

        # Remove inactive concepts from active list
        self.active_concepts = [
            name for name in self.active_concepts if self.concepts[name].activation > 0.1
        ]

    def get_concept_features(self, name: str) -> Optional[np.ndarray]:
        """Get abstract features for concept"""
        if name in self.concepts:
            return self.concepts[name].features.copy()
        return None
