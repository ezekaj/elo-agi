"""
Default Mode Network (DMN) - Idea Generation and Imagination

The DMN is active during rest and mind-wandering, enabling:
- Association of disconnected concepts
- Generation of hypothetical scenarios
- Mental simulation
- Spontaneous thought

Based on research showing DMN facilitates creativity by activating
during spontaneous thought and enabling distant associations.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
import random
from collections import defaultdict
import time


@dataclass
class Concept:
    """A concept in semantic memory"""

    id: str
    content: Any
    features: Dict[str, float] = field(default_factory=dict)
    associations: Dict[str, float] = field(default_factory=dict)  # concept_id -> strength
    activation: float = 0.0
    last_activated: float = 0.0


@dataclass
class Association:
    """An association between concepts"""

    source_id: str
    target_id: str
    strength: float
    association_type: str = "semantic"  # semantic, episodic, analogical
    creation_time: float = 0.0


@dataclass
class SpontaneousThought:
    """A thought generated spontaneously by the DMN"""

    concepts: List[str]
    associations_used: List[Association]
    novelty_score: float
    coherence_score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class HypotheticalScenario:
    """A mental simulation of a hypothetical situation"""

    premise: str
    elements: List[Concept]
    transformations: List[str]
    outcome_predictions: List[str]
    plausibility: float


class DefaultModeNetwork:
    """
    Default Mode Network - spontaneous thought and imagination.

    Active during:
    - Rest and mind-wandering
    - Daydreaming
    - Future thinking
    - Creative ideation

    Key functions:
    - Generate spontaneous associations
    - Create hypothetical scenarios
    - Enable distant concept connections
    - Support mental simulation
    """

    def __init__(
        self,
        spreading_activation_decay: float = 0.7,
        association_threshold: float = 0.3,
        novelty_bonus: float = 0.5,
    ):
        self.concepts: Dict[str, Concept] = {}
        self.associations: List[Association] = []
        self.spreading_activation_decay = spreading_activation_decay
        self.association_threshold = association_threshold
        self.novelty_bonus = novelty_bonus

        # Track activation history for novelty detection
        self._activation_history: List[Set[str]] = []
        self._association_index: Dict[str, List[Association]] = defaultdict(list)

        # Mind-wandering state
        self._wandering = False
        self._current_thought_chain: List[str] = []

    def add_concept(
        self, concept_id: str, content: Any, features: Optional[Dict[str, float]] = None
    ) -> Concept:
        """Add a concept to semantic memory"""
        concept = Concept(id=concept_id, content=content, features=features or {})
        self.concepts[concept_id] = concept
        return concept

    def create_association(
        self,
        source_id: str,
        target_id: str,
        strength: float = 0.5,
        association_type: str = "semantic",
    ) -> Optional[Association]:
        """Create association between concepts"""
        if source_id not in self.concepts or target_id not in self.concepts:
            return None

        assoc = Association(
            source_id=source_id,
            target_id=target_id,
            strength=strength,
            association_type=association_type,
            creation_time=time.time(),
        )

        self.associations.append(assoc)
        self._association_index[source_id].append(assoc)

        # Bidirectional
        reverse_assoc = Association(
            source_id=target_id,
            target_id=source_id,
            strength=strength * 0.8,  # Slightly weaker reverse
            association_type=association_type,
            creation_time=time.time(),
        )
        self.associations.append(reverse_assoc)
        self._association_index[target_id].append(reverse_assoc)

        # Update concept association maps
        self.concepts[source_id].associations[target_id] = strength
        self.concepts[target_id].associations[source_id] = strength * 0.8

        return assoc

    def spreading_activation(
        self, seed_concepts: List[str], steps: int = 3, top_k: int = 10
    ) -> Dict[str, float]:
        """
        Spread activation through concept network.

        This is how the DMN generates associations - activation
        spreads from seed concepts to related concepts.
        """
        # Initialize activation
        activations = {cid: 0.0 for cid in self.concepts}
        for seed in seed_concepts:
            if seed in activations:
                activations[seed] = 1.0

        # Spread activation
        for step in range(steps):
            new_activations = activations.copy()

            for concept_id, activation in activations.items():
                if activation < 0.01:
                    continue

                # Spread to associated concepts
                for assoc in self._association_index.get(concept_id, []):
                    spread = activation * assoc.strength * self.spreading_activation_decay
                    new_activations[assoc.target_id] = max(new_activations[assoc.target_id], spread)

            activations = new_activations

        # Update concept activations
        for cid, act in activations.items():
            self.concepts[cid].activation = act
            if act > 0.1:
                self.concepts[cid].last_activated = time.time()

        # Return top-k activated concepts
        sorted_activations = sorted(activations.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_activations[:top_k])

    def generate_spontaneous_thought(
        self, seed: Optional[str] = None, association_steps: int = 4
    ) -> SpontaneousThought:
        """
        Generate a spontaneous thought through mind-wandering.

        The DMN naturally generates thoughts by following
        associative chains, sometimes making distant connections.
        """
        # Choose seed
        if seed is None:
            # Random seed weighted by recency
            weights = []
            concept_ids = list(self.concepts.keys())
            for cid in concept_ids:
                recency = time.time() - self.concepts[cid].last_activated
                weight = 1.0 / (1.0 + recency / 3600)  # Decay over hours
                weights.append(weight)

            if weights:
                weights = np.array(weights) / sum(weights)
                seed = np.random.choice(concept_ids, p=weights)
            else:
                return SpontaneousThought([], [], 0.0, 0.0)

        # Follow association chain
        thought_chain = [seed]
        associations_used = []
        current = seed

        for _ in range(association_steps):
            # Get associations from current concept
            assocs = self._association_index.get(current, [])
            if not assocs:
                break

            # Probabilistic selection with novelty bonus
            candidates = []
            probs = []

            for assoc in assocs:
                if assoc.target_id in thought_chain:
                    continue  # Avoid loops

                prob = assoc.strength

                # Novelty bonus for distant/unusual associations
                if assoc.target_id not in self._get_recent_activations():
                    prob *= 1.0 + self.novelty_bonus

                candidates.append(assoc)
                probs.append(prob)

            if not candidates:
                break

            # Normalize and sample
            probs = np.array(probs) / sum(probs)
            chosen = np.random.choice(len(candidates), p=probs)

            assoc = candidates[chosen]
            associations_used.append(assoc)
            thought_chain.append(assoc.target_id)
            current = assoc.target_id

        # Compute novelty (how unusual is this combination?)
        novelty = self._compute_novelty(thought_chain)

        # Compute coherence (do concepts fit together?)
        coherence = self._compute_coherence(thought_chain)

        # Record activation
        self._activation_history.append(set(thought_chain))
        if len(self._activation_history) > 100:
            self._activation_history.pop(0)

        self._current_thought_chain = thought_chain

        return SpontaneousThought(
            concepts=thought_chain,
            associations_used=associations_used,
            novelty_score=novelty,
            coherence_score=coherence,
        )

    def _get_recent_activations(self, window: int = 20) -> Set[str]:
        """Get recently activated concepts"""
        recent = set()
        for activation_set in self._activation_history[-window:]:
            recent.update(activation_set)
        return recent

    def _compute_novelty(self, concepts: List[str]) -> float:
        """
        Compute novelty of concept combination.

        Novel = concepts rarely combined before
        """
        if len(concepts) < 2:
            return 0.0

        concept_set = set(concepts)

        # Check historical co-activations
        co_occurrence_count = 0
        for past_activation in self._activation_history:
            overlap = len(concept_set & past_activation)
            if overlap >= 2:
                co_occurrence_count += 1

        # Low co-occurrence = high novelty
        novelty = 1.0 - (co_occurrence_count / max(len(self._activation_history), 1))
        return novelty

    def _compute_coherence(self, concepts: List[str]) -> float:
        """
        Compute coherence of concept combination.

        Coherent = concepts have meaningful connections
        """
        if len(concepts) < 2:
            return 1.0

        # Average association strength between adjacent concepts
        total_strength = 0.0
        pairs = 0

        for i in range(len(concepts) - 1):
            c1, c2 = concepts[i], concepts[i + 1]
            if c1 in self.concepts and c2 in self.concepts[c1].associations:
                total_strength += self.concepts[c1].associations[c2]
                pairs += 1

        return total_strength / max(pairs, 1)

    def generate_hypothetical_scenario(
        self, premise: str, seed_concepts: List[str], transformations: List[str]
    ) -> HypotheticalScenario:
        """
        Generate a hypothetical scenario (mental simulation).

        The DMN excels at "what if" thinking - imagining
        counterfactual or future scenarios.
        """
        # Activate seed concepts
        activations = self.spreading_activation(seed_concepts, steps=2)

        # Get activated concepts as scenario elements
        elements = [self.concepts[cid] for cid in activations.keys() if cid in self.concepts]

        # Generate outcome predictions based on associations
        outcome_predictions = []
        for transform in transformations:
            # Simple prediction: what concepts follow from this state?
            prediction = f"If {premise} with {transform}, then likely: "

            # Find concepts associated with transformation
            related = []
            for cid, activation in activations.items():
                if activation > 0.3:
                    related.append(self.concepts[cid].content)

            prediction += ", ".join(str(r) for r in related[:3])
            outcome_predictions.append(prediction)

        # Compute plausibility based on coherence
        plausibility = self._compute_coherence(list(activations.keys())[:5])

        return HypotheticalScenario(
            premise=premise,
            elements=elements[:10],
            transformations=transformations,
            outcome_predictions=outcome_predictions,
            plausibility=plausibility,
        )

    def find_distant_associations(
        self, concept_id: str, min_distance: int = 3, max_results: int = 5
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Find distant but potentially creative associations.

        Creative insight often comes from connecting concepts
        that are not directly associated but share hidden links.
        """
        if concept_id not in self.concepts:
            return []

        # BFS to find distant concepts
        visited = {concept_id: (0, [concept_id])}
        queue = [(concept_id, 0, [concept_id])]

        while queue:
            current, distance, path = queue.pop(0)

            if distance >= min_distance + 2:
                continue

            for assoc in self._association_index.get(current, []):
                target = assoc.target_id
                if target not in visited:
                    new_path = path + [target]
                    visited[target] = (distance + 1, new_path)
                    queue.append((target, distance + 1, new_path))

        # Filter to distant concepts
        distant = [
            (cid, dist, path) for cid, (dist, path) in visited.items() if dist >= min_distance
        ]

        # Score by novelty potential
        scored = []
        for cid, dist, path in distant:
            # Longer paths = more creative potential
            # But also need some semantic coherence
            coherence = self._compute_coherence(path)
            score = dist * 0.3 + coherence * 0.7  # Balance novelty and coherence
            scored.append((cid, score, path))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_results]

    def mind_wander(self, duration_steps: int = 10) -> List[SpontaneousThought]:
        """
        Simulate mind-wandering - unconstrained thought generation.

        During mind-wandering, the DMN freely generates thoughts
        without external task constraints.
        """
        self._wandering = True
        thoughts = []

        current_seed = None

        for _ in range(duration_steps):
            # Generate spontaneous thought
            thought = self.generate_spontaneous_thought(
                seed=current_seed, association_steps=random.randint(2, 5)
            )
            thoughts.append(thought)

            # Sometimes continue from current thought, sometimes jump
            if random.random() < 0.7 and thought.concepts:
                # Continue from end of current thought
                current_seed = thought.concepts[-1]
            else:
                # Random jump
                current_seed = None

        self._wandering = False
        return thoughts

    def get_activation_state(self) -> Dict[str, float]:
        """Get current activation levels of all concepts"""
        return {cid: c.activation for cid, c in self.concepts.items()}

    def reset_activations(self):
        """Reset all activations to zero"""
        for concept in self.concepts.values():
            concept.activation = 0.0
