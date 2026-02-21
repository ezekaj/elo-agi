"""Narrative Self - Autobiographical self and personal identity

Neural basis: mPFC, hippocampus
Core experience: "I am a continuous person with a history and future"
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class NarrativeParams:
    """Parameters for narrative self"""
    n_features: int = 50
    memory_capacity: int = 100
    consolidation_rate: float = 0.1
    retrieval_threshold: float = 0.4


class AutobiographicalMemory:
    """Store and retrieve personal memories

    Supports episodic memories with temporal context
    """

    def __init__(self, params: Optional[NarrativeParams] = None):
        self.params = params or NarrativeParams()

        # Memory storage (list of episodes)
        self.episodes: List[Dict] = []

        # Hippocampal activation
        self.hippocampal_activation = np.zeros(self.params.n_features)

        # Temporal context
        self.current_time = 0

    def encode_episode(self, content: np.ndarray, context: Dict,
                      emotional_salience: float = 0.5) -> int:
        """Encode a new autobiographical memory"""
        if len(content) != self.params.n_features:
            content = np.resize(content, self.params.n_features)

        episode = {
            "content": content.copy(),
            "context": context.copy(),
            "time": self.current_time,
            "salience": emotional_salience,
            "retrieval_count": 0
        }

        self.episodes.append(episode)

        # Hippocampal encoding
        self.hippocampal_activation = np.tanh(
            self.hippocampal_activation * 0.3 + content * 0.7
        )

        # Manage capacity
        if len(self.episodes) > self.params.memory_capacity:
            # Remove least salient, least retrieved
            scores = [e["salience"] + e["retrieval_count"] * 0.1 for e in self.episodes]
            min_idx = np.argmin(scores)
            self.episodes.pop(min_idx)

        self.current_time += 1

        return len(self.episodes) - 1

    def retrieve_by_cue(self, cue: np.ndarray, n: int = 5) -> List[Dict]:
        """Retrieve memories matching cue"""
        if len(cue) != self.params.n_features:
            cue = np.resize(cue, self.params.n_features)

        # Compute similarities
        similarities = []
        for i, episode in enumerate(self.episodes):
            sim = np.dot(cue, episode["content"]) / (
                np.linalg.norm(cue) * np.linalg.norm(episode["content"]) + 1e-8
            )
            # Salience boosts retrieval
            sim *= (1 + episode["salience"] * 0.5)
            similarities.append((i, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top matches above threshold
        results = []
        for idx, sim in similarities[:n]:
            if sim > self.params.retrieval_threshold:
                self.episodes[idx]["retrieval_count"] += 1
                results.append({
                    "episode": self.episodes[idx],
                    "similarity": sim
                })

        return results

    def retrieve_recent(self, n: int = 5) -> List[Dict]:
        """Retrieve most recent memories"""
        recent = self.episodes[-n:] if len(self.episodes) >= n else self.episodes
        return [{"episode": e, "recency": i} for i, e in enumerate(reversed(recent))]

    def get_memory_count(self) -> int:
        """Get number of stored memories"""
        return len(self.episodes)


class SelfConcept:
    """Abstract representation of self

    Includes traits, values, roles, and self-schemas
    """

    def __init__(self, params: Optional[NarrativeParams] = None):
        self.params = params or NarrativeParams()

        # Self-schema (abstract self-representation)
        self.self_schema = np.random.randn(self.params.n_features) * 0.1

        # Trait dimensions
        self.traits: Dict[str, float] = {}

        # Roles and identities
        self.roles: List[str] = []

        # Values (what matters to self)
        self.values: Dict[str, float] = {}

        # mPFC activation (self-referential processing)
        self.mpfc_activation = np.zeros(self.params.n_features)

    def update_trait(self, trait: str, value: float):
        """Update a trait assessment"""
        self.traits[trait] = np.clip(value, -1, 1)

    def add_role(self, role: str):
        """Add a social role/identity"""
        if role not in self.roles:
            self.roles.append(role)

    def set_value(self, value: str, importance: float):
        """Set importance of a value"""
        self.values[value] = np.clip(importance, 0, 1)

    def process_self_relevant(self, stimulus: np.ndarray) -> Dict:
        """Process self-relevant information"""
        if len(stimulus) != self.params.n_features:
            stimulus = np.resize(stimulus, self.params.n_features)

        # mPFC activation for self-reference
        self.mpfc_activation = np.tanh(
            self.mpfc_activation * 0.3 + stimulus * 0.7
        )

        # Self-relevance = similarity to self-schema
        relevance = np.dot(stimulus, self.self_schema) / (
            np.linalg.norm(stimulus) * np.linalg.norm(self.self_schema) + 1e-8
        )

        # Update self-schema if highly self-relevant
        if relevance > 0.5:
            self.self_schema = 0.9 * self.self_schema + 0.1 * stimulus
            self.self_schema = self.self_schema / (np.linalg.norm(self.self_schema) + 1e-8)

        return {
            "self_relevance": relevance,
            "mpfc_activity": np.mean(self.mpfc_activation)
        }

    def get_self_description(self) -> Dict:
        """Get description of self-concept"""
        return {
            "traits": self.traits.copy(),
            "roles": self.roles.copy(),
            "values": self.values.copy(),
            "schema_summary": np.mean(self.self_schema)
        }


class NarrativeSelf:
    """Integrated narrative self

    Combines autobiographical memory with self-concept
    """

    def __init__(self, params: Optional[NarrativeParams] = None):
        self.params = params or NarrativeParams()

        self.memory = AutobiographicalMemory(params)
        self.self_concept = SelfConcept(params)

        # Narrative coherence
        self.narrative_coherence = 0.7

        # Current life chapter/theme
        self.current_theme = "default"

    def experience_event(self, event: np.ndarray, context: Dict,
                        emotional_impact: float = 0.5) -> Dict:
        """Experience and encode a life event"""
        # Process self-relevance
        relevance = self.self_concept.process_self_relevant(event)

        # Encode in autobiographical memory
        memory_idx = self.memory.encode_episode(
            event, context, emotional_salience=emotional_impact
        )

        # Update narrative coherence
        if relevance["self_relevance"] > 0.5:
            self.narrative_coherence = min(1.0, self.narrative_coherence + 0.05)
        else:
            self.narrative_coherence = max(0.3, self.narrative_coherence - 0.02)

        return {
            "memory_encoded": memory_idx,
            "self_relevance": relevance,
            "narrative_coherence": self.narrative_coherence
        }

    def recall_life_period(self, cue: np.ndarray) -> Dict:
        """Recall memories from a life period"""
        memories = self.memory.retrieve_by_cue(cue)

        # Narrative integration
        if memories:
            coherence_boost = len(memories) * 0.02
            self.narrative_coherence = min(1.0, self.narrative_coherence + coherence_boost)

        return {
            "memories": memories,
            "count": len(memories),
            "narrative_coherence": self.narrative_coherence
        }

    def reflect_on_self(self) -> Dict:
        """Engage in self-reflection"""
        # Retrieve self-relevant memories
        self_cue = self.self_concept.self_schema
        relevant_memories = self.memory.retrieve_by_cue(self_cue, n=10)

        # Update self-concept based on memories
        if relevant_memories:
            memory_content = np.mean([
                m["episode"]["content"] for m in relevant_memories
            ], axis=0)
            self.self_concept.process_self_relevant(memory_content)

        return {
            "self_concept": self.self_concept.get_self_description(),
            "relevant_memories": len(relevant_memories),
            "narrative_coherence": self.narrative_coherence
        }

    def get_narrative_self_state(self) -> Dict:
        """Get narrative self state"""
        return {
            "memory_count": self.memory.get_memory_count(),
            "self_concept": self.self_concept.get_self_description(),
            "narrative_coherence": self.narrative_coherence,
            "current_theme": self.current_theme,
            "mpfc_activity": np.mean(self.self_concept.mpfc_activation),
            "hippocampal_activity": np.mean(self.memory.hippocampal_activation)
        }
