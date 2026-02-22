"""Situated Cognition - Context-dependent processing

Core principle: Cognition is situated in environment; context shapes processing
Key features: External memory, environmental scaffolding, distributed cognition
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Any


@dataclass
class SituatedParams:
    """Parameters for situated cognition"""

    n_features: int = 50
    context_influence: float = 0.5
    memory_decay: float = 0.05
    scaffolding_strength: float = 0.3


class ExternalMemory:
    """Environment as cognitive extension

    Models how external representations support cognition
    """

    def __init__(self, capacity: int = 20):
        self.capacity = capacity

        # External storage (like notes, tools, displays)
        self.storage: Dict[str, Any] = {}
        self.storage_locations: Dict[str, np.ndarray] = {}

        # Attention to external items
        self.attention = {}

    def store(self, key: str, value: Any, location: Optional[np.ndarray] = None):
        """Store information externally"""
        if len(self.storage) >= self.capacity:
            # Remove least attended item
            if self.attention:
                least_attended = min(self.attention.keys(), key=lambda k: self.attention.get(k, 0))
                del self.storage[least_attended]
                if least_attended in self.storage_locations:
                    del self.storage_locations[least_attended]
                if least_attended in self.attention:
                    del self.attention[least_attended]

        self.storage[key] = value
        if location is not None:
            self.storage_locations[key] = location
        self.attention[key] = 1.0

    def retrieve(self, key: str) -> Any:
        """Retrieve from external memory"""
        if key in self.storage:
            self.attention[key] = min(1.0, self.attention.get(key, 0) + 0.3)
            return self.storage[key]
        return None

    def search_by_location(self, query_location: np.ndarray, threshold: float = 0.5) -> List[str]:
        """Find items near a location"""
        matches = []
        for key, location in self.storage_locations.items():
            distance = np.linalg.norm(query_location - location)
            if distance < threshold:
                matches.append(key)
        return matches

    def decay_attention(self, dt: float = 1.0, decay_rate: float = 0.1):
        """Decay attention to external items"""
        for key in self.attention:
            self.attention[key] *= 1 - decay_rate * dt

    def get_attended_items(self, threshold: float = 0.3) -> List[str]:
        """Get items above attention threshold"""
        return [k for k, v in self.attention.items() if v > threshold]


class SituatedContext:
    """Context representation for situated cognition

    Models how current situation shapes processing
    """

    def __init__(self, params: Optional[SituatedParams] = None):
        self.params = params or SituatedParams()

        # Context features
        self.context_features = np.zeros(self.params.n_features)

        # Context components
        self.physical_context = np.zeros(self.params.n_features // 3)
        self.social_context = np.zeros(self.params.n_features // 3)
        self.task_context = np.zeros(self.params.n_features // 3)

        # Context history
        self.context_history = []

    def set_physical_context(self, features: np.ndarray):
        """Set physical environment context"""
        if len(features) != len(self.physical_context):
            features = np.resize(features, len(self.physical_context))
        self.physical_context = features
        self._update_combined()

    def set_social_context(self, features: np.ndarray):
        """Set social context (who is present, social norms)"""
        if len(features) != len(self.social_context):
            features = np.resize(features, len(self.social_context))
        self.social_context = features
        self._update_combined()

    def set_task_context(self, features: np.ndarray):
        """Set task/goal context"""
        if len(features) != len(self.task_context):
            features = np.resize(features, len(self.task_context))
        self.task_context = features
        self._update_combined()

    def _update_combined(self):
        """Update combined context features"""
        combined = np.concatenate([self.physical_context, self.social_context, self.task_context])
        self.context_features = np.zeros(self.params.n_features)
        n = min(len(combined), self.params.n_features)
        self.context_features[:n] = combined[:n]

        self.context_history.append(self.context_features.copy())

    def modulate_processing(self, input_pattern: np.ndarray) -> np.ndarray:
        """Modulate input processing based on context"""
        if len(input_pattern) != self.params.n_features:
            input_pattern = np.resize(input_pattern, self.params.n_features)

        # Context biases processing
        modulated = input_pattern + self.context_features * self.params.context_influence

        return np.clip(modulated, -1, 1)

    def get_context_similarity(self, other_context: np.ndarray) -> float:
        """Compare to another context"""
        if len(other_context) != len(self.context_features):
            other_context = np.resize(other_context, len(self.context_features))

        return np.dot(self.context_features, other_context) / (
            np.linalg.norm(self.context_features) * np.linalg.norm(other_context) + 1e-8
        )

    def detect_context_change(self, threshold: float = 0.3) -> bool:
        """Detect if context has changed significantly"""
        if len(self.context_history) < 2:
            return False

        prev = self.context_history[-2]
        curr = self.context_history[-1]
        change = np.linalg.norm(curr - prev)

        return change > threshold


class ContextualReasoner:
    """Reasoning adapted to current situation

    Models context-dependent inference and decision making
    """

    def __init__(self, params: Optional[SituatedParams] = None):
        self.params = params or SituatedParams()

        self.context = SituatedContext(params)
        self.external_memory = ExternalMemory()

        # Context-specific knowledge
        self.context_knowledge: Dict[str, List[np.ndarray]] = {}

    def set_context(self, physical: np.ndarray, social: np.ndarray, task: np.ndarray):
        """Set full context"""
        self.context.set_physical_context(physical)
        self.context.set_social_context(social)
        self.context.set_task_context(task)

    def reason_in_context(self, problem: np.ndarray) -> Dict:
        """Reason about problem in current context"""
        if len(problem) != self.params.n_features:
            problem = np.resize(problem, self.params.n_features)

        # Context modulates problem representation
        contextualized_problem = self.context.modulate_processing(problem)

        # Check external memory for relevant information
        external_support = []
        attended = self.external_memory.get_attended_items()
        for key in attended:
            item = self.external_memory.retrieve(key)
            if item is not None:
                external_support.append(item)

        # Generate solution (simplified)
        solution = np.tanh(contextualized_problem + np.random.randn(self.params.n_features) * 0.1)

        # Apply scaffolding from environment
        if external_support:
            scaffolding = np.mean(
                [
                    np.resize(s, self.params.n_features)
                    if isinstance(s, np.ndarray)
                    else np.zeros(self.params.n_features)
                    for s in external_support
                ],
                axis=0,
            )
            solution = solution + scaffolding * self.params.scaffolding_strength

        return {
            "solution": solution,
            "context_influence": self.params.context_influence,
            "external_support_used": len(external_support),
            "context_changed": self.context.detect_context_change(),
        }

    def store_knowledge(self, context_key: str, knowledge: np.ndarray):
        """Store context-specific knowledge"""
        if context_key not in self.context_knowledge:
            self.context_knowledge[context_key] = []
        self.context_knowledge[context_key].append(knowledge.copy())

    def retrieve_contextual_knowledge(self, context_key: str) -> List[np.ndarray]:
        """Retrieve knowledge for specific context"""
        return self.context_knowledge.get(context_key, [])

    def offload_to_environment(self, key: str, value: Any, location: Optional[np.ndarray] = None):
        """Offload cognitive load to environment"""
        self.external_memory.store(key, value, location)

    def update(self, dt: float = 1.0):
        """Update situated cognition state"""
        self.external_memory.decay_attention(dt, self.params.memory_decay)

    def get_state(self) -> Dict:
        """Get reasoner state"""
        return {
            "context_features": self.context.context_features.copy(),
            "external_memory_size": len(self.external_memory.storage),
            "attended_items": self.external_memory.get_attended_items(),
            "context_knowledge_keys": list(self.context_knowledge.keys()),
        }
