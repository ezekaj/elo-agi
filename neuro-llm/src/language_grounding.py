"""
Language Grounding: Connects language to perception and action.

Provides grounding of linguistic symbols in:
- Perceptual experiences (what things look like)
- Actions (what to do)
- Goals (what to achieve)
- World model (how things work)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
import time

from llm_interface import LLMOracle, MockLLM
from semantic_bridge import SemanticBridge, SemanticConfig


@dataclass
class GroundingConfig:
    """Configuration for language grounding."""
    internal_dim: int = 64
    perceptual_dim: int = 32
    action_dim: int = 32
    learning_rate: float = 0.01
    decay_rate: float = 0.99
    min_confidence: float = 0.1


@dataclass
class GroundedConcept:
    """A concept grounded in perception and action."""
    name: str
    text: str
    perceptual_features: np.ndarray  # What it looks like
    action_affordances: np.ndarray   # What you can do with it
    goal_relevance: np.ndarray       # How it relates to goals
    confidence: float = 0.5
    occurrences: int = 1
    last_updated: float = field(default_factory=time.time)

    def update(self, perceptual: np.ndarray, action: np.ndarray, goal: np.ndarray, lr: float = 0.1) -> None:
        """Update grounding based on new experience."""
        self.perceptual_features = (1 - lr) * self.perceptual_features + lr * perceptual
        self.action_affordances = (1 - lr) * self.action_affordances + lr * action
        self.goal_relevance = (1 - lr) * self.goal_relevance + lr * goal
        self.occurrences += 1
        self.confidence = min(1.0, self.confidence + 0.1)
        self.last_updated = time.time()


class LanguageGrounding:
    """
    Grounds language in perception, action, and world knowledge.

    The grounding system:
    1. Associates words/phrases with perceptual features
    2. Links language to action affordances
    3. Connects goals to linguistic descriptions
    4. Learns from experience (observation-language pairs)
    """

    def __init__(
        self,
        llm: Optional[LLMOracle] = None,
        bridge: Optional[SemanticBridge] = None,
        config: Optional[GroundingConfig] = None,
    ):
        self.config = config or GroundingConfig()
        self.llm = llm or MockLLM()
        self.bridge = bridge or SemanticBridge(self.llm)

        # Grounded concepts
        self._concepts: Dict[str, GroundedConcept] = {}

        # Word-level groundings
        self._word_perceptual: Dict[str, np.ndarray] = {}
        self._word_action: Dict[str, np.ndarray] = {}

        # Experience buffer for learning
        self._experience_buffer: List[Tuple[str, np.ndarray, np.ndarray]] = []

        # Statistics
        self._ground_count = 0
        self._unground_count = 0

    def ground(
        self,
        text: str,
        observation: Optional[np.ndarray] = None,
        action: Optional[np.ndarray] = None,
        goal: Optional[np.ndarray] = None,
    ) -> GroundedConcept:
        """
        Ground a text expression in experience.

        Args:
            text: Natural language expression
            observation: Current perceptual observation
            action: Associated action (if any)
            goal: Associated goal (if any)

        Returns:
            Grounded concept
        """
        self._ground_count += 1

        # Normalize text
        text_key = text.lower().strip()

        # Get or create concept
        if text_key in self._concepts:
            concept = self._concepts[text_key]
        else:
            concept = GroundedConcept(
                name=text_key,
                text=text,
                perceptual_features=np.zeros(self.config.perceptual_dim, dtype=np.float32),
                action_affordances=np.zeros(self.config.action_dim, dtype=np.float32),
                goal_relevance=np.zeros(self.config.internal_dim, dtype=np.float32),
            )
            self._concepts[text_key] = concept

        # Update with experience
        if observation is not None:
            perceptual = self._extract_perceptual(observation)
            concept.perceptual_features = (
                (1 - self.config.learning_rate) * concept.perceptual_features +
                self.config.learning_rate * perceptual
            )

        if action is not None:
            action_features = self._extract_action(action)
            concept.action_affordances = (
                (1 - self.config.learning_rate) * concept.action_affordances +
                self.config.learning_rate * action_features
            )

        if goal is not None:
            goal_features = self._resize(goal, self.config.internal_dim)
            concept.goal_relevance = (
                (1 - self.config.learning_rate) * concept.goal_relevance +
                self.config.learning_rate * goal_features
            )

        concept.occurrences += 1
        concept.last_updated = time.time()

        # Ground individual words
        self._ground_words(text, observation, action)

        return concept

    def unground(
        self,
        concept: GroundedConcept,
        modality: str = "all"
    ) -> Dict[str, Any]:
        """
        Generate descriptions from grounded concept.

        Args:
            concept: Grounded concept to describe
            modality: Which aspect to describe ("perceptual", "action", "goal", "all")

        Returns:
            Dictionary of descriptions
        """
        self._unground_count += 1
        descriptions = {}

        if modality in ["perceptual", "all"]:
            # Describe perceptual features
            activation = np.mean(np.abs(concept.perceptual_features))
            if activation > 0.5:
                descriptions['perceptual'] = f"{concept.text} appears prominent and salient"
            elif activation > 0.2:
                descriptions['perceptual'] = f"{concept.text} is moderately visible"
            else:
                descriptions['perceptual'] = f"{concept.text} is barely noticeable"

        if modality in ["action", "all"]:
            # Describe action affordances
            max_action = np.argmax(concept.action_affordances)
            action_names = ["approach", "avoid", "manipulate", "observe"]
            if max_action < len(action_names):
                action_name = action_names[max_action % len(action_names)]
                descriptions['action'] = f"You can {action_name} {concept.text}"
            else:
                descriptions['action'] = f"Interactions with {concept.text} are possible"

        if modality in ["goal", "all"]:
            # Describe goal relevance
            relevance = np.mean(concept.goal_relevance)
            if relevance > 0.5:
                descriptions['goal'] = f"{concept.text} is highly relevant to current goals"
            elif relevance > 0.2:
                descriptions['goal'] = f"{concept.text} may be useful"
            else:
                descriptions['goal'] = f"{concept.text} seems unrelated to goals"

        return descriptions

    def _ground_words(
        self,
        text: str,
        observation: Optional[np.ndarray],
        action: Optional[np.ndarray],
    ) -> None:
        """Ground individual words from text."""
        words = text.lower().split()

        for word in words:
            # Skip common words
            if len(word) < 3 or word in ['the', 'a', 'an', 'is', 'are', 'was', 'were']:
                continue

            if observation is not None:
                perceptual = self._extract_perceptual(observation)
                if word in self._word_perceptual:
                    self._word_perceptual[word] = (
                        0.9 * self._word_perceptual[word] + 0.1 * perceptual
                    )
                else:
                    self._word_perceptual[word] = perceptual.copy()

            if action is not None:
                action_features = self._extract_action(action)
                if word in self._word_action:
                    self._word_action[word] = (
                        0.9 * self._word_action[word] + 0.1 * action_features
                    )
                else:
                    self._word_action[word] = action_features.copy()

    def _extract_perceptual(self, observation: np.ndarray) -> np.ndarray:
        """Extract perceptual features from observation."""
        obs = self._resize(observation, self.config.perceptual_dim)
        return obs.astype(np.float32)

    def _extract_action(self, action: np.ndarray) -> np.ndarray:
        """Extract action features."""
        act = self._resize(action, self.config.action_dim)
        return act.astype(np.float32)

    def _resize(self, vec: np.ndarray, target_dim: int) -> np.ndarray:
        """Resize vector to target dimension."""
        vec = np.asarray(vec, dtype=np.float32).flatten()
        if len(vec) < target_dim:
            return np.pad(vec, (0, target_dim - len(vec)))
        return vec[:target_dim]

    def describe_observation(self, observation: np.ndarray) -> str:
        """
        Generate natural language description of observation.

        Args:
            observation: Perceptual observation vector

        Returns:
            Natural language description
        """
        perceptual = self._extract_perceptual(observation)

        # Find matching concepts
        matches = []
        for name, concept in self._concepts.items():
            similarity = self._cosine_similarity(perceptual, concept.perceptual_features)
            if similarity > 0.3:
                matches.append((similarity, concept))

        matches.sort(key=lambda x: x[0], reverse=True)

        if matches:
            top_concepts = [c.text for _, c in matches[:3]]
            description = "I perceive: " + ", ".join(top_concepts)
        else:
            # Fall back to activation description
            activation = np.mean(np.abs(perceptual))
            if activation > 0.5:
                description = "I perceive something significant"
            elif activation > 0.2:
                description = "I perceive something in the environment"
            else:
                description = "The environment appears quiet"

        return description

    def parse_instruction(self, instruction: str) -> Dict[str, Any]:
        """
        Parse natural language instruction into structured format.

        Args:
            instruction: Natural language instruction

        Returns:
            Structured instruction with action, target, etc.
        """
        # Use LLM to parse
        parsed = self.llm.parse_action(instruction)

        # Enrich with grounded concepts
        instruction_lower = instruction.lower()
        matched_concepts = []

        for name, concept in self._concepts.items():
            if name in instruction_lower:
                matched_concepts.append(concept)

        if matched_concepts:
            parsed['grounded_concepts'] = [c.name for c in matched_concepts]
            parsed['action_affordances'] = np.mean(
                [c.action_affordances for c in matched_concepts], axis=0
            ).tolist()

        return parsed

    def instruction_to_action(self, instruction: str) -> np.ndarray:
        """
        Convert instruction to action vector.

        Args:
            instruction: Natural language instruction

        Returns:
            Action vector
        """
        # Parse instruction
        parsed = self.parse_instruction(instruction)

        # Start with base action from semantic bridge
        action = self.bridge.text_to_action(instruction)

        # Modify based on grounded concepts
        if 'action_affordances' in parsed:
            affordances = np.array(parsed['action_affordances'], dtype=np.float32)
            affordances = self._resize(affordances, self.config.action_dim)
            action[:self.config.action_dim] = (
                0.5 * action[:self.config.action_dim] + 0.5 * affordances
            )

        return action

    def goal_to_text(self, goal: np.ndarray) -> str:
        """
        Describe goal in natural language.

        Args:
            goal: Goal vector

        Returns:
            Natural language goal description
        """
        goal_features = self._resize(goal, self.config.internal_dim)

        # Find concepts relevant to goal
        matches = []
        for name, concept in self._concepts.items():
            relevance = self._cosine_similarity(goal_features, concept.goal_relevance)
            if relevance > 0.2:
                matches.append((relevance, concept))

        matches.sort(key=lambda x: x[0], reverse=True)

        if matches:
            top_goals = [c.text for _, c in matches[:2]]
            return "Goal: achieve " + " and ".join(top_goals)
        else:
            activation = np.mean(np.abs(goal_features))
            if activation > 0.5:
                return "Goal: accomplish an important objective"
            else:
                return "Goal: maintain current state"

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a > 0 and norm_b > 0:
            return float(np.dot(a, b) / (norm_a * norm_b))
        return 0.0

    def learn_from_experience(
        self,
        text: str,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
    ) -> None:
        """
        Learn grounding from experience.

        Args:
            text: Description of experience
            observation: Observation during experience
            action: Action taken
            reward: Reward received
        """
        # Add to buffer
        self._experience_buffer.append((text, observation.copy(), action.copy()))

        # Keep buffer bounded
        if len(self._experience_buffer) > 1000:
            self._experience_buffer = self._experience_buffer[-500:]

        # Ground with reward-weighted learning rate
        lr = self.config.learning_rate * (1 + reward)  # Higher reward = faster learning
        lr = np.clip(lr, 0.001, 0.1)

        self.ground(
            text=text,
            observation=observation,
            action=action,
        )

    def get_concept(self, name: str) -> Optional[GroundedConcept]:
        """Get a grounded concept by name."""
        return self._concepts.get(name.lower().strip())

    def list_concepts(self) -> List[str]:
        """List all grounded concept names."""
        return list(self._concepts.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """Get grounding statistics."""
        return {
            'concept_count': len(self._concepts),
            'word_perceptual_count': len(self._word_perceptual),
            'word_action_count': len(self._word_action),
            'experience_buffer_size': len(self._experience_buffer),
            'ground_count': self._ground_count,
            'unground_count': self._unground_count,
            'bridge_stats': self.bridge.get_statistics(),
        }

    def reset(self) -> None:
        """Reset grounding state (keeps learned concepts)."""
        self._experience_buffer = []
        self.bridge.reset()
