"""
Curiosity and Exploration

Implements the neural mechanisms of curiosity:
- Nucleus accumbens: Reward anticipation for information
- Hippocampus: Memory encoding boost from curiosity
- Ventral tegmental area: Dopamine release for novelty
- Prefrontal cortex: Information-seeking decisions

Key findings from neuroscience:
- Novelty activates dopamine neurons even without reward prediction
- Curiosity enhances memory encoding through hippocampal activation
- Information has intrinsic value independent of utility
- Dopamine is the "neuromodulator of exploration"
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class CuriosityType(Enum):
    """Types of curiosity/information-seeking"""

    PERCEPTUAL = "perceptual"  # Novel sensory stimuli
    EPISTEMIC = "epistemic"  # Knowledge gaps
    SPECIFIC = "specific"  # Targeted information need
    DIVERSIVE = "diversive"  # General exploration/boredom relief


@dataclass
class InformationPacket:
    """A piece of information that can satisfy curiosity"""

    content: np.ndarray
    novelty: float
    relevance: float
    uncertainty_reduction: float
    source: str = ""


@dataclass
class CuriosityState:
    """Current state of the curiosity system"""

    overall_level: float  # General curiosity drive
    specific_targets: List[str]  # What we're curious about
    knowledge_gaps: List[str]  # Identified unknowns
    boredom_level: float  # Drives diversive curiosity
    recent_discoveries: int  # Fuels further curiosity


class NoveltyDetector:
    """Detects novel stimuli that trigger curiosity.

    Novelty activates dopamine neurons in VTA even without
    explicit reward prediction.
    """

    def __init__(self, input_dim: int, memory_capacity: int = 1000, novelty_threshold: float = 0.3):
        self.input_dim = input_dim
        self.memory_capacity = memory_capacity
        self.novelty_threshold = novelty_threshold

        # Memory of experienced stimuli
        self.experience_memory: deque = deque(maxlen=memory_capacity)

        # Prototype representations (compressed experience)
        self.prototypes: List[np.ndarray] = []
        self.max_prototypes = 50

        # Novelty history
        self.novelty_history: deque = deque(maxlen=500)

        # Habituation (repeated exposure reduces novelty)
        self.habituation_map: Dict[tuple, int] = {}

    def compute_novelty(self, stimulus: np.ndarray) -> float:
        """Compute novelty of a stimulus.

        Novelty = distance from known experiences.
        """
        if len(self.experience_memory) < 5:
            return 1.0  # Everything is novel initially

        # Distance to nearest memory
        min_dist = float("inf")
        for memory in self.experience_memory:
            dist = np.linalg.norm(stimulus - memory)
            min_dist = min(min_dist, dist)

        # Distance to nearest prototype
        for proto in self.prototypes:
            dist = np.linalg.norm(stimulus - proto)
            min_dist = min(min_dist, dist)

        # Normalize novelty
        mean_dist = np.mean([np.linalg.norm(m) for m in self.experience_memory])
        novelty = min_dist / (mean_dist + 1e-8)
        novelty = np.clip(novelty, 0, 1)

        # Apply habituation
        key = self._hash_stimulus(stimulus)
        exposure_count = self.habituation_map.get(key, 0)
        habituation_factor = 1.0 / (1.0 + 0.2 * exposure_count)
        novelty *= habituation_factor

        return novelty

    def _hash_stimulus(self, stimulus: np.ndarray) -> tuple:
        """Create hashable key for stimulus."""
        return tuple(np.round(stimulus, 1))

    def observe(self, stimulus: np.ndarray) -> float:
        """Observe a stimulus and return its novelty."""
        novelty = self.compute_novelty(stimulus)

        # Add to memory
        self.experience_memory.append(stimulus.copy())
        self.novelty_history.append(novelty)

        # Update habituation
        key = self._hash_stimulus(stimulus)
        self.habituation_map[key] = self.habituation_map.get(key, 0) + 1

        # Update prototypes if very novel
        if novelty > 0.7:
            self._add_prototype(stimulus)

        return novelty

    def _add_prototype(self, stimulus: np.ndarray) -> None:
        """Add a new prototype (representative experience)."""
        if len(self.prototypes) < self.max_prototypes:
            self.prototypes.append(stimulus.copy())
        else:
            # Replace least representative prototype
            min_coverage = float("inf")
            min_idx = 0
            for i, proto in enumerate(self.prototypes):
                # Count how many memories this prototype represents
                coverage = sum(1 for m in self.experience_memory if np.linalg.norm(m - proto) < 0.5)
                if coverage < min_coverage:
                    min_coverage = coverage
                    min_idx = i
            self.prototypes[min_idx] = stimulus.copy()

    def get_novelty_trend(self, window: int = 50) -> float:
        """Get recent trend in novelty (increasing = more novel environment)."""
        if len(self.novelty_history) < window:
            return 0.0

        recent = list(self.novelty_history)[-window:]
        half = window // 2

        early_mean = np.mean(recent[:half])
        late_mean = np.mean(recent[half:])

        return late_mean - early_mean

    def is_novel(self, stimulus: np.ndarray) -> bool:
        """Check if stimulus exceeds novelty threshold."""
        return self.compute_novelty(stimulus) > self.novelty_threshold


class InformationValue:
    """Computes intrinsic value of information.

    Information has value independent of instrumental utility.
    Value comes from:
    - Uncertainty reduction
    - Knowledge gap filling
    - Prediction improvement
    """

    def __init__(
        self,
        base_curiosity: float = 0.5,
        uncertainty_weight: float = 1.0,
        relevance_weight: float = 0.5,
    ):
        self.base_curiosity = base_curiosity
        self.uncertainty_weight = uncertainty_weight
        self.relevance_weight = relevance_weight

        # Current knowledge state (uncertainty about different topics)
        self.uncertainty_map: Dict[str, float] = {}

        # Information seeking history
        self.info_history: deque = deque(maxlen=200)

    def set_uncertainty(self, topic: str, uncertainty: float) -> None:
        """Set uncertainty level for a topic."""
        self.uncertainty_map[topic] = np.clip(uncertainty, 0, 1)

    def compute_information_value(
        self, info: InformationPacket, topics: Optional[List[str]] = None
    ) -> float:
        """Compute intrinsic value of information.

        Value = novelty + uncertainty_reduction + relevance
        """
        value = 0.0

        # Novelty component
        value += info.novelty * self.base_curiosity

        # Uncertainty reduction component
        value += info.uncertainty_reduction * self.uncertainty_weight

        # Relevance to current goals
        value += info.relevance * self.relevance_weight

        # Bonus if addresses known knowledge gap
        if topics:
            for topic in topics:
                if topic in self.uncertainty_map:
                    gap_size = self.uncertainty_map[topic]
                    value += 0.3 * gap_size

        return value

    def record_information_received(self, info: InformationPacket, satisfaction: float) -> None:
        """Record received information and satisfaction level."""
        self.info_history.append(
            {
                "novelty": info.novelty,
                "uncertainty_reduction": info.uncertainty_reduction,
                "satisfaction": satisfaction,
            }
        )

        # Update uncertainties if info was relevant
        if info.source and info.source in self.uncertainty_map:
            current = self.uncertainty_map[info.source]
            self.uncertainty_map[info.source] = current * (1 - info.uncertainty_reduction)

    def get_curiosity_satisfaction(self) -> float:
        """Get how well curiosity has been satisfied recently."""
        if len(self.info_history) < 5:
            return 0.5

        recent = list(self.info_history)[-20:]
        return np.mean([r["satisfaction"] for r in recent])

    def identify_knowledge_gaps(self, threshold: float = 0.5) -> List[str]:
        """Identify topics with high uncertainty (knowledge gaps)."""
        gaps = []
        for topic, uncertainty in self.uncertainty_map.items():
            if uncertainty > threshold:
                gaps.append(topic)
        return sorted(gaps, key=lambda t: self.uncertainty_map[t], reverse=True)


class ExplorationController:
    """Controls exploration vs exploitation balance.

    Manages the trade-off between:
    - Exploiting known good options
    - Exploring to discover new options

    Dopamine modulates this balance.
    """

    def __init__(
        self,
        base_exploration: float = 0.3,
        curiosity_boost: float = 0.2,
        boredom_threshold: float = 0.7,
    ):
        self.base_exploration = base_exploration
        self.curiosity_boost = curiosity_boost
        self.boredom_threshold = boredom_threshold

        # Boredom from repetitive experience
        self.boredom_level = 0.0
        self.recent_actions: deque = deque(maxlen=50)

        # Exploration success history
        self.exploration_outcomes: deque = deque(maxlen=100)

    def compute_exploration_rate(
        self, curiosity_level: float, dopamine_level: float, uncertainty: float
    ) -> float:
        """Compute current exploration rate.

        Args:
            curiosity_level: Current curiosity drive
            dopamine_level: Tonic dopamine level
            uncertainty: Environmental uncertainty

        Returns:
            Probability of exploring vs exploiting
        """
        rate = self.base_exploration

        # Curiosity increases exploration
        rate += self.curiosity_boost * curiosity_level

        # Dopamine modulates exploration
        rate += 0.1 * (dopamine_level - 0.5)

        # Uncertainty increases exploration
        rate += 0.15 * uncertainty

        # Boredom strongly increases exploration
        if self.boredom_level > self.boredom_threshold:
            rate += 0.3 * (self.boredom_level - self.boredom_threshold)

        return np.clip(rate, 0.05, 0.95)

    def should_explore(
        self, curiosity_level: float, dopamine_level: float, uncertainty: float
    ) -> bool:
        """Decide whether to explore or exploit."""
        rate = self.compute_exploration_rate(curiosity_level, dopamine_level, uncertainty)
        return np.random.random() < rate

    def update_boredom(self, action: np.ndarray) -> None:
        """Update boredom level based on action repetitiveness."""
        self.recent_actions.append(action.copy())

        if len(self.recent_actions) < 10:
            self.boredom_level = 0.0
            return

        # Compute action diversity
        actions = np.array(list(self.recent_actions))
        action_variance = np.mean(np.var(actions, axis=0))

        # Low variance = repetitive = boring
        self.boredom_level = 1.0 / (1.0 + action_variance * 10)

    def record_exploration_outcome(self, value_discovered: float) -> None:
        """Record outcome of exploration."""
        self.exploration_outcomes.append(value_discovered)

    def get_exploration_value(self) -> float:
        """Get expected value of exploration based on history."""
        if len(self.exploration_outcomes) < 5:
            return 0.5

        return np.mean(list(self.exploration_outcomes))


class CuriosityModule:
    """Complete curiosity system integrating all components.

    Implements:
    - Novelty detection (VTA dopamine)
    - Information valuation (NAcc, PFC)
    - Memory enhancement (Hippocampus)
    - Exploration control
    """

    def __init__(
        self, state_dim: int, base_curiosity: float = 0.5, memory_boost_factor: float = 1.5
    ):
        self.state_dim = state_dim
        self.base_curiosity = base_curiosity
        self.memory_boost_factor = memory_boost_factor

        # Components
        self.novelty_detector = NoveltyDetector(state_dim)
        self.info_value = InformationValue(base_curiosity=base_curiosity)
        self.exploration_controller = ExplorationController()

        # Current curiosity state
        self.curiosity_level = base_curiosity
        self.specific_curiosities: Dict[str, float] = {}

        # Memory (hippocampal) - stores important experiences
        self.memory_store: List[Tuple[np.ndarray, float]] = []  # (state, importance)
        self.max_memories = 500

        # Recent curiosity events
        self.curiosity_history: deque = deque(maxlen=200)

    def process_stimulus(
        self,
        stimulus: np.ndarray,
        action: Optional[np.ndarray] = None,
        context: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """Process a stimulus through the curiosity system.

        Args:
            stimulus: Input stimulus/state
            action: Action that led to this stimulus
            context: Additional context (goals, topics, etc.)

        Returns:
            Dictionary with curiosity-related signals
        """
        # Detect novelty
        novelty = self.novelty_detector.observe(stimulus)

        # Create information packet
        info = InformationPacket(
            content=stimulus,
            novelty=novelty,
            relevance=0.5,  # Would be computed from context
            uncertainty_reduction=novelty * 0.5,
        )

        # Compute information value
        topics = context.get("topics", []) if context else []
        info_value = self.info_value.compute_information_value(info, topics)

        # Update boredom if action provided
        if action is not None:
            self.exploration_controller.update_boredom(action)

        # Update curiosity level
        self._update_curiosity(novelty, info_value)

        # Determine memory encoding strength (curiosity enhances memory)
        memory_strength = self._compute_memory_strength(novelty, info_value)

        # Store in memory if important enough
        if memory_strength > 0.5:
            self._store_memory(stimulus, memory_strength)

        # Record event
        event = {
            "novelty": novelty,
            "info_value": info_value,
            "curiosity_level": self.curiosity_level,
            "memory_strength": memory_strength,
            "boredom": self.exploration_controller.boredom_level,
        }
        self.curiosity_history.append(event)

        return event

    def _update_curiosity(self, novelty: float, info_value: float) -> None:
        """Update overall curiosity level."""
        # Novelty increases curiosity
        novelty_effect = 0.1 * novelty

        # Information satisfaction decreases curiosity temporarily
        if info_value > 0.5:
            satisfaction_effect = -0.05 * info_value
        else:
            satisfaction_effect = 0

        # Boredom increases curiosity
        boredom_effect = 0.05 * self.exploration_controller.boredom_level

        # Update
        self.curiosity_level += novelty_effect + satisfaction_effect + boredom_effect

        # Natural decay toward base level
        self.curiosity_level += 0.02 * (self.base_curiosity - self.curiosity_level)

        self.curiosity_level = np.clip(self.curiosity_level, 0.1, 1.0)

    def _compute_memory_strength(self, novelty: float, info_value: float) -> float:
        """Compute how strongly to encode in memory.

        Curiosity enhances memory encoding (hippocampal activation).
        """
        base_strength = 0.3

        # Novelty strongly enhances encoding
        novelty_boost = self.memory_boost_factor * novelty

        # Curiosity state enhances encoding
        curiosity_boost = 0.5 * self.curiosity_level

        # Information value enhances encoding
        value_boost = 0.3 * info_value

        strength = base_strength + novelty_boost + curiosity_boost + value_boost
        return np.clip(strength, 0, 1)

    def _store_memory(self, state: np.ndarray, importance: float) -> None:
        """Store experience in memory."""
        self.memory_store.append((state.copy(), importance))

        # Prune if too many memories
        if len(self.memory_store) > self.max_memories:
            # Remove least important
            self.memory_store.sort(key=lambda x: x[1], reverse=True)
            self.memory_store = self.memory_store[: self.max_memories]

    def should_explore(self, dopamine_level: float = 0.5, uncertainty: float = 0.5) -> bool:
        """Decide whether to explore."""
        return self.exploration_controller.should_explore(
            self.curiosity_level, dopamine_level, uncertainty
        )

    def get_exploration_bonus(self, state: np.ndarray) -> float:
        """Get exploration bonus for visiting a state."""
        novelty = self.novelty_detector.compute_novelty(state)
        return self.curiosity_level * novelty

    def register_curiosity_target(self, topic: str, intensity: float = 0.5) -> None:
        """Register something we're specifically curious about."""
        self.specific_curiosities[topic] = np.clip(intensity, 0, 1)
        self.info_value.set_uncertainty(topic, intensity)

    def get_knowledge_gaps(self) -> List[str]:
        """Get list of identified knowledge gaps."""
        return self.info_value.identify_knowledge_gaps()

    def get_state(self) -> CuriosityState:
        """Get current curiosity state."""
        return CuriosityState(
            overall_level=self.curiosity_level,
            specific_targets=list(self.specific_curiosities.keys()),
            knowledge_gaps=self.get_knowledge_gaps(),
            boredom_level=self.exploration_controller.boredom_level,
            recent_discoveries=sum(1 for e in self.curiosity_history if e.get("novelty", 0) > 0.7),
        )

    def get_metrics(self) -> Dict[str, float]:
        """Get curiosity system metrics."""
        return {
            "curiosity_level": self.curiosity_level,
            "boredom_level": self.exploration_controller.boredom_level,
            "novelty_trend": self.novelty_detector.get_novelty_trend(),
            "exploration_value": self.exploration_controller.get_exploration_value(),
            "memory_count": len(self.memory_store),
            "avg_memory_importance": np.mean([m[1] for m in self.memory_store])
            if self.memory_store
            else 0,
        }

    def reset(self) -> None:
        """Reset curiosity system."""
        self.novelty_detector = NoveltyDetector(self.state_dim)
        self.info_value = InformationValue(base_curiosity=self.base_curiosity)
        self.exploration_controller = ExplorationController()
        self.curiosity_level = self.base_curiosity
        self.specific_curiosities = {}
        self.memory_store = []
        self.curiosity_history.clear()
