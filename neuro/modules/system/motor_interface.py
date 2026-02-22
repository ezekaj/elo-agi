"""
Motor Interface: Generates output actions from cognitive state.

Converts internal cognitive representations to:
- Vector outputs (continuous actions)
- Discrete outputs (categorical actions)
- Text outputs (language generation)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np
import time

from .config import SystemConfig


class OutputType(Enum):
    """Types of motor output."""

    VECTOR = "vector"
    DISCRETE = "discrete"
    TEXT = "text"
    COMPOSITE = "composite"


class ActionCategory(Enum):
    """Categories of discrete actions."""

    MOVE = "move"
    INTERACT = "interact"
    COMMUNICATE = "communicate"
    WAIT = "wait"
    EXPLORE = "explore"


@dataclass
class MotorOutput:
    """A single motor output."""

    output_type: OutputType
    value: Any
    confidence: float = 0.5
    priority: float = 0.5
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionBuffer:
    """Buffer for recent actions."""

    capacity: int = 10
    actions: List[MotorOutput] = field(default_factory=list)

    def add(self, action: MotorOutput) -> None:
        """Add action to buffer."""
        self.actions.append(action)
        if len(self.actions) > self.capacity:
            self.actions.pop(0)

    def get_recent(self, n: int = 1) -> List[MotorOutput]:
        """Get n most recent actions."""
        return self.actions[-n:]

    def clear(self) -> None:
        """Clear buffer."""
        self.actions = []


class MotorInterface:
    """
    Generates motor outputs from cognitive state.

    The interface:
    1. Receives action proposals from modules
    2. Selects and prioritizes actions
    3. Converts to appropriate output format
    4. Maintains action history
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.buffer = ActionBuffer(capacity=10)

        # Pending actions from modules
        self._pending_actions: List[MotorOutput] = []

        # Statistics
        self._action_count = 0
        self._last_action: Optional[MotorOutput] = None

    def propose_action(
        self,
        action: np.ndarray,
        confidence: float = 0.5,
        priority: float = 0.5,
        output_type: OutputType = OutputType.VECTOR,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Propose an action from a module.

        Args:
            action: Action vector or value
            confidence: Confidence in the action
            priority: Priority level
            output_type: Type of output
            metadata: Additional information
        """
        output = MotorOutput(
            output_type=output_type,
            value=action,
            confidence=confidence,
            priority=priority,
            metadata=metadata or {},
        )
        self._pending_actions.append(output)

    def generate(
        self,
        cognitive_state: np.ndarray,
        output_type: OutputType = OutputType.VECTOR,
    ) -> MotorOutput:
        """
        Generate motor output from cognitive state.

        Args:
            cognitive_state: Current cognitive state vector
            output_type: Desired output type

        Returns:
            Motor output
        """
        if output_type == OutputType.VECTOR:
            return self._generate_vector(cognitive_state)
        elif output_type == OutputType.DISCRETE:
            return self._generate_discrete(cognitive_state)
        elif output_type == OutputType.TEXT:
            return self._generate_text(cognitive_state)
        elif output_type == OutputType.COMPOSITE:
            return self._generate_composite(cognitive_state)
        else:
            return self._generate_vector(cognitive_state)

    def _generate_vector(self, state: np.ndarray) -> MotorOutput:
        """Generate continuous action vector."""
        # Transform state to action space
        action = self._transform_to_action(state)

        # Compute confidence from state activation
        confidence = float(np.tanh(np.mean(np.abs(state))))

        output = MotorOutput(
            output_type=OutputType.VECTOR,
            value=action,
            confidence=confidence,
            priority=confidence,
        )

        self._record_action(output)
        return output

    def _generate_discrete(self, state: np.ndarray) -> MotorOutput:
        """Generate discrete categorical action."""
        # Partition state into action categories
        n_categories = len(ActionCategory)
        partition_size = len(state) // n_categories

        category_scores = []
        for i in range(n_categories):
            start = i * partition_size
            end = start + partition_size
            score = float(np.mean(state[start:end]))
            category_scores.append(score)

        # Select highest scoring category
        best_idx = np.argmax(category_scores)
        best_category = list(ActionCategory)[best_idx]

        confidence = float(np.max(category_scores) - np.mean(category_scores))
        confidence = np.clip(confidence + 0.5, 0, 1)

        output = MotorOutput(
            output_type=OutputType.DISCRETE,
            value=best_category,
            confidence=confidence,
            priority=confidence,
            metadata={"scores": dict(zip([c.value for c in ActionCategory], category_scores))},
        )

        self._record_action(output)
        return output

    def _generate_text(self, state: np.ndarray) -> MotorOutput:
        """Generate text output from cognitive state."""
        # Simple text generation from state
        # In production, would use language model

        # Use state to select words/tokens
        text_parts = []

        # Map state activation to basic tokens
        tokens = [
            "observe",
            "move",
            "wait",
            "think",
            "act",
            "explore",
            "remember",
            "decide",
            "communicate",
            "learn",
        ]

        # Select tokens based on state
        n_tokens = min(5, max(1, int(np.sum(np.abs(state)) / 10)))
        indices = np.argsort(state)[-n_tokens:]

        for idx in indices:
            token_idx = idx % len(tokens)
            text_parts.append(tokens[token_idx])

        text = " ".join(text_parts)

        confidence = float(np.mean(np.abs(state[:10])))
        confidence = np.clip(confidence, 0, 1)

        output = MotorOutput(
            output_type=OutputType.TEXT,
            value=text,
            confidence=confidence,
            priority=confidence,
        )

        self._record_action(output)
        return output

    def _generate_composite(self, state: np.ndarray) -> MotorOutput:
        """Generate composite output with multiple components."""
        vector = self._transform_to_action(state)
        discrete = self._generate_discrete(state)
        text = self._generate_text(state)

        composite = {
            "vector": vector,
            "discrete": discrete.value,
            "text": text.value,
        }

        confidence = (discrete.confidence + text.confidence) / 2

        output = MotorOutput(
            output_type=OutputType.COMPOSITE,
            value=composite,
            confidence=confidence,
            priority=confidence,
        )

        self._record_action(output)
        return output

    def _transform_to_action(self, state: np.ndarray) -> np.ndarray:
        """Transform cognitive state to action vector."""
        # Resize to output dimension
        if len(state) >= self.config.output_dim:
            action = state[: self.config.output_dim]
        else:
            action = np.pad(state, (0, self.config.output_dim - len(state)))

        # Apply nonlinearity
        action = np.tanh(action)

        # Apply threshold
        action = np.where(np.abs(action) > self.config.action_threshold, action, 0.0)

        return action.astype(np.float32)

    def select_action(self) -> Optional[MotorOutput]:
        """
        Select best action from pending proposals.

        Uses priority and confidence to rank actions.
        """
        if not self._pending_actions:
            return None

        # Score actions
        scored = []
        for action in self._pending_actions:
            score = action.priority * action.confidence
            scored.append((score, action))

        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)

        # Select best
        best = scored[0][1]

        # Clear pending
        self._pending_actions = []

        self._record_action(best)
        return best

    def _record_action(self, action: MotorOutput) -> None:
        """Record action in history."""
        self.buffer.add(action)
        self._last_action = action
        self._action_count += 1

    def get_last_action(self) -> Optional[MotorOutput]:
        """Get most recent action."""
        return self._last_action

    def get_action_history(self, n: int = 5) -> List[MotorOutput]:
        """Get recent action history."""
        return self.buffer.get_recent(n)

    def get_statistics(self) -> Dict[str, Any]:
        """Get interface statistics."""
        return {
            "action_count": self._action_count,
            "pending_count": len(self._pending_actions),
            "buffer_size": len(self.buffer.actions),
            "output_dim": self.config.output_dim,
            "action_threshold": self.config.action_threshold,
        }

    def reset(self) -> None:
        """Reset interface state."""
        self.buffer.clear()
        self._pending_actions = []
        self._last_action = None
        self._action_count = 0
