"""
Dialogue Environment: Conversational learning environment.

Provides a simulated dialogue partner for social cognition
and language development.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
import re

from base_env import NeuroEnvironment, EnvironmentConfig, StepResult


@dataclass
class DialogueConfig(EnvironmentConfig):
    """Configuration for dialogue environment."""
    max_turns: int = 20
    vocab_size: int = 5000
    embedding_dim: int = 64
    reward_coherent: float = 0.1
    reward_on_topic: float = 0.2
    reward_goal_achieved: float = 1.0


@dataclass
class DialogueTurn:
    """A single turn in a dialogue."""
    speaker: str  # "agent" or "partner"
    text: str
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueScenario:
    """A dialogue scenario with goals."""
    name: str
    description: str
    partner_persona: str
    agent_goal: str
    success_keywords: List[str]
    initial_message: str


class DialoguePartner:
    """
    Simulated dialogue partner.

    Uses simple rule-based responses. In production,
    would be replaced with LLM.
    """

    def __init__(self, persona: str = "helpful assistant"):
        self.persona = persona
        self._context: List[str] = []
        self._response_templates = self._build_templates()

    def _build_templates(self) -> Dict[str, List[str]]:
        """Build response templates by topic."""
        return {
            "greeting": [
                "Hello! How can I help you today?",
                "Hi there! What would you like to discuss?",
                "Greetings! I'm here to help.",
            ],
            "question": [
                "That's an interesting question. Let me think...",
                "I can help with that. The answer is...",
                "Good question! Here's what I know...",
            ],
            "statement": [
                "I see. That's interesting.",
                "I understand. Tell me more.",
                "That makes sense. What else?",
            ],
            "farewell": [
                "Goodbye! It was nice talking.",
                "See you later! Take care.",
                "Bye! Have a great day.",
            ],
            "unknown": [
                "I'm not sure I understand. Could you rephrase?",
                "Can you explain that differently?",
                "I didn't quite catch that.",
            ],
        }

    def respond(self, message: str) -> str:
        """Generate a response to the message."""
        message_lower = message.lower()
        self._context.append(message)

        # Detect message type
        if any(w in message_lower for w in ["hello", "hi", "hey", "greetings"]):
            category = "greeting"
        elif message.endswith("?"):
            category = "question"
        elif any(w in message_lower for w in ["bye", "goodbye", "farewell"]):
            category = "farewell"
        elif len(message.split()) > 3:
            category = "statement"
        else:
            category = "unknown"

        # Select response
        templates = self._response_templates[category]
        idx = hash(message) % len(templates)
        response = templates[idx]

        self._context.append(response)
        return response

    def reset(self) -> None:
        """Reset conversation context."""
        self._context = []


class DialogueEnvironment(NeuroEnvironment):
    """
    Environment for conversational learning.

    The agent learns to:
    1. Generate coherent responses
    2. Stay on topic
    3. Achieve dialogue goals
    """

    def __init__(
        self,
        config: Optional[DialogueConfig] = None,
        scenario: Optional[DialogueScenario] = None,
    ):
        self.dialogue_config = config or DialogueConfig()
        super().__init__(self.dialogue_config)

        self._scenario = scenario or self._default_scenario()
        self._partner = DialoguePartner(self._scenario.partner_persona)
        self._history: List[DialogueTurn] = []
        self._turn_count = 0
        self._goal_achieved = False

        # Build vocabulary for encoding
        self._build_vocabulary()

    def _default_scenario(self) -> DialogueScenario:
        """Create default dialogue scenario."""
        return DialogueScenario(
            name="information_gathering",
            description="Gather information through conversation",
            partner_persona="knowledgeable assistant",
            agent_goal="Learn about the topic being discussed",
            success_keywords=["understand", "learned", "thanks", "helpful"],
            initial_message="Hello! I'm ready to discuss any topic you'd like.",
        )

    def _build_vocabulary(self) -> None:
        """Build vocabulary for text encoding."""
        common_words = [
            "the", "a", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them", "my", "your", "his", "its", "our", "their",
            "this", "that", "these", "those", "what", "which", "who", "whom",
            "hello", "hi", "bye", "goodbye", "please", "thanks", "thank",
            "yes", "no", "maybe", "ok", "okay", "sure", "right", "wrong",
            "good", "bad", "great", "nice", "fine", "well", "better", "best",
            "help", "know", "think", "want", "need", "like", "love", "hate",
            "understand", "learn", "tell", "ask", "say", "speak", "talk",
        ]
        self._vocab = {word: idx for idx, word in enumerate(common_words)}
        self._idx_to_word = {idx: word for word, idx in self._vocab.items()}

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._episode_count += 1
        self._turn_count = 0
        self._goal_achieved = False
        self._history = []
        self._partner.reset()

        # Partner initiates
        initial = self._scenario.initial_message
        self._history.append(DialogueTurn(speaker="partner", text=initial))

        obs = self._encode_dialogue()
        return obs, {"message": initial, "scenario": self._scenario.name}

    def step(self, action: np.ndarray) -> StepResult:
        action = self._normalize_action(action)
        self._step_count += 1
        self._turn_count += 1

        # Decode agent's response
        agent_message = self._decode_response(action)
        self._history.append(DialogueTurn(speaker="agent", text=agent_message))

        # Calculate reward for agent's response
        reward = self._calculate_reward(agent_message)

        # Get partner's response
        partner_response = self._partner.respond(agent_message)
        self._history.append(DialogueTurn(speaker="partner", text=partner_response))

        self._total_reward += reward

        # Check termination
        terminated = self._goal_achieved
        truncated = self._turn_count >= self.dialogue_config.max_turns

        obs = self._encode_dialogue()

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "agent_message": agent_message,
                "partner_response": partner_response,
                "turn": self._turn_count,
                "goal_achieved": self._goal_achieved,
            },
        )

    def render(self) -> Optional[str]:
        if not self._history:
            return "No dialogue yet."

        lines = []
        for turn in self._history[-4:]:  # Last 4 turns
            prefix = "Partner" if turn.speaker == "partner" else "Agent"
            lines.append(f"{prefix}: {turn.text}")

        return "\n".join(lines)

    def _encode_dialogue(self) -> np.ndarray:
        """Encode recent dialogue history to observation."""
        obs = np.zeros(self.config.observation_dim, dtype=np.float32)

        # Encode last few turns
        recent = self._history[-3:]
        for i, turn in enumerate(recent):
            words = turn.text.lower().split()
            for j, word in enumerate(words[:10]):
                word = re.sub(r'[^\w]', '', word)
                if word in self._vocab:
                    idx = (i * 10 + j) % self.config.observation_dim
                    obs[idx] += (self._vocab[word] + 1) / len(self._vocab)

        # Add speaker indicator
        if recent and recent[-1].speaker == "partner":
            obs[0] += 1.0

        # Normalize
        norm = np.linalg.norm(obs)
        if norm > 0:
            obs = obs / norm

        return obs

    def _decode_response(self, action: np.ndarray) -> str:
        """Decode action vector to response text."""
        # Use action vector to select words
        words = []

        # Select up to 10 words
        for i in range(10):
            if i >= len(action):
                break

            # Map action value to vocabulary
            word_idx = int((action[i] + 1) / 2 * len(self._idx_to_word)) % len(self._idx_to_word)
            word = self._idx_to_word.get(word_idx, "")
            if word and action[i] > -0.5:  # Threshold for including word
                words.append(word)

        if not words:
            words = ["I", "understand"]

        # Simple grammar cleanup
        response = " ".join(words)
        response = response.capitalize()
        if not response.endswith((".", "?", "!")):
            response += "."

        return response

    def _calculate_reward(self, message: str) -> float:
        """Calculate reward for agent's message."""
        reward = 0.0
        message_lower = message.lower()

        # Reward coherent responses (has multiple words, proper structure)
        if len(message.split()) >= 3:
            reward += self.dialogue_config.reward_coherent

        # Reward staying on topic (references previous content)
        if self._history:
            last_partner = [t for t in self._history if t.speaker == "partner"]
            if last_partner:
                last_msg = last_partner[-1].text.lower()
                # Check for word overlap
                overlap = set(message_lower.split()) & set(last_msg.split())
                if len(overlap) > 1:
                    reward += self.dialogue_config.reward_on_topic

        # Reward goal achievement
        for keyword in self._scenario.success_keywords:
            if keyword in message_lower:
                self._goal_achieved = True
                reward += self.dialogue_config.reward_goal_achieved
                break

        return reward

    def get_history(self) -> List[DialogueTurn]:
        """Get full dialogue history."""
        return self._history.copy()


class MultiPersonaDialogue(DialogueEnvironment):
    """Dialogue environment with multiple partner personas."""

    def __init__(
        self,
        personas: Optional[List[str]] = None,
        config: Optional[DialogueConfig] = None,
    ):
        self._personas = personas or [
            "friendly assistant",
            "curious student",
            "skeptical critic",
            "enthusiastic teacher",
        ]
        super().__init__(config=config)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Switch persona randomly
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        persona = self._rng.choice(self._personas)
        self._partner = DialoguePartner(persona)

        return super().reset(seed)
