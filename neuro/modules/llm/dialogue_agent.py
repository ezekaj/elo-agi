"""
Dialogue Agent: Conversational wrapper for the cognitive system.

Provides a natural language interface to interact with the
neuro cognitive system through conversation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import time

from .llm_interface import LLMOracle, MockLLM
from .semantic_bridge import SemanticBridge
from .language_grounding import LanguageGrounding


@dataclass
class DialogueConfig:
    """Configuration for dialogue agent."""

    max_history: int = 20
    system_prompt: str = "You are a helpful cognitive agent that can perceive, think, and act."
    use_grounding: bool = True
    use_memory: bool = True
    response_max_tokens: int = 256


@dataclass
class ConversationTurn:
    """A single turn in conversation."""

    role: str  # "user" or "agent"
    content: str
    timestamp: float = field(default_factory=time.time)
    internal_state: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class NeuroDialogueAgent:
    """
    Natural language interface to the cognitive system.

    The agent:
    1. Receives user messages
    2. Processes through cognitive system (if connected)
    3. Generates contextual responses
    4. Maintains conversation history
    5. Grounds language in experience
    """

    def __init__(
        self,
        llm: Optional[LLMOracle] = None,
        cognitive_core: Optional[Any] = None,  # CognitiveCore from neuro-system
        config: Optional[DialogueConfig] = None,
    ):
        self.config = config or DialogueConfig()
        self.llm = llm or MockLLM()
        self.cognitive_core = cognitive_core

        # Language processing
        self.bridge = SemanticBridge(self.llm)
        self.grounding = LanguageGrounding(self.llm, self.bridge)

        # Conversation state
        self._history: List[ConversationTurn] = []
        self._context: Dict[str, Any] = {}
        self._current_state: Optional[np.ndarray] = None

        # Statistics
        self._turn_count = 0
        self._total_tokens = 0

    def respond(self, user_message: str) -> str:
        """
        Generate response to user message.

        This is the main entry point for conversation.

        Args:
            user_message: User's input message

        Returns:
            Agent's response
        """
        self._turn_count += 1

        # Record user turn
        user_turn = ConversationTurn(role="user", content=user_message)
        self._history.append(user_turn)

        # Process through cognitive system if available
        if self.cognitive_core is not None:
            response = self._respond_with_cognition(user_message)
        else:
            response = self._respond_without_cognition(user_message)

        # Record agent turn
        agent_turn = ConversationTurn(
            role="agent",
            content=response,
            internal_state=self._current_state,
        )
        self._history.append(agent_turn)

        # Trim history
        if len(self._history) > self.config.max_history * 2:
            self._history = self._history[-self.config.max_history * 2 :]

        return response

    def _respond_with_cognition(self, message: str) -> str:
        """Generate response using cognitive system."""
        # Encode message to internal representation
        internal = self.bridge.encode(message)

        # Process through cognitive system
        self.cognitive_core.perceive(internal)
        self.cognitive_core.think()
        output = self.cognitive_core.act()

        # Get cognitive state
        self._current_state = self.cognitive_core.state.last_input

        # Convert output to action description
        if hasattr(output, "value") and isinstance(output.value, np.ndarray):
            pass
        else:
            pass  # Fallback

        # Generate response using LLM with cognitive context
        self._build_context()
        state_description = (
            self.bridge.observation_to_text(self._current_state)
            if self._current_state is not None
            else ""
        )

        prompt = f"""Based on the conversation and cognitive state, generate a response.

Conversation:
{self._format_history()}

Current cognitive state: {state_description}

User said: {message}

Respond naturally and helpfully:"""

        response = self.llm.query(prompt, system=self.config.system_prompt)
        self._total_tokens += response.tokens_used

        return response.text

    def _respond_without_cognition(self, message: str) -> str:
        """Generate response without cognitive system (LLM only)."""
        # Encode for grounding
        internal = self.bridge.encode(message)
        self._current_state = internal

        # Ground message
        if self.config.use_grounding:
            self.grounding.ground(message, observation=internal)

        # Build prompt
        prompt = self._build_prompt(message)

        # Query LLM
        response = self.llm.query(prompt, system=self.config.system_prompt)
        self._total_tokens += response.tokens_used

        return response.text

    def _build_prompt(self, current_message: str) -> str:
        """Build prompt with conversation history."""
        parts = []

        # Add history
        if self.config.use_memory and self._history:
            parts.append("Previous conversation:")
            parts.append(self._format_history(max_turns=5))
            parts.append("")

        # Add current message
        parts.append(f"User: {current_message}")
        parts.append("")
        parts.append("Respond helpfully:")

        return "\n".join(parts)

    def _build_context(self) -> Dict[str, Any]:
        """Build context dictionary for response generation."""
        context = {
            "turn_count": self._turn_count,
            "history_length": len(self._history),
        }

        if self._current_state is not None:
            context["state_summary"] = {
                "mean_activation": float(np.mean(np.abs(self._current_state))),
                "max_activation": float(np.max(np.abs(self._current_state))),
            }

        if self.cognitive_core is not None:
            context["cognitive_stats"] = self.cognitive_core.get_statistics()

        return context

    def _format_history(self, max_turns: int = 10) -> str:
        """Format conversation history as string."""
        recent = self._history[-max_turns * 2 :]
        lines = []
        for turn in recent:
            prefix = "User" if turn.role == "user" else "Agent"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)

    def process_observation(self, observation: np.ndarray) -> str:
        """
        Process external observation and describe it.

        Args:
            observation: Observation vector

        Returns:
            Natural language description
        """
        # Ground observation
        description = self.grounding.describe_observation(observation)
        self._current_state = observation.copy()

        # Process through cognitive system if available
        if self.cognitive_core is not None:
            self.cognitive_core.perceive(observation)
            self.cognitive_core.think()

        return description

    def execute_instruction(self, instruction: str) -> Tuple[np.ndarray, str]:
        """
        Execute a natural language instruction.

        Args:
            instruction: Natural language instruction

        Returns:
            Tuple of (action_vector, action_description)
        """
        # Parse instruction
        parsed = self.grounding.parse_instruction(instruction)

        # Convert to action
        action = self.grounding.instruction_to_action(instruction)

        # Execute through cognitive system if available
        if self.cognitive_core is not None:
            self.cognitive_core.perceive(self.bridge.encode(instruction))
            self.cognitive_core.think()
            output = self.cognitive_core.act()
            if hasattr(output, "value") and isinstance(output.value, np.ndarray):
                action = output.value

        # Describe action
        action_type = parsed.get("type", "unknown")
        description = f"Executing {action_type} action"

        return action, description

    def set_goal(self, goal_description: str) -> None:
        """
        Set agent goal from natural language.

        Args:
            goal_description: Natural language goal
        """
        # Encode goal
        goal_vector = self.bridge.encode(goal_description)

        # Set in cognitive system if available
        if self.cognitive_core is not None:
            self.cognitive_core.set_goals(goal_vector)

        # Ground goal
        self.grounding.ground(goal_description, goal=goal_vector)

        self._context["current_goal"] = goal_description

    def get_goal_description(self) -> str:
        """Get current goal as natural language."""
        if "current_goal" in self._context:
            return self._context["current_goal"]

        if self.cognitive_core is not None and self.cognitive_core.state.goals is not None:
            return self.grounding.goal_to_text(self.cognitive_core.state.goals)

        return "No specific goal set"

    def get_state_description(self) -> str:
        """Get current cognitive state as natural language."""
        if self._current_state is not None:
            return self.bridge.observation_to_text(self._current_state)
        return "State unknown"

    def learn_from_feedback(self, feedback: str, reward: float) -> None:
        """
        Learn from user feedback.

        Args:
            feedback: User feedback text
            reward: Numeric reward signal (-1 to 1)
        """
        if not self._history:
            return

        # Get last exchange
        last_agent = None
        last_user = None
        for turn in reversed(self._history):
            if turn.role == "agent" and last_agent is None:
                last_agent = turn
            elif turn.role == "user" and last_user is None:
                last_user = turn
            if last_agent and last_user:
                break

        if last_agent and last_agent.internal_state is not None:
            # Learn grounding
            self.grounding.learn_from_experience(
                text=feedback,
                observation=last_agent.internal_state,
                action=last_agent.internal_state,  # Use state as proxy
                reward=reward,
            )

    def get_history(self) -> List[ConversationTurn]:
        """Get conversation history."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history = []
        self._context = {}

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "turn_count": self._turn_count,
            "total_tokens": self._total_tokens,
            "history_length": len(self._history),
            "has_cognitive_core": self.cognitive_core is not None,
            "grounding_stats": self.grounding.get_statistics(),
            "bridge_stats": self.bridge.get_statistics(),
            "llm_stats": self.llm.get_statistics(),
        }

    def reset(self) -> None:
        """Reset agent state."""
        self.clear_history()
        self._current_state = None
        self.bridge.reset()
        self.grounding.reset()


class MultiAgentDialogue:
    """
    Multi-agent dialogue system for agent-to-agent communication.
    """

    def __init__(
        self,
        agents: Optional[Dict[str, NeuroDialogueAgent]] = None,
        moderator_llm: Optional[LLMOracle] = None,
    ):
        self.agents = agents or {}
        self.moderator = moderator_llm or MockLLM()
        self._conversation_log: List[Dict[str, Any]] = []

    def add_agent(self, name: str, agent: NeuroDialogueAgent) -> None:
        """Add an agent to the dialogue."""
        self.agents[name] = agent

    def run_turn(self, speaker: str, message: str) -> Dict[str, str]:
        """
        Run a dialogue turn where one agent speaks.

        Args:
            speaker: Name of speaking agent
            message: Message to broadcast

        Returns:
            Dictionary of agent_name -> response
        """
        responses = {}

        # Record speaker's message
        self._conversation_log.append(
            {
                "speaker": speaker,
                "message": message,
                "timestamp": time.time(),
            }
        )

        # Get responses from other agents
        for name, agent in self.agents.items():
            if name != speaker:
                response = agent.respond(f"{speaker} says: {message}")
                responses[name] = response
                self._conversation_log.append(
                    {
                        "speaker": name,
                        "message": response,
                        "timestamp": time.time(),
                    }
                )

        return responses

    def run_discussion(
        self,
        topic: str,
        rounds: int = 3,
        starter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a multi-round discussion on a topic.

        Args:
            topic: Discussion topic
            rounds: Number of discussion rounds
            starter: Agent to start (random if None)

        Returns:
            Conversation log
        """
        if not self.agents:
            return []

        agent_names = list(self.agents.keys())

        # Initial prompt
        current_speaker = starter or agent_names[0]
        current_message = f"Let's discuss: {topic}"

        for round_num in range(rounds):
            responses = self.run_turn(current_speaker, current_message)

            if responses:
                # Pick next speaker and their response
                next_speaker = list(responses.keys())[round_num % len(responses)]
                current_message = responses[next_speaker]
                current_speaker = next_speaker

        return self._conversation_log

    def get_log(self) -> List[Dict[str, Any]]:
        """Get full conversation log."""
        return self._conversation_log.copy()

    def clear_log(self) -> None:
        """Clear conversation log."""
        self._conversation_log = []
