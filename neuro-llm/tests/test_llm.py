"""
Tests for the Neuro LLM Language Bridge.

Covers:
- LLM interface and mock
- Semantic bridge
- Language grounding
- Dialogue agent
"""

import pytest
import numpy as np
from pathlib import Path

# Add src to path
from neuro.modules.llm.llm_interface import LLMOracle, MockLLM, LLMConfig, LLMResponse, create_llm
from neuro.modules.llm.semantic_bridge import SemanticBridge, SemanticConfig, Embedding
from neuro.modules.llm.language_grounding import LanguageGrounding, GroundingConfig, GroundedConcept
from neuro.modules.llm.dialogue_agent import NeuroDialogueAgent, DialogueConfig, ConversationTurn, MultiAgentDialogue

# =============================================================================
# Tests: LLM Interface
# =============================================================================

class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_default_config(self):
        config = LLMConfig()
        assert config.provider == "mock"
        assert config.embedding_dim == 64

    def test_custom_config(self):
        config = LLMConfig(provider="anthropic", model="claude-3-opus")
        assert config.provider == "anthropic"
        assert config.model == "claude-3-opus"

class TestLLMResponse:
    """Tests for LLMResponse."""

    def test_creation(self):
        response = LLMResponse(text="Hello", tokens_used=5)
        assert response.text == "Hello"
        assert response.tokens_used == 5
        assert not response.cached

class TestMockLLM:
    """Tests for MockLLM."""

    def test_creation(self):
        llm = MockLLM()
        assert llm.config.provider == "mock"

    def test_query(self):
        llm = MockLLM()
        response = llm.query("Hello!")
        assert isinstance(response, LLMResponse)
        assert len(response.text) > 0

    def test_query_greeting(self):
        llm = MockLLM()
        response = llm.query("Hello there!")
        assert "Hello" in response.text or "help" in response.text.lower()

    def test_query_question(self):
        llm = MockLLM()
        response = llm.query("What is the meaning of life?")
        assert "question" in response.text.lower() or "think" in response.text.lower()

    def test_embed(self):
        llm = MockLLM()
        embedding = llm.embed("test text")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == llm.config.embedding_dim

    def test_embed_deterministic(self):
        llm = MockLLM()
        e1 = llm.embed("same text")
        e2 = llm.embed("same text")
        assert np.allclose(e1, e2)

    def test_embed_different(self):
        llm = MockLLM()
        e1 = llm.embed("text one")
        e2 = llm.embed("text two")
        assert not np.allclose(e1, e2)

    def test_parse_action_move(self):
        llm = MockLLM()
        parsed = llm.parse_action("I will move north")
        assert parsed['type'] == 'move'
        assert parsed.get('direction') == 'north'

    def test_parse_action_take(self):
        llm = MockLLM()
        parsed = llm.parse_action("Let me take the key")
        assert parsed['type'] == 'take'

    def test_cache(self):
        llm = MockLLM()
        r1 = llm.query("Test prompt")
        r2 = llm.query("Test prompt")
        assert r2.cached
        assert llm._cache_hits == 1

    def test_statistics(self):
        llm = MockLLM()
        llm.query("Test 1")
        llm.query("Test 2")
        stats = llm.get_statistics()
        assert stats['query_count'] == 2

    def test_clear_cache(self):
        llm = MockLLM()
        llm.query("Test")
        llm.clear_cache()
        assert len(llm._cache) == 0

class TestCreateLLM:
    """Tests for create_llm factory."""

    def test_create_mock(self):
        llm = create_llm(LLMConfig(provider="mock"))
        assert isinstance(llm, MockLLM)

    def test_create_default(self):
        llm = create_llm()
        assert isinstance(llm, MockLLM)

# =============================================================================
# Tests: Semantic Bridge
# =============================================================================

class TestEmbedding:
    """Tests for Embedding dataclass."""

    def test_creation(self):
        vec = np.random.randn(64).astype(np.float32)
        emb = Embedding(vector=vec, text="test")
        assert len(emb.vector) == 64
        assert emb.text == "test"

    def test_similarity(self):
        vec1 = np.array([1, 0, 0], dtype=np.float32)
        vec2 = np.array([1, 0, 0], dtype=np.float32)
        emb1 = Embedding(vector=vec1, text="a")
        emb2 = Embedding(vector=vec2, text="b")
        assert emb1.similarity(emb2) == pytest.approx(1.0)

    def test_similarity_orthogonal(self):
        vec1 = np.array([1, 0, 0], dtype=np.float32)
        vec2 = np.array([0, 1, 0], dtype=np.float32)
        emb1 = Embedding(vector=vec1, text="a")
        emb2 = Embedding(vector=vec2, text="b")
        assert emb1.similarity(emb2) == pytest.approx(0.0)

class TestSemanticBridge:
    """Tests for SemanticBridge."""

    def test_creation(self):
        bridge = SemanticBridge()
        assert bridge.llm is not None
        assert bridge.config is not None

    def test_encode(self):
        bridge = SemanticBridge()
        internal = bridge.encode("test text")
        assert isinstance(internal, np.ndarray)
        assert len(internal) == bridge.config.internal_dim

    def test_encode_normalized(self):
        bridge = SemanticBridge()
        internal = bridge.encode("test text")
        norm = np.linalg.norm(internal)
        assert norm == pytest.approx(1.0, rel=0.1)

    def test_decode(self):
        bridge = SemanticBridge()
        internal = np.random.randn(64).astype(np.float32)
        text = bridge.decode(internal)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_register_concept(self):
        bridge = SemanticBridge()
        emb = bridge.register_concept("apple", "A red fruit")
        assert bridge.get_concept("apple") is not None

    def test_observation_to_text(self):
        bridge = SemanticBridge()
        obs = np.random.randn(64).astype(np.float32)
        text = bridge.observation_to_text(obs)
        assert isinstance(text, str)

    def test_text_to_action(self):
        bridge = SemanticBridge()
        action = bridge.text_to_action("move north")
        assert isinstance(action, np.ndarray)

    def test_memory(self):
        bridge = SemanticBridge()
        bridge.encode("first")
        bridge.encode("second")
        bridge.encode("third")
        assert len(bridge._memory) == 3

    def test_statistics(self):
        bridge = SemanticBridge()
        bridge.encode("test")
        bridge.decode(np.zeros(64))
        stats = bridge.get_statistics()
        assert stats['encode_count'] == 1
        assert stats['decode_count'] == 1

    def test_reset(self):
        bridge = SemanticBridge()
        bridge.encode("test")
        bridge.reset()
        assert len(bridge._memory) == 0

# =============================================================================
# Tests: Language Grounding
# =============================================================================

class TestGroundedConcept:
    """Tests for GroundedConcept."""

    def test_creation(self):
        concept = GroundedConcept(
            name="apple",
            text="A red apple",
            perceptual_features=np.zeros(32),
            action_affordances=np.zeros(32),
            goal_relevance=np.zeros(64),
        )
        assert concept.name == "apple"
        assert concept.confidence == 0.5

    def test_update(self):
        concept = GroundedConcept(
            name="test",
            text="test",
            perceptual_features=np.zeros(32),
            action_affordances=np.zeros(32),
            goal_relevance=np.zeros(64),
        )
        concept.update(
            perceptual=np.ones(32),
            action=np.ones(32),
            goal=np.ones(64),
        )
        assert concept.occurrences == 2
        assert concept.confidence > 0.5

class TestLanguageGrounding:
    """Tests for LanguageGrounding."""

    def test_creation(self):
        grounding = LanguageGrounding()
        assert grounding.llm is not None
        assert grounding.bridge is not None

    def test_ground(self):
        grounding = LanguageGrounding()
        obs = np.random.randn(64).astype(np.float32)
        concept = grounding.ground("red apple", observation=obs)
        assert isinstance(concept, GroundedConcept)
        assert concept.name == "red apple"

    def test_ground_with_action(self):
        grounding = LanguageGrounding()
        obs = np.random.randn(64).astype(np.float32)
        action = np.random.randn(32).astype(np.float32)
        concept = grounding.ground("pick up", observation=obs, action=action)
        assert np.any(concept.action_affordances != 0)

    def test_unground(self):
        grounding = LanguageGrounding()
        obs = np.random.randn(64).astype(np.float32)
        concept = grounding.ground("test object", observation=obs)
        descriptions = grounding.unground(concept)
        assert 'perceptual' in descriptions
        assert 'action' in descriptions

    def test_describe_observation(self):
        grounding = LanguageGrounding()
        obs = np.random.randn(64).astype(np.float32)
        description = grounding.describe_observation(obs)
        assert isinstance(description, str)

    def test_parse_instruction(self):
        grounding = LanguageGrounding()
        parsed = grounding.parse_instruction("move to the door")
        assert 'type' in parsed

    def test_instruction_to_action(self):
        grounding = LanguageGrounding()
        action = grounding.instruction_to_action("go north")
        assert isinstance(action, np.ndarray)

    def test_goal_to_text(self):
        grounding = LanguageGrounding()
        goal = np.random.randn(64).astype(np.float32)
        text = grounding.goal_to_text(goal)
        assert isinstance(text, str)
        assert "goal" in text.lower() or "Goal" in text

    def test_learn_from_experience(self):
        grounding = LanguageGrounding()
        obs = np.random.randn(64).astype(np.float32)
        action = np.random.randn(32).astype(np.float32)
        grounding.learn_from_experience(
            text="good action",
            observation=obs,
            action=action,
            reward=1.0,
        )
        assert len(grounding._experience_buffer) == 1

    def test_list_concepts(self):
        grounding = LanguageGrounding()
        grounding.ground("apple", observation=np.zeros(64))
        grounding.ground("banana", observation=np.zeros(64))
        concepts = grounding.list_concepts()
        assert "apple" in concepts
        assert "banana" in concepts

    def test_statistics(self):
        grounding = LanguageGrounding()
        grounding.ground("test", observation=np.zeros(64))
        stats = grounding.get_statistics()
        assert stats['concept_count'] == 1
        assert stats['ground_count'] == 1

# =============================================================================
# Tests: Dialogue Agent
# =============================================================================

class TestConversationTurn:
    """Tests for ConversationTurn."""

    def test_creation(self):
        turn = ConversationTurn(role="user", content="Hello")
        assert turn.role == "user"
        assert turn.content == "Hello"

class TestNeuroDialogueAgent:
    """Tests for NeuroDialogueAgent."""

    def test_creation(self):
        agent = NeuroDialogueAgent()
        assert agent.llm is not None
        assert agent.bridge is not None

    def test_respond(self):
        agent = NeuroDialogueAgent()
        response = agent.respond("Hello!")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_conversation(self):
        agent = NeuroDialogueAgent()
        r1 = agent.respond("Hi there")
        r2 = agent.respond("How are you?")
        assert len(agent._history) == 4  # 2 user + 2 agent

    def test_process_observation(self):
        agent = NeuroDialogueAgent()
        obs = np.random.randn(64).astype(np.float32)
        description = agent.process_observation(obs)
        assert isinstance(description, str)

    def test_execute_instruction(self):
        agent = NeuroDialogueAgent()
        action, description = agent.execute_instruction("move forward")
        assert isinstance(action, np.ndarray)
        assert isinstance(description, str)

    def test_set_goal(self):
        agent = NeuroDialogueAgent()
        agent.set_goal("Find the treasure")
        assert agent._context.get('current_goal') == "Find the treasure"

    def test_get_goal_description(self):
        agent = NeuroDialogueAgent()
        agent.set_goal("Explore the area")
        goal_text = agent.get_goal_description()
        assert "Explore the area" in goal_text

    def test_get_state_description(self):
        agent = NeuroDialogueAgent()
        agent.respond("Hello")  # Sets internal state
        description = agent.get_state_description()
        assert isinstance(description, str)

    def test_learn_from_feedback(self):
        agent = NeuroDialogueAgent()
        agent.respond("What should I do?")
        agent.learn_from_feedback("Good response!", reward=1.0)
        # Should not crash

    def test_history(self):
        agent = NeuroDialogueAgent()
        agent.respond("Test 1")
        agent.respond("Test 2")
        history = agent.get_history()
        assert len(history) == 4

    def test_clear_history(self):
        agent = NeuroDialogueAgent()
        agent.respond("Test")
        agent.clear_history()
        assert len(agent._history) == 0

    def test_statistics(self):
        agent = NeuroDialogueAgent()
        agent.respond("Test")
        stats = agent.get_statistics()
        assert stats['turn_count'] == 1

    def test_reset(self):
        agent = NeuroDialogueAgent()
        agent.respond("Test")
        agent.reset()
        assert len(agent._history) == 0
        assert agent._current_state is None

class TestMultiAgentDialogue:
    """Tests for MultiAgentDialogue."""

    def test_creation(self):
        multi = MultiAgentDialogue()
        assert multi.agents == {}

    def test_add_agent(self):
        multi = MultiAgentDialogue()
        agent = NeuroDialogueAgent()
        multi.add_agent("alice", agent)
        assert "alice" in multi.agents

    def test_run_turn(self):
        multi = MultiAgentDialogue()
        multi.add_agent("alice", NeuroDialogueAgent())
        multi.add_agent("bob", NeuroDialogueAgent())

        responses = multi.run_turn("alice", "Hello everyone!")
        assert "bob" in responses
        assert len(multi._conversation_log) >= 2

    def test_run_discussion(self):
        multi = MultiAgentDialogue()
        multi.add_agent("alice", NeuroDialogueAgent())
        multi.add_agent("bob", NeuroDialogueAgent())

        log = multi.run_discussion("AI safety", rounds=2)
        assert len(log) > 0

    def test_get_log(self):
        multi = MultiAgentDialogue()
        multi.add_agent("alice", NeuroDialogueAgent())
        multi.add_agent("bob", NeuroDialogueAgent())

        multi.run_turn("alice", "Test")
        log = multi.get_log()
        assert isinstance(log, list)

    def test_clear_log(self):
        multi = MultiAgentDialogue()
        multi.add_agent("alice", NeuroDialogueAgent())
        multi.add_agent("bob", NeuroDialogueAgent())

        multi.run_turn("alice", "Test")
        multi.clear_log()
        assert len(multi._conversation_log) == 0

# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full language bridge."""

    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding preserves meaning."""
        bridge = SemanticBridge()
        original = "The quick brown fox jumps"
        internal = bridge.encode(original)
        decoded = bridge.decode(internal, prompt_hint="Describe what was said")
        # Can't check exact match, but should be non-empty
        assert len(decoded) > 0

    def test_grounding_with_bridge(self):
        """Test grounding uses bridge correctly."""
        llm = MockLLM()
        bridge = SemanticBridge(llm)
        grounding = LanguageGrounding(llm, bridge)

        obs = np.random.randn(64).astype(np.float32)
        concept = grounding.ground("red ball", observation=obs)

        # Concept should be retrievable
        retrieved = grounding.get_concept("red ball")
        assert retrieved is not None
        assert retrieved.name == "red ball"

    def test_dialogue_with_grounding(self):
        """Test dialogue agent uses grounding."""
        agent = NeuroDialogueAgent()

        # Have a conversation
        agent.respond("I see a red apple on the table")
        agent.respond("What can I do with it?")

        # Check grounding occurred
        stats = agent.get_statistics()
        assert stats['grounding_stats']['concept_count'] > 0

    def test_full_pipeline(self):
        """Test full language processing pipeline."""
        llm = MockLLM()
        bridge = SemanticBridge(llm)
        grounding = LanguageGrounding(llm, bridge)
        agent = NeuroDialogueAgent(llm)

        # Process observation
        obs = np.random.randn(64).astype(np.float32)
        description = agent.process_observation(obs)

        # Set goal
        agent.set_goal("Find the exit")

        # Have conversation
        response = agent.respond("Where should I go?")

        # Execute instruction
        action, action_desc = agent.execute_instruction("go north")

        # All should work without errors
        assert len(description) > 0
        assert len(response) > 0
        assert action.shape[0] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
