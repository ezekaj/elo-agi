"""Tests for long-term memory systems"""

import pytest
import time

from src.long_term_memory import (
    EpisodicMemory, Episode,
    SemanticMemory, Concept,
    ProceduralMemory, Procedure
)
from src.long_term_memory.procedural_memory import Pattern, Action


class TestEpisodicMemory:
    """Tests for episodic memory"""

    def test_encode_and_retrieve(self):
        """Test basic encoding and retrieval"""
        em = EpisodicMemory()

        episode = em.encode(
            experience="Had coffee at cafe",
            context={"location": "downtown", "time": "morning"},
            emotional_valence=0.5
        )

        results = em.retrieve_by_cue("coffee")
        assert len(results) == 1
        assert "coffee" in str(results[0].content)

    def test_context_retrieval(self):
        """Test context-dependent retrieval"""
        em = EpisodicMemory()

        em.encode("Event A", context={"location": "home"})
        em.encode("Event B", context={"location": "work"})
        em.encode("Event C", context={"location": "home"})

        results = em.retrieve_by_context({"location": "home"})
        assert len(results) == 2

    def test_emotional_retrieval(self):
        """Test mood-congruent retrieval"""
        em = EpisodicMemory()

        em.encode("Happy event", emotional_valence=0.8)
        em.encode("Sad event", emotional_valence=-0.7)
        em.encode("Neutral event", emotional_valence=0.0)

        positive = em.retrieve_by_emotion(0.5, 1.0)
        negative = em.retrieve_by_emotion(-1.0, -0.5)

        assert len(positive) == 1
        assert len(negative) == 1

    def test_temporal_retrieval(self):
        """Test time-based retrieval"""
        em = EpisodicMemory()

        current_time = 0.0
        em.set_time_function(lambda: current_time)

        current_time = 100.0
        em.encode("Old event")

        current_time = 200.0
        em.encode("New event")

        results = em.retrieve_by_time(start_time=150.0)
        assert len(results) == 1
        assert "New" in str(results[0].content)

    def test_replay_strengthens(self):
        """Test that replay strengthens memories"""
        em = EpisodicMemory()

        episode = em.encode("Test event")
        initial_strength = episode.strength

        em.replay([episode])

        assert episode.strength > initial_strength

    def test_forgetting(self):
        """Test forgetting mechanisms"""
        em = EpisodicMemory()

        episode = em.encode("Forgettable event")

        # Decay
        em.forget(episode, method="decay")
        assert episode.strength < 1.0
        assert episode.vividness < 1.0


class TestSemanticMemory:
    """Tests for semantic memory"""

    def test_store_and_retrieve(self):
        """Test basic concept storage and retrieval"""
        sm = SemanticMemory()

        sm.create_concept(
            name="dog",
            features={"furry": 1.0, "has_tail": 1.0},
            relations=[("is-a", "animal", 1.0)]
        )

        concept = sm.retrieve("dog")
        assert concept is not None
        assert concept.name == "dog"
        assert "furry" in concept.features

    def test_spreading_activation(self):
        """Test spreading activation through network"""
        sm = SemanticMemory()

        sm.create_concept("dog", relations=[("is-a", "animal", 1.0)])
        sm.create_concept("animal", relations=[("has-a", "legs", 1.0)])
        sm.create_concept("legs")

        activations = sm.spread_activation("dog", depth=3)

        assert "dog" in activations
        assert "animal" in activations
        assert activations["dog"] > activations["animal"]

    def test_find_path(self):
        """Test finding path between concepts"""
        sm = SemanticMemory()

        sm.create_concept("dog", relations=[("is-a", "mammal", 1.0)])
        sm.create_concept("mammal", relations=[("is-a", "animal", 1.0)])
        sm.create_concept("animal")

        path = sm.find_path("dog", "animal")

        assert path is not None
        assert len(path) == 2  # dog->mammal, mammal->animal

    def test_category_membership(self):
        """Test category operations"""
        sm = SemanticMemory()

        sm.create_concept("dog", relations=[("is-a", "mammal", 1.0)])
        sm.create_concept("cat", relations=[("is-a", "mammal", 1.0)])
        sm.create_concept("mammal")

        categories = sm.get_categories("dog")
        members = sm.get_members("mammal")

        assert "mammal" in categories
        assert "dog" in members
        assert "cat" in members


class TestProceduralMemory:
    """Tests for procedural memory"""

    def test_encode_and_execute(self):
        """Test skill encoding and execution"""
        pm = ProceduralMemory()

        pm.encode_simple(
            name="greet",
            trigger_features={"situation": "meeting"},
            action_names=["smile", "wave", "say_hello"]
        )

        results = pm.execute({"situation": "meeting"})

        assert results is not None
        assert len(results) == 3

    def test_automaticity_through_practice(self):
        """Test that practice increases automaticity"""
        pm = ProceduralMemory()

        procedure = pm.encode_simple(
            name="typing",
            trigger_features={"task": "write"},
            action_names=["type"]
        )

        initial_strength = procedure.strength

        # Practice
        for _ in range(10):
            pm.execute({"task": "write"})
            pm.strengthen(procedure, success=True)

        assert procedure.strength > initial_strength
        assert procedure.is_automatic(threshold=0.5)

    def test_competition(self):
        """Test procedure competition"""
        pm = ProceduralMemory()

        # Two competing procedures
        p1 = pm.encode_simple(
            name="response1",
            trigger_features={"stimulus": "beep"},
            action_names=["action1"]
        )
        p1.strength = 0.9

        p2 = pm.encode_simple(
            name="response2",
            trigger_features={"stimulus": "beep"},
            action_names=["action2"]
        )
        p2.strength = 0.3

        # Get matching procedures
        matches = pm.get_matching({"stimulus": "beep"})

        # Higher strength should win
        winner = pm.compete(matches)
        assert winner.name == "response1"

    def test_decay_from_nonuse(self):
        """Test skill decay from non-use"""
        pm = ProceduralMemory()

        procedure = pm.encode_simple(
            name="skill",
            trigger_features={"x": 1},
            action_names=["do_something"]
        )
        procedure.strength = 0.8

        pm.decay_all(amount=0.1)

        assert procedure.strength == 0.7
