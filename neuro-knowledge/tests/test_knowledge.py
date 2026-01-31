"""
Tests for neuro-knowledge: Knowledge representation and reasoning.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from semantic_network import (
    SemanticNetwork, Concept, SemanticRelation, RelationType,
    ActivationPattern,
)
from ontology import (
    Ontology, OntologyNode, OntologyRelation, HierarchyType, OntologyQuery,
)
from fact_store import (
    FactStore, Fact, Triple, FactQuery, FactIndex,
)
from inference_engine import (
    InferenceEngine, Rule, Pattern, InferenceResult, InferenceChain, InferenceMode,
)
from knowledge_graph import (
    KnowledgeGraph, Entity, Relation, GraphEmbedding, GraphQuery,
)
from common_sense import (
    CommonSenseReasoner, CommonSenseKB, PhysicsReasoner,
    SocialReasoner, TemporalReasoner, PhysicsProperty,
    SocialNorm, TemporalRelation, SocialRelation,
)


# =============================================================================
# Semantic Network Tests
# =============================================================================

class TestConcept:
    """Tests for Concept class."""

    def test_concept_creation(self):
        """Test basic concept creation."""
        concept = Concept(name="dog")
        assert concept.name == "dog"
        assert concept.activation == 0.0
        assert concept.base_activation == 0.0

    def test_concept_with_properties(self):
        """Test concept with properties."""
        concept = Concept(
            name="dog",
            properties={"has_fur": True, "legs": 4},
        )
        assert concept.properties["has_fur"] is True
        assert concept.properties["legs"] == 4

    def test_concept_hash_equality(self):
        """Test concept hashing and equality."""
        c1 = Concept(name="dog")
        c2 = Concept(name="dog")
        c3 = Concept(name="cat")
        assert c1 == c2
        assert c1 != c3
        assert hash(c1) == hash(c2)


class TestSemanticRelation:
    """Tests for SemanticRelation class."""

    def test_relation_creation(self):
        """Test basic relation creation."""
        rel = SemanticRelation(
            source="dog",
            target="animal",
            relation_type=RelationType.IS_A,
        )
        assert rel.source == "dog"
        assert rel.target == "animal"
        assert rel.relation_type == RelationType.IS_A
        assert rel.weight == 1.0

    def test_relation_with_weight(self):
        """Test relation with custom weight."""
        rel = SemanticRelation(
            source="dog",
            target="pet",
            relation_type=RelationType.IS_A,
            weight=0.8,
        )
        assert rel.weight == 0.8


class TestSemanticNetwork:
    """Tests for SemanticNetwork class."""

    @pytest.fixture
    def network(self):
        """Create a test network."""
        net = SemanticNetwork()
        net.add_concept("animal", properties={"type": "living creature"})
        net.add_concept("dog", properties={"type": "domesticated canine"})
        net.add_concept("cat", properties={"type": "domesticated feline"})
        net.add_concept("mammal", properties={"type": "warm-blooded animal"})
        net.add_relation("dog", "animal", RelationType.IS_A)
        net.add_relation("cat", "animal", RelationType.IS_A)
        net.add_relation("dog", "mammal", RelationType.IS_A)
        net.add_relation("cat", "mammal", RelationType.IS_A)
        return net

    def test_add_concept(self):
        """Test adding concepts."""
        net = SemanticNetwork()
        concept = net.add_concept("dog", properties={"type": "canine"})
        assert concept.name == "dog"
        assert net.has_concept("dog")

    def test_add_relation(self, network):
        """Test adding relations."""
        rel = network.add_relation("dog", "bark", RelationType.CAN)
        assert rel.source == "dog"
        assert rel.target == "bark"

    def test_get_relations(self, network):
        """Test getting relations."""
        relations = network.get_relations("dog", RelationType.IS_A)
        targets = [r.target for r in relations]
        assert "animal" in targets
        assert "mammal" in targets

    def test_get_ancestors(self, network):
        """Test getting all ancestors."""
        ancestors = network.get_ancestors("dog")
        assert "animal" in ancestors
        assert "mammal" in ancestors

    def test_spreading_activation(self, network):
        """Test spreading activation."""
        pattern = network.activate("dog", initial_activation=1.0)
        assert pattern.source_concept == "dog"
        assert len(pattern.activations) > 0
        # Dog should have highest activation
        assert pattern.activations.get("dog", 0) > pattern.activations.get("animal", 0)

    def test_find_path(self, network):
        """Test path finding between concepts."""
        # Dog to animal has direct path
        path = network.find_path("dog", "animal")
        assert path is not None
        assert len(path) > 0

    def test_semantic_distance(self, network):
        """Test semantic distance calculation."""
        dist = network.semantic_distance("dog", "animal")
        assert dist is not None
        assert dist > 0
        # Dog to itself should be 0
        assert network.semantic_distance("dog", "dog") == 0

    def test_query(self, network):
        """Test network queries."""
        results = network.query(subject="dog", relation=RelationType.IS_A)
        assert len(results) > 0

    def test_statistics(self, network):
        """Test network statistics."""
        stats = network.statistics()
        assert stats["n_concepts"] == 4
        assert stats["n_relations"] > 0


# =============================================================================
# Ontology Tests
# =============================================================================

class TestOntologyNode:
    """Tests for OntologyNode class."""

    def test_node_creation(self):
        """Test basic node creation."""
        node = OntologyNode(name="Animal", definition="A living creature")
        assert node.name == "Animal"
        assert node.definition == "A living creature"
        assert node.depth == 0

    def test_node_with_properties(self):
        """Test node with properties."""
        node = OntologyNode(
            name="Mammal",
            properties={"warm_blooded": True},
            constraints=["has_fur:required"],
        )
        assert node.properties["warm_blooded"] is True
        assert "has_fur:required" in node.constraints


class TestOntology:
    """Tests for Ontology class."""

    @pytest.fixture
    def ontology(self):
        """Create a test ontology."""
        ont = Ontology(name="animals")
        ont.add_node("Thing", is_abstract=True)
        ont.add_node("Animal", properties={"living": True})
        ont.add_node("Mammal", properties={"warm_blooded": True})
        ont.add_node("Dog", properties={"barks": True})
        ont.add_node("Cat", properties={"meows": True})
        ont.add_is_a("Animal", "Thing")
        ont.add_is_a("Mammal", "Animal")
        ont.add_is_a("Dog", "Mammal")
        ont.add_is_a("Cat", "Mammal")
        return ont

    def test_add_node(self):
        """Test adding nodes."""
        ont = Ontology()
        node = ont.add_node("Animal", definition="A living creature")
        assert node.name == "Animal"
        assert ont.has_node("Animal")

    def test_add_is_a(self, ontology):
        """Test IS-A relations."""
        assert ontology.is_subclass("Dog", "Mammal")
        assert ontology.is_subclass("Dog", "Animal")
        assert not ontology.is_subclass("Animal", "Dog")

    def test_get_parents(self, ontology):
        """Test getting direct parents."""
        parents = ontology.get_parents("Dog")
        assert "Mammal" in parents

    def test_get_children(self, ontology):
        """Test getting direct children."""
        children = ontology.get_children("Mammal")
        assert "Dog" in children
        assert "Cat" in children

    def test_get_ancestors(self, ontology):
        """Test getting all ancestors."""
        ancestors = ontology.get_ancestors("Dog")
        assert "Mammal" in ancestors
        assert "Animal" in ancestors
        assert "Thing" in ancestors

    def test_get_descendants(self, ontology):
        """Test getting all descendants."""
        descendants = ontology.get_descendants("Mammal")
        assert "Dog" in descendants
        assert "Cat" in descendants

    def test_inherited_properties(self, ontology):
        """Test property inheritance."""
        props = ontology.get_inherited_properties("Dog")
        assert props["living"] is True  # From Animal
        assert props["warm_blooded"] is True  # From Mammal
        assert props["barks"] is True  # Own property

    def test_lowest_common_ancestor(self, ontology):
        """Test LCA finding."""
        lca = ontology.lowest_common_ancestor("Dog", "Cat")
        assert lca == "Mammal"

    def test_get_siblings(self, ontology):
        """Test getting siblings."""
        siblings = ontology.get_siblings("Dog")
        assert "Cat" in siblings

    def test_add_part_of(self):
        """Test PART-OF relations."""
        ont = Ontology()
        ont.add_node("Car")
        ont.add_node("Wheel")
        ont.add_part_of("Wheel", "Car")
        parts = ont.get_parts("Car")
        assert "Wheel" in parts

    def test_validate_constraints(self, ontology):
        """Test constraint validation."""
        ontology._nodes["Animal"].constraints = ["name:required"]
        valid, violations = ontology.validate_constraints("Dog", {"name": "Rex"})
        assert valid
        valid, violations = ontology.validate_constraints("Dog", {})
        assert not valid

    def test_statistics(self, ontology):
        """Test ontology statistics."""
        stats = ontology.statistics()
        assert stats["n_nodes"] == 5
        assert stats["n_relations"] == 4


# =============================================================================
# Fact Store Tests
# =============================================================================

class TestTriple:
    """Tests for Triple class."""

    def test_triple_creation(self):
        """Test basic triple creation."""
        triple = Triple("dog", "is_a", "animal")
        assert triple.subject == "dog"
        assert triple.predicate == "is_a"
        assert triple.obj == "animal"

    def test_triple_hash_equality(self):
        """Test triple hashing and equality."""
        t1 = Triple("dog", "is_a", "animal")
        t2 = Triple("dog", "is_a", "animal")
        t3 = Triple("cat", "is_a", "animal")
        assert t1 == t2
        assert t1 != t3
        assert hash(t1) == hash(t2)

    def test_triple_as_tuple(self):
        """Test conversion to tuple."""
        triple = Triple("dog", "is_a", "animal")
        assert triple.as_tuple() == ("dog", "is_a", "animal")


class TestFact:
    """Tests for Fact class."""

    def test_fact_creation(self):
        """Test basic fact creation."""
        triple = Triple("dog", "is_a", "animal")
        fact = Fact(triple=triple, confidence=0.9, source="test")
        assert fact.subject == "dog"
        assert fact.predicate == "is_a"
        assert fact.obj == "animal"
        assert fact.confidence == 0.9

    def test_fact_validity(self):
        """Test temporal validity."""
        triple = Triple("dog", "is_a", "animal")
        fact = Fact(triple=triple, valid_from=0, valid_to=1000)
        assert fact.is_valid(500)
        assert not fact.is_valid(2000)


class TestFactStore:
    """Tests for FactStore class."""

    @pytest.fixture
    def store(self):
        """Create a test fact store."""
        fs = FactStore()
        fs.add("dog", "is_a", "animal")
        fs.add("cat", "is_a", "animal")
        fs.add("dog", "has", "fur")
        fs.add("cat", "has", "fur")
        fs.add("dog", "can", "bark")
        return fs

    def test_add_fact(self):
        """Test adding facts."""
        fs = FactStore()
        fact = fs.add("dog", "is_a", "animal")
        assert fact.subject == "dog"
        assert fs.exists("dog", "is_a", "animal")

    def test_add_duplicate(self):
        """Test adding duplicate facts."""
        fs = FactStore()
        fs.add("dog", "is_a", "animal", confidence=0.5)
        fs.add("dog", "is_a", "animal", confidence=0.9)
        # Should update to higher confidence
        facts = fs.query("dog", "is_a", "animal")
        assert len(facts) == 1
        assert facts[0].confidence == 0.9

    def test_query_by_subject(self, store):
        """Test querying by subject."""
        facts = store.query(subject="dog")
        assert len(facts) == 3

    def test_query_by_predicate(self, store):
        """Test querying by predicate."""
        facts = store.query(predicate="is_a")
        assert len(facts) == 2

    def test_query_by_object(self, store):
        """Test querying by object."""
        facts = store.query(obj="animal")
        assert len(facts) == 2

    def test_query_composite(self, store):
        """Test composite queries."""
        facts = store.query(subject="dog", predicate="is_a")
        assert len(facts) == 1
        assert facts[0].obj == "animal"

    def test_query_object(self, store):
        """Test convenience method for object query."""
        objs = store.query_object("dog", "is_a")
        assert "animal" in objs

    def test_query_subject(self, store):
        """Test convenience method for subject query."""
        subjs = store.query_subject("is_a", "animal")
        assert "dog" in subjs
        assert "cat" in subjs

    def test_remove_facts(self, store):
        """Test removing facts."""
        removed = store.remove(subject="dog", predicate="can")
        assert removed == 1
        assert not store.exists("dog", "can", "bark")

    def test_count(self, store):
        """Test fact counting."""
        assert store.count(predicate="is_a") == 2
        assert store.count(subject="dog") == 3

    def test_entities(self, store):
        """Test getting unique entities."""
        entities = store.entities()
        assert "dog" in entities
        assert "animal" in entities

    def test_bulk_add(self):
        """Test bulk adding."""
        fs = FactStore()
        triples = [
            ("a", "r", "b"),
            ("b", "r", "c"),
            ("c", "r", "d"),
        ]
        added = fs.bulk_add(triples)
        assert added == 3

    def test_export_triples(self, store):
        """Test exporting triples."""
        triples = store.export_triples()
        assert len(triples) == 5
        assert ("dog", "is_a", "animal") in triples

    def test_statistics(self, store):
        """Test store statistics."""
        stats = store.statistics()
        assert stats["n_facts"] == 5
        assert stats["n_subjects"] == 2


# =============================================================================
# Inference Engine Tests
# =============================================================================

class TestPattern:
    """Tests for Pattern class."""

    def test_pattern_creation(self):
        """Test basic pattern creation."""
        pattern = Pattern("?x", "is_a", "animal")
        assert pattern.subject == "?x"
        assert pattern.is_variable("?x")
        assert not pattern.is_variable("animal")

    def test_get_variables(self):
        """Test getting variables from pattern."""
        pattern = Pattern("?x", "?rel", "?y")
        vars = pattern.get_variables()
        assert "?x" in vars
        assert "?rel" in vars
        assert "?y" in vars

    def test_substitute(self):
        """Test variable substitution."""
        pattern = Pattern("?x", "is_a", "?y")
        grounded = pattern.substitute({"?x": "dog", "?y": "animal"})
        assert grounded.subject == "dog"
        assert grounded.obj == "animal"

    def test_matches(self):
        """Test pattern matching against fact."""
        pattern = Pattern("?x", "is_a", "animal")
        triple = Triple("dog", "is_a", "animal")
        fact = Fact(triple=triple)
        bindings = pattern.matches(fact)
        assert bindings is not None
        assert bindings["?x"] == "dog"

    def test_no_match(self):
        """Test pattern that doesn't match."""
        pattern = Pattern("?x", "is_a", "plant")
        triple = Triple("dog", "is_a", "animal")
        fact = Fact(triple=triple)
        bindings = pattern.matches(fact)
        assert bindings is None


class TestRule:
    """Tests for Rule class."""

    def test_rule_creation(self):
        """Test basic rule creation."""
        rule = Rule(
            name="transitivity",
            antecedents=[
                Pattern("?x", "is_a", "?y"),
                Pattern("?y", "is_a", "?z"),
            ],
            consequents=[
                Pattern("?x", "is_a", "?z"),
            ],
        )
        assert rule.name == "transitivity"
        assert len(rule.antecedents) == 2
        assert len(rule.consequents) == 1

    def test_get_variables(self):
        """Test getting all variables from rule."""
        rule = Rule(
            name="test",
            antecedents=[Pattern("?x", "r", "?y")],
            consequents=[Pattern("?y", "s", "?x")],
        )
        vars = rule.get_variables()
        assert "?x" in vars
        assert "?y" in vars


class TestInferenceEngine:
    """Tests for InferenceEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a test inference engine."""
        fs = FactStore()
        fs.add("dog", "is_a", "mammal")
        fs.add("mammal", "is_a", "animal")
        fs.add("cat", "is_a", "mammal")
        engine = InferenceEngine(fact_store=fs)
        engine.add_transitivity_rule("is_a")
        return engine

    def test_forward_chain(self, engine):
        """Test forward chaining inference."""
        results = engine.forward_chain()
        # Should derive dog is_a animal
        derived = engine.query("dog", "is_a", "animal")
        assert len(derived) > 0

    def test_backward_chain(self, engine):
        """Test backward chaining inference."""
        engine.forward_chain()  # Populate derived facts
        goal = Pattern("dog", "is_a", "animal")
        chain = engine.backward_chain(goal)
        assert chain.success

    def test_prove(self, engine):
        """Test convenience prove method."""
        engine.forward_chain()
        chain = engine.prove("dog", "is_a", "animal")
        assert chain.success

    def test_simple_rule(self):
        """Test adding a simple rule."""
        fs = FactStore()
        fs.add("socrates", "is_a", "human")
        engine = InferenceEngine(fact_store=fs)
        engine.add_simple_rule(
            name="mortality",
            if_pattern=("?x", "is_a", "human"),
            then_pattern=("?x", "is", "mortal"),
        )
        engine.forward_chain()
        derived = engine.query("socrates", "is", "mortal")
        assert len(derived) > 0

    def test_query_with_derived(self, engine):
        """Test querying includes derived facts."""
        engine.forward_chain()
        facts = engine.query(predicate="is_a")
        # Should include both base and derived
        assert len(facts) > 3

    def test_statistics(self, engine):
        """Test engine statistics."""
        engine.forward_chain()
        stats = engine.statistics()
        assert stats["n_rules"] == 1
        assert stats["inferences_made"] > 0


# =============================================================================
# Knowledge Graph Tests
# =============================================================================

class TestEntity:
    """Tests for Entity class."""

    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = Entity(name="dog", entity_type="animal")
        assert entity.name == "dog"
        assert entity.entity_type == "animal"

    def test_entity_hash_equality(self):
        """Test entity hashing and equality."""
        e1 = Entity(name="dog")
        e2 = Entity(name="dog")
        e3 = Entity(name="cat")
        assert e1 == e2
        assert e1 != e3
        assert hash(e1) == hash(e2)


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph class."""

    @pytest.fixture
    def graph(self):
        """Create a test knowledge graph."""
        kg = KnowledgeGraph(embedding_dim=32, random_seed=42)
        kg.add_entity("alice", entity_type="person")
        kg.add_entity("bob", entity_type="person")
        kg.add_entity("paris", entity_type="city")
        kg.add_entity("france", entity_type="country")
        kg.add_relation("lives_in")
        kg.add_relation("capital_of")
        kg.add_edge("alice", "lives_in", "paris")
        kg.add_edge("bob", "lives_in", "paris")
        kg.add_edge("paris", "capital_of", "france")
        return kg

    def test_add_entity(self):
        """Test adding entities."""
        kg = KnowledgeGraph()
        entity = kg.add_entity("dog", entity_type="animal")
        assert entity.name == "dog"
        assert kg.has_entity("dog")
        assert entity.embedding is not None

    def test_add_relation(self):
        """Test adding relations."""
        kg = KnowledgeGraph()
        relation = kg.add_relation("is_a", symmetric=False, transitive=True)
        assert relation.name == "is_a"
        assert relation.transitive is True

    def test_add_edge(self, graph):
        """Test adding edges."""
        assert graph.has_edge("alice", "lives_in", "paris")

    def test_add_edge_auto_create(self):
        """Test edges auto-create entities and relations."""
        kg = KnowledgeGraph()
        kg.add_edge("a", "r", "b")
        assert kg.has_entity("a")
        assert kg.has_entity("b")
        assert kg.get_relation("r") is not None

    def test_get_neighbors(self, graph):
        """Test getting neighbors."""
        neighbors = graph.get_neighbors("alice", direction="outgoing")
        assert ("lives_in", "paris") in neighbors

    def test_score_triple(self, graph):
        """Test triple scoring."""
        score = graph.score_triple("alice", "lives_in", "paris")
        assert score is not None
        assert score >= 0

    def test_predict_tail(self, graph):
        """Test tail prediction."""
        predictions = graph.predict_tail("alice", "lives_in", top_k=5)
        assert len(predictions) > 0
        # All should be entities
        for name, score in predictions:
            assert graph.has_entity(name)

    def test_predict_head(self, graph):
        """Test head prediction."""
        predictions = graph.predict_head("capital_of", "france", top_k=5)
        assert len(predictions) > 0

    def test_find_path(self, graph):
        """Test path finding."""
        path = graph.find_path("alice", "france")
        assert path is not None
        # Should go alice -> paris -> france
        assert len(path) == 2

    def test_train_step(self, graph):
        """Test training step."""
        triples = [("alice", "lives_in", "paris"), ("bob", "lives_in", "paris")]
        loss = graph.train_step(triples, negative_ratio=2)
        assert loss >= 0

    def test_get_similar_entities(self, graph):
        """Test similarity search."""
        similar = graph.get_similar_entities("alice", top_k=5)
        assert len(similar) > 0
        # Should be entity names with scores
        for name, score in similar:
            assert graph.has_entity(name)

    def test_get_entities_by_type(self, graph):
        """Test getting entities by type."""
        people = graph.get_entities_by_type("person")
        assert len(people) == 2
        names = [p.name for p in people]
        assert "alice" in names
        assert "bob" in names

    def test_query(self, graph):
        """Test graph queries."""
        query = GraphQuery(head="alice", relation="lives_in")
        results = graph.query(query)
        assert len(results) > 0
        assert results[0][2] == "paris"

    def test_statistics(self, graph):
        """Test graph statistics."""
        stats = graph.statistics()
        assert stats["n_entities"] == 4
        assert stats["n_relations"] == 2
        assert stats["n_edges"] == 3


# =============================================================================
# Common Sense Tests
# =============================================================================

class TestPhysicsReasoner:
    """Tests for PhysicsReasoner class."""

    @pytest.fixture
    def reasoner(self):
        """Create a physics reasoner."""
        return PhysicsReasoner()

    def test_predict_effect_gravity(self, reasoner):
        """Test gravity effect prediction."""
        effects = reasoner.predict_effect("ball", "the ball falls from table")
        assert len(effects) > 0
        # Should predict falling
        assert any("falls" in e[0].lower() for e in effects)

    def test_breakable_object(self, reasoner):
        """Test breakable object reasoning."""
        prob = reasoner.will_break("glass", "drop")
        assert prob > 0.5  # Glass should have high break probability when dropped

    def test_will_break_non_breakable(self, reasoner):
        """Test non-breakable object."""
        prob = reasoner.will_break("rock", "drop")
        assert prob == 0.0  # Rock is not breakable

    def test_can_contain(self, reasoner):
        """Test container reasoning."""
        # Glass can contain water
        assert reasoner.can_contain("glass", "water") is True

    def test_get_applicable_rules(self, reasoner):
        """Test getting applicable physics rules."""
        rules = reasoner.get_applicable_rules("object falls")
        assert len(rules) > 0


class TestSocialReasoner:
    """Tests for SocialReasoner class."""

    @pytest.fixture
    def reasoner(self):
        """Create a social reasoner."""
        return SocialReasoner()

    def test_get_applicable_norms_meeting(self, reasoner):
        """Test getting norms for meeting context."""
        norms = reasoner.get_applicable_norms("meeting someone")
        assert len(norms) > 0
        norm_names = [n.name for n in norms]
        assert "greeting" in norm_names

    def test_get_applicable_norms_help(self, reasoner):
        """Test getting norms for receiving help."""
        norms = reasoner.get_applicable_norms("receiving help")
        assert len(norms) > 0

    def test_infer_goal(self, reasoner):
        """Test inferring goals from actions."""
        goals = reasoner.infer_goal(["eating lunch", "cooking dinner"])
        assert len(goals) > 0
        goal_names = [g[0] for g in goals]
        assert "satisfy hunger" in goal_names

    def test_infer_emotion(self, reasoner):
        """Test inferring emotions from outcome."""
        emotions = reasoner.infer_emotion("competition", "success")
        assert len(emotions) > 0
        emotion_names = [e[0] for e in emotions]
        assert "happy" in emotion_names

    def test_trust_level(self, reasoner):
        """Test trust level estimation."""
        # Doctor should have high trust for health topics
        trust = reasoner.trust_level("doctor", "health advice")
        assert trust > 0.8


class TestTemporalReasoner:
    """Tests for TemporalReasoner class."""

    @pytest.fixture
    def reasoner(self):
        """Create a temporal reasoner."""
        return TemporalReasoner()

    def test_get_typical_duration(self, reasoner):
        """Test getting typical duration."""
        duration = reasoner.get_typical_duration("eating_meal")
        assert duration is not None
        assert duration[0] < duration[1]  # min < max

    def test_get_typical_sequence(self, reasoner):
        """Test getting typical sequence."""
        sequence = reasoner.get_typical_sequence("morning_routine")
        assert len(sequence) > 0
        assert "wake_up" in sequence

    def test_sequence_ordering(self, reasoner):
        """Test sequence ordering."""
        events = ["shower", "wake_up", "eat_breakfast"]
        ordered = reasoner.order_events(events, context="morning_routine")
        # Should reorder to typical morning sequence
        assert ordered is not None
        assert len(ordered) == 3

    def test_infer_relation(self, reasoner):
        """Test inferring temporal relation."""
        relation = reasoner.infer_relation("wake_up", "eat_breakfast")
        assert relation == "before"

    def test_is_plausible_duration(self, reasoner):
        """Test duration plausibility."""
        # 30 minutes for eating is plausible
        assert reasoner.is_plausible_duration("eating_meal", 30) is True
        # 500 minutes for eating is not plausible
        assert reasoner.is_plausible_duration("eating_meal", 500) is False


class TestCommonSenseKB:
    """Tests for CommonSenseKB class."""

    @pytest.fixture
    def kb(self):
        """Create a common sense KB."""
        return CommonSenseKB()

    def test_get_properties(self, kb):
        """Test getting object properties."""
        props = kb.get_properties("glass")
        assert PhysicsProperty.BREAKABLE in props

    def test_set_properties(self, kb):
        """Test setting object properties."""
        kb.set_properties("ball", {PhysicsProperty.SOLID, PhysicsProperty.LIGHT})
        props = kb.get_properties("ball")
        assert PhysicsProperty.LIGHT in props

    def test_get_typical_location(self, kb):
        """Test getting typical location."""
        location = kb.get_typical_location("refrigerator")
        assert location == "kitchen"

    def test_get_objects_in_location(self, kb):
        """Test getting objects in location."""
        objects = kb.get_objects_in_location("kitchen")
        assert "stove" in objects
        assert "refrigerator" in objects

    def test_get_typical_actions(self, kb):
        """Test getting typical actions."""
        actions = kb.get_typical_actions("knife")
        assert "cut" in actions

    def test_get_social_role(self, kb):
        """Test getting social role info."""
        role_info = kb.get_social_role("doctor")
        assert role_info["authority"] is True
        assert role_info["trust"] > 0.8


class TestCommonSenseReasoner:
    """Tests for CommonSenseReasoner class."""

    @pytest.fixture
    def reasoner(self):
        """Create a common sense reasoner."""
        return CommonSenseReasoner()

    def test_reason_physics(self, reasoner):
        """Test physics reasoning through main interface."""
        result = reasoner.reason(
            "what happens when it falls?",
            context={"object": "glass"}
        )
        assert result is not None
        assert "physics_effects" in result

    def test_reason_social(self, reasoner):
        """Test social reasoning through main interface."""
        result = reasoner.reason("meeting someone new")
        assert result is not None
        assert "social_norms" in result
        assert len(result["social_norms"]) > 0

    def test_is_plausible(self, reasoner):
        """Test plausibility checking."""
        # Plausible statement
        plausible, conf, reason = reasoner.is_plausible("The dog ran in the park")
        assert plausible is True

        # Implausible statement - human flying unaided
        plausible, conf, reason = reasoner.is_plausible("The human can fly by flapping arms")
        assert plausible is False

    def test_fill_gaps(self, reasoner):
        """Test filling gaps in story."""
        story = ["John was at home", "John was at work"]
        filled = reasoner.fill_gaps(story)
        assert len(filled) > len(story)  # Should add implicit travel

    def test_statistics(self, reasoner):
        """Test getting statistics."""
        stats = reasoner.statistics()
        assert stats["physics_rules"] > 0
        assert stats["social_norms"] > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestKnowledgeIntegration:
    """Integration tests across knowledge components."""

    def test_semantic_network_to_fact_store(self):
        """Test converting semantic network to fact store."""
        net = SemanticNetwork()
        net.add_concept("dog")
        net.add_concept("animal")
        net.add_relation("dog", "animal", RelationType.IS_A)

        fs = FactStore()
        # Extract relations from semantic network
        for concept in [net.get_concept("dog")]:
            relations = net.get_relations(concept.name, RelationType.IS_A)
            for rel in relations:
                fs.add(concept.name, "is_a", rel.target)

        assert fs.exists("dog", "is_a", "animal")

    def test_ontology_to_knowledge_graph(self):
        """Test mapping ontology to knowledge graph."""
        ont = Ontology()
        ont.add_node("Animal")
        ont.add_node("Dog")
        ont.add_is_a("Dog", "Animal")

        kg = KnowledgeGraph()
        for node in [ont.get_node("Animal"), ont.get_node("Dog")]:
            if node:
                kg.add_entity(node.name, entity_type="concept")

        # Add hierarchy relations
        for child in ["Dog"]:
            parents = ont.get_parents(child)
            for parent in parents:
                kg.add_edge(child, "is_a", parent)

        assert kg.has_edge("Dog", "is_a", "Animal")

    def test_inference_with_knowledge_graph(self):
        """Test inference engine using knowledge graph facts."""
        kg = KnowledgeGraph()
        kg.add_edge("socrates", "is_a", "human")
        kg.add_edge("human", "is_a", "mortal")

        fs = FactStore()
        # Export KG edges to fact store
        for triple in [("socrates", "is_a", "human"), ("human", "is_a", "mortal")]:
            fs.add(*triple)

        engine = InferenceEngine(fact_store=fs)
        engine.add_transitivity_rule("is_a")
        engine.forward_chain()

        # Should derive socrates is_a mortal
        derived = engine.query("socrates", "is_a", "mortal")
        assert len(derived) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
