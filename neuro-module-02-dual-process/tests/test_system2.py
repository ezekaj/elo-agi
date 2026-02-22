"""Tests for System 2 components"""

import pytest
import numpy as np
import time
from neuro.modules.m02_dual_process.system2.working_memory import WorkingMemory
from neuro.modules.m02_dual_process.system2.cognitive_control import (
    CognitiveControl,
    Response,
    ConflictLevel,
)
from neuro.modules.m02_dual_process.system2.relational_reasoning import (
    RelationalReasoning,
    RelationType,
)


class TestWorkingMemory:
    """Tests for limited-capacity working memory"""

    def test_capacity_limit(self):
        wm = WorkingMemory(capacity=4)

        # Fill to capacity
        for i in range(4):
            wm.store(f"item_{i}", f"content_{i}")

        assert wm.current_load == 4
        assert wm.is_full

        # Adding one more should displace oldest
        wm.store("item_4", "content_4")
        assert wm.current_load == 4

    def test_retrieval_refreshes(self):
        wm = WorkingMemory(capacity=5)

        wm.store("test", "content")
        slot = wm.slots["test"]
        original_access = slot.last_accessed

        time.sleep(0.01)
        wm.retrieve("test")

        assert slot.last_accessed > original_access
        assert slot.access_count == 1

    def test_decay(self):
        wm = WorkingMemory(capacity=5, decay_rate=0.5, decay_threshold=0.3)

        wm.store("decaying", "content", priority=0.4)

        # Force decay
        wm.slots["decaying"].last_accessed = time.time() - 10
        wm._apply_decay()

        # Should have decayed
        assert wm.slots.get("decaying") is None or wm.slots["decaying"].activation < 0.4

    def test_chunking(self):
        wm = WorkingMemory(capacity=4)

        # Store individual items
        wm.store("a", "apple")
        wm.store("b", "banana")
        wm.store("c", "cherry")

        # Chunk them
        wm.chunk(["a", "b", "c"], "fruits")

        # Now only one item
        assert wm.current_load == 1
        chunked = wm.retrieve("fruits")
        assert len(chunked.items) == 3

    def test_binding(self):
        wm = WorkingMemory()

        wm.store("X", "variable_x")
        wm.bind("X", "role", "agent")
        wm.bind("X", "value", 42)

        bindings = wm.get_bindings("X")
        assert bindings["role"] == "agent"
        assert bindings["value"] == 42

    def test_serial_processing(self):
        wm = WorkingMemory()

        wm.store("step1", "first")
        wm.store("step2", "second")
        wm.store("step3", "third")

        # Serial access
        results = wm.serial_process(["step1", "step2", "step3"])

        assert results == ["first", "second", "third"]


class TestCognitiveControl:
    """Tests for conflict detection and inhibition"""

    def test_no_conflict_single_response(self):
        cc = CognitiveControl()

        responses = [Response(id="only", activation=0.9, source="test")]
        signal = cc.detect_conflict(responses)

        assert signal.level == ConflictLevel.NONE
        assert signal.recommended_action == "proceed"

    def test_detect_conflict(self):
        cc = CognitiveControl()

        # Two strong competing responses
        responses = [
            Response(id="A", activation=0.8, source="s1"),
            Response(id="B", activation=0.75, source="s1"),
        ]
        signal = cc.detect_conflict(responses)

        assert signal.level != ConflictLevel.NONE
        assert signal.conflict_energy > 0

    def test_severe_conflict(self):
        cc = CognitiveControl()

        # Many equally strong responses
        responses = [Response(id=f"R{i}", activation=0.9, source="s1") for i in range(4)]
        signal = cc.detect_conflict(responses)

        assert signal.level in [ConflictLevel.HIGH, ConflictLevel.SEVERE]

    def test_error_signal(self):
        cc = CognitiveControl()

        signal = cc.error_signal(
            expected=np.array([1.0, 0.0]), actual=np.array([0.0, 1.0]), error_type="prediction"
        )

        assert signal.magnitude > 0
        assert signal.error_type == "prediction"

    def test_inhibition(self):
        cc = CognitiveControl(inhibition_strength=0.7)

        weak_response = Response(id="weak", activation=0.5, source="habit")
        strong_response = Response(id="strong", activation=0.95, source="habit")

        # Weak response can be inhibited
        success, remaining = cc.inhibit(weak_response)
        assert success

        # Strong response harder to inhibit
        success, remaining = cc.inhibit(strong_response)
        assert not success
        assert remaining > 0

    def test_should_engage_system2(self):
        cc = CognitiveControl()

        # Low conflict - no S2 needed
        low_conflict = [Response(id="clear", activation=0.9, source="s1")]
        assert not cc.should_engage_system2(low_conflict)

        # High conflict - S2 needed
        high_conflict = [
            Response(id="A", activation=0.8, source="s1"),
            Response(id="B", activation=0.8, source="s1"),
            Response(id="C", activation=0.75, source="s1"),
        ]
        assert cc.should_engage_system2(high_conflict)

    def test_attention_allocation(self):
        cc = CognitiveControl()

        requested = {"task_A": 0.6, "task_B": 0.6}
        allocated = cc.allocate_attention(requested)

        # Total should not exceed budget (1.0)
        assert sum(allocated.values()) <= 1.0


class TestRelationalReasoning:
    """Tests for compositional binding"""

    def test_basic_binding(self):
        rr = RelationalReasoning()

        dog = rr.create_element("dog", type_tag="animal")
        cat = rr.create_element("cat", type_tag="animal")

        structure = rr.bind(dog, RelationType.CAUSES, cat)

        assert len(structure.elements) == 2
        assert len(structure.relations) == 1
        assert structure.relations[0].relation_type == RelationType.CAUSES

    def test_composition(self):
        rr = RelationalReasoning()

        a = rr.create_element("A")
        b = rr.create_element("B")
        c = rr.create_element("C")

        s1 = rr.bind(a, RelationType.BEFORE, b)
        s2 = rr.bind(b, RelationType.BEFORE, c)

        composed = rr.compose(s1, s2)

        assert len(composed.elements) == 3
        assert len(composed.relations) == 2

    def test_modifier_binding(self):
        """Test 'jump twice' style composition"""
        rr = RelationalReasoning()

        jump = rr.create_element("jump", type_tag="action")
        twice = rr.create_element("twice", type_tag="modifier")

        structure = rr.bind_modifier(jump, twice)

        assert len(structure.relations) == 1
        assert structure.relations[0].relation_type == RelationType.MODIFIER

    def test_action_structure(self):
        rr = RelationalReasoning()

        bite = rr.create_element("bite", type_tag="action")
        dog = rr.create_element("dog", type_tag="entity")
        cat = rr.create_element("cat", type_tag="entity")

        structure = rr.create_action_structure(action=bite, agent=dog, patient=cat)

        assert len(structure.elements) == 3

        # Check roles
        agent_rels = [r for r in structure.relations if r.relation_type == RelationType.AGENT]
        patient_rels = [r for r in structure.relations if r.relation_type == RelationType.PATIENT]

        assert len(agent_rels) == 1
        assert len(patient_rels) == 1

    def test_decomposition(self):
        rr = RelationalReasoning()

        a = rr.create_element("A")
        b = rr.create_element("B")
        structure = rr.bind(a, RelationType.EQUALS, b)

        elements, relations = rr.decompose(structure)

        assert len(elements) == 2
        assert len(relations) == 1

    def test_analogy(self):
        """Test analogical mapping"""
        rr = RelationalReasoning()

        # Source: dog is to puppy
        dog = rr.create_element("dog", element_id="dog")
        puppy = rr.create_element("puppy", element_id="puppy")
        source = rr.bind(dog, RelationType.IS_A, puppy)

        # Target: cat is to ?
        cat = rr.create_element("cat", element_id="cat")
        kitten = rr.create_element("kitten", element_id="kitten")
        target_elements = {"cat": cat, "kitten": kitten}

        # Map analogy
        result = rr.analogy(source, target_elements)

        assert result is not None
        assert len(result.relations) == 1
        assert result.relations[0].relation_type == RelationType.IS_A


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
