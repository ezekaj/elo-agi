"""Tests for skill library."""

import pytest
import numpy as np

from neuro.modules.planning.skill_library import (
    Skill,
    SkillMetadata,
    SkillType,
    SkillLibrary,
    SkillComposer,
)

class TestSkillMetadata:
    """Tests for SkillMetadata class."""

    def test_initial_values(self):
        meta = SkillMetadata()
        assert meta.use_count == 0
        assert meta.success_count == 0
        assert meta.success_rate == 0.0

    def test_success_rate_calculation(self):
        meta = SkillMetadata(use_count=10, success_count=7)
        assert meta.success_rate == pytest.approx(0.7)

    def test_record_use(self):
        meta = SkillMetadata()
        meta.record_use(success=True, steps=5, reward=1.0, timestamp=1.0)

        assert meta.use_count == 1
        assert meta.success_count == 1
        assert meta.average_steps == 5.0
        assert meta.average_reward == 1.0
        assert meta.last_used == 1.0

    def test_record_multiple_uses(self):
        meta = SkillMetadata()
        meta.record_use(success=True, steps=10, reward=2.0, timestamp=1.0)
        meta.record_use(success=False, steps=20, reward=0.0, timestamp=2.0)

        assert meta.use_count == 2
        assert meta.success_count == 1
        assert meta.success_rate == 0.5
        assert meta.average_steps == pytest.approx(15.0)
        assert meta.average_reward == pytest.approx(1.0)

class TestSkill:
    """Tests for Skill class."""

    def test_skill_creation(self):
        skill = Skill(
            name="test_skill",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "action",
            termination=lambda s, p: False,
        )
        assert skill.name == "test_skill"
        assert skill.skill_type == SkillType.PRIMITIVE

    def test_can_execute(self):
        skill = Skill(
            name="conditional",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: s > 0,
            effects=lambda s: s,
            policy=lambda s, p: "a",
            termination=lambda s, p: False,
        )
        assert skill.can_execute(5)
        assert not skill.can_execute(-1)

    def test_execute_step(self):
        skill = Skill(
            name="test",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s + 1,
            policy=lambda s, p: s * 2,
            termination=lambda s, p: s > 10,
        )
        action, terminated = skill.execute_step(3)
        assert action == 6
        assert not terminated

        action, terminated = skill.execute_step(15)
        assert terminated

    def test_predict_effects(self):
        skill = Skill(
            name="doubler",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s * 2,
            policy=lambda s, p: "a",
            termination=lambda s, p: False,
        )
        assert skill.predict_effects(5) == 10

    def test_parameters(self):
        skill = Skill(
            name="param_skill",
            skill_type=SkillType.PARAMETERIZED,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: p.get("speed", 1) if p else 1,
            termination=lambda s, p: False,
            parameters={"speed": 5},
        )
        action, _ = skill.execute_step(0)
        assert action == 5

        skill.set_parameter("speed", 10)
        action, _ = skill.execute_step(0)
        assert action == 10

    def test_similarity(self):
        skill1 = Skill(
            name="s1",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "a",
            termination=lambda s, p: False,
            embedding=np.array([1.0, 0.0, 0.0]),
        )
        skill2 = Skill(
            name="s2",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "b",
            termination=lambda s, p: False,
            embedding=np.array([1.0, 0.0, 0.0]),
        )
        skill3 = Skill(
            name="s3",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "c",
            termination=lambda s, p: False,
            embedding=np.array([0.0, 1.0, 0.0]),
        )
        assert skill1.similarity(skill2) == pytest.approx(1.0)
        assert skill1.similarity(skill3) == pytest.approx(0.0)

class TestSkillComposer:
    """Tests for SkillComposer class."""

    def test_sequence_composition(self):
        composer = SkillComposer(random_seed=42)

        skill1 = Skill(
            name="s1",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s + 1,
            policy=lambda s, p: "a1",
            termination=lambda s, p: True,
        )
        skill2 = Skill(
            name="s2",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s * 2,
            policy=lambda s, p: "a2",
            termination=lambda s, p: True,
        )

        composed = composer.sequence([skill1, skill2], "seq")

        assert composed.name == "seq"
        assert composed.skill_type == SkillType.COMPOSITE
        assert "s1" in composed.child_skills
        assert "s2" in composed.child_skills

        predicted = composed.predict_effects(5)
        assert predicted == 12

    def test_conditional_composition(self):
        composer = SkillComposer(random_seed=42)

        skill_pos = Skill(
            name="positive",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: abs(s),
            policy=lambda s, p: "pos",
            termination=lambda s, p: True,
        )
        skill_neg = Skill(
            name="negative",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: -abs(s),
            policy=lambda s, p: "neg",
            termination=lambda s, p: True,
        )

        composed = composer.conditional(
            condition=lambda s: s >= 0,
            if_skill=skill_pos,
            else_skill=skill_neg,
            name="cond",
        )

        assert composed.predict_effects(5) == 5
        assert composed.predict_effects(-5) == -5

class TestSkillLibrary:
    """Tests for SkillLibrary class."""

    def test_creation(self):
        library = SkillLibrary(embedding_dim=64, random_seed=42)
        assert library.embedding_dim == 64

    def test_register_skill(self):
        library = SkillLibrary(random_seed=42)
        skill = Skill(
            name="test",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "a",
            termination=lambda s, p: False,
        )
        name = library.register_skill(skill)

        assert name == "test"
        assert library.get_skill("test") is not None
        assert skill.embedding is not None

    def test_get_skill(self):
        library = SkillLibrary(random_seed=42)
        skill = Skill(
            name="test",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "a",
            termination=lambda s, p: False,
        )
        library.register_skill(skill)

        retrieved = library.get_skill("test")
        assert retrieved.name == "test"
        assert library.get_skill("nonexistent") is None

    def test_retrieve_applicable(self):
        library = SkillLibrary(random_seed=42)

        always_skill = Skill(
            name="always",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "a",
            termination=lambda s, p: False,
        )
        conditional_skill = Skill(
            name="positive_only",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: s > 0,
            effects=lambda s: s,
            policy=lambda s, p: "b",
            termination=lambda s, p: False,
        )

        library.register_skill(always_skill)
        library.register_skill(conditional_skill)

        applicable = library.retrieve_applicable(5)
        names = [s.name for s in applicable]
        assert "always" in names
        assert "positive_only" in names

        applicable = library.retrieve_applicable(-5)
        names = [s.name for s in applicable]
        assert "always" in names
        assert "positive_only" not in names

    def test_retrieve_similar(self):
        library = SkillLibrary(embedding_dim=3, random_seed=42)

        skill1 = Skill(
            name="s1",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "a",
            termination=lambda s, p: False,
            embedding=np.array([1.0, 0.0, 0.0]),
        )
        skill2 = Skill(
            name="s2",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "b",
            termination=lambda s, p: False,
            embedding=np.array([0.9, 0.1, 0.0]),
        )

        library.register_skill(skill1)
        library.register_skill(skill2)

        similar = library.retrieve_similar(np.array([1.0, 0.0, 0.0]), top_k=2)
        assert len(similar) == 2
        assert similar[0][0].name == "s1"

    def test_compose_skills(self):
        library = SkillLibrary(random_seed=42)

        skill1 = Skill(
            name="s1",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s + 1,
            policy=lambda s, p: "a",
            termination=lambda s, p: True,
        )
        skill2 = Skill(
            name="s2",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s * 2,
            policy=lambda s, p: "b",
            termination=lambda s, p: True,
        )

        library.register_skill(skill1)
        library.register_skill(skill2)

        composed = library.compose_skills(["s1", "s2"], "sequence", "composed")
        assert composed is not None
        assert composed.name == "composed"
        assert library.get_skill("composed") is not None

    def test_transfer_skill(self):
        library = SkillLibrary(random_seed=42)

        skill = Skill(
            name="original",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "a",
            termination=lambda s, p: False,
        )
        skill.metadata.domains.add("source_domain")
        library.register_skill(skill)

        transferred = library.transfer_skill("original", "target_domain")
        assert transferred is not None
        assert "target_domain" in transferred.metadata.domains
        assert transferred.name == "original_target_domain"

    def test_record_skill_use(self):
        library = SkillLibrary(random_seed=42)
        skill = Skill(
            name="test",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "a",
            termination=lambda s, p: False,
        )
        library.register_skill(skill)

        library.record_skill_use("test", success=True, steps=10, reward=1.5)

        retrieved = library.get_skill("test")
        assert retrieved.metadata.use_count == 1
        assert retrieved.metadata.success_count == 1

    def test_get_best_skill(self):
        library = SkillLibrary(random_seed=42)

        skill1 = Skill(
            name="good",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "a",
            termination=lambda s, p: False,
        )
        skill2 = Skill(
            name="bad",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "b",
            termination=lambda s, p: False,
        )

        library.register_skill(skill1)
        library.register_skill(skill2)

        for _ in range(10):
            library.record_skill_use("good", success=True, steps=5, reward=2.0)
            library.record_skill_use("bad", success=False, steps=10, reward=0.5)

        best = library.get_best_skill(0, metric="success_rate")
        assert best.name == "good"

        best = library.get_best_skill(0, metric="reward")
        assert best.name == "good"

    def test_domain_filtering(self):
        library = SkillLibrary(random_seed=42)

        skill1 = Skill(
            name="domain_a",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "a",
            termination=lambda s, p: False,
        )
        skill1.metadata.domains.add("A")

        skill2 = Skill(
            name="domain_b",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "b",
            termination=lambda s, p: False,
        )
        skill2.metadata.domains.add("B")

        library.register_skill(skill1)
        library.register_skill(skill2)

        applicable = library.retrieve_applicable(0, domain="A")
        names = [s.name for s in applicable]
        assert "domain_a" in names
        assert "domain_b" not in names

    def test_list_skills(self):
        library = SkillLibrary(random_seed=42)

        for name in ["s1", "s2", "s3"]:
            skill = Skill(
                name=name,
                skill_type=SkillType.PRIMITIVE,
                preconditions=lambda s: True,
                effects=lambda s: s,
                policy=lambda s, p: "a",
                termination=lambda s, p: False,
            )
            library.register_skill(skill)

        skills = library.list_skills()
        assert len(skills) == 3

    def test_statistics(self):
        library = SkillLibrary(random_seed=42)
        skill = Skill(
            name="test",
            skill_type=SkillType.PRIMITIVE,
            preconditions=lambda s: True,
            effects=lambda s: s,
            policy=lambda s, p: "a",
            termination=lambda s, p: False,
        )
        library.register_skill(skill)

        stats = library.statistics()
        assert stats["total_skills"] == 1
        assert "by_type" in stats
