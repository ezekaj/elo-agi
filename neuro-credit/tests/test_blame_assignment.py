"""Tests for blame assignment."""

import pytest
import numpy as np

from src.blame_assignment import (
    Failure,
    FailureType,
    BlameResult,
    ModuleAction,
    BlameAssignment,
    CounterfactualBlame,
)


class TestFailure:
    """Tests for Failure class."""

    def test_failure_creation(self):
        failure = Failure(
            failure_type=FailureType.GOAL_UNREACHED,
            description="Failed to reach target",
            timestamp=100,
            severity=0.8,
        )
        assert failure.failure_type == FailureType.GOAL_UNREACHED
        assert failure.severity == 0.8


class TestModuleAction:
    """Tests for ModuleAction class."""

    def test_action_creation(self):
        action = ModuleAction(
            module_id="mod1",
            action="move_forward",
            state_before={"x": 0},
            state_after={"x": 1},
            timestamp=10,
        )
        assert action.module_id == "mod1"


class TestCounterfactualBlame:
    """Tests for CounterfactualBlame class."""

    def test_creation(self):
        cf = CounterfactualBlame(random_seed=42)
        assert cf is not None

    def test_set_world_model(self):
        cf = CounterfactualBlame(random_seed=42)
        cf.set_world_model(lambda s, a: s + a)
        assert cf._world_model is not None

    def test_compute_counterfactual(self):
        cf = CounterfactualBlame(random_seed=42)
        cf.set_world_model(lambda s, a: s + a if s is not None else a)

        trajectory = [
            ModuleAction("m1", 1, 0, 1, 0),
            ModuleAction("m1", 1, 1, 2, 1),
        ]

        outcome, simulated = cf.compute_counterfactual(
            trajectory, "m1", [2, 2]
        )

        assert len(simulated) == 2

    def test_compute_counterfactual_no_world_model(self):
        cf = CounterfactualBlame(random_seed=42)

        trajectory = [ModuleAction("m1", 1, 0, 1, 0)]

        outcome, simulated = cf.compute_counterfactual(
            trajectory, "m1", [2]
        )

        assert outcome == 0.0
        assert len(simulated) == 0


class TestBlameAssignment:
    """Tests for BlameAssignment class."""

    def test_creation(self):
        ba = BlameAssignment(random_seed=42)
        assert ba is not None

    def test_identify_root_cause(self):
        ba = BlameAssignment(random_seed=42)

        failure = Failure(
            FailureType.GOAL_UNREACHED, "Test failure", 100, 1.0
        )

        trajectory = [
            ModuleAction("m1", "a", "s", "s'", 95),
            ModuleAction("m2", "b", "s", "s'", 98),
            ModuleAction("m1", "c", "s", "s'", 99),
        ]

        root, confidence = ba.identify_root_cause(
            failure, trajectory, ["m1", "m2"]
        )

        assert root in ["m1", "m2"]
        assert 0 <= confidence <= 1

    def test_compute_blame(self):
        ba = BlameAssignment(random_seed=42)

        failure = Failure(
            FailureType.CONSTRAINT_VIOLATION, "Violated constraint", 50, 0.9
        )

        trajectory = [
            ModuleAction("m1", "a", "s1", "s2", 45),
            ModuleAction("m1", "b", "s2", "s3", 48),
        ]

        result = ba.compute_blame(failure, trajectory, "m1")

        assert isinstance(result, BlameResult)
        assert result.module_id == "m1"
        assert 0 <= result.blame_score <= 1
        assert len(result.evidence) > 0

    def test_propagate_blame(self):
        ba = BlameAssignment(random_seed=42)

        failure = Failure(FailureType.TIMEOUT, "Timed out", 100, 1.0)
        distribution = {"m1": 0.7, "m2": 0.3}

        ba.propagate_blame(failure, distribution)

        stats = ba.statistics()
        assert stats["total_failures"] == 1

    def test_get_module_blame_history(self):
        ba = BlameAssignment(random_seed=42)

        failure1 = Failure(FailureType.TIMEOUT, "F1", 100, 1.0)
        failure2 = Failure(FailureType.GOAL_UNREACHED, "F2", 200, 0.8)

        ba.propagate_blame(failure1, {"m1": 0.8})
        ba.propagate_blame(failure2, {"m1": 0.6, "m2": 0.4})

        history = ba.get_module_blame_history("m1")
        assert len(history) == 2

    def test_get_chronic_offenders(self):
        ba = BlameAssignment(random_seed=42)

        for i in range(5):
            failure = Failure(FailureType.TIMEOUT, f"F{i}", i * 10, 1.0)
            ba.propagate_blame(failure, {"m1": 0.9})

        offenders = ba.get_chronic_offenders(threshold=3)
        assert len(offenders) == 1
        assert offenders[0][0] == "m1"
        assert offenders[0][1] == 5

    def test_temporal_blame(self):
        ba = BlameAssignment(random_seed=42)

        failure = Failure(FailureType.GOAL_UNREACHED, "Test", 100, 1.0)

        trajectory = [
            ModuleAction("m1", "a", "s", "s'", 10),
            ModuleAction("m2", "b", "s", "s'", 99),
        ]

        result_m1 = ba.compute_blame(failure, trajectory, "m1")
        result_m2 = ba.compute_blame(failure, trajectory, "m2")

        assert result_m2.blame_score >= result_m1.blame_score

    def test_statistics(self):
        ba = BlameAssignment(random_seed=42)

        failure = Failure(FailureType.TIMEOUT, "Test", 10, 1.0)
        trajectory = [ModuleAction("m1", "a", "s", "s'", 5)]

        ba.compute_blame(failure, trajectory, "m1")
        ba.propagate_blame(failure, {"m1": 0.5})

        stats = ba.statistics()

        assert "total_failures" in stats
        assert "total_blame_assignments" in stats
        assert stats["total_blame_assignments"] == 1


class TestBlameWithWorldModel:
    """Tests for blame assignment with world model."""

    def test_counterfactual_blame(self):
        ba = BlameAssignment(random_seed=42)

        def world_model(state, action):
            if state is None:
                return action
            return state + action

        ba.set_world_model(world_model)

        failure = Failure(FailureType.GOAL_UNREACHED, "Test", 100, 1.0)

        trajectory = [
            ModuleAction("m1", 1, 0, 1, 10),
            ModuleAction("m1", 1, 1, 2, 20),
        ]

        result = ba.compute_blame(failure, trajectory, "m1")

        assert isinstance(result.counterfactual_impact, float)
