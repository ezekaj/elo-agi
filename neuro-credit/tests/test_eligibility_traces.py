"""Tests for eligibility traces."""

import pytest
import numpy as np

from src.eligibility_traces import (
    EligibilityTrace,
    TraceConfig,
    TraceType,
    EligibilityTraceManager,
)


class TestEligibilityTrace:
    """Tests for EligibilityTrace class."""

    def test_trace_creation(self):
        trace = EligibilityTrace(
            state_key="s1",
            action_key="a1",
            module_id="mod1",
            trace_value=1.0,
            timestamp=0,
        )
        assert trace.trace_value == 1.0
        assert trace.module_id == "mod1"

    def test_decay(self):
        trace = EligibilityTrace("s", "a", "m", 1.0, 0)
        trace.decay(0.9)
        assert trace.trace_value == pytest.approx(0.9)

    def test_accumulate(self):
        trace = EligibilityTrace("s", "a", "m", 0.5, 0)
        trace.accumulate(0.3, max_trace=10.0)
        assert trace.trace_value == pytest.approx(0.8)

    def test_accumulate_capped(self):
        trace = EligibilityTrace("s", "a", "m", 0.9, 0)
        trace.accumulate(0.5, max_trace=1.0)
        assert trace.trace_value == 1.0

    def test_replace(self):
        trace = EligibilityTrace("s", "a", "m", 0.5, 0)
        trace.replace(0.8)
        assert trace.trace_value == 0.8

    def test_dutch_update(self):
        trace = EligibilityTrace("s", "a", "m", 0.5, 0)
        trace.dutch_update(1.0, 0.9, max_trace=5.0)
        assert 0 < trace.trace_value < 5.0


class TestTraceConfig:
    """Tests for TraceConfig class."""

    def test_default_config(self):
        config = TraceConfig()
        assert config.trace_type == TraceType.ACCUMULATING
        assert config.lambda_param == 0.9
        assert config.gamma == 0.99

    def test_custom_config(self):
        config = TraceConfig(
            trace_type=TraceType.REPLACING,
            lambda_param=0.8,
        )
        assert config.trace_type == TraceType.REPLACING
        assert config.lambda_param == 0.8


class TestEligibilityTraceManager:
    """Tests for EligibilityTraceManager class."""

    def test_creation(self):
        manager = EligibilityTraceManager(random_seed=42)
        assert manager is not None

    def test_mark_eligible(self):
        manager = EligibilityTraceManager(random_seed=42)
        manager.mark_eligible("state1", "action1", "module1")

        stats = manager.statistics()
        assert stats["active_traces"] == 1

    def test_mark_eligible_multiple(self):
        manager = EligibilityTraceManager(random_seed=42)
        manager.mark_eligible("s1", "a1", "m1")
        manager.mark_eligible("s2", "a2", "m1")
        manager.mark_eligible("s1", "a1", "m2")

        stats = manager.statistics()
        assert stats["active_traces"] == 3

    def test_mark_eligible_same_twice_accumulating(self):
        config = TraceConfig(trace_type=TraceType.ACCUMULATING)
        manager = EligibilityTraceManager(config=config, random_seed=42)

        manager.mark_eligible("s1", "a1", "m1")
        manager.mark_eligible("s1", "a1", "m1")

        trace = manager.get_trace("s1", "a1", "m1")
        assert trace.trace_value > 1.0

    def test_mark_eligible_same_twice_replacing(self):
        config = TraceConfig(trace_type=TraceType.REPLACING)
        manager = EligibilityTraceManager(config=config, random_seed=42)

        manager.mark_eligible("s1", "a1", "m1")
        manager.mark_eligible("s1", "a1", "m1")

        trace = manager.get_trace("s1", "a1", "m1")
        assert trace.trace_value == 1.0

    def test_decay_traces(self):
        manager = EligibilityTraceManager(random_seed=42)
        manager.mark_eligible("s1", "a1", "m1")

        initial_trace = manager.get_trace("s1", "a1", "m1")
        initial_value = initial_trace.trace_value

        manager.decay_traces()

        decayed_trace = manager.get_trace("s1", "a1", "m1")
        assert decayed_trace.trace_value < initial_value

    def test_decay_removes_small_traces(self):
        config = TraceConfig(min_trace=0.1, lambda_param=0.1, gamma=0.1)
        manager = EligibilityTraceManager(config=config, random_seed=42)

        manager.mark_eligible("s1", "a1", "m1")

        for _ in range(10):
            manager.decay_traces()

        stats = manager.statistics()
        assert stats["active_traces"] == 0

    def test_distribute_credit(self):
        manager = EligibilityTraceManager(random_seed=42)
        manager.mark_eligible("s1", "a1", "m1")
        manager.mark_eligible("s2", "a2", "m2")

        credits = manager.distribute_credit(1.0)

        assert "m1" in credits
        assert "m2" in credits
        assert sum(credits.values()) == pytest.approx(1.0)

    def test_distribute_credit_proportional(self):
        config = TraceConfig(trace_type=TraceType.ACCUMULATING)
        manager = EligibilityTraceManager(config=config, random_seed=42)

        manager.mark_eligible("s1", "a1", "m1")
        manager.mark_eligible("s1", "a1", "m1")
        manager.mark_eligible("s2", "a2", "m2")

        credits = manager.distribute_credit(1.0)

        assert credits["m1"] > credits["m2"]

    def test_get_module_traces(self):
        manager = EligibilityTraceManager(random_seed=42)
        manager.mark_eligible("s1", "a1", "m1")
        manager.mark_eligible("s2", "a2", "m1")
        manager.mark_eligible("s3", "a3", "m2")

        m1_traces = manager.get_module_traces("m1")
        assert len(m1_traces) == 2

    def test_get_total_trace(self):
        manager = EligibilityTraceManager(random_seed=42)
        manager.mark_eligible("s1", "a1", "m1")
        manager.mark_eligible("s2", "a2", "m2")

        total = manager.get_total_trace()
        assert total == 2.0

        m1_total = manager.get_total_trace("m1")
        assert m1_total == 1.0

    def test_get_cumulative_credits(self):
        manager = EligibilityTraceManager(random_seed=42)
        manager.mark_eligible("s1", "a1", "m1")

        manager.distribute_credit(1.0)
        manager.distribute_credit(2.0)

        cumulative = manager.get_cumulative_credits()
        assert cumulative["m1"] == pytest.approx(3.0, rel=0.1)

    def test_reset_traces(self):
        manager = EligibilityTraceManager(random_seed=42)
        manager.mark_eligible("s1", "a1", "m1")
        manager.reset_traces()

        stats = manager.statistics()
        assert stats["active_traces"] == 0

    def test_prune_old_traces(self):
        manager = EligibilityTraceManager(random_seed=42)
        manager.mark_eligible("s1", "a1", "m1")

        for _ in range(10):
            manager.decay_traces()

        manager.mark_eligible("s2", "a2", "m2")

        removed = manager.prune_old_traces(max_age=5)
        assert removed == 1

    def test_set_lambda(self):
        manager = EligibilityTraceManager(random_seed=42)
        manager.set_lambda(0.5)
        assert manager.config.lambda_param == 0.5

    def test_importance_sampling(self):
        config = TraceConfig(use_importance_sampling=True)
        manager = EligibilityTraceManager(config=config, random_seed=42)

        manager.mark_eligible("s1", "a1", "m1", importance_weight=2.0)

        trace = manager.get_trace("s1", "a1", "m1")
        assert trace.trace_value == 2.0

    def test_statistics(self):
        manager = EligibilityTraceManager(random_seed=42)
        manager.mark_eligible("s1", "a1", "m1")
        manager.decay_traces()
        manager.distribute_credit(1.0)

        stats = manager.statistics()

        assert "active_traces" in stats
        assert "total_credits_distributed" in stats
        assert stats["total_decay_events"] == 1


class TestTDLambda:
    """Tests for TD(lambda) specific behavior."""

    def test_lambda_zero_is_td0(self):
        config = TraceConfig(lambda_param=0.0, gamma=0.99)
        manager = EligibilityTraceManager(config=config, random_seed=42)

        manager.mark_eligible("s1", "a1", "m1")
        manager.decay_traces()

        trace = manager.get_trace("s1", "a1", "m1")
        assert trace is None or trace.trace_value < 0.01

    def test_lambda_one_is_mc(self):
        config = TraceConfig(lambda_param=1.0, gamma=0.99, min_trace=0.01)
        manager = EligibilityTraceManager(config=config, random_seed=42)

        manager.mark_eligible("s1", "a1", "m1")

        for _ in range(5):
            manager.decay_traces()

        trace = manager.get_trace("s1", "a1", "m1")
        assert trace is not None
        assert trace.trace_value > 0.5
