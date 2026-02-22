"""Tests for credit assignment integration."""

import numpy as np

from neuro.modules.credit.integration import CreditConfig, CreditAssignmentSystem
from neuro.modules.credit.blame_assignment import FailureType


class TestCreditConfig:
    """Tests for CreditConfig class."""

    def test_default_config(self):
        config = CreditConfig()
        assert config.gamma == 0.99
        assert config.trace_lambda == 0.9

    def test_custom_config(self):
        config = CreditConfig(gamma=0.95, use_shapley=False)
        assert config.gamma == 0.95
        assert not config.use_shapley


class TestCreditAssignmentSystem:
    """Tests for CreditAssignmentSystem class."""

    def test_creation(self):
        system = CreditAssignmentSystem()
        assert system is not None

    def test_creation_with_config(self):
        config = CreditConfig(random_seed=42)
        system = CreditAssignmentSystem(config=config)
        assert system.config.random_seed == 42

    def test_register_module(self):
        system = CreditAssignmentSystem()
        system.register_module("module1")
        assert "module1" in system._active_modules

    def test_unregister_module(self):
        system = CreditAssignmentSystem()
        system.register_module("module1")
        system.unregister_module("module1")
        assert "module1" not in system._active_modules

    def test_record_action(self):
        system = CreditAssignmentSystem()
        system.register_module("m1")

        system.record_action("m1", "state1", "action1")

        stats = system.statistics()
        assert stats["action_history_size"] == 1

    def test_receive_reward(self):
        system = CreditAssignmentSystem()
        system.register_module("m1")
        system.register_module("m2")

        system.record_action("m1", "s1", "a1")
        system.record_action("m2", "s2", "a2")

        credits = system.receive_reward(1.0)

        assert "m1" in credits or "m2" in credits

    def test_receive_reward_with_td_error(self):
        system = CreditAssignmentSystem()
        system.register_module("m1")
        system.record_action("m1", "s1", "a1")

        credits = system.receive_reward(reward=1.0, td_error=0.5)

        assert isinstance(credits, dict)

    def test_process_trajectory(self):
        system = CreditAssignmentSystem()
        system.register_module("m1")

        trajectory = {
            "rewards": [1.0, 1.0, 1.0],
            "values": [0.5, 0.5, 0.5],
            "log_probs": np.array([-0.5, -0.5, -0.5]),
        }

        result = system.process_trajectory(trajectory, "m1")

        assert result.policy_loss is not None
        assert result.value_loss is not None

    def test_handle_failure(self):
        system = CreditAssignmentSystem()
        system.register_module("m1")
        system.register_module("m2")

        system.record_action("m1", "s1", "a1")

        blame_results = system.handle_failure(
            failure_type=FailureType.TIMEOUT,
            description="Task timed out",
            severity=0.8,
        )

        assert "m1" in blame_results
        assert "m2" in blame_results

    def test_get_adaptive_learning_rate(self):
        system = CreditAssignmentSystem()
        system.register_module("m1")

        lr = system.get_adaptive_learning_rate("m1", 1.0, 2.0)

        assert isinstance(lr, float)
        assert lr > 0

    def test_compute_surprise(self):
        system = CreditAssignmentSystem()

        metrics = system.compute_surprise(1.0, 5.0)

        assert metrics.surprise_value > 0

    def test_get_module_credits(self):
        system = CreditAssignmentSystem()
        system.register_module("m1")
        system.record_action("m1", "s", "a")
        system.receive_reward(1.0)

        credits = system.get_module_credits()
        assert isinstance(credits, dict)

    def test_get_underperforming_modules(self):
        config = CreditConfig(use_shapley=False)
        system = CreditAssignmentSystem(config=config)
        system.register_module("good")
        system.register_module("bad")

        for _ in range(10):
            system.record_action("good", "s", "a")
            system._contributions.record_contribution("good", 2.0)
            system._contributions.record_contribution("bad", 0.1)

        underperforming = system.get_underperforming_modules(threshold=0.5)
        assert "bad" in underperforming

    def test_get_top_contributors(self):
        system = CreditAssignmentSystem()

        system._contributions.record_contribution("top", 5.0)
        system._contributions.record_contribution("mid", 3.0)
        system._contributions.record_contribution("low", 1.0)

        top = system.get_top_contributors(n=2)

        assert len(top) == 2
        assert top[0][0] == "top"

    def test_get_chronic_failure_modules(self):
        system = CreditAssignmentSystem()
        system.register_module("failing")

        for _ in range(5):
            system.record_action("failing", "s", "a")
            system.handle_failure(FailureType.TIMEOUT, "Timeout", severity=1.0)

        chronic = system.get_chronic_failure_modules(threshold=3)
        module_names = [m[0] for m in chronic]
        assert "failing" in module_names

    def test_clear_history(self):
        system = CreditAssignmentSystem()
        system.register_module("m1")
        system.record_action("m1", "s", "a")

        system.clear_history()

        stats = system.statistics()
        assert stats["action_history_size"] == 0

    def test_reset_traces(self):
        system = CreditAssignmentSystem()
        system.register_module("m1")
        system.record_action("m1", "s", "a")

        system.reset_traces()

        stats = system.statistics()
        assert stats["trace_stats"]["active_traces"] == 0

    def test_set_world_model(self):
        system = CreditAssignmentSystem()

        def model(s, a):
            return s + a

        system.set_world_model(model)
        assert system._blame._counterfactual._world_model is not None

    def test_statistics(self):
        system = CreditAssignmentSystem()
        system.register_module("m1")
        system.record_action("m1", "s", "a")
        system.receive_reward(1.0)

        stats = system.statistics()

        assert "active_modules" in stats
        assert "timestep" in stats
        assert "trace_stats" in stats
        assert "policy_gradient_stats" in stats
        assert "blame_stats" in stats
        assert "surprise_stats" in stats
        assert "contribution_stats" in stats


class TestEndToEndCredit:
    """End-to-end credit assignment tests."""

    def test_full_episode(self):
        config = CreditConfig(random_seed=42)
        system = CreditAssignmentSystem(config=config)

        system.register_module("perception")
        system.register_module("decision")
        system.register_module("action")

        for step in range(10):
            system.record_action("perception", f"obs_{step}", "process")
            system.record_action("decision", f"state_{step}", f"decide_{step}")
            system.record_action("action", f"decision_{step}", f"act_{step}")

            if step == 9:
                system.receive_reward(10.0)

        credits = system.get_module_credits()
        assert len(credits) > 0

    def test_failure_and_recovery(self):
        config = CreditConfig(random_seed=42)
        system = CreditAssignmentSystem(config=config)

        system.register_module("planner")
        system.register_module("executor")

        system.record_action("planner", "s1", "plan_a")
        system.record_action("executor", "s2", "exec_a")

        blame = system.handle_failure(
            FailureType.GOAL_UNREACHED,
            "Failed to reach goal",
            severity=0.8,
        )

        planner_blame = blame["planner"].blame_score
        executor_blame = blame["executor"].blame_score

        assert planner_blame is not None
        assert executor_blame is not None

    def test_adaptive_learning(self):
        config = CreditConfig(random_seed=42)
        system = CreditAssignmentSystem(config=config)
        system.register_module("learner")

        lr_expected = system.get_adaptive_learning_rate("learner", 1.0, 1.1)
        lr_surprised = system.get_adaptive_learning_rate("learner", 1.0, 5.0)

        assert lr_surprised > lr_expected
