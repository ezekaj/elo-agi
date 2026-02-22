"""Tests for temporal abstraction with options framework."""

import pytest

from neuro.modules.planning.temporal_abstraction import (
    Option,
    OptionPolicy,
    TerminationCondition,
    OptionState,
    OptionsFramework,
    IntraOptionLearning,
)


class TestTerminationCondition:
    """Tests for TerminationCondition class."""

    def test_deterministic_termination(self):
        term = TerminationCondition(
            name="test",
            predicate=lambda s: s > 5,
        )
        assert not term.should_terminate(3)
        assert term.should_terminate(7)

    def test_termination_probability(self):
        term = TerminationCondition(
            name="test",
            predicate=lambda s: s > 10,
            probability=lambda s: s / 20.0,
        )
        assert term.termination_probability(5) == pytest.approx(0.25)
        assert term.termination_probability(15) == 1.0


class TestOptionPolicy:
    """Tests for OptionPolicy class."""

    def test_deterministic_policy(self):
        policy = OptionPolicy(
            name="go_right",
            policy_fn=lambda s: "right",
            stochastic=False,
        )
        assert policy.select_action(None) == "right"
        assert policy.action_probability(None, "right") == 1.0
        assert policy.action_probability(None, "left") == 0.0

    def test_state_dependent_policy(self):
        policy = OptionPolicy(
            name="adaptive",
            policy_fn=lambda s: "up" if s > 0 else "down",
        )
        assert policy.select_action(1) == "up"
        assert policy.select_action(-1) == "down"


class TestOption:
    """Tests for Option class."""

    def test_option_creation(self):
        option = Option(
            name="test_option",
            initiation_set=lambda s: True,
            policy=OptionPolicy("p", lambda s: "action"),
            termination=TerminationCondition("t", lambda s: False),
        )
        assert option.name == "test_option"
        assert option.state == OptionState.INACTIVE

    def test_can_initiate(self):
        option = Option(
            name="conditional",
            initiation_set=lambda s: s > 0,
            policy=OptionPolicy("p", lambda s: "a"),
            termination=TerminationCondition("t", lambda s: False),
        )
        assert option.can_initiate(5)
        assert not option.can_initiate(-3)

    def test_initiate(self):
        option = Option(
            name="test",
            initiation_set=lambda s: True,
            policy=OptionPolicy("p", lambda s: "a"),
            termination=TerminationCondition("t", lambda s: False),
        )
        assert option.initiate(0)
        assert option.state == OptionState.EXECUTING
        assert option.start_state == 0

    def test_initiate_fails_outside_initiation_set(self):
        option = Option(
            name="test",
            initiation_set=lambda s: s > 0,
            policy=OptionPolicy("p", lambda s: "a"),
            termination=TerminationCondition("t", lambda s: False),
        )
        assert not option.initiate(-1)
        assert option.state == OptionState.INACTIVE

    def test_step(self):
        step_count = [0]

        option = Option(
            name="test",
            initiation_set=lambda s: True,
            policy=OptionPolicy("p", lambda s: f"action_{step_count[0]}"),
            termination=TerminationCondition("t", lambda s: step_count[0] >= 3),
        )
        option.initiate(0)

        for i in range(5):
            action, terminated = option.step(i)
            step_count[0] += 1
            if terminated:
                break

        assert option.steps_taken >= 1

    def test_add_reward(self):
        option = Option(
            name="test",
            initiation_set=lambda s: True,
            policy=OptionPolicy("p", lambda s: "a"),
            termination=TerminationCondition("t", lambda s: False),
        )
        option.initiate(0)
        option.steps_taken = 1
        option.add_reward(1.0, discount=0.9)
        option.steps_taken = 2
        option.add_reward(1.0, discount=0.9)

        assert option.cumulative_reward > 0

    def test_reset(self):
        option = Option(
            name="test",
            initiation_set=lambda s: True,
            policy=OptionPolicy("p", lambda s: "a"),
            termination=TerminationCondition("t", lambda s: False),
        )
        option.initiate(0)
        option.steps_taken = 5
        option.cumulative_reward = 10.0

        option.reset()

        assert option.state == OptionState.INACTIVE
        assert option.steps_taken == 0
        assert option.cumulative_reward == 0.0


class TestIntraOptionLearning:
    """Tests for IntraOptionLearning class."""

    def test_creation(self):
        learner = IntraOptionLearning(discount=0.99, learning_rate=0.1)
        assert learner.discount == 0.99
        assert learner.learning_rate == 0.1

    def test_get_option_value_unknown(self):
        learner = IntraOptionLearning()
        value = learner.get_option_value("unknown", "state")
        assert value == 0.0

    def test_update_option_value(self):
        learner = IntraOptionLearning(learning_rate=0.5)
        learner.update_option_value("opt1", "s1", 1.0)

        value = learner.get_option_value("opt1", "s1")
        assert value == pytest.approx(0.5)

        learner.update_option_value("opt1", "s1", 1.0)
        value = learner.get_option_value("opt1", "s1")
        assert value > 0.5

    def test_update(self):
        learner = IntraOptionLearning(learning_rate=0.1, discount=0.9)

        option = Option(
            name="test_opt",
            initiation_set=lambda s: True,
            policy=OptionPolicy("p", lambda s: "a"),
            termination=TerminationCondition("t", lambda s: False, probability=lambda s: 0.1),
        )
        option.initiate("s1")

        td_error = learner.update(option, "s1", "a", 1.0, "s2", terminated=False)
        assert isinstance(td_error, float)
        assert learner.update_count == 1

    def test_statistics(self):
        learner = IntraOptionLearning()
        learner.update_option_value("opt1", "s1", 1.0)

        stats = learner.statistics()
        assert stats["options_tracked"] == 1


class TestOptionsFramework:
    """Tests for OptionsFramework class."""

    def test_creation(self):
        framework = OptionsFramework(random_seed=42)
        assert framework.discount == 0.99
        assert framework.learning_rate == 0.1

    def test_create_option(self):
        framework = OptionsFramework(random_seed=42)
        option = framework.create_option(
            name="test",
            initiation_set=lambda s: True,
            policy=lambda s: "action",
            termination=lambda s: False,
        )
        assert option.name == "test"
        assert "test" in framework.list_options()

    def test_get_option(self):
        framework = OptionsFramework(random_seed=42)
        framework.create_option("opt1", lambda s: True, lambda s: "a", lambda s: False)

        option = framework.get_option("opt1")
        assert option is not None
        assert option.name == "opt1"

        assert framework.get_option("nonexistent") is None

    def test_get_available_options(self):
        framework = OptionsFramework(random_seed=42)
        framework.create_option("always", lambda s: True, lambda s: "a", lambda s: False)
        framework.create_option("never", lambda s: False, lambda s: "a", lambda s: False)
        framework.create_option("positive", lambda s: s > 0, lambda s: "a", lambda s: False)

        available = framework.get_available_options(5)
        names = [o.name for o in available]
        assert "always" in names
        assert "positive" in names
        assert "never" not in names

    def test_select_option_exploration(self):
        framework = OptionsFramework(random_seed=42)
        framework.create_option("opt1", lambda s: True, lambda s: "a", lambda s: False)
        framework.create_option("opt2", lambda s: True, lambda s: "b", lambda s: False)

        selections = set()
        for _ in range(100):
            selected = framework.select_option(0, epsilon=1.0)
            if selected:
                selections.add(selected)

        assert len(selections) == 2

    def test_select_option_exploitation(self):
        framework = OptionsFramework(random_seed=42)
        framework.create_option("high", lambda s: True, lambda s: "a", lambda s: False)
        framework.create_option("low", lambda s: True, lambda s: "b", lambda s: False)

        framework.set_option_value("high", "state", 10.0)
        framework.set_option_value("low", "state", 1.0)

        selected = framework.select_option("state", epsilon=0.0)
        assert selected == "high"

    def test_initiate_option(self):
        framework = OptionsFramework(random_seed=42)
        framework.create_option("test", lambda s: True, lambda s: "a", lambda s: False)

        result = framework.initiate_option("test", 0)
        assert result
        assert framework.get_active_option() == "test"

    def test_step_option(self):
        framework = OptionsFramework(random_seed=42)
        step_count = [0]
        framework.create_option(
            "counter",
            lambda s: True,
            lambda s: f"step_{step_count[0]}",
            lambda s: step_count[0] >= 3,
        )

        framework.initiate_option("counter", 0)

        for _ in range(5):
            action, terminated = framework.step_option(step_count[0])
            step_count[0] += 1
            if terminated:
                break

        assert framework.get_active_option() is None

    def test_update_from_transition(self):
        framework = OptionsFramework(random_seed=42)
        framework.create_option("test", lambda s: True, lambda s: "a", lambda s: False)
        framework.initiate_option("test", "s1")

        td_error = framework.update_from_transition("s1", "a", 1.0, "s2", False)
        assert isinstance(td_error, float)

    def test_option_hierarchy(self):
        framework = OptionsFramework(random_seed=42)
        framework.create_option("parent", lambda s: True, lambda s: "a", lambda s: False)
        framework.create_option("child1", lambda s: True, lambda s: "b", lambda s: False)
        framework.create_option("child2", lambda s: True, lambda s: "c", lambda s: False)

        framework.add_option_hierarchy("parent", ["child1", "child2"])

        children = framework.get_child_options("parent")
        assert len(children) == 2
        assert "child1" in children
        assert "child2" in children

    def test_option_stack(self):
        framework = OptionsFramework(random_seed=42)
        framework.create_option("opt1", lambda s: True, lambda s: "a", lambda s: False)
        framework.create_option("opt2", lambda s: True, lambda s: "b", lambda s: False)

        framework.initiate_option("opt1", 0)
        framework.initiate_option("opt2", 0)

        stack = framework.get_option_stack()
        assert len(stack) == 2
        assert stack[0] == "opt1"
        assert stack[1] == "opt2"

    def test_reset(self):
        framework = OptionsFramework(random_seed=42)
        framework.create_option("test", lambda s: True, lambda s: "a", lambda s: False)
        framework.initiate_option("test", 0)

        framework.reset()

        assert framework.get_active_option() is None
        assert len(framework.get_option_stack()) == 0

    def test_statistics(self):
        framework = OptionsFramework(random_seed=42)
        framework.create_option("test", lambda s: True, lambda s: "a", lambda s: False)
        framework.initiate_option("test", 0)

        stats = framework.statistics()
        assert stats["total_options"] == 1
        assert stats["total_options_executed"] == 1


class TestOptionTerminationBehavior:
    """Tests for option termination behaviors."""

    def test_goal_based_termination(self):
        goal_state = 10

        option = Option(
            name="reach_goal",
            initiation_set=lambda s: True,
            policy=OptionPolicy("p", lambda s: 1),
            termination=TerminationCondition("t", lambda s: s >= goal_state),
        )
        option.initiate(0)

        for state in range(15):
            _, terminated = option.step(state)
            if terminated:
                assert state >= goal_state
                break

    def test_probabilistic_termination(self):
        framework = OptionsFramework(random_seed=42)

        step_count = [0]

        framework.create_option(
            "prob_term",
            lambda s: True,
            lambda s: "a",
            lambda s: False,
            termination_prob=lambda s: 0.3,
        )
        framework.initiate_option("prob_term", 0)

        for _ in range(100):
            step_count[0] += 1
            option = framework.get_option("prob_term")
            prob = option.termination.termination_probability(0)
            assert prob == pytest.approx(0.3)
            break
