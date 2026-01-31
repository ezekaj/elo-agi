"""Tests for executive network"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.executive_network import (
    ConflictMonitor, PFCController, ExecutiveNetwork, ExecutiveParams
)


class TestConflictMonitor:
    """Tests for conflict monitoring"""

    def test_initialization(self):
        """Test monitor initialization"""
        monitor = ConflictMonitor(n_responses=4)

        assert monitor.n_responses == 4
        assert monitor.conflict == 0.0

    def test_low_conflict_single_response(self):
        """Test low conflict with single dominant response"""
        monitor = ConflictMonitor()

        # Single strong response
        responses = np.array([0.9, 0.1, 0.1, 0.1])
        conflict = monitor.compute_conflict(responses)

        assert conflict < 0.5

    def test_high_conflict_competing_responses(self):
        """Test high conflict with competing responses"""
        monitor = ConflictMonitor()

        # Two competing responses
        responses = np.array([0.8, 0.8, 0.1, 0.1])
        conflict = monitor.compute_conflict(responses)

        assert conflict > 0.3

    def test_needs_control(self):
        """Test control need detection"""
        monitor = ConflictMonitor(threshold=0.3)

        # High conflict
        monitor.compute_conflict(np.array([0.9, 0.9, 0.1, 0.1]))

        assert monitor.needs_control()

    def test_control_signal(self):
        """Test control signal generation"""
        monitor = ConflictMonitor(threshold=0.3)

        # High conflict
        monitor.compute_conflict(np.array([0.9, 0.9, 0.1, 0.1]))
        signal = monitor.get_control_signal()

        assert signal > 0

    def test_mean_conflict(self):
        """Test mean conflict calculation"""
        monitor = ConflictMonitor()

        # Record some conflicts
        for _ in range(5):
            monitor.compute_conflict(np.random.rand(4))

        mean = monitor.get_mean_conflict(window=5)

        assert mean >= 0


class TestPFCController:
    """Tests for PFC controller"""

    def test_initialization(self):
        """Test controller initialization"""
        controller = PFCController(n_goals=3)

        assert controller.n_goals == 3
        assert controller.goals.shape[0] == 3

    def test_set_goal(self):
        """Test setting goals"""
        controller = PFCController(n_goals=3)

        goal = np.random.rand(100)
        controller.set_goal(0, goal)

        assert controller.goal_activation[0] == 1.0
        assert controller.current_goal == 0

    def test_goal_maintenance(self):
        """Test goals are maintained"""
        controller = PFCController(n_goals=3)
        controller.set_goal(0, np.random.rand(100))

        initial = controller.goal_activation[0]

        for _ in range(10):
            controller.maintain_goals(dt=1.0)

        # Current goal should stay active
        assert controller.goal_activation[0] > 0.5

    def test_biasing_signal(self):
        """Test biasing signal generation"""
        controller = PFCController(n_goals=3)
        controller.set_goal(0, np.ones(100))

        input_pattern = np.zeros(100)
        bias = controller.get_biasing_signal(input_pattern)

        assert len(bias) == 100
        assert np.sum(np.abs(bias)) > 0

    def test_control_adjustment(self):
        """Test control level adjustment"""
        controller = PFCController(n_goals=3)
        initial = controller.control_level

        controller.adjust_control(0.5)  # High conflict

        assert controller.control_level > initial

    def test_state_getter(self):
        """Test state retrieval"""
        controller = PFCController(n_goals=3)

        state = controller.get_state()

        assert "current_goal" in state
        assert "goal_activation" in state
        assert "control_level" in state


class TestExecutiveNetwork:
    """Tests for integrated executive network"""

    def test_initialization(self):
        """Test network initialization"""
        network = ExecutiveNetwork()

        assert network.inhibition is not None
        assert network.working_memory is not None
        assert network.flexibility is not None
        assert network.conflict_monitor is not None
        assert network.controller is not None

    def test_process_stimulus(self):
        """Test stimulus processing"""
        network = ExecutiveNetwork()

        stimulus = np.random.rand(100)
        result = network.process_stimulus(stimulus)

        assert "selected_response" in result
        assert "conflict" in result
        assert "wm_load" in result

    def test_goal_setting(self):
        """Test goal setting"""
        network = ExecutiveNetwork()

        goal = np.random.rand(100)
        network.set_goal(0, goal)

        state = network.controller.get_state()
        assert state["current_goal"] == 0

    def test_task_switching(self):
        """Test task switching"""
        network = ExecutiveNetwork()

        network.switch_task(1)

        assert network.flexibility.task_switcher.current_task == 1

    def test_update(self):
        """Test system update"""
        network = ExecutiveNetwork()

        # Should run without error
        network.update(dt=1.0)

    def test_executive_state(self):
        """Test comprehensive state retrieval"""
        network = ExecutiveNetwork()
        network.process_stimulus(np.random.rand(100))

        state = network.get_executive_state()

        assert "controller" in state
        assert "conflict" in state
        assert "flexibility" in state
        assert "wm_load" in state
        assert "arousal" in state
        assert "fatigue" in state

    def test_stroop_trial(self):
        """Test Stroop-like trial"""
        network = ExecutiveNetwork()

        result = network.run_stroop_trial("red", "blue", task="color")

        assert "congruent" in result
        assert not result["congruent"]  # red != blue
        assert "conflict" in result

    def test_fatigue_accumulates(self):
        """Test fatigue accumulates with use"""
        network = ExecutiveNetwork()
        initial_fatigue = network.fatigue

        for _ in range(20):
            network.process_stimulus(np.random.rand(100))

        assert network.fatigue > initial_fatigue

    def test_arousal_fluctuates(self):
        """Test arousal varies"""
        network = ExecutiveNetwork()

        network.update(dt=10.0)

        # Arousal should be in valid range
        assert 0.2 <= network.arousal <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
