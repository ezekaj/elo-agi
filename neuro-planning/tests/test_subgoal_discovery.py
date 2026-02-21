"""Tests for subgoal discovery."""

import pytest
import numpy as np

from neuro.modules.planning.subgoal_discovery import (
    Subgoal,
    SubgoalType,
    Trajectory,
    BottleneckDetector,
    OptionTerminationDetector,
    SubgoalDiscovery,
)

class TestTrajectory:
    """Tests for Trajectory class."""

    def test_trajectory_creation(self):
        traj = Trajectory(
            states=[np.array([0, 0]), np.array([1, 0]), np.array([2, 0])],
            actions=["right", "right"],
            rewards=[0.0, 1.0],
        )
        assert traj.length == 3
        assert len(traj.actions) == 2
        assert len(traj.rewards) == 2

    def test_get_returns(self):
        traj = Trajectory(
            states=[np.array([i]) for i in range(4)],
            actions=["a", "a", "a"],
            rewards=[1.0, 1.0, 1.0],
        )
        returns = traj.get_returns(discount=1.0)
        assert returns[0] == pytest.approx(3.0)
        assert returns[-1] == pytest.approx(1.0)

        returns_discounted = traj.get_returns(discount=0.5)
        assert returns_discounted[0] == pytest.approx(1 + 0.5 + 0.25)

class TestSubgoal:
    """Tests for Subgoal class."""

    def test_subgoal_creation(self):
        subgoal = Subgoal(
            name="test",
            subgoal_type=SubgoalType.BOTTLENECK,
            state_representation=np.array([1.0, 0.0, 0.0]),
        )
        assert subgoal.name == "test"
        assert subgoal.subgoal_type == SubgoalType.BOTTLENECK

    def test_matches(self):
        subgoal = Subgoal(
            name="test",
            subgoal_type=SubgoalType.BOTTLENECK,
            state_representation=np.array([1.0, 0.0, 0.0]),
        )
        assert subgoal.matches(np.array([1.0, 0.0, 0.0]), threshold=0.9)
        assert subgoal.matches(np.array([0.95, 0.05, 0.0]), threshold=0.9)
        assert not subgoal.matches(np.array([0.0, 1.0, 0.0]), threshold=0.9)

    def test_update_statistics(self):
        subgoal = Subgoal(
            name="test",
            subgoal_type=SubgoalType.BOTTLENECK,
            state_representation=np.array([1.0]),
        )
        subgoal.update_statistics(visited=True, reached_goal=True)
        assert subgoal.frequency == 1
        assert subgoal.utility > 0

class TestBottleneckDetector:
    """Tests for BottleneckDetector class."""

    def test_creation(self):
        detector = BottleneckDetector(state_dim=64, random_seed=42)
        assert detector.state_dim == 64
        assert detector.min_frequency == 3

    def test_process_trajectory(self):
        detector = BottleneckDetector(state_dim=2, random_seed=42)

        traj = Trajectory(
            states=[np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([2.0, 0.0])],
            actions=["r", "r"],
            rewards=[0.0, 1.0],
            success=True,
        )
        detector.process_trajectory(traj)

        stats = detector.statistics()
        assert stats["total_visits"] > 0

    def test_identify_bottlenecks(self):
        detector = BottleneckDetector(
            state_dim=2,
            similarity_threshold=0.8,
            min_frequency=2,
            random_seed=42,
        )

        trajectories = []
        common_state = np.array([5.0, 5.0])

        for i in range(10):
            states = [
                np.array([float(i), 0.0]),
                common_state + np.random.randn(2) * 0.1,
                np.array([10.0, float(i)]),
            ]
            traj = Trajectory(
                states=states,
                actions=["a", "b"],
                rewards=[0.0, 1.0],
                success=True,
            )
            trajectories.append(traj)

        bottlenecks = detector.identify_bottlenecks(trajectories, top_k=5)
        assert len(bottlenecks) > 0

        for bn in bottlenecks:
            assert bn.subgoal_type == SubgoalType.BOTTLENECK

    def test_success_rate_tracking(self):
        detector = BottleneckDetector(state_dim=2, min_frequency=1, random_seed=42)

        success_traj = Trajectory(
            states=[np.array([1.0, 0.0])],
            actions=[],
            rewards=[],
            success=True,
        )
        fail_traj = Trajectory(
            states=[np.array([1.0, 0.0])],
            actions=[],
            rewards=[],
            success=False,
        )

        detector.process_trajectory(success_traj)
        detector.process_trajectory(fail_traj)

        stats = detector.statistics()
        assert stats["success_states"] > 0

class TestOptionTerminationDetector:
    """Tests for OptionTerminationDetector class."""

    def test_creation(self):
        detector = OptionTerminationDetector(state_dim=64, random_seed=42)
        assert detector.state_dim == 64

    def test_identify_value_peaks(self):
        detector = OptionTerminationDetector(
            state_dim=2,
            value_threshold=0.1,
            random_seed=42,
        )

        traj = Trajectory(
            states=[np.array([i]) for i in range(10)],
            actions=["a"] * 9,
            rewards=[0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        )

        peaks = detector.identify_value_peaks(traj, discount=0.99)
        assert isinstance(peaks, list)

    def test_identify_action_changes(self):
        detector = OptionTerminationDetector(
            state_dim=2,
            action_change_threshold=0.5,
            random_seed=42,
        )

        traj = Trajectory(
            states=[np.array([i]) for i in range(5)],
            actions=[0, 0, 1, 1],
            rewards=[0.0] * 4,
        )

        def action_encoder(a):
            return np.array([1.0, 0.0]) if a == 0 else np.array([0.0, 1.0])

        changes = detector.identify_action_changes(traj, action_encoder)
        assert isinstance(changes, list)

    def test_identify_option_terminations(self):
        detector = OptionTerminationDetector(state_dim=4, random_seed=42)

        trajectories = []
        for _ in range(10):
            states = [np.random.randn(4) for _ in range(10)]
            states[5] = np.array([1.0, 1.0, 1.0, 1.0])
            traj = Trajectory(
                states=states,
                actions=list(range(9)),
                rewards=[0.0] * 4 + [1.0] + [0.0] * 4,
            )
            trajectories.append(traj)

        terminations = detector.identify_option_terminations(trajectories)
        assert isinstance(terminations, list)

class TestSubgoalDiscovery:
    """Tests for SubgoalDiscovery class."""

    def test_creation(self):
        discovery = SubgoalDiscovery(state_dim=64, random_seed=42)
        assert discovery.state_dim == 64

    def test_discover_subgoals_bottleneck(self):
        discovery = SubgoalDiscovery(state_dim=4, random_seed=42)

        trajectories = []
        for _ in range(20):
            states = [np.random.randn(4) for _ in range(5)]
            states[2] = np.array([1.0, 1.0, 1.0, 1.0]) + np.random.randn(4) * 0.1
            traj = Trajectory(
                states=states,
                actions=["a"] * 4,
                rewards=[0.0] * 4,
                success=True,
            )
            trajectories.append(traj)

        subgoals = discovery.discover_subgoals(
            trajectories,
            methods=["bottleneck"],
            top_k_per_method=5,
        )
        assert isinstance(subgoals, list)

    def test_discover_subgoals_cluster(self):
        discovery = SubgoalDiscovery(state_dim=4, random_seed=42)

        trajectories = []
        for _ in range(20):
            states = [np.random.randn(4) for _ in range(10)]
            traj = Trajectory(
                states=states,
                actions=["a"] * 9,
                rewards=[0.0] * 9,
                success=True,
            )
            trajectories.append(traj)

        subgoals = discovery.discover_subgoals(
            trajectories,
            methods=["cluster"],
            top_k_per_method=3,
        )
        assert len(subgoals) <= 3
        for sg in subgoals:
            assert sg.subgoal_type == SubgoalType.STATE_CLUSTER

    def test_verify_subgoal_utility(self):
        discovery = SubgoalDiscovery(state_dim=4, random_seed=42)

        subgoal = Subgoal(
            name="test",
            subgoal_type=SubgoalType.BOTTLENECK,
            state_representation=np.array([1.0, 1.0, 1.0, 1.0]),
        )

        task_trajectories = []
        for i in range(10):
            states = [np.random.randn(4) for _ in range(5)]
            if i < 7:
                states[2] = np.array([1.0, 1.0, 1.0, 1.0])
            success = i < 7
            traj = Trajectory(
                states=states,
                actions=["a"] * 4,
                rewards=[float(success)] * 4,
                success=success,
            )
            task_trajectories.append(traj)

        utility = discovery.verify_subgoal_utility(subgoal, task_trajectories)
        assert isinstance(utility, float)

    def test_get_subgoal(self):
        discovery = SubgoalDiscovery(state_dim=4, random_seed=42)

        trajectories = []
        for _ in range(10):
            traj = Trajectory(
                states=[np.random.randn(4) for _ in range(5)],
                actions=["a"] * 4,
                rewards=[0.0] * 4,
                success=True,
            )
            trajectories.append(traj)

        subgoals = discovery.discover_subgoals(trajectories)

        if subgoals:
            retrieved = discovery.get_subgoal(subgoals[0].name)
            assert retrieved is not None
            assert retrieved.name == subgoals[0].name

    def test_list_subgoals(self):
        discovery = SubgoalDiscovery(state_dim=4, random_seed=42)

        trajectories = []
        for _ in range(20):
            traj = Trajectory(
                states=[np.random.randn(4) for _ in range(5)],
                actions=["a"] * 4,
                rewards=[0.0] * 4,
                success=True,
            )
            trajectories.append(traj)

        discovery.discover_subgoals(trajectories, methods=["bottleneck", "cluster"])

        all_subgoals = discovery.list_subgoals()
        assert isinstance(all_subgoals, list)

        bottlenecks = discovery.list_subgoals(subgoal_type=SubgoalType.BOTTLENECK)
        for sg in bottlenecks:
            assert sg.subgoal_type == SubgoalType.BOTTLENECK

    def test_prune_subgoals(self):
        discovery = SubgoalDiscovery(state_dim=4, random_seed=42)

        trajectories = []
        for _ in range(20):
            traj = Trajectory(
                states=[np.random.randn(4) for _ in range(5)],
                actions=["a"] * 4,
                rewards=[0.0] * 4,
                success=True,
            )
            trajectories.append(traj)

        discovery.discover_subgoals(trajectories, methods=["cluster"])

        initial_count = len(discovery.list_subgoals())
        removed = discovery.prune_subgoals(min_utility=0.5, min_frequency=100)

        assert removed >= 0
        assert len(discovery.list_subgoals()) <= initial_count

    def test_statistics(self):
        discovery = SubgoalDiscovery(state_dim=4, random_seed=42)

        trajectories = []
        for _ in range(10):
            traj = Trajectory(
                states=[np.random.randn(4) for _ in range(5)],
                actions=["a"] * 4,
                rewards=[0.0] * 4,
                success=True,
            )
            trajectories.append(traj)

        discovery.discover_subgoals(trajectories)

        stats = discovery.statistics()
        assert "total_subgoals" in stats
        assert "trajectories_processed" in stats
        assert stats["trajectories_processed"] == 10

class TestMultipleMethodDiscovery:
    """Tests for combining multiple discovery methods."""

    def test_combined_methods(self):
        discovery = SubgoalDiscovery(state_dim=4, random_seed=42)

        trajectories = []
        for _ in range(30):
            states = [np.random.randn(4) for _ in range(8)]
            states[4] = np.array([2.0, 2.0, 2.0, 2.0]) + np.random.randn(4) * 0.1
            traj = Trajectory(
                states=states,
                actions=["a"] * 7,
                rewards=[0.0] * 3 + [1.0] + [0.0] * 3,
                success=True,
            )
            trajectories.append(traj)

        subgoals = discovery.discover_subgoals(
            trajectories,
            methods=["bottleneck", "termination", "cluster"],
            top_k_per_method=3,
        )

        types_found = set(sg.subgoal_type for sg in subgoals)
        assert len(types_found) >= 1
