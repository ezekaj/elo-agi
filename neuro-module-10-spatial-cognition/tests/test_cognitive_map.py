"""Tests for cognitive map integration."""

import numpy as np
import pytest
from neuro.modules.m10_spatial_cognition.cognitive_map import CognitiveMap, Environment, Landmark
from neuro.modules.m10_spatial_cognition.border_cells import WallDirection
from neuro.modules.m10_spatial_cognition.head_direction_cells import HeadDirectionSystem
from neuro.modules.m10_spatial_cognition.path_integration import PathIntegrator


class TestEnvironment:
    """Test environment representation."""

    def test_environment_creation(self):
        env = Environment(bounds=(0, 1, 0, 1), name="test_env")
        assert env.width == 1.0
        assert env.height == 1.0
        np.testing.assert_array_equal(env.center, [0.5, 0.5])

    def test_position_validation(self):
        env = Environment(bounds=(0, 1, 0, 1))
        assert env.is_valid_position(np.array([0.5, 0.5]))
        assert not env.is_valid_position(np.array([1.5, 0.5]))
        assert not env.is_valid_position(np.array([-0.1, 0.5]))

    def test_add_landmark(self):
        env = Environment(bounds=(0, 1, 0, 1))
        landmark = env.add_landmark("tree", np.array([0.3, 0.7]), color="green")
        assert len(env.landmarks) == 1
        assert env.landmarks[0].name == "tree"
        assert env.landmarks[0].features["color"] == "green"

    def test_visible_landmarks(self):
        env = Environment(bounds=(0, 1, 0, 1))
        env.add_landmark("tree", np.array([0.7, 0.5]))

        position = np.array([0.5, 0.5])
        heading = 0.0  # Facing east (toward tree)

        visible = env.get_visible_landmarks(position, heading, fov=np.pi)
        assert len(visible) == 1
        landmark, distance, bearing = visible[0]
        assert landmark.name == "tree"
        assert distance == pytest.approx(0.2, rel=0.1)


class TestCognitiveMap:
    """Test integrated cognitive map."""

    def test_cognitive_map_creation(self):
        cog_map = CognitiveMap(n_place_cells=50, n_grid_modules=2, n_head_direction_cells=30)
        assert len(cog_map.place_cells) == 50
        assert len(cog_map.grid_cells.modules) == 2
        assert len(cog_map.head_direction) == 30

    def test_get_state(self):
        cog_map = CognitiveMap(n_place_cells=50)
        state = cog_map.get_state()

        assert hasattr(state, "position")
        assert hasattr(state, "heading")
        assert hasattr(state, "place_activity")
        assert len(state.place_activity) == 50

    def test_update_with_movement(self):
        cog_map = CognitiveMap(n_place_cells=50)
        initial_pos = cog_map.get_position().copy()

        # Move forward
        velocity = np.array([0.1, 0.0])
        for _ in range(5):
            cog_map.update(velocity, angular_velocity=0.0, dt=0.1)

        final_pos = cog_map.get_position()
        displacement = np.linalg.norm(final_pos - initial_pos)
        assert displacement > 0

    def test_heading_update(self):
        cog_map = CognitiveMap()
        cog_map.set_heading(0.0)

        # Turn 90 degrees (angular_velocity * dt * steps = π/2)
        # π/2 per step * 0.1 dt * 10 steps = π/2
        for _ in range(10):
            cog_map.update(
                velocity=np.array([0.0, 0.0]),
                angular_velocity=np.pi / 2,  # π/2 rad/s
                dt=0.1,  # 0.1 s per step -> 10 steps = 1s total -> π/2 rad
            )

        heading = cog_map.get_heading()
        assert heading == pytest.approx(np.pi / 2, rel=0.2)

    def test_encode_and_recall_location(self):
        cog_map = CognitiveMap()
        cog_map.set_position(np.array([0.3, 0.7]))
        cog_map.encode_location("home")

        recalled = cog_map.recall_location("home")
        assert recalled is not None
        np.testing.assert_array_almost_equal(recalled, [0.3, 0.7])

    def test_navigate_to_goal(self):
        cog_map = CognitiveMap()
        cog_map.set_position(np.array([0.2, 0.2]))

        goal = np.array([0.8, 0.8])
        velocity, distance = cog_map.navigate_to(goal, speed=0.1)

        assert distance > 0
        assert np.linalg.norm(velocity) == pytest.approx(0.1, rel=1e-6)

    def test_remapping(self):
        cog_map = CognitiveMap(n_place_cells=50, random_seed=42)
        pos = np.array([0.5, 0.5])
        cog_map.set_position(pos)
        activity_before = cog_map.place_cells.get_population_activity(pos)

        # Remap to new environment
        new_env = Environment(bounds=(0, 2, 0, 2), name="room2")
        cog_map.remap(new_env)

        activity_after = cog_map.place_cells.get_population_activity(pos)

        # Activity patterns should differ significantly
        correlation = np.corrcoef(activity_before, activity_after)[0, 1]
        assert abs(correlation) < 0.8

    def test_boundary_detection(self):
        cog_map = CognitiveMap()
        # Move near west wall
        cog_map.set_position(np.array([0.05, 0.5]))
        state = cog_map.get_state()

        assert WallDirection.WEST in state.nearby_walls

    def test_distance_to_location(self):
        cog_map = CognitiveMap()
        cog_map.set_position(np.array([0.0, 0.0]))
        cog_map.encode_location("target", np.array([0.3, 0.4]))

        dist = cog_map.distance_to_location("target")
        assert dist == pytest.approx(0.5, rel=0.01)

    def test_bearing_to_location(self):
        cog_map = CognitiveMap()
        cog_map.set_position(np.array([0.0, 0.0]))
        cog_map.set_heading(0.0)  # Facing east
        cog_map.encode_location("target", np.array([1.0, 0.0]))  # Due east

        bearing = cog_map.bearing_to_location("target")
        assert abs(bearing) < 0.1  # Should be nearly 0

    def test_trajectory_recording(self):
        cog_map = CognitiveMap()
        cog_map.set_position(np.array([0.0, 0.0]))

        for _ in range(5):
            cog_map.update(np.array([0.1, 0.0]), dt=0.1)

        trajectory = cog_map.get_trajectory()
        assert len(trajectory) >= 5


class TestHeadDirectionSystem:
    """Test head direction cells."""

    def test_head_direction_creation(self):
        hd = HeadDirectionSystem(n_cells=60)
        assert len(hd) == 60

    def test_population_activity(self):
        hd = HeadDirectionSystem(n_cells=60)
        activity = hd.get_population_activity(np.pi / 4)
        assert len(activity) == 60
        assert np.max(activity) > 0

    def test_decode_heading(self):
        hd = HeadDirectionSystem(n_cells=60)
        true_heading = np.pi / 3
        activity = hd.get_population_activity(true_heading)
        decoded = hd.decode_heading(activity)

        error = abs(decoded - true_heading)
        if error > np.pi:
            error = 2 * np.pi - error
        assert error < 0.2

    def test_heading_update(self):
        hd = HeadDirectionSystem(n_cells=60)
        hd.anchor_to_landmark(0.0)

        for _ in range(10):
            hd.update_heading(angular_velocity=0.1, dt=0.1)

        heading = hd.get_current_heading()
        assert heading == pytest.approx(0.1, rel=0.2)


class TestPathIntegrator:
    """Test path integration."""

    def test_path_integrator_creation(self):
        pi = PathIntegrator(initial_position=np.array([0.5, 0.5]))
        assert np.allclose(pi.position_estimate, [0.5, 0.5])

    def test_integrate_velocity(self):
        pi = PathIntegrator(initial_position=np.array([0.0, 0.0]), noise_scale=0.0)

        for _ in range(10):
            pi.integrate(velocity=np.array([0.1, 0.0]), dt=0.1)

        pos = pi.position_estimate
        assert pos[0] == pytest.approx(0.1, rel=0.2)

    def test_uncertainty_grows(self):
        pi = PathIntegrator(initial_position=np.array([0.0, 0.0]), uncertainty_growth=0.01)
        initial_uncertainty = pi.uncertainty

        for _ in range(10):
            pi.integrate(velocity=np.array([0.1, 0.0]), dt=0.1)

        assert pi.uncertainty > initial_uncertainty

    def test_reset_position(self):
        pi = PathIntegrator()
        pi.integrate(np.array([1.0, 1.0]), dt=1.0)

        pi.reset(np.array([0.0, 0.0]))
        assert np.allclose(pi.position_estimate, [0.0, 0.0])
        assert pi.uncertainty == 0.0

    def test_total_distance(self):
        pi = PathIntegrator(initial_position=np.array([0.0, 0.0]), noise_scale=0.0)

        for _ in range(10):
            pi.integrate(velocity=np.array([0.1, 0.0]), dt=0.1)

        total = pi.get_total_distance()
        assert total == pytest.approx(0.1, rel=0.2)


class TestLandmark:
    """Test landmark dataclass."""

    def test_landmark_creation(self):
        lm = Landmark(name="tree", position=np.array([0.5, 0.5]), features={"height": 10})
        assert lm.name == "tree"
        assert lm.features["height"] == 10
