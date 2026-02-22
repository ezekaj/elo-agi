#!/usr/bin/env python3
"""
STRESS TEST: Comprehensive testing of Module 10 Spatial Cognition

Tests include:
- Place cell firing patterns
- Grid cell hexagonal patterns
- Head direction accuracy
- Border cell boundary detection
- Cognitive map integration
- Path integration accuracy
- Conceptual space mapping
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")

from src import (
    PlaceCell,
    PlaceCellPopulation,
    GridCell,
    GridCellPopulation,
    HeadDirectionCell,
    HeadDirectionSystem,
    BorderCell,
    BorderCellPopulation,
    WallDirection,
    CognitiveMap,
    Environment,
    PathIntegrator,
    ConceptCell,
    SocialDistanceGrid,
    ConceptualMap,
)


class TestResults:
    """Track test results"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, name, passed, error=None):
        if passed:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            self.errors.append((name, error))
            print(f"  ✗ {name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"RESULTS: {self.passed}/{total} passed ({100 * self.passed / total:.1f}%)")
        if self.errors:
            print("\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'=' * 60}")
        return self.failed == 0


results = TestResults()


def test_section(name):
    print(f"\n{'=' * 60}")
    print(f"TEST: {name}")
    print(f"{'=' * 60}")


# =============================================================================
# 1. PLACE CELLS TESTS
# =============================================================================


def test_place_cells():
    test_section("PLACE CELLS")

    # Test 1: Single place cell firing
    try:
        cell = PlaceCell(center=[0.5, 0.5], radius=0.2, peak_rate=20.0)

        # At center - should fire maximally
        rate_center = cell.compute_firing(np.array([0.5, 0.5]))
        # Away from center - should fire less
        rate_away = cell.compute_firing(np.array([0.8, 0.8]))

        results.record(
            "Place cell fires maximally at center",
            rate_center > rate_away and rate_center > 19.0,
            f"center: {rate_center:.2f}, away: {rate_away:.2f}",
        )
    except Exception as e:
        results.record("Place cell fires maximally at center", False, str(e))

    # Test 2: Population covers environment
    try:
        pop = PlaceCellPopulation(n_cells=100, environment_size=(1.0, 1.0), random_seed=42)

        # Check coverage - every point should have some activity
        positions = [np.array([0.1, 0.1]), np.array([0.5, 0.5]), np.array([0.9, 0.9])]

        all_covered = all(np.max(pop.get_population_activity(pos)) > 0 for pos in positions)

        results.record("Place cell population covers environment", all_covered)
    except Exception as e:
        results.record("Place cell population covers environment", False, str(e))

    # Test 3: Position decoding accuracy
    try:
        pop = PlaceCellPopulation(n_cells=200, environment_size=(1.0, 1.0), random_seed=42)

        test_pos = np.array([0.3, 0.7])
        activity = pop.get_population_activity(test_pos)
        decoded = pop.decode_position(activity)

        error = np.linalg.norm(decoded - test_pos)

        results.record("Position decoding accuracy < 0.2", error < 0.2, f"error: {error:.3f}")
    except Exception as e:
        results.record("Position decoding accuracy < 0.2", False, str(e))

    # Test 4: Remapping changes fields
    try:
        pop = PlaceCellPopulation(n_cells=50, random_seed=42)

        # Get initial centers
        initial_centers = [cell.place_field.center.copy() for cell in pop.cells]

        # Remap
        pop.remap()

        # Check if centers changed
        changed = sum(
            1
            for i, cell in enumerate(pop.cells)
            if not np.allclose(cell.place_field.center, initial_centers[i])
        )

        results.record(
            "Remapping changes place fields",
            changed > 40,  # Most should change
            f"changed: {changed}/50",
        )
    except Exception as e:
        results.record("Remapping changes place fields", False, str(e))


# =============================================================================
# 2. GRID CELLS TESTS
# =============================================================================


def test_grid_cells():
    test_section("GRID CELLS")

    # Test 1: Hexagonal pattern
    try:
        cell = GridCell(spacing=0.3, orientation=0.0)

        # Sample firing at multiple positions
        rates = []
        for x in np.linspace(0, 1, 20):
            for y in np.linspace(0, 1, 20):
                rate = cell.compute_firing(np.array([x, y]))
                rates.append(rate)

        # Should have peaks and valleys (not uniform)
        rate_range = max(rates) - min(rates)

        results.record(
            "Grid cell has hexagonal pattern", rate_range > 5.0, f"rate range: {rate_range:.2f}"
        )
    except Exception as e:
        results.record("Grid cell has hexagonal pattern", False, str(e))

    # Test 2: Multi-scale modules
    try:
        pop = GridCellPopulation(n_modules=4, base_spacing=0.2, scale_ratio=1.4)

        spacings = [m.spacing for m in pop.modules]

        # Check increasing spacing
        increasing = all(spacings[i] < spacings[i + 1] for i in range(len(spacings) - 1))

        results.record(
            "Grid modules have increasing spacing",
            increasing,
            f"spacings: {[f'{s:.2f}' for s in spacings]}",
        )
    except Exception as e:
        results.record("Grid modules have increasing spacing", False, str(e))

    # Test 3: Path integration
    try:
        pop = GridCellPopulation(random_seed=42)
        pop.reset_position(np.array([0.0, 0.0]))

        # Move in a known direction
        for _ in range(10):
            pop.path_integrate(np.array([0.1, 0.0]), dt=1.0)

        final_pos = pop.get_position_estimate()

        # Should be approximately at (1.0, 0.0)
        error = np.linalg.norm(final_pos - np.array([1.0, 0.0]))

        results.record(
            "Path integration tracks movement",
            error < 0.1,
            f"final: {final_pos}, error: {error:.3f}",
        )
    except Exception as e:
        results.record("Path integration tracks movement", False, str(e))


# =============================================================================
# 3. HEAD DIRECTION TESTS
# =============================================================================


def test_head_direction():
    test_section("HEAD DIRECTION CELLS")

    # Test 1: Cell tuning
    try:
        cell = HeadDirectionCell(preferred_direction=np.pi / 2, tuning_width=2.0)

        # At preferred - max firing
        rate_pref = cell.compute_firing(np.pi / 2)
        # Opposite - min firing
        rate_opp = cell.compute_firing(3 * np.pi / 2)

        results.record(
            "HD cell fires at preferred direction",
            rate_pref > rate_opp * 5,
            f"pref: {rate_pref:.2f}, opp: {rate_opp:.2f}",
        )
    except Exception as e:
        results.record("HD cell fires at preferred direction", False, str(e))

    # Test 2: Ring covers 360 degrees
    try:
        system = HeadDirectionSystem(n_cells=60)

        # Check all directions have representation
        directions = [0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2]
        all_covered = True

        for direction in directions:
            activity = system.get_population_activity(direction)
            if np.max(activity) < 10:
                all_covered = False
                break

        results.record("HD system covers all directions", all_covered)
    except Exception as e:
        results.record("HD system covers all directions", False, str(e))

    # Test 3: Heading decode accuracy
    try:
        system = HeadDirectionSystem(n_cells=60)

        test_heading = np.pi / 3
        activity = system.get_population_activity(test_heading)
        decoded = system.decode_heading(activity)

        # Angular error
        error = abs(decoded - test_heading)
        if error > np.pi:
            error = 2 * np.pi - error

        results.record("Heading decode accuracy < 0.2 rad", error < 0.2, f"error: {error:.3f} rad")
    except Exception as e:
        results.record("Heading decode accuracy < 0.2 rad", False, str(e))

    # Test 4: Angular integration
    try:
        system = HeadDirectionSystem()
        system._current_heading = 0.0

        # Rotate 90 degrees
        for _ in range(10):
            system.update_heading(np.pi / 20, dt=1.0)

        final = system.get_current_heading()

        results.record(
            "Angular integration tracks rotation",
            abs(final - np.pi / 2) < 0.1,
            f"final: {final:.3f}, expected: {np.pi / 2:.3f}",
        )
    except Exception as e:
        results.record("Angular integration tracks rotation", False, str(e))


# =============================================================================
# 4. BORDER CELLS TESTS
# =============================================================================


def test_border_cells():
    test_section("BORDER CELLS")

    # Test 1: Cell fires near wall
    try:
        cell = BorderCell(preferred_wall=WallDirection.NORTH, distance_tuning=0.1)
        bounds = (0, 1, 0, 1)

        # Near north wall
        rate_near = cell.compute_firing(np.array([0.5, 0.95]), bounds)
        # Far from north wall
        rate_far = cell.compute_firing(np.array([0.5, 0.5]), bounds)

        results.record(
            "Border cell fires near preferred wall",
            rate_near > rate_far * 5,
            f"near: {rate_near:.2f}, far: {rate_far:.2f}",
        )
    except Exception as e:
        results.record("Border cell fires near preferred wall", False, str(e))

    # Test 2: Population detects all walls
    try:
        pop = BorderCellPopulation(cells_per_wall=5, environment_bounds=(0, 1, 0, 1))

        # Test each corner
        corners = {
            (0.05, 0.95): [WallDirection.WEST, WallDirection.NORTH],
            (0.95, 0.95): [WallDirection.EAST, WallDirection.NORTH],
            (0.05, 0.05): [WallDirection.WEST, WallDirection.SOUTH],
            (0.95, 0.05): [WallDirection.EAST, WallDirection.SOUTH],
        }

        all_detected = True
        for pos, expected_walls in corners.items():
            detected = pop.detect_boundary(np.array(pos), threshold=0.2)
            for wall in expected_walls:
                if wall not in detected:
                    all_detected = False
                    break

        results.record("Border population detects all walls", all_detected)
    except Exception as e:
        results.record("Border population detects all walls", False, str(e))

    # Test 3: Distance to walls
    try:
        pop = BorderCellPopulation(environment_bounds=(0, 1, 0, 1))

        pos = np.array([0.3, 0.7])
        distances = pop.get_distance_to_walls(pos)

        expected = {
            WallDirection.NORTH: 0.3,
            WallDirection.SOUTH: 0.7,
            WallDirection.EAST: 0.7,
            WallDirection.WEST: 0.3,
        }

        all_correct = all(abs(distances[wall] - expected[wall]) < 0.01 for wall in expected)

        results.record("Distance to walls computed correctly", all_correct)
    except Exception as e:
        results.record("Distance to walls computed correctly", False, str(e))


# =============================================================================
# 5. COGNITIVE MAP INTEGRATION TESTS
# =============================================================================


def test_cognitive_map():
    test_section("COGNITIVE MAP INTEGRATION")

    # Test 1: Create and navigate
    try:
        env = Environment(bounds=(0, 1, 0, 1))
        cmap = CognitiveMap(environment=env, random_seed=42)

        cmap.set_position(np.array([0.2, 0.2]))

        # Navigate toward goal
        goal = np.array([0.8, 0.8])
        velocity, distance = cmap.navigate_to(goal)

        results.record(
            "Navigation computes velocity to goal",
            np.linalg.norm(velocity) > 0 and distance > 0.5,
            f"velocity: {velocity}, distance: {distance:.2f}",
        )
    except Exception as e:
        results.record("Navigation computes velocity to goal", False, str(e))

    # Test 2: Update with movement
    try:
        env = Environment(bounds=(0, 1, 0, 1))
        cmap = CognitiveMap(environment=env, random_seed=42)

        initial = cmap.get_position().copy()
        cmap.update(velocity=np.array([0.1, 0.0]), dt=1.0)
        final = cmap.get_position()

        moved = np.linalg.norm(final - initial) > 0.05

        results.record("Update moves position", moved, f"initial: {initial}, final: {final}")
    except Exception as e:
        results.record("Update moves position", False, str(e))

    # Test 3: Encode and recall locations
    try:
        cmap = CognitiveMap(random_seed=42)

        cmap.set_position(np.array([0.3, 0.4]))
        cmap.encode_location("home")

        cmap.set_position(np.array([0.8, 0.8]))

        recalled = cmap.recall_location("home")

        results.record(
            "Location encoding and recall",
            recalled is not None and np.allclose(recalled, [0.3, 0.4]),
            f"recalled: {recalled}",
        )
    except Exception as e:
        results.record("Location encoding and recall", False, str(e))

    # Test 4: State contains all activities
    try:
        cmap = CognitiveMap(random_seed=42)
        state = cmap.get_state()

        has_all = (
            len(state.place_activity) > 0
            and len(state.grid_activity) > 0
            and len(state.head_direction_activity) > 0
            and len(state.border_activity) > 0
        )

        results.record("State contains all cell activities", has_all)
    except Exception as e:
        results.record("State contains all cell activities", False, str(e))

    # Test 5: Remapping
    try:
        cmap = CognitiveMap(random_seed=42)

        # Store original place activity
        original_activity = cmap.get_state().place_activity.copy()

        # Remap to new environment
        new_env = Environment(bounds=(0, 2, 0, 2), name="new")
        cmap.remap(new_env)

        new_activity = cmap.get_state().place_activity

        # Activities should be different
        different = not np.allclose(original_activity, new_activity)

        results.record("Remapping changes place activities", different)
    except Exception as e:
        results.record("Remapping changes place activities", False, str(e))


# =============================================================================
# 6. PATH INTEGRATION TESTS
# =============================================================================


def test_path_integration():
    test_section("PATH INTEGRATION")

    # Test 1: Basic integration
    try:
        integrator = PathIntegrator(initial_position=np.array([0.0, 0.0]))

        # Move in a square
        for _ in range(10):
            integrator.integrate(np.array([0.1, 0.0]), dt=1.0)
        for _ in range(10):
            integrator.integrate(np.array([0.0, 0.1]), dt=1.0)

        pos = integrator.position_estimate

        results.record(
            "Path integration tracks square path", pos[0] > 0.9 and pos[1] > 0.9, f"position: {pos}"
        )
    except Exception as e:
        results.record("Path integration tracks square path", False, str(e))

    # Test 2: Uncertainty grows
    try:
        integrator = PathIntegrator(uncertainty_growth=0.01)

        initial_uncertainty = integrator.uncertainty

        for _ in range(100):
            integrator.integrate(np.array([0.01, 0.01]), dt=1.0)

        final_uncertainty = integrator.uncertainty

        results.record(
            "Uncertainty grows over time",
            final_uncertainty > initial_uncertainty,
            f"initial: {initial_uncertainty:.3f}, final: {final_uncertainty:.3f}",
        )
    except Exception as e:
        results.record("Uncertainty grows over time", False, str(e))

    # Test 3: Reset reduces uncertainty
    try:
        integrator = PathIntegrator()

        # Accumulate uncertainty
        for _ in range(50):
            integrator.integrate(np.array([0.01, 0.01]), dt=1.0)

        integrator.reset(np.array([0.5, 0.5]))

        results.record("Reset clears uncertainty", integrator.uncertainty == 0.0)
    except Exception as e:
        results.record("Reset clears uncertainty", False, str(e))

    # Test 4: Trajectory recording
    try:
        integrator = PathIntegrator(initial_position=np.array([0.0, 0.0]))

        for i in range(10):
            integrator.integrate(np.array([0.1, 0.0]), dt=1.0)

        trajectory = integrator.get_trajectory()

        results.record(
            "Trajectory recorded correctly",
            len(trajectory) == 11,  # Initial + 10 steps
            f"trajectory length: {len(trajectory)}",
        )
    except Exception as e:
        results.record("Trajectory recorded correctly", False, str(e))


# =============================================================================
# 7. CONCEPTUAL SPACE TESTS
# =============================================================================


def test_conceptual_space():
    test_section("CONCEPTUAL SPACE (2025 DISCOVERY)")

    # Test 1: Concept cell activation
    try:
        cell = ConceptCell(
            concept_center=np.array([0.5, 0.5, 0.5]),
            concept_radius=0.3,
            associated_concept="democracy",
        )

        # At center - max activation
        act_center = cell.compute_activation(np.array([0.5, 0.5, 0.5]))
        # Away - less activation
        act_away = cell.compute_activation(np.array([0.9, 0.9, 0.9]))

        results.record(
            "Concept cell fires for associated concept",
            act_center > act_away * 2,
            f"center: {act_center:.3f}, away: {act_away:.3f}",
        )
    except Exception as e:
        results.record("Concept cell fires for associated concept", False, str(e))

    # Test 2: Conceptual distance
    try:
        cmap = ConceptualMap(concept_dimensions=5, random_seed=42)

        # Embed concepts
        cmap.embed_concept("dog", np.array([1.0, 0.0, 0.0, 0.5, 0.0]))
        cmap.embed_concept("cat", np.array([0.9, 0.0, 0.0, 0.4, 0.0]))
        cmap.embed_concept("car", np.array([0.0, 1.0, 0.0, 0.0, 0.5]))

        dist_dog_cat = cmap.conceptual_distance("dog", "cat")
        dist_dog_car = cmap.conceptual_distance("dog", "car")

        results.record(
            "Similar concepts are closer in concept space",
            dist_dog_cat < dist_dog_car,
            f"dog-cat: {dist_dog_cat:.3f}, dog-car: {dist_dog_car:.3f}",
        )
    except Exception as e:
        results.record("Similar concepts are closer in concept space", False, str(e))

    # Test 3: Find similar concepts
    try:
        cmap = ConceptualMap(concept_dimensions=3, random_seed=42)

        cmap.embed_concept("apple", np.array([1.0, 0.0, 0.0]))
        cmap.embed_concept("orange", np.array([0.9, 0.1, 0.0]))
        cmap.embed_concept("banana", np.array([0.8, 0.2, 0.0]))
        cmap.embed_concept("hammer", np.array([0.0, 0.0, 1.0]))

        similar = cmap.find_similar("apple", n=2)

        results.record(
            "Find similar concepts",
            len(similar) == 2 and similar[0][0] in ["orange", "banana"],
            f"similar to apple: {similar}",
        )
    except Exception as e:
        results.record("Find similar concepts", False, str(e))

    # Test 4: Analogy computation (a:b :: c:?)
    try:
        cmap = ConceptualMap(concept_dimensions=3, random_seed=42)

        # King - Man + Woman = Queen pattern
        cmap.embed_concept("king", np.array([1.0, 1.0, 0.0]))
        cmap.embed_concept("man", np.array([0.0, 1.0, 0.0]))
        cmap.embed_concept("woman", np.array([0.0, 0.0, 1.0]))
        cmap.embed_concept("queen", np.array([1.0, 0.0, 1.0]))

        result = cmap.compute_analogy("man", "king", "woman")

        results.record(
            "Analogy computation (man:king :: woman:?)",
            result is not None and result[0] == "queen",
            f"result: {result}",
        )
    except Exception as e:
        results.record("Analogy computation (man:king :: woman:?)", False, str(e))

    # Test 5: Social distance grid
    try:
        social = SocialDistanceGrid(dimensions=2)

        # Set social positions (power, affiliation)
        social.set_social_position("boss", np.array([1.0, 0.5]))
        social.set_social_position("colleague", np.array([0.5, 0.8]))
        social.set_social_position("friend", np.array([0.3, 0.9]))

        dist_colleague = social.compute_social_distance("boss", "colleague")
        dist_friend = social.compute_social_distance("boss", "friend")

        results.record(
            "Social distance computed",
            dist_colleague is not None and dist_friend is not None,
            f"boss-colleague: {dist_colleague:.3f}, boss-friend: {dist_friend:.3f}",
        )
    except Exception as e:
        results.record("Social distance computed", False, str(e))

    # Test 6: Navigate through concept space
    try:
        cmap = ConceptualMap(concept_dimensions=3, random_seed=42)

        cmap.embed_concept("start", np.array([0.0, 0.0, 0.0]))
        cmap.embed_concept("end", np.array([1.0, 1.0, 1.0]))

        path = cmap.navigate_concepts("start", "end", steps=5)

        results.record(
            "Navigate through concept space",
            len(path) == 6,  # start + 5 intermediate + end
            f"path length: {len(path)}",
        )
    except Exception as e:
        results.record("Navigate through concept space", False, str(e))


# =============================================================================
# 8. EDGE CASES AND STRESS
# =============================================================================


def test_edge_cases():
    test_section("EDGE CASES AND STRESS")

    # Test 1: Large place cell population
    try:
        pop = PlaceCellPopulation(n_cells=1000, random_seed=42)
        activity = pop.get_population_activity(np.array([0.5, 0.5]))

        results.record("Large place cell population (1000 cells)", len(activity) == 1000)
    except Exception as e:
        results.record("Large place cell population (1000 cells)", False, str(e))

    # Test 2: Long navigation
    try:
        cmap = CognitiveMap(random_seed=42)

        for _ in range(1000):
            cmap.update(velocity=np.array([0.001, 0.001]), dt=0.1)

        pos = cmap.get_position()

        results.record("Long navigation (1000 steps)", cmap.environment.is_valid_position(pos))
    except Exception as e:
        results.record("Long navigation (1000 steps)", False, str(e))

    # Test 3: Boundary handling
    try:
        cmap = CognitiveMap(environment=Environment(bounds=(0, 1, 0, 1)))

        # Try to move outside bounds
        cmap.set_position(np.array([0.99, 0.99]))
        cmap.update(velocity=np.array([0.1, 0.1]), dt=1.0)

        pos = cmap.get_position()

        results.record(
            "Position stays within bounds",
            cmap.environment.is_valid_position(pos),
            f"position: {pos}",
        )
    except Exception as e:
        results.record("Position stays within bounds", False, str(e))

    # Test 4: Many concepts
    try:
        cmap = ConceptualMap(concept_dimensions=10, random_seed=42)

        for i in range(100):
            features = np.random.randn(10)
            cmap.embed_concept(f"concept_{i}", features)

        results.record("Embed many concepts (100)", len(cmap) == 100)
    except Exception as e:
        results.record("Embed many concepts (100)", False, str(e))

    # Test 5: Full circular path
    try:
        integrator = PathIntegrator(initial_position=np.array([0.5, 0.5]))

        # Move in a circle
        for angle in np.linspace(0, 2 * np.pi, 100):
            velocity = 0.01 * np.array([np.cos(angle), np.sin(angle)])
            integrator.integrate(velocity, dt=0.1)

        # Should return close to start (with some drift)
        final = integrator.position_estimate
        start = np.array([0.5, 0.5])
        drift = np.linalg.norm(final - start)

        results.record(
            "Circular path integration (drift < 0.5)", drift < 0.5, f"drift: {drift:.3f}"
        )
    except Exception as e:
        results.record("Circular path integration (drift < 0.5)", False, str(e))


# =============================================================================
# 9. ADVANCED PLACE CELL STRESS TESTS
# =============================================================================


def test_advanced_place_cells():
    test_section("ADVANCED PLACE CELL STRESS")

    # Test 1: Decoding at 1000 random positions
    try:
        pop = PlaceCellPopulation(n_cells=500, environment_size=(1.0, 1.0), random_seed=42)

        errors = []
        for _ in range(1000):
            true_pos = np.random.rand(2)
            activity = pop.get_population_activity(true_pos)
            decoded = pop.decode_position(activity)
            errors.append(np.linalg.norm(decoded - true_pos))

        mean_error = np.mean(errors)
        max_error = np.max(errors)

        results.record(
            "1000 position decode (mean < 0.15)",
            mean_error < 0.15,
            f"mean: {mean_error:.4f}, max: {max_error:.4f}",
        )
    except Exception as e:
        results.record("1000 position decode (mean < 0.15)", False, str(e))

    # Test 2: Sparse activity pattern
    try:
        pop = PlaceCellPopulation(n_cells=200, field_radius=0.05, random_seed=42)

        # Smaller fields = sparser activity
        activity = pop.get_population_activity(np.array([0.5, 0.5]))
        active_fraction = np.sum(activity > 1.0) / len(activity)

        results.record(
            "Sparse place field coverage",
            0.01 < active_fraction < 0.3,
            f"active fraction: {active_fraction:.3f}",
        )
    except Exception as e:
        results.record("Sparse place field coverage", False, str(e))

    # Test 3: Large environment
    try:
        pop = PlaceCellPopulation(n_cells=500, environment_size=(10.0, 10.0), random_seed=42)

        # Test multiple positions
        test_positions = [
            np.array([1.0, 1.0]),
            np.array([5.0, 5.0]),
            np.array([9.0, 9.0]),
        ]

        activities = []
        for pos in test_positions:
            max_act = np.max(pop.get_population_activity(pos))
            activities.append(max_act)

        all_active = all(a > 0 for a in activities)

        results.record(
            "Large environment (10x10) coverage",
            all_active,
            f"activities: {[f'{a:.2f}' for a in activities]}",
        )
    except Exception as e:
        results.record("Large environment (10x10) coverage", False, str(e))


# =============================================================================
# 10. ADVANCED GRID CELL STRESS TESTS
# =============================================================================


def test_advanced_grid_cells():
    test_section("ADVANCED GRID CELL STRESS")

    # Test 1: Periodicity verification
    try:
        cell = GridCell(spacing=0.25, orientation=0.0)

        # Sample along x-axis
        rates = [cell.compute_firing(np.array([x, 0])) for x in np.linspace(0, 1, 50)]

        # Count high activity regions (rate > 10)
        high_regions = 0
        in_high = False
        for rate in rates:
            if rate > 10 and not in_high:
                high_regions += 1
                in_high = True
            elif rate <= 8:
                in_high = False

        max_rate = max(rates)
        min_rate = min(rates)
        dynamic_range = max_rate - min_rate

        # Grid cells should show clear periodicity with high dynamic range
        results.record(
            "Grid periodicity (2+ high regions, range > 10)",
            high_regions >= 2 and dynamic_range > 10,
            f"regions: {high_regions}, range: {dynamic_range:.2f}",
        )
    except Exception as e:
        results.record("Grid periodicity (2+ high regions, range > 10)", False, str(e))

    # Test 2: Multi-module independence
    try:
        pop = GridCellPopulation(n_modules=5, cells_per_module=15, random_seed=42)

        # Each module should have different spacing
        spacings = [m.spacing for m in pop.modules]
        all_different = len(set([round(s, 3) for s in spacings])) == len(spacings)

        results.record(
            "5 modules with different spacings",
            all_different,
            f"spacings: {[f'{s:.3f}' for s in spacings]}",
        )
    except Exception as e:
        results.record("5 modules with different spacings", False, str(e))

    # Test 3: Long path integration accuracy
    try:
        pop = GridCellPopulation(random_seed=42)
        pop.reset_position(np.array([0.0, 0.0]))

        # Move 100 steps in a straight line
        for _ in range(100):
            pop.path_integrate(np.array([0.05, 0.03]), dt=1.0)

        expected = np.array([5.0, 3.0])
        actual = pop.get_position_estimate()
        error = np.linalg.norm(actual - expected)

        results.record(
            "Long path integration (100 steps)",
            error < 0.5,
            f"expected: {expected}, actual: {actual}, error: {error:.3f}",
        )
    except Exception as e:
        results.record("Long path integration (100 steps)", False, str(e))


# =============================================================================
# 11. ADVANCED HEAD DIRECTION STRESS TESTS
# =============================================================================


def test_advanced_head_direction():
    test_section("ADVANCED HEAD DIRECTION STRESS")

    # Test 1: Full rotation tracking
    try:
        system = HeadDirectionSystem(n_cells=60)
        system._current_heading = 0.0

        # Rotate through full 360 degrees
        headings = []
        for i in range(36):
            system.update_heading(np.pi / 18, dt=1.0)  # 10 degrees per step
            headings.append(system.get_heading_degrees())

        # Should complete full rotation
        final_heading = system.get_current_heading()
        expected = 2 * np.pi
        error = abs(final_heading - expected)

        results.record(
            "Full 360-degree rotation tracking",
            error < 0.1,
            f"final: {np.degrees(final_heading):.1f} deg",
        )
    except Exception as e:
        results.record("Full 360-degree rotation tracking", False, str(e))

    # Test 2: Bidirectional rotation
    try:
        system = HeadDirectionSystem()
        system._current_heading = np.pi  # Start facing south

        # Rotate left (positive)
        for _ in range(5):
            system.update_heading(np.pi / 10, dt=1.0)

        # Rotate right (negative)
        for _ in range(10):
            system.update_heading(-np.pi / 10, dt=1.0)

        # Should end up at pi - pi/2 = pi/2 (facing east)
        expected = np.pi / 2
        actual = system.get_current_heading()

        # Account for wraparound
        error = abs(actual - expected)
        if error > np.pi:
            error = 2 * np.pi - error

        results.record(
            "Bidirectional rotation accuracy",
            error < 0.2,
            f"expected: {np.degrees(expected):.1f}, actual: {np.degrees(actual):.1f}",
        )
    except Exception as e:
        results.record("Bidirectional rotation accuracy", False, str(e))

    # Test 3: Population vector stability
    try:
        system = HeadDirectionSystem(n_cells=100)

        # Test at 8 compass directions
        directions = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        errors = []

        for direction in directions:
            activity = system.get_population_activity(direction)
            decoded = system.decode_heading(activity)

            error = abs(decoded - direction)
            if error > np.pi:
                error = 2 * np.pi - error
            errors.append(error)

        max_error = np.max(errors)

        results.record(
            "8 compass directions decode (max error < 0.15)",
            max_error < 0.15,
            f"max error: {max_error:.4f} rad",
        )
    except Exception as e:
        results.record("8 compass directions decode (max error < 0.15)", False, str(e))


# =============================================================================
# 12. ADVANCED PATH INTEGRATION STRESS TESTS
# =============================================================================


def test_advanced_path_integration():
    test_section("ADVANCED PATH INTEGRATION STRESS")

    # Test 1: Return to home
    try:
        integrator = PathIntegrator(initial_position=np.array([0.5, 0.5]), noise_scale=0.0)

        # Outward journey
        for _ in range(20):
            integrator.integrate(np.array([0.05, 0.0]), dt=1.0)
        for _ in range(20):
            integrator.integrate(np.array([0.0, 0.05]), dt=1.0)

        # Return journey (opposite)
        for _ in range(20):
            integrator.integrate(np.array([0.0, -0.05]), dt=1.0)
        for _ in range(20):
            integrator.integrate(np.array([-0.05, 0.0]), dt=1.0)

        final = integrator.position_estimate
        start = np.array([0.5, 0.5])
        error = np.linalg.norm(final - start)

        results.record("Return to home (square path, no noise)", error < 0.1, f"error: {error:.4f}")
    except Exception as e:
        results.record("Return to home (square path, no noise)", False, str(e))

    # Test 2: Long journey with noise
    try:
        integrator = PathIntegrator(initial_position=np.array([0.0, 0.0]), noise_scale=0.001)

        # 500 steps in random directions
        np.random.seed(42)
        total_displacement = np.array([0.0, 0.0])
        for _ in range(500):
            velocity = np.random.randn(2) * 0.02
            total_displacement += velocity
            integrator.integrate(velocity, dt=1.0)

        expected = total_displacement
        actual = integrator.position_estimate
        error = np.linalg.norm(actual - expected)

        results.record(
            "500 step random walk with noise",
            error < 1.0,  # Allow for accumulated noise
            f"error: {error:.3f}",
        )
    except Exception as e:
        results.record("500 step random walk with noise", False, str(e))

    # Test 3: Spiral path
    try:
        integrator = PathIntegrator(initial_position=np.array([0.5, 0.5]), noise_scale=0.0)

        # Spiral outward
        for i in range(100):
            angle = i * 0.2
            radius = 0.01 + i * 0.001
            velocity = radius * np.array([np.cos(angle), np.sin(angle)])
            integrator.integrate(velocity, dt=1.0)

        trajectory = integrator.get_trajectory()
        total_dist = integrator.get_total_distance()

        results.record(
            "Spiral path tracking (100 steps)",
            len(trajectory) == 101 and total_dist > 0,
            f"distance: {total_dist:.2f}",
        )
    except Exception as e:
        results.record("Spiral path tracking (100 steps)", False, str(e))


# =============================================================================
# 13. ADVANCED CONCEPTUAL SPACE STRESS TESTS
# =============================================================================


def test_advanced_conceptual_space():
    test_section("ADVANCED CONCEPTUAL SPACE STRESS")

    # Test 1: Large concept network
    try:
        cmap = ConceptualMap(concept_dimensions=20, n_concept_cells=200, random_seed=42)

        # Embed 500 concepts in clusters
        np.random.seed(42)
        for cluster in range(5):
            base = np.random.randn(20) * 0.5
            for i in range(100):
                features = base + np.random.randn(20) * 0.1
                cmap.embed_concept(f"cluster{cluster}_item{i}", features)

        results.record("Large concept network (500 concepts)", len(cmap) == 500)
    except Exception as e:
        results.record("Large concept network (500 concepts)", False, str(e))

    # Test 2: Cluster similarity
    try:
        cmap = ConceptualMap(concept_dimensions=10, random_seed=42)

        # Create two distinct clusters
        for i in range(10):
            cmap.embed_concept(
                f"animal_{i}",
                np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                + np.random.randn(10) * 0.1,
            )
            cmap.embed_concept(
                f"vehicle_{i}",
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                + np.random.randn(10) * 0.1,
            )

        # Animals should be closer to animals
        similar_to_animal = cmap.find_similar("animal_0", n=5)
        animal_count = sum(1 for name, _ in similar_to_animal if "animal" in name)

        results.record(
            "Cluster similarity preserved", animal_count >= 4, f"animals in top 5: {animal_count}"
        )
    except Exception as e:
        results.record("Cluster similarity preserved", False, str(e))

    # Test 3: Complex analogy chains
    try:
        cmap = ConceptualMap(concept_dimensions=4, random_seed=42)

        # Create structured relationships
        # country:capital :: country:capital
        cmap.embed_concept("france", np.array([1.0, 0.0, 0.0, 0.0]))
        cmap.embed_concept("paris", np.array([1.0, 1.0, 0.0, 0.0]))
        cmap.embed_concept("germany", np.array([0.0, 0.0, 1.0, 0.0]))
        cmap.embed_concept("berlin", np.array([0.0, 1.0, 1.0, 0.0]))
        cmap.embed_concept("italy", np.array([0.0, 0.0, 0.0, 1.0]))
        cmap.embed_concept("rome", np.array([0.0, 1.0, 0.0, 1.0]))

        # france:paris :: germany:?
        result = cmap.compute_analogy("france", "paris", "germany")

        results.record(
            "Country:capital analogy",
            result is not None and result[0] == "berlin",
            f"result: {result}",
        )
    except Exception as e:
        results.record("Country:capital analogy", False, str(e))

    # Test 4: High-dimensional concept space
    try:
        cmap = ConceptualMap(concept_dimensions=100, random_seed=42)

        for i in range(50):
            features = np.zeros(100)
            features[i * 2 : (i + 1) * 2] = 1.0  # Each concept has unique features
            cmap.embed_concept(f"hd_concept_{i}", features)

        # All should be equidistant
        dist1 = cmap.conceptual_distance("hd_concept_0", "hd_concept_1")
        dist2 = cmap.conceptual_distance("hd_concept_0", "hd_concept_25")

        results.record(
            "100-dimensional concept space",
            abs(dist1 - dist2) < 0.1,
            f"dist1: {dist1:.3f}, dist2: {dist2:.3f}",
        )
    except Exception as e:
        results.record("100-dimensional concept space", False, str(e))


# =============================================================================
# 14. COGNITIVE MAP EXTREME STRESS TESTS
# =============================================================================


def test_extreme_cognitive_map():
    test_section("EXTREME COGNITIVE MAP STRESS")

    # Test 1: Many location memories
    try:
        cmap = CognitiveMap(random_seed=42)

        for i in range(100):
            pos = np.random.rand(2)
            cmap.set_position(pos)
            cmap.encode_location(f"location_{i}")

        # Recall random locations
        errors = []
        for i in [0, 25, 50, 75, 99]:
            recalled = cmap.recall_location(f"location_{i}")
            if recalled is not None:
                errors.append(1)
            else:
                errors.append(0)

        results.record("100 location memories stored and recalled", sum(errors) == 5)
    except Exception as e:
        results.record("100 location memories stored and recalled", False, str(e))

    # Test 2: Rapid environment switching
    try:
        cmap = CognitiveMap(random_seed=42)

        for i in range(10):
            env = Environment(bounds=(0, i + 1, 0, i + 1), name=f"env_{i}")
            cmap.remap(env)
            cmap.set_position(np.array([(i + 1) / 2, (i + 1) / 2]))

        final_pos = cmap.get_position()

        results.record(
            "10 environment remappings",
            cmap.environment.name == "env_9" and cmap.environment.is_valid_position(final_pos),
        )
    except Exception as e:
        results.record("10 environment remappings", False, str(e))

    # Test 3: Very long navigation
    try:
        cmap = CognitiveMap(environment=Environment(bounds=(0, 100, 0, 100)), random_seed=42)
        cmap.set_position(np.array([50.0, 50.0]))

        # Navigate 5000 steps
        for _ in range(5000):
            velocity = np.random.randn(2) * 0.1
            cmap.update(velocity=velocity, dt=0.1)

        pos = cmap.get_position()

        results.record(
            "5000 step navigation", cmap.environment.is_valid_position(pos), f"final: {pos}"
        )
    except Exception as e:
        results.record("5000 step navigation", False, str(e))

    # Test 4: Full state retrieval under load
    try:
        cmap = CognitiveMap(n_place_cells=500, n_grid_modules=6, random_seed=42)

        states = []
        for _ in range(100):
            cmap.update(velocity=np.random.randn(2) * 0.01, dt=0.1)
            states.append(cmap.get_state())

        # Verify state integrity
        all_valid = all(
            len(s.place_activity) == 500
            and len(s.grid_activity) > 0
            and len(s.head_direction_activity) > 0
            for s in states
        )

        results.record("100 state retrievals with large populations", all_valid)
    except Exception as e:
        results.record("100 state retrievals with large populations", False, str(e))


# =============================================================================
# 15. PERFORMANCE AND MEMORY STRESS
# =============================================================================


def test_performance():
    test_section("PERFORMANCE STRESS")

    # Test 1: Speed test - place cells
    try:
        pop = PlaceCellPopulation(n_cells=1000, random_seed=42)

        start = time.time()
        for _ in range(1000):
            pop.get_population_activity(np.random.rand(2))
        elapsed = time.time() - start

        results.record("1000 place cell queries (< 2s)", elapsed < 2.0, f"time: {elapsed:.3f}s")
    except Exception as e:
        results.record("1000 place cell queries (< 2s)", False, str(e))

    # Test 2: Speed test - cognitive map updates
    try:
        cmap = CognitiveMap(n_place_cells=200, random_seed=42)

        start = time.time()
        for _ in range(500):
            cmap.update(velocity=np.random.randn(2) * 0.1, dt=0.1)
        elapsed = time.time() - start

        results.record("500 cognitive map updates (< 2s)", elapsed < 2.0, f"time: {elapsed:.3f}s")
    except Exception as e:
        results.record("500 cognitive map updates (< 2s)", False, str(e))

    # Test 3: Speed test - concept similarity search
    try:
        cmap = ConceptualMap(concept_dimensions=50, random_seed=42)

        for i in range(200):
            cmap.embed_concept(f"c_{i}", np.random.randn(50))

        start = time.time()
        for i in range(100):
            cmap.find_similar(f"c_{i}", n=10)
        elapsed = time.time() - start

        results.record(
            "100 similarity searches (200 concepts) (< 1s)", elapsed < 1.0, f"time: {elapsed:.3f}s"
        )
    except Exception as e:
        results.record("100 similarity searches (200 concepts) (< 1s)", False, str(e))

    # Test 4: Memory - large populations
    try:
        # Create large populations
        place_pop = PlaceCellPopulation(n_cells=2000, random_seed=42)
        grid_pop = GridCellPopulation(n_modules=8, cells_per_module=50, random_seed=42)
        hd_system = HeadDirectionSystem(n_cells=200)

        total_cells = len(place_pop) + len(grid_pop) + len(hd_system)

        results.record(
            "Large neural populations (2000+ cells)",
            total_cells > 2000,
            f"total: {total_cells} cells",
        )
    except Exception as e:
        results.record("Large neural populations (2000+ cells)", False, str(e))


# =============================================================================
# RUN ALL TESTS
# =============================================================================


def run_all_tests():
    print("\n" + "=" * 60)
    print("MODULE 10: SPATIAL COGNITION - COMPREHENSIVE STRESS TESTS")
    print("=" * 60)

    start_time = time.time()

    test_place_cells()
    test_grid_cells()
    test_head_direction()
    test_border_cells()
    test_cognitive_map()
    test_path_integration()
    test_conceptual_space()
    test_edge_cases()
    test_advanced_place_cells()
    test_advanced_grid_cells()
    test_advanced_head_direction()
    test_advanced_path_integration()
    test_advanced_conceptual_space()
    test_extreme_cognitive_map()
    test_performance()

    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed:.2f}s")

    return results.summary()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
