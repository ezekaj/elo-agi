#!/usr/bin/env python3
"""
Demo: Spatial Navigation with Cognitive Map

Demonstrates:
- Place cell firing at specific locations
- Grid cell path integration
- Head direction tracking
- Border cell boundary detection
- Navigation to remembered locations
"""

import sys

sys.path.insert(0, "..")

import numpy as np
from src import (
    CognitiveMap,
    Environment,
)


def demo_navigation():
    print("=" * 60)
    print("SPATIAL NAVIGATION DEMO")
    print("=" * 60)

    # 1. CREATE ENVIRONMENT
    print("\n1. CREATING ENVIRONMENT")
    print("-" * 40)

    env = Environment(bounds=(0, 1, 0, 1), name="arena")
    env.add_landmark("food", np.array([0.8, 0.8]), type="resource")
    env.add_landmark("water", np.array([0.2, 0.8]), type="resource")
    env.add_landmark("nest", np.array([0.5, 0.2]), type="home")

    print(f"Environment: {env.name}")
    print(f"Size: {env.width} x {env.height}")
    print(f"Landmarks: {[lm.name for lm in env.landmarks]}")

    # 2. CREATE COGNITIVE MAP
    print("\n2. CREATING COGNITIVE MAP")
    print("-" * 40)

    cmap = CognitiveMap(environment=env, n_place_cells=200, n_grid_modules=4, random_seed=42)

    print(f"Place cells: {len(cmap.place_cells)}")
    print(f"Grid modules: {len(cmap.grid_cells.modules)}")
    print(f"Head direction cells: {len(cmap.head_direction)}")
    print(f"Border cells: {len(cmap.border_cells)}")

    # 3. EXPLORE ENVIRONMENT
    print("\n3. EXPLORING ENVIRONMENT")
    print("-" * 40)

    # Start at nest
    cmap.set_position(np.array([0.5, 0.2]))
    cmap.encode_location("start")

    print(f"Starting position: {cmap.get_position()}")

    # Random exploration
    np.random.seed(42)
    positions_visited = [cmap.get_position().copy()]

    for step in range(50):
        # Random velocity
        velocity = np.random.randn(2) * 0.05
        cmap.update(velocity=velocity, dt=0.1)
        positions_visited.append(cmap.get_position().copy())

    print(f"Explored {len(positions_visited)} positions")

    # 4. PLACE CELL ACTIVITY
    print("\n4. PLACE CELL ACTIVITY")
    print("-" * 40)

    test_positions = [
        ("nest area", np.array([0.5, 0.2])),
        ("food area", np.array([0.8, 0.8])),
        ("center", np.array([0.5, 0.5])),
    ]

    for name, pos in test_positions:
        activity = cmap.place_cells.get_population_activity(pos)
        active_cells = len([a for a in activity if a > 5.0])
        max_rate = np.max(activity)
        print(f"  {name}: {active_cells} active cells, max rate: {max_rate:.1f} Hz")

    # 5. NAVIGATION TO GOAL
    print("\n5. NAVIGATION TO GOAL")
    print("-" * 40)

    # Return to food
    cmap.set_position(np.array([0.3, 0.3]))
    goal = np.array([0.8, 0.8])  # Food location

    print(f"Current position: {cmap.get_position()}")
    print(f"Goal (food): {goal}")

    # Navigate
    nav_steps = 0
    while nav_steps < 50:
        velocity, distance = cmap.navigate_to(goal, speed=0.05)
        cmap.update(velocity=velocity, dt=1.0)
        nav_steps += 1

        if distance < 0.1:
            print(f"Reached goal in {nav_steps} steps!")
            break

    print(f"Final position: {cmap.get_position()}")
    print(f"Distance to goal: {np.linalg.norm(cmap.get_position() - goal):.3f}")

    # 6. HEAD DIRECTION
    print("\n6. HEAD DIRECTION")
    print("-" * 40)

    # Turn in place
    cmap.set_heading(0)
    print(f"Initial heading: {cmap.get_heading_degrees():.1f}°")

    for _ in range(4):
        cmap.update(velocity=np.zeros(2), angular_velocity=np.pi / 4, dt=1.0)
        state = cmap.get_state()

        # Find peak head direction cell
        hd_activity = state.head_direction_activity
        peak_idx = np.argmax(hd_activity)
        peak_dir = (360 * peak_idx) / len(hd_activity)

        print(f"  Heading: {cmap.get_heading_degrees():.1f}°, Peak HD cell: {peak_dir:.1f}°")

    # 7. BORDER CELLS
    print("\n7. BORDER CELL ACTIVITY")
    print("-" * 40)

    border_positions = [
        ("center", np.array([0.5, 0.5])),
        ("near north", np.array([0.5, 0.95])),
        ("near west", np.array([0.05, 0.5])),
        ("corner", np.array([0.05, 0.95])),
    ]

    for name, pos in border_positions:
        cmap.set_position(pos)
        state = cmap.get_state()
        walls = [w.value for w in state.nearby_walls]
        print(f"  {name}: nearby walls = {walls}")

    # 8. ENCODE AND RECALL LOCATIONS
    print("\n8. LOCATION MEMORY")
    print("-" * 40)

    # Encode important locations
    cmap.set_position(np.array([0.8, 0.8]))
    cmap.encode_location("food_found")

    cmap.set_position(np.array([0.2, 0.8]))
    cmap.encode_location("water_found")

    # Move away
    cmap.set_position(np.array([0.5, 0.5]))

    # Recall
    food_loc = cmap.recall_location("food_found")
    water_loc = cmap.recall_location("water_found")

    print(f"Recalled food location: {food_loc}")
    print(f"Recalled water location: {water_loc}")

    # Compute vectors to remembered locations
    dist_food = cmap.distance_to_location("food_found")
    bearing_food = cmap.bearing_to_location("food_found")

    print("\nFrom center to food:")
    print(f"  Distance: {dist_food:.3f}")
    print(f"  Bearing: {np.degrees(bearing_food):.1f}°")

    # 9. PATH INTEGRATION
    print("\n9. PATH INTEGRATION TRAJECTORY")
    print("-" * 40)

    cmap.set_position(np.array([0.5, 0.5]))
    cmap.set_heading(0)

    # Walk in a square
    for direction, steps in [("east", 10), ("north", 10), ("west", 10), ("south", 10)]:
        heading_map = {"east": 0, "north": np.pi / 2, "west": np.pi, "south": 3 * np.pi / 2}
        cmap.set_heading(heading_map[direction])

        for _ in range(steps):
            velocity = 0.02 * np.array([np.cos(cmap.get_heading()), np.sin(cmap.get_heading())])
            cmap.update(velocity=velocity, dt=1.0)

    trajectory = cmap.get_trajectory()
    print(f"Trajectory: {len(trajectory)} points")
    print(f"Start: {trajectory[0]}")
    print(f"End: {trajectory[-1]}")
    print(f"Return error: {np.linalg.norm(trajectory[-1] - trajectory[0]):.4f}")

    print("\n" + "=" * 60)
    print("NAVIGATION DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo_navigation()
