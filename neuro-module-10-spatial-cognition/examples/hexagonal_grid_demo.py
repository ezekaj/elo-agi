#!/usr/bin/env python3
"""
Demo: Grid Cell Hexagonal Firing Patterns

Demonstrates:
- Hexagonal grid cell firing patterns
- Multiple scale modules
- Grid-based distance encoding
- Path integration using grids
"""

import sys

sys.path.insert(0, "..")

import numpy as np
from src import (
    GridCell,
    GridCellModule,
    GridCellPopulation,
)


def demo_hexagonal_grids():
    print("=" * 60)
    print("HEXAGONAL GRID CELL DEMO")
    print("=" * 60)

    # 1. SINGLE GRID CELL
    print("\n1. SINGLE GRID CELL PATTERN")
    print("-" * 40)

    cell = GridCell(spacing=0.3, orientation=0.0, peak_rate=15.0)

    # Sample firing pattern
    print("Firing rates across environment:")
    print("  Position    | Rate (Hz)")
    print("  " + "-" * 25)

    for x in [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]:
        rate = cell.compute_firing(np.array([x, 0.0]))
        bar = "#" * int(rate)
        print(f"  ({x:.2f}, 0.00) | {rate:5.2f} {bar}")

    # 2. HEXAGONAL PATTERN VERIFICATION
    print("\n2. HEXAGONAL PATTERN")
    print("-" * 40)

    directions = [
        ("East (0 deg)", 0),
        ("Northeast (60 deg)", np.pi / 3),
        ("Northwest (120 deg)", 2 * np.pi / 3),
    ]

    print("Peak count when moving in different directions:")
    for name, angle in directions:
        peaks = 0
        prev_rate = 0
        direction = np.array([np.cos(angle), np.sin(angle)])

        for t in range(100):
            pos = direction * t * 0.01
            rate = cell.compute_firing(pos)
            if rate > 10 and prev_rate <= 10:
                peaks += 1
            prev_rate = rate

        print(f"  {name}: {peaks} peaks in 1.0 units")

    # 3. MULTI-SCALE GRID MODULES
    print("\n3. MULTI-SCALE GRID MODULES")
    print("-" * 40)

    pop = GridCellPopulation(
        n_modules=4, cells_per_module=10, base_spacing=0.15, scale_ratio=1.5, random_seed=42
    )

    print(f"Total cells: {len(pop)}")
    print("\nModule properties:")
    for i, module in enumerate(pop.modules):
        print(f"  Module {i + 1}:")
        print(f"    Spacing: {module.spacing:.3f}")
        print(f"    Cells: {len(module.cells)}")
        print(f"    Orientation: {np.degrees(module.cells[0].params.orientation):.1f} deg")

    # 4. GRID CELL FIRING MAP
    print("\n4. GRID CELL FIRING MAP (Text Visualization)")
    print("-" * 40)

    cell = GridCell(spacing=0.25, orientation=0.0)

    print("Firing rate map (. = low, o = medium, O = high, @ = peak):")
    for y in np.linspace(1.0, 0.0, 15):
        row = ""
        for x in np.linspace(0.0, 1.0, 30):
            rate = cell.compute_firing(np.array([x, y]))
            if rate > 12:
                row += "@"
            elif rate > 8:
                row += "O"
            elif rate > 4:
                row += "o"
            else:
                row += "."
        print(f"  {row}")

    # 5. PATH INTEGRATION
    print("\n5. PATH INTEGRATION WITH GRID CELLS")
    print("-" * 40)

    pop = GridCellPopulation(random_seed=42)
    pop.reset_position(np.array([0.0, 0.0]))

    print("Moving in steps of 0.1 units east:")
    print("  Step | Position Estimate")
    print("  " + "-" * 30)

    for step in range(10):
        pos = pop.path_integrate(velocity=np.array([0.1, 0.0]), dt=1.0)
        if step % 2 == 0:
            print(f"    {step + 1:2d} | ({pos[0]:.2f}, {pos[1]:.2f})")

    print(f"\n  Final position: ({pos[0]:.2f}, {pos[1]:.2f})")
    print(f"  Expected: (1.00, 0.00)")
    print(f"  Error: {np.linalg.norm(pos - np.array([1.0, 0.0])):.4f}")

    # 6. CIRCULAR PATH
    print("\n6. CIRCULAR PATH INTEGRATION")
    print("-" * 40)

    pop = GridCellPopulation(random_seed=42)
    pop.reset_position(np.array([0.5, 0.5]))

    radius = 0.3
    n_steps = 50
    start_pos = pop.get_position_estimate().copy()

    for i in range(n_steps):
        angle = 2 * np.pi * i / n_steps
        velocity = 2 * np.pi * radius / n_steps * np.array([-np.sin(angle), np.cos(angle)])
        pop.path_integrate(velocity, dt=1.0)

    end_pos = pop.get_position_estimate()
    drift = np.linalg.norm(end_pos - start_pos)

    print(f"Circular path (radius={radius}):")
    print(f"  Start: ({start_pos[0]:.3f}, {start_pos[1]:.3f})")
    print(f"  End:   ({end_pos[0]:.3f}, {end_pos[1]:.3f})")
    print(f"  Drift: {drift:.4f}")

    # 7. GRID SCALE COMPARISON
    print("\n7. GRID SCALE COMPARISON")
    print("-" * 40)

    spacings = [0.1, 0.2, 0.4, 0.8]
    print("Resolution at different grid scales:")
    print("  Spacing | Peaks in 1.0 unit | Resolution")
    print("  " + "-" * 45)

    for spacing in spacings:
        cell = GridCell(spacing=spacing)
        peaks = 0
        prev_rate = 0
        for t in range(200):
            rate = cell.compute_firing(np.array([t * 0.005, 0]))
            if rate > 10 and prev_rate <= 10:
                peaks += 1
            prev_rate = rate

        resolution = "Fine" if spacing < 0.2 else "Medium" if spacing < 0.5 else "Coarse"
        print(f"    {spacing:.1f}  |        {peaks:2d}         | {resolution}")

    # 8. PHASE RELATIONSHIPS
    print("\n8. GRID CELL PHASE RELATIONSHIPS")
    print("-" * 40)

    module = GridCellModule(n_cells=5, spacing=0.3, orientation=0.0, random_seed=42)

    print("Cells in module have same spacing but different phases:")
    print("  Cell | Phase (x, y) | Rate at origin")
    print("  " + "-" * 40)

    for i, cell in enumerate(module.cells):
        phase = cell.params.phase
        rate = cell.compute_firing(np.array([0.0, 0.0]))
        print(f"    {i + 1}  | ({phase[0]:.2f}, {phase[1]:.2f})  | {rate:.2f} Hz")

    print("\n  Different phases = different coverage of space")
    print("  Together they tile the entire environment")

    print("\n" + "=" * 60)
    print("HEXAGONAL GRID DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo_hexagonal_grids()
