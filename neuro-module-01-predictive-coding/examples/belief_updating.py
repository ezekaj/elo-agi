"""
Demo: Belief Updating with Cognitive Manifold

This example demonstrates:
1. Cognitive states as points on a manifold
2. Thinking as gradient flow minimizing potential
3. Dual-process emergence (System 1/2) from geometry
4. Precision-weighted belief updating

The key insight: different cognitive styles emerge from the SAME
manifold with different local metrics, not from separate modules.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cognitive_manifold import (
    CognitiveState,
    CognitiveManifold,
    DualProcess,
    AttractorLandscape,
    FlowType
)
from src.precision_weighting import AdaptivePrecision
from src.predictive_hierarchy import PredictiveHierarchy


def demo_belief_flow():
    """Demonstrate belief updating as gradient flow."""
    print("=" * 60)
    print("Demo 1: Belief Updating as Gradient Flow")
    print("=" * 60)

    # Create manifold with goal (optimal belief state)
    manifold = CognitiveManifold(
        dim=2,
        accuracy_weight=1.0,
        parsimony_weight=0.1,
        utility_weight=0.8
    )

    # Set goal: where we want beliefs to converge
    goal = np.array([3.0, 2.0])
    manifold.set_goal(goal)

    # Start from initial belief
    manifold.state.position = np.array([-2.0, -1.0])

    print(f"\nInitial belief: {manifold.state.position}")
    print(f"Goal belief: {goal}")

    # Flow toward optimal beliefs
    trajectories = {
        'gradient': [],
        'natural': []
    }

    # Standard gradient descent
    manifold.reset(np.array([-2.0, -1.0]))
    for _ in range(100):
        trajectories['gradient'].append(manifold.state.position.copy())
        manifold.flow(dt=0.1, flow_type=FlowType.GRADIENT_DESCENT)

    # Natural gradient (metric-aware)
    manifold.reset(np.array([-2.0, -1.0]))
    # Use anisotropic metric
    manifold.state.metric = np.array([[2.0, 0.0], [0.0, 0.5]])
    manifold.state._update_inverse_metric()
    for _ in range(100):
        trajectories['natural'].append(manifold.state.position.copy())
        manifold.flow(dt=0.1, flow_type=FlowType.NATURAL_GRADIENT)

    # Convert to arrays
    for key in trajectories:
        trajectories[key] = np.array(trajectories[key])

    print(f"\nGradient descent final: {trajectories['gradient'][-1]}")
    print(f"Natural gradient final: {trajectories['natural'][-1]}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Trajectory plot
    ax = axes[0]
    ax.plot(trajectories['gradient'][:, 0], trajectories['gradient'][:, 1],
            'b-', label='Gradient Descent', linewidth=2)
    ax.plot(trajectories['natural'][:, 0], trajectories['natural'][:, 1],
            'r--', label='Natural Gradient', linewidth=2)

    ax.plot(-2.0, -1.0, 'ko', markersize=12, label='Start')
    ax.plot(goal[0], goal[1], 'g*', markersize=15, label='Goal')

    # Draw potential contours
    x = np.linspace(-3, 4, 50)
    y = np.linspace(-2, 3, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            state = np.array([X[i, j], Y[i, j]])
            Z[i, j] = manifold.parsimony_weight * np.sum(state ** 2) + \
                      manifold.utility_weight * np.sum((state - goal) ** 2)

    ax.contour(X, Y, Z, levels=20, alpha=0.3, cmap='viridis')

    ax.set_xlabel('Belief Dimension 1')
    ax.set_ylabel('Belief Dimension 2')
    ax.set_title('Belief Trajectories on Cognitive Manifold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 4)
    ax.set_ylim(-2, 3)

    # Potential over time
    ax = axes[1]

    pot_gradient = []
    pot_natural = []

    manifold.reset(np.array([-2.0, -1.0]))
    for pos in trajectories['gradient']:
        pot_gradient.append(manifold.potential(pos))

    manifold.reset(np.array([-2.0, -1.0]))
    for pos in trajectories['natural']:
        pot_natural.append(manifold.potential(pos))

    ax.plot(pot_gradient, 'b-', label='Gradient Descent', linewidth=2)
    ax.plot(pot_natural, 'r--', label='Natural Gradient', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cognitive Potential')
    ax.set_title('Potential Minimization (Thinking)')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('belief_flow.png', dpi=150)
    print("\nFigure saved: belief_flow.png")


def demo_dual_process_emergence():
    """Demonstrate System 1/2 emergence from geometry."""
    print("\n" + "=" * 60)
    print("Demo 2: Dual-Process Emergence from Geometry")
    print("=" * 60)

    # Create manifold
    manifold = CognitiveManifold(
        dim=2,
        parsimony_weight=1.0,
        utility_weight=0.5,
        accuracy_weight=0.0
    )

    manifold.set_goal(np.array([0.0, 0.0]))

    # Create dual-process system
    dp = DualProcess(manifold, fast_threshold=0.8, slow_threshold=0.2)

    # Test different starting positions
    scenarios = [
        ('Far from equilibrium', np.array([5.0, 5.0])),
        ('Near equilibrium', np.array([0.5, 0.5])),
        ('Medium distance', np.array([2.0, 2.0]))
    ]

    results = {}

    for name, start in scenarios:
        manifold.reset(start)
        final_pos, systems = dp.think(max_steps=100, dt=0.1)
        s1_prop, s2_prop = dp.get_system_balance(systems)

        results[name] = {
            'trajectory': manifold.get_trajectory(),
            'systems': systems,
            's1_prop': s1_prop,
            's2_prop': s2_prop,
            'start': start
        }

        print(f"\n{name}:")
        print(f"  System 1 (fast): {s1_prop * 100:.1f}%")
        print(f"  System 2 (slow): {s2_prop * 100:.1f}%")
        print(f"  Steps to converge: {len(systems)}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Trajectories colored by system
    ax = axes[0, 0]
    colors = {'Far from equilibrium': 'blue',
              'Near equilibrium': 'green',
              'Medium distance': 'orange'}

    for name, data in results.items():
        traj = data['trajectory']
        ax.plot(traj[:, 0], traj[:, 1], color=colors[name],
                linewidth=2, label=name)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=colors[name], markersize=10)

    ax.plot(0, 0, 'k*', markersize=15, label='Goal')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Thinking Trajectories')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # System usage over time (far from equilibrium case)
    ax = axes[0, 1]
    systems = results['Far from equilibrium']['systems']
    ax.plot(systems, 'b-', linewidth=1.5)
    ax.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('System (1=Fast, 2=Slow)')
    ax.set_title('System Switching: Far from Equilibrium')
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['System 1\n(Fast)', 'System 2\n(Slow)'])
    ax.grid(True, alpha=0.3)

    # System proportions comparison
    ax = axes[1, 0]
    names = list(results.keys())
    s1_props = [results[n]['s1_prop'] for n in names]
    s2_props = [results[n]['s2_prop'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, s1_props, width, label='System 1 (Fast)', color='coral')
    bars2 = ax.bar(x + width/2, s2_props, width, label='System 2 (Slow)', color='steelblue')

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Proportion')
    ax.set_title('System Usage by Scenario')
    ax.set_xticks(x)
    ax.set_xticklabels(['Far', 'Near', 'Medium'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gradient magnitude (determines system)
    ax = axes[1, 1]

    manifold.reset(np.array([5.0, 5.0]))
    gradients = []
    for _ in range(50):
        grad = manifold.gradient()
        gradients.append(np.linalg.norm(grad))
        manifold.flow(dt=0.1)

    ax.plot(gradients, 'b-', linewidth=2)
    ax.axhline(y=dp.fast_threshold, color='coral', linestyle='--',
               label=f'Fast threshold ({dp.fast_threshold})')
    ax.axhline(y=dp.slow_threshold, color='steelblue', linestyle='--',
               label=f'Slow threshold ({dp.slow_threshold})')

    ax.fill_between(range(len(gradients)),
                    dp.fast_threshold, max(gradients) + 0.5,
                    alpha=0.2, color='coral', label='System 1 region')
    ax.fill_between(range(len(gradients)),
                    0, dp.slow_threshold,
                    alpha=0.2, color='steelblue', label='System 2 region')

    ax.set_xlabel('Step')
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('System Selection by Gradient Steepness')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dual_process.png', dpi=150)
    print("\nFigure saved: dual_process.png")


def demo_attractor_dynamics():
    """Demonstrate belief attractors (stable cognitive states)."""
    print("\n" + "=" * 60)
    print("Demo 3: Attractor Dynamics (Stable Beliefs)")
    print("=" * 60)

    # Create manifold
    manifold = CognitiveManifold(
        dim=2,
        parsimony_weight=0.01,
        utility_weight=0.0,
        accuracy_weight=0.0
    )

    # Create attractor landscape with multiple belief attractors
    landscape = AttractorLandscape(manifold)

    # Add attractors (stable beliefs)
    attractors = [
        (np.array([-2.0, 0.0]), 1.5, 'Belief A'),
        (np.array([2.0, 0.0]), 1.0, 'Belief B'),
        (np.array([0.0, 2.0]), 0.8, 'Belief C')
    ]

    for pos, strength, name in attractors:
        landscape.add_attractor(pos, strength)
        print(f"Added attractor: {name} at {pos} (strength={strength})")

    # Override manifold potential with attractor potential
    manifold._custom_utility = landscape.attractor_potential

    # Test convergence from different starting points
    starts = [
        np.array([-1.0, 0.5]),
        np.array([1.0, 0.5]),
        np.array([0.0, 1.5]),
        np.array([0.0, -0.5])
    ]

    trajectories = []
    final_attractors = []

    for start in starts:
        manifold.reset(start)
        final, steps = manifold.flow_until_convergence(dt=0.05, max_steps=200)
        trajectories.append(manifold.get_trajectory())

        nearest = landscape.find_nearest_attractor(final)
        final_attractors.append(nearest)
        print(f"\nStart {start} -> Attractor {nearest} ({attractors[nearest][2]})")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Attractor landscape
    ax = axes[0]

    # Draw potential surface
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-2, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            state = np.array([X[i, j], Y[i, j]])
            Z[i, j] = landscape.attractor_potential(state)

    contour = ax.contourf(X, Y, Z, levels=30, cmap='RdYlBu_r', alpha=0.7)
    plt.colorbar(contour, ax=ax, label='Potential')

    # Draw attractors
    colors_attr = ['red', 'green', 'blue']
    for i, (pos, strength, name) in enumerate(attractors):
        ax.plot(pos[0], pos[1], 'o', color=colors_attr[i],
                markersize=15 * strength, label=name)

    # Draw trajectories
    traj_colors = ['purple', 'orange', 'cyan', 'magenta']
    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], color=traj_colors[i],
                linewidth=2, alpha=0.8)
        ax.plot(traj[0, 0], traj[0, 1], 's', color=traj_colors[i], markersize=10)

    ax.set_xlabel('Belief Dimension 1')
    ax.set_ylabel('Belief Dimension 2')
    ax.set_title('Attractor Landscape (Stable Beliefs)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Basin membership
    ax = axes[1]

    # Sample many starting points
    n_samples = 500
    sample_starts = np.random.randn(n_samples, 2) * 2

    basin_membership = []
    for start in sample_starts:
        manifold.reset(start)
        final, _ = manifold.flow_until_convergence(dt=0.05, max_steps=100)
        nearest = landscape.find_nearest_attractor(final)
        basin_membership.append(nearest)

    # Plot with colors by basin
    for i in range(len(attractors)):
        mask = np.array(basin_membership) == i
        ax.scatter(sample_starts[mask, 0], sample_starts[mask, 1],
                   c=colors_attr[i], alpha=0.3, s=20,
                   label=f'{attractors[i][2]} basin')

    # Draw attractors
    for i, (pos, strength, name) in enumerate(attractors):
        ax.plot(pos[0], pos[1], 'o', color=colors_attr[i],
                markersize=15, markeredgecolor='black', markeredgewidth=2)

    ax.set_xlabel('Starting Position X')
    ax.set_ylabel('Starting Position Y')
    ax.set_title('Basin of Attraction Membership')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('attractor_dynamics.png', dpi=150)
    print("\nFigure saved: attractor_dynamics.png")


def demo_precision_belief_updating():
    """Demonstrate precision-weighted belief updating."""
    print("\n" + "=" * 60)
    print("Demo 4: Precision-Weighted Belief Updating")
    print("=" * 60)

    dim = 3

    # Create precision tracker
    precision = AdaptivePrecision(dim=dim, learning_rate=0.3)

    # Create predictive hierarchy
    hierarchy = PredictiveHierarchy(
        layer_dims=[dim, 4, 3],
        learning_rate=0.2
    )

    # Simulate noisy observations of a true state
    true_state = np.array([1.0, -0.5, 0.8])

    # Phase 1: High noise (low precision)
    print("\nPhase 1: High noise environment")
    high_noise_errors = []
    for t in range(30):
        noise = np.random.randn(dim) * 0.8  # High noise
        observation = true_state + noise
        result = hierarchy.step(observation, dt=0.1)

        error = observation - hierarchy.layers[0].generate_prediction()
        prec, vol = precision.update(error)
        high_noise_errors.append(result['total_error'])

    print(f"  Mean precision: {np.mean(precision.precision):.3f}")
    print(f"  Mean volatility: {np.mean(precision.volatility):.3f}")
    print(f"  Mean error: {np.mean(high_noise_errors):.4f}")

    # Phase 2: Low noise (high precision)
    print("\nPhase 2: Low noise environment")
    low_noise_errors = []
    for t in range(30):
        noise = np.random.randn(dim) * 0.1  # Low noise
        observation = true_state + noise
        result = hierarchy.step(observation, dt=0.1)

        error = observation - hierarchy.layers[0].generate_prediction()
        prec, vol = precision.update(error)
        low_noise_errors.append(result['total_error'])

    print(f"  Mean precision: {np.mean(precision.precision):.3f}")
    print(f"  Mean volatility: {np.mean(precision.volatility):.3f}")
    print(f"  Mean error: {np.mean(low_noise_errors):.4f}")

    # Phase 3: Sudden change (regime shift)
    print("\nPhase 3: Regime shift (new true state)")
    new_true_state = np.array([-1.0, 0.5, -0.8])
    shift_errors = []
    for t in range(30):
        noise = np.random.randn(dim) * 0.1
        observation = new_true_state + noise
        result = hierarchy.step(observation, dt=0.1)

        error = observation - hierarchy.layers[0].generate_prediction()
        prec, vol = precision.update(error)
        shift_errors.append(result['total_error'])

    print(f"  Mean precision: {np.mean(precision.precision):.3f}")
    print(f"  Mean volatility: {np.mean(precision.volatility):.3f}")
    print(f"  Mean error: {np.mean(shift_errors):.4f}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Error over time
    ax = axes[0, 0]
    all_errors = high_noise_errors + low_noise_errors + shift_errors
    ax.plot(all_errors, 'b-', linewidth=1.5)
    ax.axvline(x=30, color='gray', linestyle='--', label='Noise reduction')
    ax.axvline(x=60, color='red', linestyle='--', label='Regime shift')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Prediction Error Across Phases')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision history
    ax = axes[0, 1]
    prec_history = np.array(precision.precision_history)
    for i in range(min(3, prec_history.shape[1])):
        ax.plot(prec_history[:, i], label=f'Dim {i}')
    ax.axvline(x=30, color='gray', linestyle='--')
    ax.axvline(x=60, color='red', linestyle='--')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Precision')
    ax.set_title('Precision Adaptation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Confidence over phases
    ax = axes[1, 0]
    phases = ['High Noise\n(Phase 1)', 'Low Noise\n(Phase 2)', 'After Shift\n(Phase 3)']
    errors = [np.mean(high_noise_errors), np.mean(low_noise_errors), np.mean(shift_errors)]
    colors = ['red', 'green', 'orange']
    ax.bar(phases, errors, color=colors)
    ax.set_ylabel('Mean Error')
    ax.set_title('Error by Phase')
    ax.grid(True, alpha=0.3)

    # Hierarchy state representation
    ax = axes[1, 1]
    beliefs = hierarchy.get_beliefs()
    layer_labels = [f'Layer {i}' for i in range(len(beliefs))]
    belief_norms = [np.linalg.norm(b) for b in beliefs]

    ax.barh(layer_labels, belief_norms, color='steelblue')
    ax.set_xlabel('Belief Magnitude')
    ax.set_title('Hierarchical Belief States')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('precision_updating.png', dpi=150)
    print("\nFigure saved: precision_updating.png")


def main():
    """Run all belief updating demos."""
    print("\n" + "=" * 60)
    print("BELIEF UPDATING DEMOS")
    print("Cognitive Manifold and Dual-Process Cognition")
    print("=" * 60)

    demo_belief_flow()
    demo_dual_process_emergence()
    demo_attractor_dynamics()
    demo_precision_belief_updating()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
