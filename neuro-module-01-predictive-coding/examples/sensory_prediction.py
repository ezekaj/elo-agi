"""
Demo: Sensory Prediction with Hierarchical Predictive Coding

This example demonstrates:
1. Learning to predict sensory sequences (birdsong-like patterns)
2. Hierarchical representation at multiple timescales
3. Omission detection when expected stimuli are missing
4. Lesion simulation showing hierarchy necessity

Based on empirical findings from predictive coding research.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.predictive_hierarchy import PredictiveHierarchy
from src.omission_detector import OmissionDetector, SequenceOmissionDetector
from src.temporal_dynamics import TemporalHierarchy


def generate_birdsong_sequence(n_samples: int = 100, dim: int = 5) -> np.ndarray:
    """Generate synthetic birdsong-like temporal sequence.

    Creates hierarchically structured sequences with:
    - Fast syllables (high frequency)
    - Phrases (medium frequency)
    - Motifs (low frequency)
    """
    sequence = np.zeros((n_samples, dim))

    # Syllable level: fast oscillations
    for i in range(dim):
        freq = 0.3 + i * 0.1  # Different frequencies per channel
        sequence[:, i] += 0.5 * np.sin(2 * np.pi * freq * np.arange(n_samples))

    # Phrase level: slower modulation
    phrase_freq = 0.05
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * phrase_freq * np.arange(n_samples))
    sequence *= modulation[:, np.newaxis]

    # Motif level: repeated patterns
    motif_length = 20
    n_motifs = n_samples // motif_length
    for m in range(n_motifs):
        start = m * motif_length
        end = start + motif_length
        # Add motif-specific signature
        sequence[start:end, m % dim] += 0.3

    return sequence


def demo_sequence_learning():
    """Demonstrate learning to predict sequences."""
    print("=" * 60)
    print("Demo 1: Sequence Learning (Birdsong-like)")
    print("=" * 60)

    # Create hierarchy
    input_dim = 5
    hierarchy = PredictiveHierarchy(
        layer_dims=[input_dim, 8, 6, 4], learning_rate=0.15, timescale_factor=3.0
    )

    # Generate training sequence
    sequence = generate_birdsong_sequence(n_samples=200, dim=input_dim)

    # Training phase
    print("\nTraining on sequence...")
    errors_over_time = []

    for epoch in range(3):
        epoch_errors = []
        for t in range(len(sequence)):
            result = hierarchy.step(sequence[t], dt=0.1, update_weights=True)
            epoch_errors.append(result["total_error"])

        errors_over_time.append(np.mean(epoch_errors))
        print(f"  Epoch {epoch + 1}: Mean error = {errors_over_time[-1]:.4f}")

    # Test prediction
    print("\nTesting prediction accuracy...")
    hierarchy.reset()

    predictions = []
    actuals = []

    for t in range(len(sequence) - 1):
        hierarchy.step(sequence[t], dt=0.1, update_weights=False)
        pred = hierarchy.predict_next()
        predictions.append(pred)
        actuals.append(sequence[t + 1])

    # Compute prediction accuracy
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    mse = np.mean((predictions - actuals) ** 2)
    correlation = np.corrcoef(predictions.flatten(), actuals.flatten())[0, 1]

    print(f"  Prediction MSE: {mse:.4f}")
    print(f"  Prediction Correlation: {correlation:.4f}")

    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Learning curve
    axes[0].plot(errors_over_time, "b-o", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Mean Error")
    axes[0].set_title("Learning Curve")
    axes[0].grid(True)

    # Actual vs Predicted (first channel)
    axes[1].plot(actuals[:50, 0], "b-", label="Actual", linewidth=1.5)
    axes[1].plot(predictions[:50, 0], "r--", label="Predicted", linewidth=1.5)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Prediction vs Actual (Channel 0)")
    axes[1].legend()
    axes[1].grid(True)

    # Layer activities
    layer_activities = [np.mean(np.abs(layer.hidden_state)) for layer in hierarchy.layers]
    axes[2].bar(range(len(layer_activities)), layer_activities)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Mean Activation")
    axes[2].set_title("Hierarchical Layer Activities")

    plt.tight_layout()
    plt.savefig("sensory_prediction_results.png", dpi=150)
    print("\nFigure saved: sensory_prediction_results.png")

    return hierarchy, sequence


def demo_omission_response(hierarchy: PredictiveHierarchy, sequence: np.ndarray):
    """Demonstrate vigorous prediction error for ABSENT expected input."""
    print("\n" + "=" * 60)
    print("Demo 2: Omission Response")
    print("=" * 60)

    input_dim = sequence.shape[1]

    # Create omission detector
    omission_detector = OmissionDetector(input_dim)
    sequence_detector = SequenceOmissionDetector(input_dim, sequence_length=5)

    # Reset and warm up hierarchy
    hierarchy.reset()
    for t in range(20):
        hierarchy.step(sequence[t], dt=0.1, update_weights=False)
        sequence_detector.observe(sequence[t], timestamp=t * 0.1)

    # Normal response to expected stimulus
    normal_result = hierarchy.step(sequence[20], dt=0.1, update_weights=False)
    normal_error = normal_result["total_error"]
    print(f"\nNormal stimulus error: {normal_error:.4f}")

    # Now OMIT the expected stimulus (replace with zeros)
    hierarchy.reset()
    for t in range(20):
        hierarchy.step(sequence[t], dt=0.1, update_weights=False)

    # Set up expectation based on prediction
    expected_value = hierarchy.predict_next()
    omission_detector.add_expectation(expected_value, time_window=21 * 0.1)

    # Omit the expected input (provide zeros instead)
    omission_result = hierarchy.step(np.zeros(input_dim), dt=0.1, update_weights=False)
    omission_error = omission_result["total_error"]

    # Check for omission
    omission_event = omission_detector.check_omissions(21 * 0.1)

    print(f"Omission error: {omission_error:.4f}")
    print(f"Error increase: {(omission_error / normal_error - 1) * 100:.1f}%")

    if omission_event:
        print("\nOmission detected!")
        print(f"  Type: {omission_event[0].omission_type.value}")
        print(f"  Magnitude: {omission_event[0].error_magnitude:.4f}")

    # Visualize omission response
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Error comparison
    errors = ["Normal\nStimulus", "Omitted\nStimulus"]
    error_values = [normal_error, omission_error]
    colors = ["green", "red"]
    axes[0].bar(errors, error_values, color=colors)
    axes[0].set_ylabel("Prediction Error")
    axes[0].set_title("Prediction Error: Normal vs Omission")

    # Per-layer errors
    normal_layer_errors = [np.sum(e**2) for e in normal_result["errors"]]
    omission_layer_errors = [np.sum(e**2) for e in omission_result["errors"]]

    x = np.arange(len(normal_layer_errors))
    width = 0.35
    axes[1].bar(x - width / 2, normal_layer_errors, width, label="Normal", color="green")
    axes[1].bar(x + width / 2, omission_layer_errors, width, label="Omission", color="red")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Per-Layer Errors")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("omission_response.png", dpi=150)
    print("\nFigure saved: omission_response.png")


def demo_lesion_simulation(sequence: np.ndarray):
    """Demonstrate degradation when connections are lesioned."""
    print("\n" + "=" * 60)
    print("Demo 3: Lesion Simulation")
    print("=" * 60)

    input_dim = sequence.shape[1]

    def train_and_evaluate(hierarchy_name: str, hierarchy: PredictiveHierarchy) -> float:
        """Train and return prediction error."""
        # Train
        for epoch in range(2):
            for t in range(len(sequence)):
                hierarchy.step(sequence[t], dt=0.1, update_weights=True)

        # Evaluate
        hierarchy.reset()
        errors = []
        for t in range(len(sequence)):
            result = hierarchy.step(sequence[t], dt=0.1, update_weights=False)
            errors.append(result["total_error"])

        return np.mean(errors)

    # Intact hierarchy
    intact_hierarchy = PredictiveHierarchy(layer_dims=[input_dim, 8, 6, 4], learning_rate=0.15)
    intact_error = train_and_evaluate("Intact", intact_hierarchy)
    print(f"\nIntact hierarchy error: {intact_error:.4f}")

    # Lesion: Remove top-down connections
    topdown_lesioned = PredictiveHierarchy(layer_dims=[input_dim, 8, 6, 4], learning_rate=0.15)
    # Zero out generative weights in higher layers
    for layer in topdown_lesioned.layers[1:]:
        layer.W_g = np.zeros_like(layer.W_g)
    topdown_error = train_and_evaluate("Top-down lesioned", topdown_lesioned)
    print(f"Top-down lesioned error: {topdown_error:.4f}")

    # Lesion: Remove intrinsic connections (dynamics)
    intrinsic_lesioned = PredictiveHierarchy(layer_dims=[input_dim, 8, 6, 4], learning_rate=0.15)
    for layer in intrinsic_lesioned.layers:
        layer.W_f = np.zeros_like(layer.W_f)
    intrinsic_error = train_and_evaluate("Intrinsic lesioned", intrinsic_lesioned)
    print(f"Intrinsic lesioned error: {intrinsic_error:.4f}")

    # Lesion: Single layer only
    single_layer = PredictiveHierarchy(layer_dims=[input_dim, 8], learning_rate=0.15)
    single_error = train_and_evaluate("Single layer", single_layer)
    print(f"Single layer error: {single_error:.4f}")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 6))

    conditions = ["Intact\nHierarchy", "Top-down\nLesioned", "Intrinsic\nLesioned", "Single\nLayer"]
    errors = [intact_error, topdown_error, intrinsic_error, single_error]
    colors = ["green", "orange", "red", "purple"]

    bars = ax.bar(conditions, errors, color=colors)

    # Add value labels
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax.annotate(
            f"{error:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    ax.set_ylabel("Prediction Error")
    ax.set_title("Effect of Lesions on Predictive Processing")

    # Add degradation percentages
    [(e / intact_error - 1) * 100 for e in errors[1:]]
    ax.axhline(y=intact_error, color="gray", linestyle="--", alpha=0.5, label="Intact baseline")

    plt.tight_layout()
    plt.savefig("lesion_simulation.png", dpi=150)
    print("\nFigure saved: lesion_simulation.png")


def demo_temporal_hierarchy():
    """Demonstrate multi-timescale processing."""
    print("\n" + "=" * 60)
    print("Demo 4: Temporal Hierarchy")
    print("=" * 60)

    # Create temporal hierarchy
    temporal = TemporalHierarchy(
        layer_dims=[5, 4, 3, 2], base_timescale=0.01, timescale_factor=10.0
    )

    print(f"\nTimescales: {temporal.get_timescales()}")

    # Generate input with multiple timescales
    n_steps = 500
    dt = 0.01

    fast_signal = np.sin(2 * np.pi * 10 * np.arange(n_steps) * dt)  # 10 Hz
    slow_signal = np.sin(2 * np.pi * 0.5 * np.arange(n_steps) * dt)  # 0.5 Hz

    # Track layer responses
    layer_responses = [[] for _ in range(4)]

    for t in range(n_steps):
        # Combine fast and slow signals
        input_val = np.zeros(5)
        input_val[0] = fast_signal[t]
        input_val[1] = slow_signal[t]
        input_val[2] = fast_signal[t] * slow_signal[t]

        states = temporal.step(input_val, dt)

        for i, state in enumerate(states):
            layer_responses[i].append(np.mean(np.abs(state)))

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    time = np.arange(n_steps) * dt

    # Input signals
    axes[0, 0].plot(time, fast_signal, "b-", alpha=0.7, label="Fast (10 Hz)")
    axes[0, 0].plot(time, slow_signal, "r-", alpha=0.7, label="Slow (0.5 Hz)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 0].set_title("Input Signals")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Layer responses
    for i, resp in enumerate(layer_responses):
        axes[0, 1].plot(time, resp, label=f"Layer {i} (τ={temporal.get_timescales()[i]:.3f}s)")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Mean Activity")
    axes[0, 1].set_title("Layer Responses")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Frequency analysis of layer responses
    from scipy import signal as sig

    for i, resp in enumerate(layer_responses):
        resp_arr = np.array(resp)
        freqs, psd = sig.welch(resp_arr, fs=1 / dt, nperseg=min(128, len(resp_arr) // 2))
        n_freqs = min(50, len(freqs))
        axes[1, 0].semilogy(freqs[:n_freqs], psd[:n_freqs], label=f"Layer {i}")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Power")
    axes[1, 0].set_title("Frequency Content per Layer")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Timescale vs layer
    timescales = temporal.get_timescales()
    axes[1, 1].semilogy(range(len(timescales)), timescales, "bo-", markersize=10)
    axes[1, 1].set_xlabel("Layer")
    axes[1, 1].set_ylabel("Timescale (s)")
    axes[1, 1].set_title("Timescale Hierarchy")
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("temporal_hierarchy.png", dpi=150)
    print("\nFigure saved: temporal_hierarchy.png")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("SENSORY PREDICTION DEMOS")
    print("Hierarchical Predictive Coding in Action")
    print("=" * 60)

    # Run demos
    hierarchy, sequence = demo_sequence_learning()
    demo_omission_response(hierarchy, sequence)
    demo_lesion_simulation(sequence)
    demo_temporal_hierarchy()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
