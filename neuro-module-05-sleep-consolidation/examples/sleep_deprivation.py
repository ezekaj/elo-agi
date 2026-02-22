"""
Demo: Sleep Deprivation Effects

Demonstrates the effects of missing specific sleep stages:
1. Full sleep vs SWS deprivation (impaired declarative memory)
2. Full sleep vs REM deprivation (impaired emotional processing)
3. Full sleep vs partial sleep (partial consolidation)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sleep_stages import SleepStage
from src.synaptic_homeostasis import SleepWakeCycle
from src.sleep_cycle import SleepCycleOrchestrator


def encode_test_memories(orchestrator: SleepCycleOrchestrator, n_memories: int = 10):
    """Encode standardized test memories."""
    experiences = []
    emotional_saliences = []

    for i in range(n_memories):
        pattern = np.zeros(20)
        pattern[i * 2 : (i * 2) + 2] = 1.0
        pattern += np.random.randn(20) * 0.1
        experiences.append(pattern)

        # Mix of emotional and neutral memories
        if i % 3 == 0:
            emotional_saliences.append(np.random.uniform(0.5, 0.9))
        elif i % 3 == 1:
            emotional_saliences.append(np.random.uniform(-0.9, -0.5))
        else:
            emotional_saliences.append(np.random.uniform(-0.2, 0.2))

    return orchestrator.wake_encoding(experiences, emotional_saliences)


def demo_sws_deprivation():
    """Demonstrate effects of SWS deprivation on declarative memory."""
    print("=" * 60)
    print("Demo 1: SWS Deprivation Effects")
    print("(Systems consolidation impaired)")
    print("=" * 60)

    # Full sleep condition
    print("\n--- Full Sleep Condition ---")
    full_sleep = SleepCycleOrchestrator(n_neurons=100, memory_dim=20)
    encode_test_memories(full_sleep)
    full_stats = full_sleep.sleep_consolidation(sleep_hours=8.0)

    # SWS deprivation condition
    print("\n--- SWS Deprivation Condition ---")
    sws_deprived = SleepCycleOrchestrator(n_neurons=100, memory_dim=20)
    encode_test_memories(sws_deprived)
    deprived_stats = sws_deprived.simulate_sleep_deprivation(skip_stages=[SleepStage.SWS])

    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON: Full Sleep vs SWS Deprivation")
    print("=" * 50)

    comparisons = [
        ("Total SWS (min)", full_stats.total_sws_minutes, deprived_stats.total_sws_minutes),
        ("Total REM (min)", full_stats.total_rem_minutes, deprived_stats.total_rem_minutes),
        (
            "Memories Replayed",
            full_stats.total_memories_replayed,
            deprived_stats.total_memories_replayed,
        ),
        (
            "Memories Consolidated",
            full_stats.total_memories_consolidated,
            deprived_stats.total_memories_consolidated,
        ),
        (
            "Cortical Transfer",
            full_stats.total_memories_transferred,
            deprived_stats.total_memories_transferred,
        ),
        (
            "Synaptic Reduction",
            full_stats.wake_synaptic_strength - full_stats.post_sleep_synaptic_strength,
            deprived_stats.wake_synaptic_strength - deprived_stats.post_sleep_synaptic_strength,
        ),
    ]

    for name, full_val, deprived_val in comparisons:
        diff = ((deprived_val / full_val) - 1) * 100 if full_val != 0 else 0
        print(f"{name:25s}: {full_val:8.1f} vs {deprived_val:8.1f} ({diff:+.1f}%)")

    print("\nKey finding: SWS deprivation impairs systems consolidation")
    print("(hippocampal → cortical transfer)")

    return full_stats, deprived_stats


def demo_rem_deprivation():
    """Demonstrate effects of REM deprivation on emotional processing."""
    print("\n" + "=" * 60)
    print("Demo 2: REM Deprivation Effects")
    print("(Emotional processing impaired)")
    print("=" * 60)

    # Full sleep condition
    print("\n--- Full Sleep Condition ---")
    full_sleep = SleepCycleOrchestrator(n_neurons=100, memory_dim=20)
    encode_test_memories(full_sleep)
    full_stats = full_sleep.sleep_consolidation(sleep_hours=8.0)

    # REM deprivation condition
    print("\n--- REM Deprivation Condition ---")
    rem_deprived = SleepCycleOrchestrator(n_neurons=100, memory_dim=20)
    encode_test_memories(rem_deprived)
    deprived_stats = rem_deprived.simulate_sleep_deprivation(skip_stages=[SleepStage.REM])

    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON: Full Sleep vs REM Deprivation")
    print("=" * 50)

    comparisons = [
        ("Total SWS (min)", full_stats.total_sws_minutes, deprived_stats.total_sws_minutes),
        ("Total REM (min)", full_stats.total_rem_minutes, deprived_stats.total_rem_minutes),
        ("Dreams Generated", full_stats.total_dreams, deprived_stats.total_dreams),
        (
            "Memories Replayed",
            full_stats.total_memories_replayed,
            deprived_stats.total_memories_replayed,
        ),
    ]

    for name, full_val, deprived_val in comparisons:
        if full_val != 0:
            diff = ((deprived_val / full_val) - 1) * 100
        else:
            diff = 0 if deprived_val == 0 else 100
        print(f"{name:25s}: {full_val:8.1f} vs {deprived_val:8.1f} ({diff:+.1f}%)")

    print("\nKey finding: REM deprivation eliminates dreaming and")
    print("impairs emotional memory processing")

    return full_stats, deprived_stats


def demo_partial_sleep():
    """Demonstrate effects of shortened sleep (partial consolidation)."""
    print("\n" + "=" * 60)
    print("Demo 3: Partial Sleep Effects")
    print("(Reduced total consolidation)")
    print("=" * 60)

    conditions = [
        ("Full sleep (8h)", 8.0),
        ("Moderate (6h)", 6.0),
        ("Short (4h)", 4.0),
        ("Very short (2h)", 2.0),
    ]

    results = []

    for name, hours in conditions:
        print(f"\n--- {name} ---")
        orchestrator = SleepCycleOrchestrator(n_neurons=100, memory_dim=20)
        encode_test_memories(orchestrator)
        stats = orchestrator.sleep_consolidation(sleep_hours=hours)
        results.append((name, hours, stats))

        print(f"   Cycles: {stats.total_cycles}")
        print(f"   SWS: {stats.total_sws_minutes:.1f} min")
        print(f"   REM: {stats.total_rem_minutes:.1f} min")
        print(f"   Consolidated: {stats.total_memories_consolidated}")

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    names = [r[0] for r in results]
    hours = [r[1] for r in results]
    sws_times = [r[2].total_sws_minutes for r in results]
    rem_times = [r[2].total_rem_minutes for r in results]
    consolidated = [r[2].total_memories_consolidated for r in results]
    replayed = [r[2].total_memories_replayed for r in results]

    # Sleep duration breakdown
    x = np.arange(len(names))
    width = 0.35
    axes[0, 0].bar(x - width / 2, sws_times, width, label="SWS", color="navy")
    axes[0, 0].bar(x + width / 2, rem_times, width, label="REM", color="coral")
    axes[0, 0].set_xlabel("Sleep Condition")
    axes[0, 0].set_ylabel("Duration (minutes)")
    axes[0, 0].set_title("Sleep Stage Duration by Condition")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([n.split()[0] for n in names], rotation=15)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Consolidation by sleep duration
    axes[0, 1].plot(hours, consolidated, "g-o", linewidth=2, markersize=10)
    axes[0, 1].set_xlabel("Sleep Duration (hours)")
    axes[0, 1].set_ylabel("Memories Consolidated")
    axes[0, 1].set_title("Consolidation vs Sleep Duration")
    axes[0, 1].grid(True, alpha=0.3)

    # Replays by sleep duration
    axes[1, 0].plot(hours, replayed, "b-s", linewidth=2, markersize=10)
    axes[1, 0].set_xlabel("Sleep Duration (hours)")
    axes[1, 0].set_ylabel("Memories Replayed")
    axes[1, 0].set_title("Memory Replay vs Sleep Duration")
    axes[1, 0].grid(True, alpha=0.3)

    # SWS to REM ratio
    ratios = [sws / (rem + 0.1) for sws, rem in zip(sws_times, rem_times)]
    axes[1, 1].bar(x, ratios, color="purple")
    axes[1, 1].set_xlabel("Sleep Condition")
    axes[1, 1].set_ylabel("SWS/REM Ratio")
    axes[1, 1].set_title("SWS/REM Balance by Condition")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([n.split()[0] for n in names], rotation=15)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("partial_sleep_effects.png", dpi=150)
    print("\nFigure saved: partial_sleep_effects.png")

    return results


def demo_synaptic_homeostasis_disruption():
    """Demonstrate synaptic homeostasis disruption from sleep loss."""
    print("\n" + "=" * 60)
    print("Demo 4: Synaptic Homeostasis Disruption")
    print("(Sleep deprivation prevents downscaling)")
    print("=" * 60)

    # Run multiple wake-sleep cycles with and without sleep
    cycles_to_run = 5

    # Normal sleep condition
    print("\n--- Normal Sleep-Wake Cycles ---")
    normal_cycle = SleepWakeCycle(n_neurons=100, connectivity=0.1)
    normal_strengths = [normal_cycle.homeostasis.measure_total_strength()]

    for i in range(cycles_to_run):
        normal_cycle.wake_learning(duration=16.0, learning_intensity=0.05)
        normal_strengths.append(normal_cycle.homeostasis.measure_total_strength())

        normal_cycle.sleep_consolidation(duration=8.0)
        normal_strengths.append(normal_cycle.homeostasis.measure_total_strength())

        print(
            f"   Cycle {i + 1}: Wake → {normal_strengths[-2]:.1f}, Sleep → {normal_strengths[-1]:.1f}"
        )

    # Sleep deprived condition
    print("\n--- Sleep Deprived (No Sleep) ---")
    deprived_cycle = SleepWakeCycle(n_neurons=100, connectivity=0.1)
    deprived_strengths = [deprived_cycle.homeostasis.measure_total_strength()]

    for i in range(cycles_to_run):
        deprived_cycle.wake_learning(duration=16.0, learning_intensity=0.05)
        deprived_strengths.append(deprived_cycle.homeostasis.measure_total_strength())

        # No sleep - just minimal downscaling
        deprived_cycle.homeostasis.downscale(factor=0.99)  # Minimal
        deprived_strengths.append(deprived_cycle.homeostasis.measure_total_strength())

        print(
            f"   Cycle {i + 1}: Wake → {deprived_strengths[-2]:.1f}, No Sleep → {deprived_strengths[-1]:.1f}"
        )

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Synaptic strength over time
    time_points = np.arange(len(normal_strengths))
    axes[0].plot(time_points, normal_strengths, "g-o", linewidth=2, label="Normal Sleep")
    axes[0].plot(time_points, deprived_strengths, "r-s", linewidth=2, label="Sleep Deprived")

    # Mark wake/sleep transitions
    for i in range(1, len(time_points), 2):
        axes[0].axvline(x=i, color="gray", linestyle="--", alpha=0.3)

    axes[0].set_xlabel("Time Point (alternating Wake/Sleep)")
    axes[0].set_ylabel("Total Synaptic Strength")
    axes[0].set_title("Synaptic Strength: Normal vs Sleep Deprived")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Final comparison
    conditions = ["Normal\nSleep", "Sleep\nDeprived"]
    final_strengths = [normal_strengths[-1], deprived_strengths[-1]]
    colors = ["green", "red"]

    axes[1].bar(conditions, final_strengths, color=colors)
    axes[1].set_ylabel("Final Synaptic Strength")
    axes[1].set_title("Final State After 5 Cycles")
    axes[1].grid(True, alpha=0.3)

    # Add percentage increase
    initial = normal_strengths[0]
    for i, (cond, strength) in enumerate(zip(conditions, final_strengths)):
        pct = (strength / initial - 1) * 100
        axes[1].annotate(
            f"{pct:+.1f}%", xy=(i, strength), xytext=(0, 5), textcoords="offset points", ha="center"
        )

    plt.tight_layout()
    plt.savefig("homeostasis_disruption.png", dpi=150)
    print("\nFigure saved: homeostasis_disruption.png")

    print("\nKey finding: Without sleep, synaptic strength accumulates")
    print("This leads to reduced signal-to-noise ratio")


def demo_comprehensive_comparison():
    """Comprehensive comparison of sleep conditions."""
    print("\n" + "=" * 60)
    print("Demo 5: Comprehensive Sleep Condition Comparison")
    print("=" * 60)

    conditions = {
        "Full Sleep": {"skip": []},
        "No SWS": {"skip": [SleepStage.SWS]},
        "No REM": {"skip": [SleepStage.REM]},
        "No SWS + No REM": {"skip": [SleepStage.SWS, SleepStage.REM]},
    }

    results = {}

    for name, config in conditions.items():
        print(f"\nRunning: {name}")
        orchestrator = SleepCycleOrchestrator(n_neurons=100, memory_dim=20)
        encode_test_memories(orchestrator)

        if config["skip"]:
            stats = orchestrator.simulate_sleep_deprivation(skip_stages=config["skip"])
        else:
            stats = orchestrator.sleep_consolidation(sleep_hours=8.0)

        results[name] = stats

    # Create summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(
        f"{'Condition':<20} {'SWS(min)':<10} {'REM(min)':<10} {'Replay':<10} {'Consol':<10} {'Dreams':<10}"
    )
    print("-" * 70)

    for name, stats in results.items():
        print(
            f"{name:<20} {stats.total_sws_minutes:<10.1f} {stats.total_rem_minutes:<10.1f} "
            f"{stats.total_memories_replayed:<10} {stats.total_memories_consolidated:<10} "
            f"{stats.total_dreams:<10}"
        )

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = list(results.keys())
    x = np.arange(len(names))

    # SWS and REM times
    sws = [results[n].total_sws_minutes for n in names]
    rem = [results[n].total_rem_minutes for n in names]

    width = 0.35
    axes[0, 0].bar(x - width / 2, sws, width, label="SWS", color="navy")
    axes[0, 0].bar(x + width / 2, rem, width, label="REM", color="coral")
    axes[0, 0].set_ylabel("Duration (minutes)")
    axes[0, 0].set_title("Sleep Stage Duration")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=15)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Memory replay
    replay = [results[n].total_memories_replayed for n in names]
    colors = ["green", "orange", "orange", "red"]
    axes[0, 1].bar(names, replay, color=colors)
    axes[0, 1].set_ylabel("Memories Replayed")
    axes[0, 1].set_title("Memory Replay Count")
    axes[0, 1].tick_params(axis="x", rotation=15)
    axes[0, 1].grid(True, alpha=0.3)

    # Consolidation
    consol = [results[n].total_memories_consolidated for n in names]
    axes[1, 0].bar(names, consol, color=colors)
    axes[1, 0].set_ylabel("Memories Consolidated")
    axes[1, 0].set_title("Consolidation Success")
    axes[1, 0].tick_params(axis="x", rotation=15)
    axes[1, 0].grid(True, alpha=0.3)

    # Dreams
    dreams = [results[n].total_dreams for n in names]
    axes[1, 1].bar(names, dreams, color=colors)
    axes[1, 1].set_ylabel("Dreams Generated")
    axes[1, 1].set_title("Dream Generation")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("comprehensive_comparison.png", dpi=150)
    print("\nFigure saved: comprehensive_comparison.png")


def main():
    """Run all sleep deprivation demos."""
    print("\n" + "=" * 60)
    print("SLEEP DEPRIVATION EFFECTS DEMOS")
    print("What Happens When We Don't Sleep Properly")
    print("=" * 60)

    demo_sws_deprivation()
    demo_rem_deprivation()
    demo_partial_sleep()
    demo_synaptic_homeostasis_disruption()
    demo_comprehensive_comparison()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("1. SWS deprivation → Impaired systems consolidation")
    print("   (memories don't transfer to cortex)")
    print("2. REM deprivation → No dreams, impaired emotional processing")
    print("3. Partial sleep → Proportional reduction in consolidation")
    print("4. No sleep → Synaptic strength accumulates dangerously")
    print("=" * 60)


if __name__ == "__main__":
    main()
