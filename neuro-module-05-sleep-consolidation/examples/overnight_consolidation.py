"""
Demo: Overnight Memory Consolidation

Demonstrates memory improvement after a full night of sleep:
1. Encode experiences during "wake" period
2. Run complete sleep cycles with consolidation
3. Measure memory improvement (strength, cortical transfer)
4. Compare full sleep vs partial sleep
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sleep_stages import SleepStage, SleepStageController
from src.memory_replay import MemoryTrace, HippocampalReplay
from src.systems_consolidation import HippocampalCorticalDialogue
from src.synaptic_homeostasis import SynapticHomeostasis, SelectiveConsolidation
from src.dream_generator import DreamGenerator
from src.sleep_cycle import SleepCycleOrchestrator, SleepArchitecture


def demo_basic_consolidation():
    """Demonstrate basic memory consolidation overnight."""
    print("=" * 60)
    print("Demo 1: Basic Overnight Consolidation")
    print("=" * 60)

    # Create sleep system
    orchestrator = SleepCycleOrchestrator(n_neurons=100, memory_dim=20)

    # Encode experiences during the "day"
    print("\n1. Encoding experiences during wake period...")
    experiences = []
    for i in range(10):
        # Create distinct memory patterns
        pattern = np.zeros(20)
        pattern[i * 2 : (i * 2) + 2] = 1.0  # Each memory has unique signature
        pattern += np.random.randn(20) * 0.1  # Add noise
        experiences.append(pattern)

    emotional_saliences = [0.0, 0.3, -0.5, 0.8, 0.0, -0.2, 0.6, 0.0, 0.0, 0.9]

    memories = orchestrator.wake_encoding(experiences, emotional_saliences)

    print(f"   Encoded {len(memories)} memories")

    # Check pre-sleep state
    pre_sleep_stats = orchestrator.get_consolidation_statistics()
    print(f"\n2. Pre-sleep state:")
    print(f"   Hippocampal memories: {pre_sleep_stats['hippocampal_memories']}")
    print(f"   Cortical memories: {pre_sleep_stats['cortical_memories']}")

    # Run overnight sleep
    print("\n3. Running overnight consolidation (8 hours)...")
    night_stats = orchestrator.sleep_consolidation(sleep_hours=8.0)

    # Check post-sleep state
    post_sleep_stats = orchestrator.get_consolidation_statistics()
    print(f"\n4. Post-sleep state:")
    print(f"   Total cycles: {night_stats.total_cycles}")
    print(f"   SWS time: {night_stats.total_sws_minutes:.1f} minutes")
    print(f"   REM time: {night_stats.total_rem_minutes:.1f} minutes")
    print(f"   Memories replayed: {night_stats.total_memories_replayed}")
    print(f"   Memories consolidated: {night_stats.total_memories_consolidated}")
    print(f"   Memories transferred to cortex: {night_stats.total_memories_transferred}")
    print(f"   Dreams generated: {night_stats.total_dreams}")

    # Synaptic changes
    print(f"\n5. Synaptic homeostasis:")
    print(f"   Pre-sleep strength: {night_stats.wake_synaptic_strength:.2f}")
    print(f"   Post-sleep strength: {night_stats.post_sleep_synaptic_strength:.2f}")
    print(
        f"   Reduction: {(1 - night_stats.post_sleep_synaptic_strength / night_stats.wake_synaptic_strength) * 100:.1f}%"
    )

    return orchestrator, night_stats


def demo_memory_strength_improvement():
    """Demonstrate that memory strength improves after sleep."""
    print("\n" + "=" * 60)
    print("Demo 2: Memory Strength Improvement")
    print("=" * 60)

    # Create replay system
    replay = HippocampalReplay(compression_factor=20.0, replay_strength_boost=0.1)

    # Encode memories
    print("\n1. Encoding memories...")
    memories = []
    for i in range(5):
        trace = replay.encode_experience(
            pattern=np.random.randn(20), emotional_salience=np.random.uniform(-0.5, 0.5)
        )
        memories.append(trace)

    # Record initial strengths
    initial_strengths = [m.strength for m in memories]
    print(f"   Initial mean strength: {np.mean(initial_strengths):.3f}")

    # Simulate sleep replay
    print("\n2. Simulating sleep replay...")
    n_replay_sessions = 50

    for _ in range(n_replay_sessions):
        selected = replay.select_for_replay(n_select=3)
        for mem in selected:
            replay.replay_memory(mem, ripple_present=np.random.random() > 0.5)

    # Record final strengths
    final_strengths = [m.strength for m in memories]
    print(f"   Final mean strength: {np.mean(final_strengths):.3f}")
    print(
        f"   Improvement: {(np.mean(final_strengths) / np.mean(initial_strengths) - 1) * 100:.1f}%"
    )

    # Show individual memory changes
    print("\n3. Individual memory changes:")
    for i, (initial, final) in enumerate(zip(initial_strengths, final_strengths)):
        print(f"   Memory {i}: {initial:.3f} -> {final:.3f} ({(final / initial - 1) * 100:.1f}%)")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Strength comparison
    x = np.arange(len(memories))
    width = 0.35
    axes[0].bar(x - width / 2, initial_strengths, width, label="Before Sleep", color="steelblue")
    axes[0].bar(x + width / 2, final_strengths, width, label="After Sleep", color="coral")
    axes[0].set_xlabel("Memory")
    axes[0].set_ylabel("Strength")
    axes[0].set_title("Memory Strength: Before vs After Sleep")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Replay counts
    replay_counts = [m.replay_count for m in memories]
    axes[1].bar(x, replay_counts, color="green")
    axes[1].set_xlabel("Memory")
    axes[1].set_ylabel("Replay Count")
    axes[1].set_title("Number of Replays per Memory")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("memory_strength_improvement.png", dpi=150)
    print("\nFigure saved: memory_strength_improvement.png")


def demo_systems_consolidation():
    """Demonstrate hippocampal to cortical transfer."""
    print("\n" + "=" * 60)
    print("Demo 3: Systems Consolidation (Hippocampus → Cortex)")
    print("=" * 60)

    dialogue = HippocampalCorticalDialogue(transfer_threshold=0.6)

    # Encode memories in hippocampus
    print("\n1. Encoding in hippocampus...")
    memory_ids = []
    for i in range(10):
        trace = MemoryTrace(
            content=np.random.randn(20), emotional_salience=np.random.uniform(-0.5, 0.5)
        )
        mid = dialogue.hippocampus.encode(trace)
        memory_ids.append(mid)

    print(f"   Hippocampal memories: {dialogue.hippocampus.get_memory_count()}")
    print(f"   Cortical memories: {dialogue.cortex.get_memory_count()}")

    # Simulate consolidation over multiple sessions
    print("\n2. Running consolidation sessions...")
    consolidation_history = []

    for session in range(20):
        # Set optimal consolidation window
        dialogue.initiate_dialogue(slow_osc_phase="up", spindle=True, ripple=True)

        # Consolidate each memory
        for mid in memory_ids:
            dialogue.consolidate_memory(mid, replay_strength=1.0)

        stats = dialogue.get_statistics()
        consolidation_history.append(
            {
                "session": session,
                "hippocampal": stats["hippocampal_memories"],
                "cortical": stats["cortical_memories"],
                "transferred": stats["total_transferred"],
            }
        )

        if session % 5 == 4:
            print(f"   Session {session + 1}: {stats['total_transferred']} transferred to cortex")

    print(f"\n3. Final state:")
    final_stats = dialogue.get_statistics()
    print(f"   Total memories transferred: {final_stats['total_transferred']}")
    print(f"   Consolidation events: {final_stats['total_consolidation_events']}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sessions = [h["session"] for h in consolidation_history]
    transferred = [h["transferred"] for h in consolidation_history]

    axes[0].plot(sessions, transferred, "b-o", linewidth=2)
    axes[0].set_xlabel("Consolidation Session")
    axes[0].set_ylabel("Memories Transferred to Cortex")
    axes[0].set_title("Cumulative Memory Transfer")
    axes[0].grid(True, alpha=0.3)

    # Consolidation levels
    levels = []
    for mid in memory_ids:
        mem = dialogue.hippocampus.retrieve(mid)
        if mem:
            levels.append(mem.consolidation_level)
        else:
            levels.append(0)

    axes[1].bar(range(len(levels)), levels, color="purple")
    axes[1].axhline(
        y=dialogue.transfer_threshold,
        color="red",
        linestyle="--",
        label=f"Transfer threshold ({dialogue.transfer_threshold})",
    )
    axes[1].set_xlabel("Memory")
    axes[1].set_ylabel("Consolidation Level")
    axes[1].set_title("Final Consolidation Levels")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("systems_consolidation.png", dpi=150)
    print("\nFigure saved: systems_consolidation.png")


def demo_sleep_architecture():
    """Demonstrate sleep architecture across the night."""
    print("\n" + "=" * 60)
    print("Demo 4: Sleep Architecture Across the Night")
    print("=" * 60)

    architecture = SleepArchitecture(target_sleep_hours=8.0)
    cycles = architecture.plan_night()

    print("\nPlanned sleep cycles:")
    print("-" * 50)

    for cycle in cycles:
        print(f"\nCycle {cycle['cycle_number'] + 1}:")
        print(f"  NREM1: {cycle['nrem1_duration']:.1f} min")
        print(f"  NREM2: {cycle['nrem2_duration']:.1f} min")
        print(f"  SWS:   {cycle['sws_duration']:.1f} min")
        print(f"  REM:   {cycle['rem_duration']:.1f} min")

    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Stage durations per cycle
    cycle_nums = [c["cycle_number"] + 1 for c in cycles]
    sws_durations = [c["sws_duration"] for c in cycles]
    rem_durations = [c["rem_duration"] for c in cycles]
    nrem2_durations = [c["nrem2_duration"] for c in cycles]

    width = 0.25
    x = np.arange(len(cycle_nums))

    axes[0].bar(x - width, sws_durations, width, label="SWS", color="navy")
    axes[0].bar(x, nrem2_durations, width, label="NREM2", color="steelblue")
    axes[0].bar(x + width, rem_durations, width, label="REM", color="coral")

    axes[0].set_xlabel("Sleep Cycle")
    axes[0].set_ylabel("Duration (minutes)")
    axes[0].set_title("Sleep Stage Durations by Cycle")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cycle_nums)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SWS vs REM trends
    axes[1].plot(cycle_nums, sws_durations, "b-o", linewidth=2, markersize=8, label="SWS")
    axes[1].plot(cycle_nums, rem_durations, "r-s", linewidth=2, markersize=8, label="REM")

    axes[1].set_xlabel("Sleep Cycle")
    axes[1].set_ylabel("Duration (minutes)")
    axes[1].set_title("SWS Decreases, REM Increases Across Night")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sleep_architecture.png", dpi=150)
    print("\nFigure saved: sleep_architecture.png")


def demo_dream_content():
    """Demonstrate dream generation during REM."""
    print("\n" + "=" * 60)
    print("Demo 5: Dream Generation During REM")
    print("=" * 60)

    replay = HippocampalReplay()
    dream_gen = DreamGenerator(pfc_suppression=0.6)

    # Create memories with different emotional content
    print("\n1. Creating memories with varying emotional content...")
    memory_labels = ["happy_event", "sad_event", "neutral_event", "fearful_event", "exciting_event"]
    emotions = [0.8, -0.6, 0.0, -0.8, 0.7]

    memories = []
    for label, emotion in zip(memory_labels, emotions):
        trace = replay.encode_experience(
            pattern=np.random.randn(20), emotional_salience=emotion, context={"label": label}
        )
        memories.append(trace)

    # Generate dreams
    print("\n2. Generating dreams from memory replay...")
    n_dreams = 5
    dreams = []

    for i in range(n_dreams):
        dream = dream_gen.generate_dream(memories, duration=10.0)
        dreams.append(dream)

        print(f"\n   Dream {i + 1}:")
        print(f"     Source memories: {len(dream.source_memories)}")
        print(f"     Elements: {len(dream.elements)}")
        print(f"     Coherence: {dream.narrative_coherence:.2f}")
        print(f"     Bizarreness: {dream.bizarreness_index:.2f}")
        print(f"     Emotional tone: {dream.emotional_tone.value}")

    # Statistics
    dream_stats = dream_gen.get_dream_statistics()
    print(f"\n3. Dream statistics:")
    print(f"   Total dreams: {dream_stats['n_dreams']}")
    print(f"   Mean bizarreness: {dream_stats['mean_bizarreness']:.2f}")
    print(f"   Mean coherence: {dream_stats['mean_coherence']:.2f}")
    print(f"   Tone distribution: {dream_stats['tone_distribution']}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Dream properties
    bizarreness = [d.bizarreness_index for d in dreams]
    coherence = [d.narrative_coherence for d in dreams]

    x = np.arange(len(dreams))
    width = 0.35

    axes[0].bar(x - width / 2, bizarreness, width, label="Bizarreness", color="purple")
    axes[0].bar(x + width / 2, coherence, width, label="Coherence", color="green")
    axes[0].set_xlabel("Dream")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Dream Properties")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Emotional tones
    tones = dream_stats["tone_distribution"]
    if tones:
        axes[1].bar(tones.keys(), tones.values(), color="coral")
        axes[1].set_xlabel("Emotional Tone")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Dream Emotional Tone Distribution")
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dream_content.png", dpi=150)
    print("\nFigure saved: dream_content.png")


def main():
    """Run all overnight consolidation demos."""
    print("\n" + "=" * 60)
    print("OVERNIGHT CONSOLIDATION DEMOS")
    print("Sleep and Memory Consolidation in Action")
    print("=" * 60)

    demo_basic_consolidation()
    demo_memory_strength_improvement()
    demo_systems_consolidation()
    demo_sleep_architecture()
    demo_dream_content()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
