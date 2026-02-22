#!/usr/bin/env python3
"""
Demo: Engram Manipulation - Erase, Trigger, and Implant Memories

Based on experimental breakthroughs that can:
- Selectively erase specific memories
- Artificially trigger recall of specific memories
- Implant synthetic memories that never occurred
"""

import sys

sys.path.insert(0, "..")

import numpy as np
from src.engram import Engram, EngramState


def demo_engram_manipulation():
    """Demonstrate engram manipulation capabilities"""
    print("=" * 60)
    print("ENGRAM MANIPULATION DEMO")
    print("=" * 60)

    # 1. CREATE AND CONSOLIDATE A MEMORY
    print("\n1. CREATE A MEMORY")
    print("-" * 40)

    np.random.seed(42)

    # Create engram with reasonable parameters
    engram = Engram(n_neurons=100, connectivity=0.3, learning_rate=0.2)

    # Create a specific pattern (representing a memory)
    # Pattern: "Memory of seeing a red apple"
    memory_pattern = np.zeros(100)
    # Activate specific "neurons" for this memory
    memory_pattern[0:20] = 1.0  # "Visual" neurons
    memory_pattern[20:30] = 0.8  # "Color" neurons (red)
    memory_pattern[40:50] = 0.9  # "Object" neurons (apple)

    # Encode the memory
    engram.encode(memory_pattern)
    print(f"Encoded memory pattern")
    print(f"State: {engram.state.value}")
    print(f"Strength: {engram.strength:.2f}")

    # Consolidate (simulate sleep)
    for _ in range(5):
        engram.consolidate()
    print(f"After consolidation - Strength: {engram.strength:.2f}")
    print(f"State: {engram.state.value}")

    # 2. DEMONSTRATE RECALL
    print("\n2. NORMAL RECALL")
    print("-" * 40)

    # Create partial cue (only visual and color info)
    partial_cue = np.zeros(100)
    partial_cue[0:20] = 1.0  # Visual neurons
    partial_cue[20:30] = 0.8  # Color neurons

    print("Providing partial cue (visual + color only)...")
    recalled = engram.reactivate(partial_cue, iterations=15)

    # Check if object neurons activated (pattern completion)
    similarity = engram.similarity(recalled)
    object_activation = np.mean(recalled[40:50])
    print(f"Similarity to original: {similarity:.2f}")
    print(f"Object neurons activation: {object_activation:.2f}")
    print(
        "Pattern completion successful!" if object_activation > 0.3 else "Pattern completion weak"
    )

    # 3. ERASE MEMORY (Selective Forgetting)
    print("\n3. ERASE MEMORY (Selective Forgetting)")
    print("-" * 40)

    # Create a new engram for erasure demo
    erase_engram = Engram(n_neurons=100, connectivity=0.3)
    erase_engram.encode(memory_pattern)
    for _ in range(5):
        erase_engram.consolidate()

    print(f"Before erasure - Strength: {erase_engram.strength:.2f}")
    stats_before = erase_engram.get_connection_stats()
    print(f"Connections before: {stats_before['count']}")

    # ERASE by aggressive pruning (simulates targeted synaptic weakening)
    pruned = erase_engram.prune(threshold=0.3)  # Remove weak connections
    pruned += erase_engram.prune(threshold=0.5)  # Remove more
    pruned += erase_engram.prune(threshold=0.7)  # Even more aggressive

    stats_after = erase_engram.get_connection_stats()
    print(f"Connections after erasure: {stats_after['count']}")
    print(f"Pruned {pruned} connections")

    # Try to recall after erasure
    print("\nAttempting recall after erasure...")
    erased_recall = erase_engram.reactivate(partial_cue, iterations=15)
    erased_similarity = erase_engram.similarity(erased_recall)
    print(f"Similarity after erasure: {erased_similarity:.2f}")
    print("Memory successfully erased!" if erased_similarity < 0.3 else "Some trace remains")

    # 4. ARTIFICIALLY TRIGGER RECALL
    print("\n4. ARTIFICIALLY TRIGGER RECALL")
    print("-" * 40)

    # Create fresh engram
    trigger_engram = Engram(n_neurons=100, connectivity=0.3, learning_rate=0.3)
    trigger_engram.encode(memory_pattern)
    for _ in range(5):
        trigger_engram.consolidate()

    # Create minimal artificial trigger (very sparse cue)
    artificial_trigger = np.zeros(100)
    artificial_trigger[0:5] = 0.5  # Just 5 neurons with weak activation

    print("Triggering with minimal artificial cue (5 neurons, 50% strength)...")
    triggered_recall = trigger_engram.reactivate(artificial_trigger, iterations=20)

    # Check how much of original pattern was recovered
    trigger_similarity = trigger_engram.similarity(triggered_recall)
    print(f"Recovery similarity: {trigger_similarity:.2f}")

    # Check specific regions
    visual_recovery = np.mean(triggered_recall[0:20])
    color_recovery = np.mean(triggered_recall[20:30])
    object_recovery = np.mean(triggered_recall[40:50])
    print(f"Visual region: {visual_recovery:.2f}")
    print(f"Color region: {color_recovery:.2f}")
    print(f"Object region: {object_recovery:.2f}")

    # 5. IMPLANT SYNTHETIC MEMORY
    print("\n5. IMPLANT SYNTHETIC MEMORY")
    print("-" * 40)

    # Create engram for implantation
    implant_engram = Engram(n_neurons=100, connectivity=0.3, learning_rate=0.4)

    # Create synthetic pattern (a memory that never happened)
    # "Memory of seeing a blue elephant"
    synthetic_pattern = np.zeros(100)
    synthetic_pattern[0:20] = 1.0  # Visual neurons
    synthetic_pattern[30:40] = 0.9  # Different color neurons (blue instead of red)
    synthetic_pattern[60:70] = 0.9  # Different object neurons (elephant instead of apple)

    print("Implanting synthetic memory pattern...")

    # Encode the synthetic memory
    implant_engram.encode(synthetic_pattern)

    # Strong consolidation (artificially strengthen the false memory)
    for _ in range(10):
        implant_engram.consolidate()

    print(f"Implanted memory strength: {implant_engram.strength:.2f}")

    # Test retrieval of implanted memory
    implant_cue = np.zeros(100)
    implant_cue[0:20] = 1.0  # Just visual cue

    print("\nAttempting to recall implanted memory with visual cue...")
    implant_recall = implant_engram.reactivate(implant_cue, iterations=15)

    # Check if synthetic memory is recalled
    blue_activation = np.mean(implant_recall[30:40])
    elephant_activation = np.mean(implant_recall[60:70])

    print(f"Blue region activation: {blue_activation:.2f}")
    print(f"Elephant region activation: {elephant_activation:.2f}")

    if blue_activation > 0.3 and elephant_activation > 0.3:
        print("SUCCESS: Synthetic memory recalled! (False memory implanted)")
    else:
        print("Implanted memory recall weak")

    # 6. RECONSOLIDATION (Memory Modification)
    print("\n6. RECONSOLIDATION (Modify Existing Memory)")
    print("-" * 40)

    # Create and consolidate original memory
    recon_engram = Engram(n_neurons=100, connectivity=0.3, learning_rate=0.3)
    original = np.zeros(100)
    original[0:30] = 1.0  # Original content

    recon_engram.encode(original)
    for _ in range(5):
        recon_engram.consolidate()

    print(f"Original memory consolidated: {recon_engram.is_consolidated()}")

    # Reactivate (destabilize)
    print("Reactivating memory (making labile)...")
    recon_engram.destabilize()
    print(f"Memory is labile: {recon_engram.is_labile()}")

    # Modify during labile window
    modification = np.zeros(100)
    modification[0:30] = 1.0  # Keep original
    modification[50:70] = 0.8  # Add new content

    print("Modifying memory during labile window...")
    recon_engram.restabilize(modification)

    print(f"Memory reconsolidated: {recon_engram.is_consolidated()}")

    # Test recall shows modification
    recon_recall = recon_engram.reactivate(original, iterations=15)
    new_content = np.mean(recon_recall[50:70])
    print(f"New content region activation: {new_content:.2f}")

    if new_content > 0.2:
        print("SUCCESS: Memory successfully modified!")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo_engram_manipulation()
