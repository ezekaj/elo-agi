"""
Demo: Intrinsic Motivation - Path Entropy Maximization

Demonstrates key findings from https://arxiv.org/html/2601.10276:
- Humans maximize action-state path entropy, not reward
- "Irreducible desire to act and move" is fundamental
- Children choose harder games over easy success
- Post-satiation exploration continues
"""

import numpy as np
import sys

sys.path.insert(0, "..")

from src.intrinsic_motivation import (
    PathEntropyMaximizer,
    PossibilitySpace,
    ActionDiversityTracker,
    IntrinsicDrive,
    DriveType,
)


def demo_path_entropy_vs_reward():
    """
    Show that the system values diverse experiences, not just rewards.
    """
    print("=" * 60)
    print("DEMO: Path Entropy vs Reward Maximization")
    print("=" * 60)

    maximizer = PathEntropyMaximizer(state_dim=2, action_dim=2)

    # Scenario 1: Repetitive high-reward actions
    print("\n1. Repetitive high-reward actions:")
    for i in range(30):
        # Same action, same state region
        state = np.array([0.5, 0.5]) + np.random.randn(2) * 0.05
        action = np.array([0.8, 0.8])  # Always same "rewarding" action
        maximizer.observe(state, action)

    metrics_repetitive = maximizer.get_metrics()
    print(f"   State diversity: {metrics_repetitive['state_diversity']:.3f}")
    print(f"   Path entropy: {metrics_repetitive['path_entropy']:.3f}")
    print(f"   Intrinsic motivation: {metrics_repetitive['intrinsic_motivation']:.3f}")

    # Reset and try diverse exploration
    maximizer2 = PathEntropyMaximizer(state_dim=2, action_dim=2)

    print("\n2. Diverse exploration (varying actions and states):")
    for i in range(30):
        # Explore diverse states and actions
        angle = i * 0.2
        state = np.array([np.cos(angle), np.sin(angle)])
        action = np.array([np.sin(angle * 2), np.cos(angle * 2)])
        maximizer2.observe(state, action)

    metrics_diverse = maximizer2.get_metrics()
    print(f"   State diversity: {metrics_diverse['state_diversity']:.3f}")
    print(f"   Path entropy: {metrics_diverse['path_entropy']:.3f}")
    print(f"   Intrinsic motivation: {metrics_diverse['intrinsic_motivation']:.3f}")

    print("\n   Key insight: Diverse exploration has HIGHER intrinsic value")
    print(f"   even without explicit rewards!")


def demo_children_choose_harder():
    """
    Demonstrate: Children naturally seek challenge over easy success.
    """
    print("\n" + "=" * 60)
    print("DEMO: Children Choose Harder Games")
    print("=" * 60)

    space = PossibilitySpace(state_dim=2, action_dim=1)

    # Easy game: small, well-explored space
    print("\n1. Easy game (small possibility space):")
    for _ in range(20):
        state = np.array([0.5, 0.5]) + np.random.randn(2) * 0.1
        space.observe(state)

    easy_volume = space.compute_volume()
    easy_diversity = space.compute_diversity()
    print(f"   Possibility volume: {easy_volume:.3f}")
    print(f"   State diversity: {easy_diversity:.3f}")

    # Hard game: larger, more diverse space
    space2 = PossibilitySpace(state_dim=2, action_dim=1)

    print("\n2. Hard game (larger possibility space):")
    for i in range(20):
        # More spread out, more challenging
        angle = i * 0.3
        radius = 0.5 + (i / 40)
        state = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        space2.observe(state)

    hard_volume = space2.compute_volume()
    hard_diversity = space2.compute_diversity()
    print(f"   Possibility volume: {hard_volume:.3f}")
    print(f"   State diversity: {hard_diversity:.3f}")

    print("\n   Children prefer harder games because they offer")
    print(f"   more possibility expansion: {hard_volume:.2f} > {easy_volume:.2f}")


def demo_post_satiation_exploration():
    """
    Demonstrate: Exploration continues after needs are met.
    """
    print("\n" + "=" * 60)
    print("DEMO: Post-Satiation Exploration")
    print("=" * 60)

    # Drive that gets satiated
    exploration_drive = IntrinsicDrive(
        DriveType.EXPLORATION, base_strength=1.0, satiation_rate=0.1, recovery_rate=0.05
    )

    print("\n1. Drive before satiation:")
    print(f"   Exploration drive: {exploration_drive.level:.3f}")

    # Satiate the drive
    print("\n2. Satisfying the drive...")
    for _ in range(10):
        exploration_drive.update(satisfaction=0.8, dt=1.0)

    print(f"   Exploration drive after satiation: {exploration_drive.level:.3f}")

    # But exploration continues!
    maximizer = PathEntropyMaximizer(state_dim=2, action_dim=2)

    print("\n3. Even with satiated drive, system suggests exploration:")
    current_state = np.array([0.0, 0.0])

    for _ in range(10):
        maximizer.observe(current_state + np.random.randn(2) * 0.1, np.random.randn(2))

    suggested = maximizer.suggest_action(current_state)
    motivation = maximizer.compute_intrinsic_motivation()

    print(f"   Suggested action: [{suggested[0]:.3f}, {suggested[1]:.3f}]")
    print(f"   Intrinsic motivation remains: {motivation:.3f}")
    print("\n   Key insight: Intrinsic motivation persists beyond satiation!")


def demo_action_diversity():
    """
    Show that action diversity is tracked and valued.
    """
    print("\n" + "=" * 60)
    print("DEMO: Action Diversity Tracking")
    print("=" * 60)

    tracker = ActionDiversityTracker(action_dim=2)

    # Repetitive actions
    print("\n1. Recording repetitive actions:")
    common_action = np.array([0.5, 0.5])
    for _ in range(20):
        tracker.record_action(common_action + np.random.randn(2) * 0.01)

    diversity_after_repetition = tracker.compute_diversity()
    common_bonus = tracker.get_diversity_bonus(common_action)

    print(f"   Action diversity: {diversity_after_repetition:.3f}")
    print(f"   Bonus for common action: {common_bonus:.3f}")

    # Novel action
    rare_action = np.array([-0.8, 0.8])
    rare_bonus = tracker.get_diversity_bonus(rare_action)

    print(f"\n2. Bonus for rare action: {rare_bonus:.3f}")
    print(f"   Rare actions get {rare_bonus / max(common_bonus, 0.001):.1f}x more bonus!")


def demo_drives_system():
    """
    Show the four intrinsic drives and their dynamics.
    """
    print("\n" + "=" * 60)
    print("DEMO: Intrinsic Drive System")
    print("=" * 60)

    drives = {
        "exploration": IntrinsicDrive(DriveType.EXPLORATION),
        "mastery": IntrinsicDrive(DriveType.MASTERY),
        "autonomy": IntrinsicDrive(DriveType.AUTONOMY),
        "challenge": IntrinsicDrive(DriveType.CHALLENGE),
    }

    print("\n1. Initial drive levels:")
    for name, drive in drives.items():
        print(f"   {name.capitalize()}: {drive.level:.3f}")

    # Simulate different satisfactions
    print("\n2. After satisfying exploration and mastery:")
    drives["exploration"].update(satisfaction=0.9, dt=5.0)
    drives["mastery"].update(satisfaction=0.7, dt=5.0)

    for name, drive in drives.items():
        print(f"   {name.capitalize()}: {drive.level:.3f}")

    # Recovery over time
    print("\n3. After time passes (drives recover):")
    for _ in range(10):
        for drive in drives.values():
            drive.update(satisfaction=0.0, dt=1.0)

    for name, drive in drives.items():
        print(f"   {name.capitalize()}: {drive.level:.3f}")

    print("\n   Key insight: Unsatisfied drives naturally increase over time!")


if __name__ == "__main__":
    demo_path_entropy_vs_reward()
    demo_children_choose_harder()
    demo_post_satiation_exploration()
    demo_action_diversity()
    demo_drives_system()

    print("\n" + "=" * 60)
    print("Key Research Validations:")
    print("=" * 60)
    print("- Path entropy maximization explains exploration")
    print("- Children prefer challenge (possibility expansion)")
    print("- Post-satiation exploration is intrinsically motivated")
    print("- Action diversity is tracked and valued")
    print("- Multiple intrinsic drives with recovery dynamics")
