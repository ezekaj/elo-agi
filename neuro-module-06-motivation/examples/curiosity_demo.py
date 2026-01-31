"""
Demo: Curiosity and Exploration

Demonstrates key findings from https://www.sciencedirect.com/science/article/pii/S0166223623002400:
- Novelty activates dopamine even without reward
- Curiosity enhances memory encoding
- Information has intrinsic value
- Dopamine is the "neuromodulator of exploration"
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from src.curiosity_drive import (
    CuriosityModule, NoveltyDetector, InformationValue,
    ExplorationController, InformationPacket
)


def demo_novelty_detection():
    """
    Novel stimuli trigger dopamine even without explicit reward.
    """
    print("=" * 60)
    print("DEMO: Novelty Detection")
    print("=" * 60)

    detector = NoveltyDetector(input_dim=4)

    print("\n1. Everything is novel initially:")
    stimulus = np.array([0.5, 0.3, 0.7, 0.2])
    novelty = detector.compute_novelty(stimulus)
    print(f"   First stimulus novelty: {novelty:.3f}")

    print("\n2. Learning reduces novelty of familiar patterns:")
    # Observe similar stimuli repeatedly
    for _ in range(20):
        similar = stimulus + np.random.randn(4) * 0.05
        detector.observe(similar)

    novelty_after = detector.compute_novelty(stimulus)
    print(f"   Same stimulus after exposure: {novelty_after:.3f}")

    print("\n3. Distant stimuli remain novel:")
    distant = np.array([-0.8, 0.9, -0.5, 0.1])
    novelty_distant = detector.compute_novelty(distant)
    print(f"   Distant stimulus novelty: {novelty_distant:.3f}")

    print("\n   Key insight: System naturally tracks what's familiar vs novel!")


def demo_information_value():
    """
    Information has intrinsic value independent of utility.
    """
    print("\n" + "=" * 60)
    print("DEMO: Information Has Intrinsic Value")
    print("=" * 60)

    info_value = InformationValue()

    # Set up knowledge gaps
    info_value.set_uncertainty('quantum_physics', 0.9)  # High uncertainty
    info_value.set_uncertainty('cooking', 0.2)          # Low uncertainty

    print("\n1. Value of information depends on knowledge gaps:")

    # Information about high-uncertainty topic
    quantum_info = InformationPacket(
        content=np.zeros(3),
        novelty=0.6,
        relevance=0.7,
        uncertainty_reduction=0.3,
        source='quantum_physics'
    )

    # Information about low-uncertainty topic
    cooking_info = InformationPacket(
        content=np.zeros(3),
        novelty=0.6,
        relevance=0.7,
        uncertainty_reduction=0.3,
        source='cooking'
    )

    quantum_value = info_value.compute_information_value(
        quantum_info, topics=['quantum_physics']
    )
    cooking_value = info_value.compute_information_value(
        cooking_info, topics=['cooking']
    )

    print(f"   Quantum physics info value: {quantum_value:.3f}")
    print(f"   Cooking info value: {cooking_value:.3f}")
    print("   -> Information about knowledge gaps is MORE valuable!")

    print("\n2. Identifying knowledge gaps:")
    gaps = info_value.identify_knowledge_gaps(threshold=0.5)
    print(f"   Current knowledge gaps: {gaps}")


def demo_curiosity_memory_enhancement():
    """
    Curiosity enhances memory encoding.
    """
    print("\n" + "=" * 60)
    print("DEMO: Curiosity Enhances Memory")
    print("=" * 60)

    module = CuriosityModule(state_dim=4, memory_boost_factor=0.5)

    print("\n1. Processing boring (familiar) stimulus:")
    # Build familiarity
    for _ in range(20):
        module.process_stimulus(np.array([0.5, 0.5, 0.5, 0.5]))

    boring_result = module.process_stimulus(np.array([0.5, 0.5, 0.5, 0.5]))
    print(f"   Novelty: {boring_result['novelty']:.3f}")
    print(f"   Memory strength: {boring_result['memory_strength']:.3f}")

    print("\n2. Processing novel (curious) stimulus:")
    novel_result = module.process_stimulus(np.array([-0.8, 0.9, -0.3, 0.7]))
    print(f"   Novelty: {novel_result['novelty']:.3f}")
    print(f"   Memory strength: {novel_result['memory_strength']:.3f}")

    print("\n   Key insight: Curiosity BOOSTS memory encoding!")
    print("   (This is why we remember surprising events better)")


def demo_exploration_vs_exploitation():
    """
    Exploration controller balances novelty-seeking vs efficiency.
    """
    print("\n" + "=" * 60)
    print("DEMO: Exploration vs Exploitation")
    print("=" * 60)

    controller = ExplorationController(base_exploration=0.3)

    print("\n1. Exploration rate varies with curiosity:")
    for curiosity in [0.1, 0.3, 0.5, 0.7, 0.9]:
        rate = controller.compute_exploration_rate(
            curiosity_level=curiosity,
            dopamine_level=0.5,
            uncertainty=0.5
        )
        bar = "#" * int(rate * 20)
        print(f"   Curiosity={curiosity}: rate={rate:.3f} {bar}")

    print("\n2. Boredom increases exploration:")
    controller2 = ExplorationController(boredom_threshold=0.4)

    # Build up boredom with repetition
    same_action = np.array([0.5, 0.5])
    for _ in range(30):
        controller2.update_boredom(same_action)

    rate_bored = controller2.compute_exploration_rate(
        curiosity_level=0.3,
        dopamine_level=0.5,
        uncertainty=0.5
    )

    print(f"   Boredom level: {controller2.boredom_level:.3f}")
    print(f"   Exploration rate (bored): {rate_bored:.3f}")
    print("   -> Boredom naturally pushes toward exploration!")


def demo_full_curiosity_system():
    """
    Full curiosity module integration.
    """
    print("\n" + "=" * 60)
    print("DEMO: Full Curiosity System")
    print("=" * 60)

    module = CuriosityModule(state_dim=3)

    print("\n1. Registering curiosity targets (knowledge gaps):")
    module.register_curiosity_target('machine_learning', intensity=0.9)
    module.register_curiosity_target('history', intensity=0.4)

    gaps = module.get_knowledge_gaps()
    print(f"   Knowledge gaps: {gaps}")

    print("\n2. Processing a series of stimuli:")
    for i in range(10):
        stimulus = np.random.randn(3)
        result = module.process_stimulus(stimulus)

    print(f"   Curiosity level: {module.curiosity_level:.3f}")

    print("\n3. System state after exploration:")
    state = module.get_state()
    print(f"   Overall curiosity: {state.overall_level:.3f}")
    print(f"   Boredom level: {state.boredom_level:.3f}")
    print(f"   Recent discoveries: {state.recent_discoveries}")

    print("\n4. Exploration bonus for novel states:")
    # Familiar state
    for _ in range(20):
        module.process_stimulus(np.array([0.0, 0.0, 0.0]))

    familiar_bonus = module.get_exploration_bonus(np.array([0.0, 0.0, 0.0]))
    novel_bonus = module.get_exploration_bonus(np.array([5.0, 5.0, 5.0]))

    print(f"   Familiar state bonus: {familiar_bonus:.3f}")
    print(f"   Novel state bonus: {novel_bonus:.3f}")


def demo_dopamine_novelty_connection():
    """
    Show that novelty activates dopamine even without reward.
    """
    print("\n" + "=" * 60)
    print("DEMO: Novelty Activates Dopamine Without Reward")
    print("=" * 60)

    module = CuriosityModule(state_dim=3)

    # Build familiarity with one region
    print("\n1. Building familiarity with one region...")
    for _ in range(30):
        module.process_stimulus(np.array([0.5, 0.5, 0.5]) + np.random.randn(3) * 0.05)

    print("\n2. Processing familiar vs novel stimuli:")

    # Familiar stimulus
    familiar_result = module.process_stimulus(np.array([0.5, 0.5, 0.5]))
    print(f"\n   Familiar stimulus:")
    print(f"     Novelty: {familiar_result['novelty']:.3f}")
    print(f"     Info value: {familiar_result['info_value']:.3f}")
    print(f"     Curiosity level: {familiar_result['curiosity_level']:.3f}")

    # Novel stimulus (no reward, just different!)
    novel_result = module.process_stimulus(np.array([-1.0, 1.0, -0.5]))
    print(f"\n   Novel stimulus (no explicit reward):")
    print(f"     Novelty: {novel_result['novelty']:.3f}")
    print(f"     Info value: {novel_result['info_value']:.3f}")
    print(f"     Curiosity level: {novel_result['curiosity_level']:.3f}")

    print("\n   Key insight: Novel stimuli activate the reward system")
    print("   even without explicit rewards - information IS the reward!")


if __name__ == '__main__':
    demo_novelty_detection()
    demo_information_value()
    demo_curiosity_memory_enhancement()
    demo_exploration_vs_exploitation()
    demo_full_curiosity_system()
    demo_dopamine_novelty_connection()

    print("\n" + "=" * 60)
    print("Key Research Validations:")
    print("=" * 60)
    print("- Novelty activates dopamine even without reward")
    print("- Curiosity enhances memory encoding")
    print("- Information has intrinsic value (fills knowledge gaps)")
    print("- Boredom naturally drives exploration")
    print("- Dopamine is the 'neuromodulator of exploration'")
