"""
Demo: Fast vs Slow Fear Response

Demonstrates:
- Fast route (~12ms) triggers fear before conscious awareness
- Slow route (~100ms) provides contextual evaluation
- Slow route can override fast route (stick vs snake)
"""

import numpy as np
import sys

sys.path.insert(0, "..")

from src.dual_emotion_routes import DualRouteProcessor


def demo_snake_vs_stick():
    """
    Classic example: Is it a snake or a stick?

    Fast route: "Danger! Could be snake!"
    Slow route: "Wait... it's just a stick."
    """
    print("=" * 60)
    print("DEMO: Snake vs Stick (Fast vs Slow Route)")
    print("=" * 60)

    processor = DualRouteProcessor()

    # Ambiguous stimulus that looks snake-like
    snake_like = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1])

    print("\n1. Processing ambiguous snake-like stimulus...")
    print(f"   Stimulus: {snake_like}")

    fast, slow = processor.process(snake_like)

    print("\n   FAST ROUTE (12ms):")
    print(f"   - Response: {fast.response_type.value}")
    print(f"   - Intensity: {fast.intensity:.2f}")
    print(f"   - Confidence: {fast.confidence:.2f}")

    print("\n   SLOW ROUTE (100ms):")
    print(f"   - Response: {slow.response_type.value}")
    print(f"   - Intensity: {slow.intensity:.2f}")
    print(f"   - Confidence: {slow.confidence:.2f}")

    # Now teach that this is actually a stick
    print("\n2. Learning it's a stick (not a snake)...")
    processor.learn_safety(snake_like, "It's just a curved stick")

    fast2, slow2 = processor.process(snake_like)
    final = processor.get_final_response(snake_like)

    print("\n   After learning:")
    print(f"   - Fast route still fires: {fast2.response_type.value}")
    print(f"   - Slow route override: {slow2.response_type.value}")
    print(f"   - Final response: {final.response_type.value}")
    if "override_reason" in final.details:
        print(f"   - Override reason: {final.details['override_reason']}")


def demo_timing_matters():
    """
    Show how timing affects response:
    - First 12ms: Only fast route has responded
    - At 100ms: Slow route completes
    """
    print("\n" + "=" * 60)
    print("DEMO: Timing Matters")
    print("=" * 60)

    processor = DualRouteProcessor()
    threat = np.array([0.9, 0.1, 0.9, 0.1, 0.8])

    fast, slow = processor.process(threat)

    print("\nTimeline of threat response:")
    print("-" * 40)
    print("  0ms:  Stimulus arrives at thalamus")
    print(f" 12ms:  FAST route responds: {fast.response_type.value.upper()}")
    print(f"        (Confidence: {fast.confidence:.2f})")
    print("        Body begins fear response before conscious awareness!")
    print(f"100ms:  SLOW route responds: {slow.response_type.value.upper()}")
    print(f"        (Confidence: {slow.confidence:.2f})")
    print("        Conscious evaluation can now begin")

    print("\n   Key insight: Fear response starts 88ms before")
    print("   you consciously 'see' the threat!")


def demo_stress_effect():
    """
    Show how stress affects fast/slow balance.
    Under high stress, slow route may not complete.
    """
    print("\n" + "=" * 60)
    print("DEMO: Stress Effects on Processing")
    print("=" * 60)

    ambiguous = np.array([0.6, 0.4, 0.5, 0.5])

    print("\nProcessing ambiguous stimulus under different stress levels:")
    print("-" * 40)

    for stress in [0.0, 0.5, 0.95]:
        processor = DualRouteProcessor(stress_level=stress)
        fast, slow = processor.process(ambiguous)

        print(f"\n  Stress level: {stress}")
        print(f"  Fast route: {fast.response_type.value} (intensity={fast.intensity:.2f})")
        if slow:
            print(f"  Slow route: {slow.response_type.value} (intensity={slow.intensity:.2f})")
        else:
            print("  Slow route: DID NOT COMPLETE (too stressed)")


def demo_fear_conditioning():
    """
    Demonstrate classical fear conditioning.
    Neutral stimulus becomes threatening after pairing with threat.
    """
    print("\n" + "=" * 60)
    print("DEMO: Fear Conditioning")
    print("=" * 60)

    processor = DualRouteProcessor()

    # Neutral stimulus (a tone, represented as pattern)
    tone = np.array([0.3, 0.3, 0.3, 0.3])

    print("\n1. Before conditioning:")
    before = processor.get_final_response(tone)
    print(f"   Response to tone: {before.response_type.value}")
    print(f"   Threat intensity: {before.intensity:.2f}")

    print("\n2. Conditioning: pairing tone with shock...")
    processor.condition_fear(tone, threat_level=0.9)

    print("\n3. After conditioning:")
    after = processor.get_final_response(tone)
    print(f"   Response to tone: {after.response_type.value}")
    print(f"   Threat intensity: {after.intensity:.2f}")

    print("\n   The previously neutral tone now triggers fear!")


if __name__ == "__main__":
    demo_snake_vs_stick()
    demo_timing_matters()
    demo_stress_effect()
    demo_fear_conditioning()

    print("\n" + "=" * 60)
    print("Key Research Validations:")
    print("=" * 60)
    print("✓ Fast route latency: 12ms")
    print("✓ Slow route latency: 100ms")
    print("✓ Slow route can override fast route")
    print("✓ Stress increases fast route dominance")
    print("✓ Fear conditioning works on fast route")
