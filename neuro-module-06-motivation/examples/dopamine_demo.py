"""
Demo: Dopamine System - Not Pleasure, But Prediction

Demonstrates key findings from https://pmc.ncbi.nlm.nih.gov/articles/PMC11602011/:
- Dopamine signals prediction error, not pleasure
- Incentive salience ("wanting" vs "liking")
- Benefit/cost ratio computation
- Motivation can enhance AND impair decisions
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from src.dopamine_system import (
    DopamineSystem, PredictionErrorComputer, IncentiveSalience,
    BenefitCostEvaluator
)


def demo_prediction_error():
    """
    Dopamine signals surprise (prediction error), not pleasure.
    """
    print("=" * 60)
    print("DEMO: Dopamine = Prediction Error, Not Pleasure")
    print("=" * 60)

    pe_computer = PredictionErrorComputer()

    # Define states
    state = np.array([0.5, 0.5])
    next_state = np.array([0.6, 0.6])

    # Scenario 1: Expected reward - learn first
    print("\n1. Learning phase - building expectations:")
    for _ in range(10):
        pe = pe_computer.compute_prediction_error(
            actual_reward=1.0,
            current_state=state,
            next_state=next_state
        )
        pe_computer.update_values(state, pe)
        print(f"   TD error: {pe:.3f}")

    print(f"\n   Value of state after learning: {pe_computer.get_value(state):.3f}")

    # Scenario 2: Now reward is expected
    print("\n2. After learning (reward expected):")
    pe = pe_computer.compute_prediction_error(
        actual_reward=1.0,
        current_state=state,
        next_state=next_state
    )
    print(f"   Same reward, PE now: {pe:.3f}")
    print("   -> No dopamine burst (no surprise)")

    # Scenario 3: Unexpected extra reward
    print("\n3. Unexpected EXTRA reward:")
    pe = pe_computer.compute_prediction_error(
        actual_reward=2.0,  # Double!
        current_state=state,
        next_state=next_state
    )
    print(f"   Double reward, PE: {pe:.3f}")
    print("   -> LARGE positive PE (pleasant surprise)")

    # Scenario 4: Expected reward omitted
    print("\n4. Expected reward NOT received:")
    pe = pe_computer.compute_prediction_error(
        actual_reward=0.0,
        current_state=state,
        next_state=next_state
    )
    print(f"   No reward, PE: {pe:.3f}")
    print("   -> Negative PE (disappointment)")


def demo_wanting_vs_liking():
    """
    Incentive salience: "wanting" is separate from "liking".
    """
    print("\n" + "=" * 60)
    print("DEMO: Wanting vs Liking (Incentive Salience)")
    print("=" * 60)

    salience = IncentiveSalience(cue_dim=3)

    cue = np.array([0.8, 0.5, 0.3])

    # Train salience on a cue via update_salience
    print("\n1. Learning cue-reward association:")
    for i in range(10):
        # Simulate prediction error (reward - expected)
        pe = 1.0 - i * 0.08  # PE decreases as learning occurs
        salience.update_salience(cue, reward=1.0, prediction_error=pe)

    wanting = salience.compute_salience(cue)
    print(f"   Trained cue salience (wanting): {wanting:.3f}")

    # Novel cue has no salience
    novel_cue = np.array([0.1, 0.9, 0.1])
    novel_wanting = salience.compute_salience(novel_cue)
    print(f"   Novel cue salience: {novel_wanting:.3f}")

    # Internal state modulation
    print("\n2. Internal state modulates wanting:")
    # Hungry state increases wanting for food cues
    wanting_hungry = salience.compute_salience(
        cue,
        internal_state={'energy': 0.1}  # Very hungry
    )
    wanting_full = salience.compute_salience(
        cue,
        internal_state={'energy': 0.9}  # Full
    )
    print(f"   Wanting when hungry: {wanting_hungry:.3f}")
    print(f"   Wanting when full: {wanting_full:.3f}")
    print("   -> Same cue, different wanting based on state!")


def demo_benefit_cost_ratio():
    """
    Dopamine modulates sensitivity to benefits vs costs.
    """
    print("\n" + "=" * 60)
    print("DEMO: Benefit/Cost Ratio Computation")
    print("=" * 60)

    evaluator = BenefitCostEvaluator()

    # Same task, different dopamine levels
    print("\n1. Same decision, different dopamine:")
    print("   Task: benefits=[0.8], costs=[0.5]")

    benefits = [0.8]
    costs = [0.5]

    # Low dopamine
    evaluator.set_dopamine_level(0.2)
    value_low = evaluator.evaluate(benefits, costs)
    print(f"\n   Low dopamine (0.2):")
    print(f"     Net value: {value_low:.3f}")
    print(f"     Decision: {'DO IT' if value_low > 0.3 else 'SKIP IT'}")

    # High dopamine
    evaluator.set_dopamine_level(0.8)
    value_high = evaluator.evaluate(benefits, costs)
    print(f"\n   High dopamine (0.8):")
    print(f"     Net value: {value_high:.3f}")
    print(f"     Decision: {'DO IT' if value_high > 0.3 else 'SKIP IT'}")

    print("\n   Key insight: Dopamine makes benefits loom larger!")


def demo_effort_discounting():
    """
    Effort costs and how they change with dopamine.
    """
    print("\n" + "=" * 60)
    print("DEMO: Effort Discounting")
    print("=" * 60)

    evaluator = BenefitCostEvaluator()

    # High-effort task
    print("\n1. High-effort task evaluation:")
    print("   Task: benefit=1.0, effort_cost=0.8")

    benefits = [1.0]
    high_effort_cost = [0.8]

    for da_level in [0.2, 0.5, 0.8]:
        evaluator.set_dopamine_level(da_level)
        value = evaluator.evaluate(benefits, high_effort_cost)
        print(f"   DA={da_level}: net_value={value:.3f}")

    print("\n   Low dopamine makes effort feel MORE costly")
    print("   High dopamine makes effort feel LESS costly")
    print("   (This explains effort avoidance in depression)")


def demo_full_dopamine_system():
    """
    Full dopamine system integration.
    """
    print("\n" + "=" * 60)
    print("DEMO: Full Dopamine System")
    print("=" * 60)

    system = DopamineSystem(state_dim=2, cue_dim=3)

    print("\n1. Processing a sequence of rewarding transitions:")

    state = np.array([0.5, 0.5])
    cue = np.array([0.8, 0.3, 0.5])

    # Initially surprising rewards
    print("\n   First few rewards (surprising):")
    for i in range(3):
        next_state = state + np.random.randn(2) * 0.1
        signal = system.process_transition(
            state=state,
            action=np.array([0.5, 0.5]),
            reward=1.0,
            next_state=next_state,
            cue=cue
        )
        print(f"     Trial {i+1}: PE={signal.prediction_error:.3f}, salience={signal.incentive_salience:.3f}")
        state = next_state

    print("\n2. After learning (rewards become expected):")
    for i in range(5):
        next_state = state + np.random.randn(2) * 0.1
        signal = system.process_transition(
            state=state,
            action=np.array([0.5, 0.5]),
            reward=1.0,
            next_state=next_state,
            cue=cue
        )
        state = next_state

    signal = system.process_transition(
        state=state,
        action=np.array([0.5, 0.5]),
        reward=1.0,
        next_state=state,
        cue=cue
    )
    print(f"   After learning: PE={signal.prediction_error:.3f}, salience={signal.incentive_salience:.3f}")
    print("   -> Prediction error diminishes as cue salience increases!")

    print("\n3. System state:")
    summary = system.get_state_summary()
    print(f"   Tonic dopamine: {summary['tonic_level']:.3f}")
    print(f"   Motivation level: {system.get_motivation_level():.3f}")


if __name__ == '__main__':
    demo_prediction_error()
    demo_wanting_vs_liking()
    demo_benefit_cost_ratio()
    demo_effort_discounting()
    demo_full_dopamine_system()

    print("\n" + "=" * 60)
    print("Key Research Validations:")
    print("=" * 60)
    print("- Dopamine = prediction error (surprise), not pleasure")
    print("- Wanting (dopamine) is separate from liking (opioid)")
    print("- Dopamine modulates benefit/cost sensitivity")
    print("- Low dopamine makes effort feel more costly")
    print("- Learning shifts dopamine signal to predictive cues")
