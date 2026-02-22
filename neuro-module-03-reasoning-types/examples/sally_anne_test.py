"""
Sally-Anne Test - Theory of Mind Demonstration

Classic false belief task demonstrating understanding of others' mental states.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interactive.theory_of_mind import TheoryOfMind, Observation, MentalStateModel


def run_sally_anne_test():
    """
    Sally-Anne Test:

    1. Sally puts a marble in her basket
    2. Sally leaves the room
    3. Anne moves the marble to her box
    4. Sally returns

    Question: Where will Sally look for the marble?

    Correct answer (understanding false belief): The basket
    Incorrect answer (no ToM): The box (where it actually is)
    """
    print("=" * 60)
    print("SALLY-ANNE FALSE BELIEF TEST")
    print("=" * 60)

    tom = TheoryOfMind(self_id="observer")

    print("\n--- Scene Setup ---")
    print("Sally and Anne are in a room with a basket and a box")

    tom.agent_models["sally"] = MentalStateModel(agent_id="sally")
    tom.agent_models["anne"] = MentalStateModel(agent_id="anne")

    print("\n--- Step 1: Sally puts marble in basket ---")
    tom.update_world_state("marble_location", "basket", inform_agents=["sally", "anne"])

    print("Sally knows: marble is in basket")
    print("Anne knows: marble is in basket")
    print("World state: marble is in basket")

    print("\n--- Step 2: Sally leaves the room ---")
    print("(Sally can no longer observe changes)")

    print("\n--- Step 3: Anne moves marble to box ---")
    tom.update_world_state("marble_location", "box", inform_agents=["anne"])

    print("Anne was informed of this change")
    print("Sally was NOT informed of this change")
    print("World state: marble is in box")
    print("Sally still believes: marble is in basket")

    print("\n--- Step 4: Sally returns ---")
    print("\n" + "=" * 60)
    print("QUESTION: Where will Sally look for the marble?")
    print("=" * 60)

    result = tom.false_belief_test(
        agent_id="sally", object_id="marble", true_location="box", believed_location="basket"
    )

    print(f"\nAnalysis:")
    print(f"  True location of marble: {result['true_location']}")
    print(f"  Sally believes marble is at: {result['agent_believes']}")
    print(f"  Has false belief: {result['has_false_belief']}")
    print(f"\n  PREDICTED: Sally will look in the {result['predicted_search_location']}")

    print("\n--- Explanation ---")
    if result["has_false_belief"]:
        print("Sally has a FALSE BELIEF because:")
        print("  - She last saw the marble in the basket")
        print("  - She was not present when Anne moved it")
        print("  - She has no way of knowing about the change")
        print("\nA system with Theory of Mind understands that:")
        print("  - Sally's beliefs differ from reality")
        print("  - Sally will act based on HER beliefs, not the true state")
        print("  - Therefore, Sally will look in the BASKET (not the box)")
    else:
        print("No false belief scenario detected")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    return result


def run_second_order_belief():
    """
    Second-order belief test:
    What does Sally think Anne thinks about where the marble is?
    """
    print("\n" + "=" * 60)
    print("SECOND-ORDER BELIEF TEST")
    print("=" * 60)

    tom = TheoryOfMind(self_id="observer")

    print("\nScenario: Both Sally and Anne know marble WAS in basket")
    print("Anne moved it to box while Sally was away")
    print("Sally returns but doesn't know about the move")

    print("\n--- First-order beliefs ---")
    print("Sally believes: marble is in basket (FALSE)")
    print("Anne believes: marble is in box (TRUE)")

    print("\n--- Second-order beliefs ---")
    print("What does Sally think Anne believes?")
    print("  Sally thinks Anne saw her put marble in basket")
    print("  Sally doesn't know Anne moved it")
    print("  So Sally thinks Anne believes marble is in basket")
    print("  (This is also FALSE - Anne knows it's in the box)")

    print("\n--- Third-order belief (recursive) ---")
    recursive = tom.recursive_belief("Anne", "the marble is in the basket", depth=2)
    print(f"  {recursive}")

    return recursive


if __name__ == "__main__":
    result = run_sally_anne_test()
    run_second_order_belief()
