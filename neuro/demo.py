"""
ELO-AGI Demo: Shows how cognitive modules enhance LLM responses.

Usage:
    python -m neuro.demo              # Run all demos
    python -m neuro.demo --scenario 1 # Run specific scenario
"""

import argparse
import sys

from neuro.wrapper import SmartWrapper


def print_header(title):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_result(response):
    print("\n  Processing Steps:")
    for step in response.processing_steps:
        print(f"    > {step}")

    print(f"\n  Modules Used: {', '.join(response.modules_used)}")
    print(f"  Confidence:   {response.confidence:.2f}")
    print(f"  Latency:      {response.latency:.2f}s")
    print(f"  Provider:     {response.provider}")

    text = response.text.strip()
    if len(text) > 300:
        text = text[:300] + "..."
    print(f"\n  Response:\n    {text}")
    print("-" * 60)


def scenario_1():
    print_header("Scenario 1: Causal Reasoning")
    query = "If a company raises product prices by 20%, what happens to their revenue?"
    print(f'\n  Query: "{query}"\n')

    wrapper = SmartWrapper()
    response = wrapper.query(query)
    print_result(response)


def scenario_2():
    print_header("Scenario 2: Multi-step Planning")
    query = "Plan a study schedule for learning Python in 30 days"
    print(f'\n  Query: "{query}"\n')

    wrapper = SmartWrapper()
    response = wrapper.query(query)
    print_result(response)


def scenario_3():
    print_header("Scenario 3: Memory-dependent Conversation")

    wrapper = SmartWrapper()

    query_1 = "My cat Luna is a British Shorthair. She's 3 years old."
    print(f'\n  Query 1: "{query_1}"\n')
    response_1 = wrapper.query(query_1)
    print_result(response_1)

    query_2 = "What breed is my cat and how old is she?"
    print(f'\n  Query 2: "{query_2}"\n')
    response_2 = wrapper.query(query_2)
    print_result(response_2)


def run_all():
    print("\n" + "#" * 60)
    print("#" + " " * 12 + "ELO-AGI Cognitive Demo" + " " * 18 + "#")
    print("#" * 60)
    print("\nDemonstrating SmartWrapper cognitive capabilities.\n")

    scenario_1()
    scenario_2()
    scenario_3()

    print("\nAll scenarios complete.")


def main():
    parser = argparse.ArgumentParser(
        prog="neuro.demo",
        description="ELO-AGI Demo: Shows how cognitive modules enhance LLM responses.",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        choices=[1, 2, 3],
        help="Run a specific scenario (1-3)",
    )
    args = parser.parse_args()

    scenarios = {1: scenario_1, 2: scenario_2, 3: scenario_3}

    if args.scenario:
        scenarios[args.scenario]()
    else:
        run_all()


if __name__ == "__main__":
    main()
