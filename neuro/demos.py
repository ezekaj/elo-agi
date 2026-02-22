"""
Neuro AGI Demo Runner.

Entry point for `neuro-demo` CLI command.

Usage:
    neuro-demo all        # Run all demos
    neuro-demo medical    # Medical diagnosis reasoning
    neuro-demo physics    # Physics concept learning
    neuro-demo language   # Language understanding
    neuro-demo arc        # ARC-style pattern reasoning
    neuro-demo continual  # Continual learning demo
"""

import argparse
import sys
from pathlib import Path


def get_version():
    """Get version from __init__.py."""
    return "0.9.0"


def run_medical():
    """Run medical diagnosis demo."""
    import sys
    from pathlib import Path

    # Ensure demos directory is in path first
    demos_dir = Path(__file__).parent.parent / "demos" / "agi_integration"
    sys.path.insert(0, str(demos_dir))
    try:
        from run_demo import demo_medical_diagnosis

        demo_medical_diagnosis()
    finally:
        sys.path.remove(str(demos_dir))


def run_physics():
    """Run physics learning demo."""
    import sys
    from pathlib import Path

    demos_dir = Path(__file__).parent.parent / "demos" / "agi_integration"
    sys.path.insert(0, str(demos_dir))
    try:
        from run_demo import demo_physics_learning

        demo_physics_learning()
    finally:
        sys.path.remove(str(demos_dir))


def run_language():
    """Run language understanding demo."""
    import sys
    from pathlib import Path

    demos_dir = Path(__file__).parent.parent / "demos" / "agi_integration"
    sys.path.insert(0, str(demos_dir))
    try:
        from run_demo import demo_language_concepts

        demo_language_concepts()
    finally:
        sys.path.remove(str(demos_dir))


def run_arc():
    """Run ARC-style reasoning demo."""
    import sys
    from pathlib import Path

    demos_dir = Path(__file__).parent.parent / "demos" / "agi_integration"
    sys.path.insert(0, str(demos_dir))
    try:
        from scenario_arc import demo_arc_reasoning

        demo_arc_reasoning()
    except ImportError as e:
        print(f"ARC demo import error: {e}")
        print("Make sure neuro-abstract, neuro-causal, neuro-meta-reasoning are available.")
        return 1
    finally:
        if str(demos_dir) in sys.path:
            sys.path.remove(str(demos_dir))
    return 0


def run_continual():
    """Run continual learning demo."""
    import sys
    from pathlib import Path

    demos_dir = Path(__file__).parent.parent / "demos" / "agi_integration"
    sys.path.insert(0, str(demos_dir))
    try:
        from scenario_continual import demo_continual_learning

        demo_continual_learning()
    except ImportError as e:
        print(f"Continual learning demo import error: {e}")
        print("Make sure neuro-continual is available.")
        return 1
    finally:
        if str(demos_dir) in sys.path:
            sys.path.remove(str(demos_dir))
    return 0


def run_all():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("#" + " " * 18 + "NEURO AGI DEMOS" + " " * 19 + "#")
    print("#" * 60)
    print(f"\nVersion: {get_version()}")
    print("Running all available demos...\n")

    run_medical()
    run_physics()
    run_language()
    run_arc()
    run_continual()

    print("\n" + "=" * 60)
    print("All demos complete!")
    print("=" * 60 + "\n")


def list_demos():
    """List available demos."""
    print("Available demos:")
    print("  medical   - Medical diagnosis reasoning")
    print("  physics   - Physics concept learning")
    print("  language  - Language understanding")
    print("  arc       - ARC-style pattern reasoning")
    print("  continual - Continual learning demo")
    print("  all       - Run all 5 demos")


def main():
    """Main entry point for neuro-demo CLI."""
    parser = argparse.ArgumentParser(
        prog="neuro-demo",
        description="Neuro AGI Demo Runner - Showcase cognitive capabilities",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available demos",
    )
    parser.add_argument(
        "demo",
        nargs="?",
        choices=["medical", "physics", "language", "arc", "continual", "all"],
        help="Demo to run",
    )

    args = parser.parse_args()

    if args.version:
        print(f"neuro-demo {get_version()}")
        return 0

    if args.list:
        list_demos()
        return 0

    if args.demo is None:
        parser.print_help()
        print("\nRun 'neuro-demo --list' to see available demos.")
        return 0

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    demos = {
        "medical": run_medical,
        "physics": run_physics,
        "language": run_language,
        "arc": run_arc,
        "continual": run_continual,
        "all": run_all,
    }

    return demos[args.demo]() or 0


if __name__ == "__main__":
    sys.exit(main())
