"""
Neuro AGI Command Line Interface.

Entry points:
    neuro --version     Show version
    neuro info          Show system info
    neuro check         Verify installation
"""

import argparse
import sys
from pathlib import Path


def get_version():
    """Get version from __init__.py."""
    return "0.9.0"


def cmd_version(args):
    """Print version."""
    print(f"neuro-agi {get_version()}")


def cmd_info(args):
    """Show system information."""
    print(f"Neuro AGI v{get_version()}")
    print("-" * 40)
    print("Neuroscience-inspired cognitive architecture")
    print("38 cognitive modules across 4 tiers:")
    print("  - Tier 1: Cognitive modules (00-19)")
    print("  - Tier 2: Infrastructure (system, llm, knowledge, ground, scale, transfer)")
    print("  - Tier 3: Support (env, bench, inference, perception, integrate)")
    print("  - New AGI: causal, abstract, robust, planning, credit, continual, meta-reasoning")
    print()
    print("Usage:")
    print("  from neuro import CognitiveCore")
    print("  core = CognitiveCore()")
    print("  core.initialize()")


def cmd_research(args):
    """Research assistant for document Q&A."""
    from neuro.apps import ResearchAssistant

    assistant = ResearchAssistant()

    # Ingest files if provided
    if args.ingest:
        for file_path in args.ingest:
            path = Path(file_path)
            if not path.exists():
                print(f"File not found: {file_path}")
                continue

            text = path.read_text()
            n = assistant.ingest_text(text, source=path.name)
            print(f"Ingested {path.name}: {n} facts extracted")

    # Answer query if provided
    if args.query:
        if not assistant.statistics()["total_facts"]:
            print("No documents ingested. Use --ingest first.")
            return 1

        result = assistant.query(args.query)
        print(f"\nQ: {args.query}")
        print(f"A: {result.answer}")
        print(f"   Confidence: {result.confidence:.1%}")
        return 0

    # Interactive mode
    if args.interactive:
        print("Research Assistant Interactive Mode")
        print("Commands: /ingest <file>, /sources, /quit")
        print("-" * 40)

        while True:
            try:
                user_input = input("\nQ: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input == "/quit":
                print("Goodbye!")
                break

            if user_input == "/sources":
                sources = assistant.get_sources()
                print(f"Sources: {', '.join(sources) if sources else 'none'}")
                continue

            if user_input.startswith("/ingest "):
                file_path = user_input[8:].strip()
                path = Path(file_path)
                if path.exists():
                    text = path.read_text()
                    n = assistant.ingest_text(text, source=path.name)
                    print(f"Ingested: {n} facts from {path.name}")
                else:
                    print(f"File not found: {file_path}")
                continue

            # Regular query
            result = assistant.query(user_input)
            print(f"A: {result.answer}")
            if result.confidence > 0:
                print(f"   Confidence: {result.confidence:.1%}")

        return 0

    # No action specified - show help
    print("Research Assistant")
    print("-" * 40)
    print("Usage:")
    print("  neuro research --ingest file.txt --query 'question'")
    print("  neuro research --interactive")
    stats = assistant.statistics()
    print(f"\nCurrent state: {stats['total_facts']} facts from {stats['total_sources']} sources")
    return 0


def cmd_check(args):
    """Verify installation and importability."""
    print(f"Neuro AGI Installation Check v{get_version()}")
    print("-" * 40)

    # Check core dependencies
    checks = []

    try:
        import numpy
        checks.append(("numpy", numpy.__version__, True))
    except ImportError as e:
        checks.append(("numpy", str(e), False))

    try:
        import scipy
        checks.append(("scipy", scipy.__version__, True))
    except ImportError as e:
        checks.append(("scipy", str(e), False))

    # Check neuro imports
    try:
        from neuro import __version__
        checks.append(("neuro", __version__, True))
    except ImportError as e:
        checks.append(("neuro", str(e), False))

    # Check key classes
    key_imports = [
        ("CognitiveCore", "cognitive_core", "CognitiveCore"),
        ("SharedSpace", "shared_space", "SharedSpace"),
        ("GlobalWorkspace", "global_workspace", "GlobalWorkspace"),
    ]

    for name, module, cls in key_imports:
        try:
            import importlib
            mod = importlib.import_module(module)
            getattr(mod, cls)
            checks.append((name, "available", True))
        except (ImportError, AttributeError) as e:
            checks.append((name, str(e)[:50], False))

    # Print results
    print("\nDependencies:")
    for name, version, ok in checks[:2]:
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}: {version}")

    print("\nCore imports:")
    for name, version, ok in checks[2:]:
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}: {version}")

    # Summary
    passed = sum(1 for _, _, ok in checks if ok)
    total = len(checks)
    print(f"\nResult: {passed}/{total} checks passed")

    if passed == total:
        print("Installation OK!")
        return 0
    else:
        print("Some checks failed. Run 'pip install -e .' from the neuro directory.")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="neuro",
        description="Neuro AGI - Neuroscience-inspired cognitive architecture",
    )
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show system information")
    info_parser.set_defaults(func=cmd_info)

    # Check command
    check_parser = subparsers.add_parser("check", help="Verify installation")
    check_parser.set_defaults(func=cmd_check)

    # Research command
    research_parser = subparsers.add_parser("research", help="Research assistant")
    research_parser.add_argument(
        "--ingest", "-i",
        nargs="+",
        help="Files to ingest",
    )
    research_parser.add_argument(
        "--query", "-q",
        help="Question to answer",
    )
    research_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode",
    )
    research_parser.set_defaults(func=cmd_research)

    args = parser.parse_args()

    if args.version:
        cmd_version(args)
        return 0

    if args.command is None:
        # Default: show info
        cmd_info(args)
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
