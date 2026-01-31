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
