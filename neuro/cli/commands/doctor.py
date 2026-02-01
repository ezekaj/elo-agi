"""Doctor command - check installation health."""

import os
import sys


def cmd_doctor(args):
    """Check NEURO installation health."""
    print(f"\n  NEURO Doctor")
    print(f"  {'─' * 50}")

    issues = []

    # Check Ollama
    print(f"  Checking Ollama...", end=" ")
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"✓ ({len(models)} models)")
        else:
            print(f"✗ (status {r.status_code})")
            issues.append("Ollama not responding correctly")
    except Exception as e:
        print(f"✗ ({e})")
        issues.append("Ollama not running - run: ollama serve")

    # Check aiohttp
    print(f"  Checking aiohttp...", end=" ")
    try:
        import aiohttp
        print(f"✓ (installed)")
    except ImportError:
        print(f"✗ (not installed)")
        issues.append("aiohttp not installed - run: pip install aiohttp")

    # Check config directory
    print(f"  Checking config...", end=" ")
    config_dir = os.path.expanduser("~/.neuro")
    if os.path.exists(config_dir):
        print(f"✓ ({config_dir})")
    else:
        print(f"! (not found)")
        issues.append(f"Config directory not found - run: mkdir -p {config_dir}")

    # Summary
    print(f"\n  {'─' * 50}")
    if issues:
        print(f"  Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  ✓ All checks passed!")

    print()
    return 0 if not issues else 1
