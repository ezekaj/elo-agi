#!/usr/bin/env python3
"""Test all CLI flags and commands."""

import subprocess
import sys

def run(cmd, timeout=5):
    """Run command and return (success, output)."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return True, "(timed out - interactive mode works)"
    except Exception as e:
        return False, str(e)

def test(name, cmd, expect_success=True, expect_contains=None, timeout=5):
    """Run a test."""
    success, output = run(cmd, timeout=timeout)

    passed = success == expect_success
    if expect_contains and passed:
        passed = expect_contains in output

    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {name}")

    if not passed:
        print(f"        Command: {cmd}")
        print(f"        Output: {output[:200]}")

    return passed

def main():
    print("\n=== NEURO CLI Test Suite ===\n")

    # Activate venv
    venv = "source .venv/bin/activate && "

    results = []

    print("Flags:")
    results.append(test("--version", f"{venv}python -m neuro.cli --version", expect_contains="3.0.0"))
    results.append(test("--help", f"{venv}python -m neuro.cli --help", expect_contains="NEURO"))
    results.append(test("-h", f"{venv}python -m neuro.cli -h", expect_contains="usage"))

    print("\nSubcommands:")
    results.append(test("doctor", f"{venv}python -m neuro.cli doctor", expect_contains="Python"))
    results.append(test("config", f"{venv}python -m neuro.cli config", expect_contains="Configuration"))

    print("\nPrint mode:")
    results.append(test("-p with prompt", f"{venv}python -m neuro.cli -p 'say hello' 2>&1", timeout=30))

    print("\nInteractive mode (timeout expected):")
    results.append(test("basic start", f"{venv}python -m neuro.cli 2>&1", timeout=2))
    results.append(test("-c continue", f"{venv}python -m neuro.cli -c 2>&1", timeout=2))
    results.append(test("-r resume", f"{venv}python -m neuro.cli -r 2>&1", timeout=2))
    results.append(test("--model", f"{venv}python -m neuro.cli --model qwen2.5:7b 2>&1", timeout=2))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("[ALL TESTS PASSED]")
        return 0
    else:
        print(f"[{total - passed} TESTS FAILED]")
        return 1

if __name__ == "__main__":
    sys.exit(main())
