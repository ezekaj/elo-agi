"""
Sandboxed Python execution environment for the ELO-AGI REPL API.

Executes Python code with neuro_agi modules available while restricting
dangerous operations like filesystem access, network calls, and os.system.
Uses a strict allowlist for builtins, memory limits, output size limits,
and execution timeouts.
"""

import sys
import io
import time
import signal
import resource
import traceback
from typing import Dict, Any
from contextlib import redirect_stdout, redirect_stderr


BLOCKED_IMPORTS = {
    "subprocess", "shutil", "socket", "http", "urllib",
    "requests", "ftplib", "smtplib", "telnetlib",
    "ctypes", "multiprocessing", "signal", "resource",
    "os", "pathlib", "importlib", "code", "codeop",
    "pickle", "shelve", "marshal", "tempfile",
}

SAFE_BUILTINS = {
    "print", "len", "range", "str", "int", "float",
    "list", "dict", "tuple", "set", "bool",
    "abs", "round", "min", "max", "sum",
    "sorted", "reversed", "enumerate", "zip", "map", "filter",
    "isinstance", "hasattr",
    "True", "False", "None",
    "frozenset", "bytes", "bytearray",
    "hex", "oct", "bin", "ord", "chr",
    "all", "any", "repr", "hash",
    "slice", "complex",
    "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration",
    "ZeroDivisionError", "OverflowError", "ImportError",
    "Exception", "ArithmeticError", "LookupError",
}

MAX_OUTPUT_BYTES = 10 * 1024  # 10KB
MAX_MEMORY_BYTES = 256 * 1024 * 1024  # 256MB
DEFAULT_TIMEOUT = 10


def _make_safe_globals() -> Dict[str, Any]:
    builtins_source = __builtins__.__dict__ if hasattr(__builtins__, '__dict__') else __builtins__

    safe_builtins = {}
    for name in SAFE_BUILTINS:
        if name in builtins_source:
            safe_builtins[name] = builtins_source[name]

    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def safe_import(name, *args, **kwargs):
        top_level = name.split(".")[0]
        if top_level in BLOCKED_IMPORTS:
            raise ImportError(f"Import of '{name}' is restricted in the sandbox")
        return original_import(name, *args, **kwargs)

    safe_builtins["__import__"] = safe_import
    safe_builtins["__builtins__"] = safe_builtins

    return safe_builtins


class SandboxTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise SandboxTimeoutError("Execution timed out (10 second limit)")


def _set_memory_limit():
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (MAX_MEMORY_BYTES, hard))
    except (ValueError, resource.error):
        pass


def execute_code(code: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Execute Python code in a sandboxed environment.

    Security measures:
    - Allowlist-only builtins (no getattr, setattr, globals, locals, etc.)
    - Blocked dangerous imports
    - Memory limit: 256MB
    - Output size limit: 10KB
    - Execution timeout: 10 seconds

    Returns:
        {
            "output": str,
            "error": str|None,
            "execution_time": float
        }
    """
    if len(code) > 5000:
        return {
            "output": "",
            "error": "Input too long (max 5000 characters)",
            "execution_time": 0.0,
        }

    start_time = time.time()
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
    except (ValueError, AttributeError):
        old_handler = None

    _set_memory_limit()

    safe_globals = _make_safe_globals()
    safe_globals["__name__"] = "__main__"

    try:
        _setup_neuro_modules(safe_globals)
    except Exception:
        pass

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            compiled = compile(code, "<repl>", "exec")
            exec(compiled, safe_globals)

        output = stdout_capture.getvalue()
        error = stderr_capture.getvalue() if stderr_capture.getvalue() else None

    except SandboxTimeoutError as e:
        output = stdout_capture.getvalue()
        error = str(e)

    except ImportError as e:
        output = stdout_capture.getvalue()
        error = f"ImportError: {e}"

    except MemoryError:
        output = stdout_capture.getvalue()
        error = "MemoryError: Execution exceeded 256MB memory limit"

    except Exception as e:
        output = stdout_capture.getvalue()
        tb = traceback.format_exc()
        last_line = tb.strip().split("\n")[-1]
        error = last_line

    finally:
        try:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
        except (ValueError, AttributeError):
            pass

    execution_time = time.time() - start_time

    if output and len(output) > MAX_OUTPUT_BYTES:
        output = output[:MAX_OUTPUT_BYTES] + "\n... (output truncated at 10KB)"

    return {
        "output": output,
        "error": error,
        "execution_time": round(execution_time, 4),
    }


def _setup_neuro_modules(globals_dict: Dict[str, Any]):
    """Pre-load neuro_agi modules into the sandbox globals."""
    import os
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    neuro_path = project_root / "neuro"
    if str(neuro_path) not in sys.path:
        sys.path.insert(0, str(neuro_path))

    for module_dir in project_root.glob("neuro-*"):
        src_dir = module_dir / "src"
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))

    try:
        import neuro as neuro_agi
        globals_dict["neuro_agi"] = neuro_agi
        globals_dict["neuro"] = neuro_agi
    except Exception:
        pass

    try:
        import numpy as np
        globals_dict["np"] = np
        globals_dict["numpy"] = np
    except ImportError:
        pass

    try:
        from neuro.benchmark import Benchmark
        globals_dict["Benchmark"] = Benchmark
    except ImportError:
        pass

    try:
        from neuro.emotions import EmotionSystem, BasicEmotion
        globals_dict["EmotionSystem"] = EmotionSystem
        globals_dict["BasicEmotion"] = BasicEmotion
    except ImportError:
        pass
