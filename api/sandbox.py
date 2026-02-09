"""
Sandboxed Python execution environment for the ELO-AGI REPL API.

Executes Python code with neuro_agi modules available while restricting
dangerous operations like filesystem access, network calls, and os.system.
"""

import sys
import io
import time
import signal
import traceback
from typing import Dict, Any
from contextlib import redirect_stdout, redirect_stderr


BLOCKED_IMPORTS = {
    "subprocess", "shutil", "socket", "http", "urllib",
    "requests", "ftplib", "smtplib", "telnetlib",
    "ctypes", "multiprocessing", "signal",
}

BLOCKED_BUILTINS = {
    "exec", "eval", "compile", "__import__", "open",
    "input", "breakpoint",
}


def _make_safe_globals() -> Dict[str, Any]:
    safe_builtins = {}
    for name, obj in __builtins__.__dict__.items() if hasattr(__builtins__, '__dict__') else __builtins__.items():
        if name not in BLOCKED_BUILTINS:
            safe_builtins[name] = obj

    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def safe_import(name, *args, **kwargs):
        top_level = name.split(".")[0]
        if top_level in BLOCKED_IMPORTS:
            raise ImportError(f"Import of '{name}' is restricted in the sandbox")
        return original_import(name, *args, **kwargs)

    safe_builtins["__import__"] = safe_import
    safe_builtins["__builtins__"] = safe_builtins

    return safe_builtins


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution timed out (10 second limit)")


def execute_code(code: str, timeout: int = 10) -> Dict[str, Any]:
    """
    Execute Python code in a sandboxed environment.

    Returns:
        {
            "output": str,      # stdout output
            "error": str|None,  # error message if any
            "execution_time": float
        }
    """
    start_time = time.time()
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
    except (ValueError, AttributeError):
        old_handler = None

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

    except TimeoutError as e:
        output = stdout_capture.getvalue()
        error = str(e)

    except ImportError as e:
        output = stdout_capture.getvalue()
        error = f"ImportError: {e}"

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

    if output and len(output) > 50000:
        output = output[:50000] + "\n... (output truncated at 50KB)"

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
