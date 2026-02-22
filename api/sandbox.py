"""
Sandboxed Python execution environment for the ELO-AGI REPL API.

Executes Python code in an isolated subprocess with restricted imports,
blocked attribute access, memory limits, and execution timeouts.
"""

import sys
import json
import subprocess
import time
import textwrap
from pathlib import Path
from typing import Dict, Any


BLOCKED_IMPORTS = {
    # System / process
    "subprocess", "multiprocessing", "signal", "resource", "os", "pathlib",
    "shutil", "tempfile", "_thread", "threading", "concurrent",
    "_posixsubprocess", "pty", "fcntl", "termios",
    # Network
    "socket", "http", "urllib", "requests", "ftplib", "smtplib", "telnetlib",
    "aiohttp", "httpx",
    # Code introspection
    "importlib", "code", "codeop", "inspect", "ast", "dis", "sys",
    # Serialization
    "pickle", "shelve", "marshal",
    # IO / compression
    "_io", "zipfile", "tarfile", "gzip", "bz2", "lzma", "mmap",
    # Unsafe
    "ctypes", "webbrowser", "antigravity", "turtle", "tkinter", "gc",
    # System info
    "grp", "pwd", "crypt",
    # Database
    "dbm", "sqlite3",
    # Parsing
    "xml", "html.parser",
    # Async
    "asyncio", "selectors", "select",
}

BLOCKED_ATTRIBUTES = {
    "__class__", "__bases__", "__subclasses__", "__globals__",
    "__code__", "__closure__", "__func__", "__self__",
    "__dict__", "__module__", "__mro__", "__init_subclass__",
    "__import__", "__builtins__", "__loader__", "__spec__",
    "__file__", "__path__", "__cached__", "__reduce__", "__reduce_ex__",
}

SAFE_BUILTINS_NAMES = [
    "print", "len", "range", "str", "int", "float", "list", "dict",
    "tuple", "set", "bool", "abs", "round", "min", "max", "sum",
    "sorted", "reversed", "enumerate", "zip", "map", "filter",
    "isinstance", "hasattr", "True", "False", "None",
    "frozenset", "bytes", "bytearray", "hex", "oct", "bin",
    "ord", "chr", "all", "any", "repr", "hash", "slice", "complex",
    "ValueError", "TypeError", "KeyError", "IndexError",
    "AttributeError", "RuntimeError", "StopIteration",
    "ZeroDivisionError", "OverflowError", "ImportError",
    "Exception", "ArithmeticError", "LookupError",
]

MAX_OUTPUT_BYTES = 10 * 1024  # 10KB
MAX_MEMORY_BYTES = 256 * 1024 * 1024  # 256MB
DEFAULT_TIMEOUT = 10
MAX_INPUT_LENGTH = 5000
RESULT_SENTINEL = "__SANDBOX_RESULT_7f3a9b2c__"


def _get_project_paths() -> list:
    """Get sys.path entries needed for neuro modules."""
    project_root = Path(__file__).parent.parent
    paths = [str(project_root)]

    neuro_path = project_root / "neuro"
    if neuro_path.exists():
        paths.append(str(neuro_path))

    for module_dir in project_root.glob("neuro-*"):
        src_dir = module_dir / "src"
        if src_dir.exists():
            paths.append(str(src_dir))

    return paths


def _build_runner_script(code: str) -> str:
    """Build the Python script that runs inside the subprocess."""
    return textwrap.dedent('''\
import sys, io, time, json, traceback

# Memory limit (Linux only)
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, ({max_mem}, {max_mem}))
except Exception:
    pass

# Add project paths
for p in {paths}:
    if p not in sys.path:
        sys.path.insert(0, p)

BLOCKED_IMPORTS = set({blocked_imports})
SAFE_NAMES = {safe_names}
MAX_OUTPUT = {max_output}
CODE = json.loads({code_json})
SENTINEL = {sentinel}

# Build safe builtins
_src = __builtins__.__dict__ if hasattr(__builtins__, "__dict__") else __builtins__
safe_builtins = {{}}
for name in SAFE_NAMES:
    if name in _src:
        safe_builtins[name] = _src[name]

_orig_import = __import__
def _safe_import(name, *args, **kwargs):
    top = name.split(".")[0]
    if top in BLOCKED_IMPORTS:
        raise ImportError(f"Import of '{{name}}' is restricted in the sandbox")
    return _orig_import(name, *args, **kwargs)

safe_builtins["__import__"] = _safe_import
safe_builtins["__builtins__"] = safe_builtins

safe_globals = {{"__name__": "__main__", "__builtins__": safe_builtins}}

# Pre-load allowed modules
try:
    import neuro as neuro_agi
    safe_globals["neuro_agi"] = neuro_agi
    safe_globals["neuro"] = neuro_agi
except Exception:
    pass

try:
    import numpy as np
    safe_globals["np"] = np
    safe_globals["numpy"] = np
except ImportError:
    pass

try:
    from neuro.benchmark import Benchmark
    safe_globals["Benchmark"] = Benchmark
except ImportError:
    pass

try:
    from neuro.emotions import EmotionSystem, BasicEmotion
    safe_globals["EmotionSystem"] = EmotionSystem
    safe_globals["BasicEmotion"] = BasicEmotion
except ImportError:
    pass

# Execute
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()
start = time.time()
error = None

try:
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_capture, stderr_capture
    try:
        compiled = compile(CODE, "<repl>", "exec")
        exec(compiled, safe_globals)
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    output = stdout_capture.getvalue()
    err_out = stderr_capture.getvalue()
    if err_out:
        error = err_out

except MemoryError:
    output = stdout_capture.getvalue()
    error = "MemoryError: Execution exceeded 256MB memory limit"

except ImportError as e:
    output = stdout_capture.getvalue()
    error = f"ImportError: {{e}}"

except Exception:
    output = stdout_capture.getvalue()
    tb = traceback.format_exc()
    error = tb.strip().split("\\n")[-1]

elapsed = time.time() - start

if len(output) > MAX_OUTPUT:
    output = output[:MAX_OUTPUT] + "\\n... (output truncated at 10KB)"

result = {{"output": output, "error": error, "execution_time": round(elapsed, 4)}}
print(SENTINEL)
print(json.dumps(result))
''').format(
        max_mem=MAX_MEMORY_BYTES,
        paths=json.dumps(_get_project_paths()),
        blocked_imports=json.dumps(sorted(BLOCKED_IMPORTS)),
        safe_names=json.dumps(SAFE_BUILTINS_NAMES),
        max_output=MAX_OUTPUT_BYTES,
        code_json=repr(json.dumps(code)),
        sentinel=repr(RESULT_SENTINEL),
    )


def execute_code(code: str, timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """
    Execute Python code in a subprocess sandbox.

    Security measures:
    - Subprocess isolation (code never runs in API process)
    - Clean environment (PYTHONPATH cleared)
    - Allowlist-only builtins
    - 50+ blocked imports
    - 20+ blocked attribute patterns
    - Memory limit: 256MB (Linux)
    - Output size limit: 10KB
    - Execution timeout: 10 seconds

    Returns:
        {"output": str, "error": str|None, "execution_time": float}
    """
    if len(code) > MAX_INPUT_LENGTH:
        return {
            "output": "",
            "error": f"Input too long (max {MAX_INPUT_LENGTH} characters)",
            "execution_time": 0.0,
        }

    for attr in BLOCKED_ATTRIBUTES:
        if attr in code:
            return {
                "output": "",
                "error": f"Access to '{attr}' is restricted in the sandbox",
                "execution_time": 0.0,
            }

    runner_script = _build_runner_script(code)

    # Build clean environment
    import os

    clean_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "en_US.UTF-8"),
    }
    # Do NOT pass PYTHONPATH — subprocess only gets project paths added in-script

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, "-c", runner_script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=clean_env,
        )

        stdout = result.stdout
        stderr = result.stderr

        if RESULT_SENTINEL in stdout:
            parts = stdout.split(RESULT_SENTINEL, 1)
            try:
                parsed = json.loads(parts[1].strip())
                return {
                    "output": parsed.get("output", ""),
                    "error": parsed.get("error"),
                    "execution_time": parsed.get("execution_time", round(time.time() - start_time, 4)),
                }
            except (json.JSONDecodeError, IndexError):
                pass

        # Fallback: no sentinel found (subprocess crashed before printing result)
        error_msg = stderr.strip() if stderr.strip() else "Execution failed"
        # Extract last line for cleaner error
        if "\n" in error_msg:
            error_msg = error_msg.strip().split("\n")[-1]

        return {
            "output": stdout[:MAX_OUTPUT_BYTES] if stdout else "",
            "error": error_msg,
            "execution_time": round(time.time() - start_time, 4),
        }

    except subprocess.TimeoutExpired:
        return {
            "output": "",
            "error": f"Execution timed out ({timeout} second limit)",
            "execution_time": float(timeout),
        }

    except Exception as e:
        return {
            "output": "",
            "error": f"Sandbox error: {e}",
            "execution_time": round(time.time() - start_time, 4),
        }
