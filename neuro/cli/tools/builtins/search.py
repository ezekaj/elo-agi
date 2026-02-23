"""Search tools - Glob and Grep for fast codebase exploration."""

import os
import re
import subprocess
from pathlib import Path


def glob_files(pattern: str, path: str = ".") -> str:
    """Fast file pattern matching using pathlib glob.

    Returns matching file paths sorted by modification time.
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: Path not found: {path}"

    try:
        base = Path(path)
        matches = []
        for m in base.glob(pattern):
            try:
                matches.append((m, m.stat().st_mtime))
            except OSError:
                matches.append((m, 0))

        matches.sort(key=lambda x: x[1], reverse=True)

        if not matches:
            return f"No files matching '{pattern}' in {path}"

        results = []
        for match, _ in matches[:100]:
            try:
                rel = match.relative_to(base)
            except ValueError:
                rel = match
            results.append(str(rel))

        return "\n".join(results)
    except Exception as e:
        return f"Error: {e}"


def grep_content(
    pattern: str,
    path: str = ".",
    output_mode: str = "content",
    glob_filter: str = "",
    case_insensitive: bool = False,
    context_lines: int = 0,
    max_results: int = 50,
) -> str:
    """Search file contents with regex support.

    Tries ripgrep first for speed, falls back to Python re.
    output_mode: 'content' (lines), 'files_with_matches' (paths), 'count'
    """
    path = os.path.expanduser(path)

    # Try ripgrep first
    try:
        cmd = ["rg", "--no-heading", "--color=never"]
        if output_mode == "files_with_matches":
            cmd.append("-l")
        elif output_mode == "count":
            cmd.append("-c")

        if case_insensitive:
            cmd.append("-i")
        if context_lines > 0:
            cmd.extend(["-C", str(context_lines)])
        if glob_filter:
            cmd.extend(["--glob", glob_filter])

        cmd.extend(["-m", str(max_results), pattern, path])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = result.stdout
        if not output:
            return f"No matches for '{pattern}'"
        return output[:30000]

    except FileNotFoundError:
        pass  # ripgrep not installed

    # Python fallback
    try:
        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(pattern, flags)
        results = []
        skip_dirs = {".git", "node_modules", "__pycache__", "venv", ".venv", ".neuro"}

        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]

            for fname in files:
                if glob_filter:
                    from fnmatch import fnmatch

                    if not fnmatch(fname, glob_filter.lstrip("*")):
                        continue

                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", errors="ignore") as f:
                        for i, line in enumerate(f, 1):
                            if regex.search(line):
                                rel = os.path.relpath(fpath, path)
                                if output_mode == "files_with_matches":
                                    results.append(rel)
                                    break
                                else:
                                    results.append(f"{rel}:{i}:{line.rstrip()}")

                                if len(results) >= max_results:
                                    break
                except (OSError, UnicodeDecodeError):
                    continue

            if len(results) >= max_results:
                break

        if not results:
            return f"No matches for '{pattern}'"
        return "\n".join(results[:max_results])

    except Exception as e:
        return f"Error: {e}"
