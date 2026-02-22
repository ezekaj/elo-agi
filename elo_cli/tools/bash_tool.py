"""Bash tool for shell command execution."""

import subprocess
import shlex
from .base import Tool, ToolResult


# Commands that are safe to run without asking
SAFE_COMMANDS = {
    "ls",
    "pwd",
    "echo",
    "cat",
    "head",
    "tail",
    "wc",
    "git status",
    "git log",
    "git branch",
    "git diff",
    "which",
    "whoami",
    "date",
    "uname",
    "env",
    "printenv",
    "python --version",
    "python3 --version",
    "node --version",
    "pip list",
    "npm list",
    "cargo --version",
}

# Commands that should always require permission
DANGEROUS_PATTERNS = [
    "rm ",
    "rm -",
    "rmdir",
    "mv ",
    "cp ",
    "sudo",
    "chmod",
    "chown",
    "git push",
    "git reset",
    "git rebase",
    "git merge",
    "curl | ",
    "wget | ",
    "| bash",
    "| sh",
    "dd ",
    "mkfs",
    "fdisk",
    "> /",
    "rm -rf",
]


class BashTool(Tool):
    """Execute shell commands."""

    name = "bash"
    description = "Execute a shell command and return the output."
    requires_permission = True  # Default to requiring permission

    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
        },
        "required": ["command"],
    }

    def is_safe(self, command: str) -> bool:
        """Check if command is safe to run without permission."""
        cmd_lower = command.lower().strip()

        # Check dangerous patterns first
        for pattern in DANGEROUS_PATTERNS:
            if pattern in cmd_lower:
                return False

        # Check if it's a known safe command
        for safe in SAFE_COMMANDS:
            if cmd_lower.startswith(safe):
                return True

        return False

    def execute(self, command: str, timeout: int = 30, **kwargs) -> ToolResult:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=kwargs.get("cwd", None),
            )

            output = result.stdout
            if result.stderr:
                output += f"\n\nStderr:\n{result.stderr}"

            # Truncate long output
            if len(output) > 10000:
                output = output[:10000] + f"\n\n... (truncated, {len(output)} total bytes)"

            return ToolResult(
                success=result.returncode == 0,
                output=output if output else "(no output)",
                error="" if result.returncode == 0 else f"Exit code: {result.returncode}",
                data={"return_code": result.returncode, "command": command},
            )

        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"Command timed out after {timeout}s")
        except Exception as e:
            return ToolResult(False, "", str(e))
