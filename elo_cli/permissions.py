"""Permission system for tool execution."""

import json
from enum import Enum
from rich.console import Console
from rich.prompt import Confirm

console = Console()


class Permission(Enum):
    ALLOW = "allow"  # Auto-approve
    ASK = "ask"  # Prompt user
    DENY = "deny"  # Block entirely


# Default permissions per tool
TOOL_PERMISSIONS = {
    "read_file": Permission.ALLOW,
    "glob": Permission.ALLOW,
    "grep": Permission.ALLOW,
    "ls": Permission.ALLOW,
    "write_file": Permission.ASK,
    "edit_file": Permission.ASK,
    "bash": Permission.ASK,
}

# Dangerous command patterns - always require permission
DANGEROUS_PATTERNS = [
    "rm ",
    "rm -",
    "rmdir",
    "sudo",
    "chmod",
    "chown",
    "git push",
    "git reset --hard",
    "> /",
    ">> /",
    "curl | ",
    "wget |",
    "mv /",
    "cp /",
]

# Safe bash commands - auto-approve
SAFE_COMMANDS = [
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
    "python --version",
    "python3 --version",
    "node --version",
    "npm --version",
    "pip list",
    "pip show",
]


def is_dangerous_command(command: str) -> bool:
    """Check if a bash command is dangerous."""
    cmd_lower = command.lower().strip()
    for pattern in DANGEROUS_PATTERNS:
        if pattern in cmd_lower:
            return True
    return False


def is_safe_command(command: str) -> bool:
    """Check if a bash command is safe to auto-execute."""
    cmd_lower = command.lower().strip()
    for safe in SAFE_COMMANDS:
        if cmd_lower.startswith(safe):
            return True
    return False


def check_permission(tool_name: str, args: dict) -> bool:
    """Check if tool execution is allowed.

    Returns True if allowed, False if denied.
    """
    perm = TOOL_PERMISSIONS.get(tool_name, Permission.ASK)

    # Special handling for bash commands
    if tool_name == "bash":
        command = args.get("command", "")
        if is_dangerous_command(command):
            perm = Permission.ASK
        elif is_safe_command(command):
            perm = Permission.ALLOW

    # Auto-allow
    if perm == Permission.ALLOW:
        return True

    # Auto-deny
    if perm == Permission.DENY:
        console.print(f"[red]⛔ Blocked: {tool_name}[/red]")
        return False

    # ASK mode - prompt user
    console.print()
    console.print(f"[bold magenta]🔒 Permission Required[/bold magenta]")
    console.print(f"[cyan]Tool:[/cyan] {tool_name}")

    for key, value in args.items():
        # Truncate long values
        display_value = str(value)
        if len(display_value) > 100:
            display_value = display_value[:100] + "..."
        console.print(f"[dim]  {key}:[/dim] {display_value}")

    return Confirm.ask("[yellow]Allow this action?[/yellow]", default=True)


class PermissionManager:
    """Manages tool permissions with customization."""

    def __init__(self):
        self.overrides = {}  # {tool_name: Permission}
        self.session_allowed = set()  # Tools allowed for this session

    def set_permission(self, tool_name: str, permission: Permission):
        """Override default permission for a tool."""
        self.overrides[tool_name] = permission

    def allow_for_session(self, tool_name: str):
        """Allow a tool for the rest of the session."""
        self.session_allowed.add(tool_name)

    def check(self, tool_name: str, args: dict) -> bool:
        """Check permission with overrides."""
        # Session override
        if tool_name in self.session_allowed:
            return True

        # Custom override
        if tool_name in self.overrides:
            perm = self.overrides[tool_name]
            if perm == Permission.ALLOW:
                return True
            if perm == Permission.DENY:
                console.print(f"[red]⛔ Blocked: {tool_name}[/red]")
                return False

        # Default check
        return check_permission(tool_name, args)
