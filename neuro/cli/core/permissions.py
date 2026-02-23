"""
Permission Manager - Tool permission system.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Set
from enum import Enum
import json
import os
import re


class PermissionMode(Enum):
    """Permission modes."""

    DEFAULT = "default"  # Prompt for each action
    ACCEPT_EDITS = "acceptEdits"  # Auto-accept file edits
    DONT_ASK = "dontAsk"  # Auto-deny (only pre-allowed tools)
    BYPASS = "bypassPermissions"  # Allow all (dangerous)
    PLAN = "plan"  # Read-only, no modifications


class PermissionDecision(Enum):
    """Permission decision types."""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass
class PermissionRule:
    """A permission rule (e.g., 'Bash(npm *)' or 'Edit')."""

    tool: str
    specifier: Optional[str] = None
    decision: PermissionDecision = PermissionDecision.ASK

    @classmethod
    def parse(cls, rule_str: str, decision: PermissionDecision) -> "PermissionRule":
        """Parse rule like 'Bash(npm *)' or 'Edit'."""
        match = re.match(r"(\w+)(?:\((.*)\))?", rule_str)
        if match:
            tool = match.group(1)
            specifier = match.group(2)
            return cls(tool=tool, specifier=specifier, decision=decision)
        return cls(tool=rule_str, decision=decision)

    def matches(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        """Check if this rule matches a tool call."""
        if self.tool != tool_name and self.tool != "*":
            return False

        if self.specifier is None or self.specifier == "*":
            return True

        # Special handling for Bash commands
        if tool_name in ("Bash", "run_command"):
            command = tool_input.get("command", "")
            pattern = self.specifier.replace("*", ".*")
            return bool(re.match(pattern, command))

        # For file operations, match path
        if tool_name in ("Read", "Write", "Edit", "read_file", "write_file", "edit_file"):
            path = tool_input.get("file_path", tool_input.get("path", ""))
            pattern = self.specifier.replace("*", ".*")
            return bool(re.match(pattern, path))

        return False


class PermissionManager:
    """
    Manages tool permissions.

    Evaluation order:
    1. Deny rules (always block)
    2. Session-level overrides
    3. Allow rules (auto-approve)
    4. Permission mode fallback
    5. User prompt
    """

    # Read-only tools that are always safe
    SAFE_TOOLS = {"read_file", "Read", "Glob", "Grep", "list_files", "glob_files", "grep_content"}

    def __init__(
        self,
        mode: PermissionMode = PermissionMode.DEFAULT,
        project_dir: str = ".",
    ):
        self.mode = mode
        self.project_dir = os.path.abspath(project_dir)

        # Rules
        self.allow_rules: List[PermissionRule] = []
        self.deny_rules: List[PermissionRule] = []

        # Runtime state
        self._session_allows: Set[str] = set()
        self._session_denies: Set[str] = set()

        # Load rules from settings
        self._load_rules()

    def _load_rules(self):
        """Load permission rules from settings files."""
        settings_paths = [
            os.path.join(self.project_dir, ".neuro", "settings.json"),
            os.path.expanduser("~/.neuro/settings.json"),
        ]

        for path in settings_paths:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)

                    perms = data.get("permissions", {})

                    for rule_str in perms.get("allow", []):
                        self.allow_rules.append(
                            PermissionRule.parse(rule_str, PermissionDecision.ALLOW)
                        )
                    for rule_str in perms.get("deny", []):
                        self.deny_rules.append(
                            PermissionRule.parse(rule_str, PermissionDecision.DENY)
                        )
                except Exception:
                    pass

    def set_mode(self, mode: PermissionMode):
        """Change permission mode."""
        self.mode = mode

    async def check(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> bool:
        """
        Check if a tool call is permitted.
        Returns True if allowed, False if denied.
        """
        # 1. Check deny rules first
        for rule in self.deny_rules:
            if rule.matches(tool_name, tool_input):
                return False

        # 2. Check session denies
        if tool_name in self._session_denies:
            return False

        # 3. Check session allows
        if tool_name in self._session_allows:
            return True

        # 4. Check allow rules
        for rule in self.allow_rules:
            if rule.matches(tool_name, tool_input):
                return True

        # 5. Apply permission mode
        if self.mode == PermissionMode.BYPASS:
            return True

        if self.mode == PermissionMode.PLAN:
            return tool_name in self.SAFE_TOOLS

        if self.mode == PermissionMode.ACCEPT_EDITS:
            if tool_name in ("Edit", "Write", "edit_file", "write_file", *self.SAFE_TOOLS):
                return True

        if self.mode == PermissionMode.DONT_ASK:
            return tool_name in self.SAFE_TOOLS

        # 6. Safe tools always allowed
        if tool_name in self.SAFE_TOOLS:
            return True

        # 7. Default: prompt user
        return await self._prompt_user(tool_name, tool_input)

    async def _prompt_user(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> bool:
        """Interactively prompt user for permission with Rich UI."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text

        console = Console()

        # Build content
        content = Text()
        content.append(f"Tool: ", style="dim")
        content.append(f"{tool_name}\n", style="bold")

        for k, v in list(tool_input.items())[:3]:
            val_str = str(v)[:60]
            if len(str(v)) > 60:
                val_str += "..."
            content.append(f"  {k}: ", style="dim")
            content.append(f"{val_str}\n")

        panel = Panel(
            content,
            title="[bold]Permission Required[/bold]",
            border_style="purple",
            padding=(0, 1),
        )
        console.print()
        console.print(panel)
        console.print(
            "  [purple]y[/purple] Allow  [purple]n[/purple] Deny  "
            "[purple]a[/purple] Always allow  [purple]d[/purple] Always deny"
        )

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: input("  > ").strip().lower())
        except (EOFError, KeyboardInterrupt):
            return False

        if response in ("y", "yes", ""):
            return True
        elif response in ("a", "always"):
            self._session_allows.add(tool_name)
            return True
        elif response in ("d", "never"):
            self._session_denies.add(tool_name)
            return False
        else:
            return False

    def grant_always(self, tool_name: str):
        """Grant 'always allow' for a tool in this session."""
        self._session_allows.add(tool_name)

    def deny_always(self, tool_name: str):
        """Set 'always deny' for a tool in this session."""
        self._session_denies.add(tool_name)

    def reset_session(self):
        """Reset session-level permissions."""
        self._session_allows.clear()
        self._session_denies.clear()
