"""
Hooks Manager - Lifecycle hooks for extensibility.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import json
import os
import subprocess
import asyncio


class HookEvent(Enum):
    """Hook event types."""
    SESSION_START = "SessionStart"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"
    SESSION_END = "SessionEnd"


@dataclass
class HookHandler:
    """A hook handler configuration."""
    type: str = "command"  # "command" or "prompt"
    command: Optional[str] = None
    timeout: int = 60


@dataclass
class HookResult:
    """Result from running a hook."""
    success: bool = True
    decision: Optional[str] = None  # "allow", "deny", "block"
    reason: Optional[str] = None
    additional_context: Optional[str] = None
    continue_: bool = True


class HooksManager:
    """
    Manages lifecycle hooks.

    Features:
    - Command hooks (shell scripts)
    - Hook chaining
    - Async execution
    """

    def __init__(
        self,
        project_dir: str = ".",
    ):
        self.project_dir = os.path.abspath(project_dir)
        self._hooks: Dict[HookEvent, List[HookHandler]] = {}
        self._load_hooks()

    def _load_hooks(self):
        """Load hooks from settings files."""
        settings_paths = [
            os.path.join(self.project_dir, ".neuro", "settings.json"),
            os.path.expanduser("~/.neuro/settings.json"),
        ]

        for path in settings_paths:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)

                    hooks_config = data.get("hooks", {})

                    for event_name, handlers in hooks_config.items():
                        try:
                            event = HookEvent(event_name)
                        except ValueError:
                            continue

                        if event not in self._hooks:
                            self._hooks[event] = []

                        for handler_config in handlers:
                            handler = HookHandler(
                                type=handler_config.get("type", "command"),
                                command=handler_config.get("command"),
                                timeout=handler_config.get("timeout", 60),
                            )
                            self._hooks[event].append(handler)
                except Exception:
                    pass

    async def run_hook(
        self,
        event: Union[HookEvent, str],
        **context
    ) -> Dict[str, Any]:
        """
        Run hooks for an event.

        Args:
            event: The hook event
            **context: Event-specific context

        Returns:
            Dict with hook results
        """
        if isinstance(event, str):
            try:
                event = HookEvent(event)
            except ValueError:
                return {"success": True}

        handlers = self._hooks.get(event, [])
        if not handlers:
            return {"success": True}

        # Build input for hooks
        hook_input = {
            "event": event.value,
            "cwd": self.project_dir,
            **context
        }

        results = []
        for handler in handlers:
            result = await self._run_handler(handler, hook_input)
            results.append(result)

            # Stop on blocking result
            if result.decision in ("block", "deny"):
                return {
                    "success": False,
                    "decision": result.decision,
                    "reason": result.reason,
                }

        return {"success": True}

    async def _run_handler(
        self,
        handler: HookHandler,
        hook_input: Dict,
    ) -> HookResult:
        """Run a single hook handler."""
        if handler.type != "command" or not handler.command:
            return HookResult()

        try:
            # Expand variables
            command = os.path.expandvars(handler.command)
            command = command.replace("$PROJECT_DIR", self.project_dir)

            # Run command with input on stdin
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_dir,
            )

            input_json = json.dumps(hook_input).encode()
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input_json),
                timeout=handler.timeout
            )

            exit_code = process.returncode

            if exit_code == 0:
                # Success - try to parse JSON output
                try:
                    output = json.loads(stdout.decode())
                    return HookResult(
                        success=True,
                        decision=output.get("decision"),
                        reason=output.get("reason"),
                        additional_context=output.get("additionalContext"),
                    )
                except json.JSONDecodeError:
                    return HookResult(success=True)
            elif exit_code == 2:
                # Blocking error
                return HookResult(
                    success=False,
                    decision="block",
                    reason=stderr.decode() or "Blocked by hook",
                )
            else:
                return HookResult(success=True)

        except asyncio.TimeoutError:
            return HookResult(success=False, reason="Hook timed out")
        except Exception as e:
            return HookResult(success=False, reason=str(e))

    def register_hook(
        self,
        event: HookEvent,
        handler: HookHandler,
    ):
        """Register a hook handler programmatically."""
        if event not in self._hooks:
            self._hooks[event] = []
        self._hooks[event].append(handler)

    def get_hooks(self, event: HookEvent) -> List[HookHandler]:
        """Get all handlers for an event."""
        return self._hooks.get(event, [])
