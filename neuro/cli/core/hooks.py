"""
Hooks Manager - Lifecycle hooks for extensibility.

Hooks allow external scripts to intercept and modify ELO behavior.
They can be used for:
- Custom validation before tool execution
- Audit logging
- Custom permissions logic
- Integration with external systems

Configuration in ~/.neuro/settings.json or .neuro/settings.json:
{
  "hooks": {
    "PreToolUse": [
      {
        "type": "command",
        "command": "./scripts/validate-tool.sh",
        "timeout": 30,
        "matcher": {
          "tool_name": "write_file",
          "file_pattern": "*.py"
        }
      }
    ]
  }
}
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import json
import os
import asyncio
import fnmatch
import re


class HookEvent(Enum):
    """Hook event types."""

    SESSION_START = "SessionStart"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"
    SESSION_END = "SessionEnd"
    NOTIFICATION = "Notification"
    SUBAGENT_SPAWN = "SubagentSpawn"


@dataclass
class HookMatcher:
    """Matcher to filter when hooks run."""

    tool_name: Optional[str] = None  # Match specific tool
    tool_pattern: Optional[str] = None  # Glob pattern for tool names
    file_pattern: Optional[str] = None  # Glob pattern for file paths
    regex: Optional[str] = None  # Regex pattern for content

    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if context matches this matcher."""
        # Match tool name
        if self.tool_name:
            if context.get("tool_name") != self.tool_name:
                return False

        # Match tool pattern
        if self.tool_pattern:
            tool_name = context.get("tool_name", "")
            if not fnmatch.fnmatch(tool_name, self.tool_pattern):
                return False

        # Match file pattern
        if self.file_pattern:
            file_path = context.get("file_path") or context.get("path", "")
            if not fnmatch.fnmatch(file_path, self.file_pattern):
                return False

        # Match regex
        if self.regex:
            content = context.get("content") or context.get("prompt", "")
            if not re.search(self.regex, str(content)):
                return False

        return True


@dataclass
class HookHandler:
    """A hook handler configuration."""

    type: str = "command"  # "command" or "url"
    command: Optional[str] = None
    url: Optional[str] = None  # For webhook-style hooks
    timeout: int = 60
    matcher: Optional[HookMatcher] = None
    env: Dict[str, str] = field(default_factory=dict)


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
                            # Parse matcher if present
                            matcher = None
                            if "matcher" in handler_config:
                                m = handler_config["matcher"]
                                matcher = HookMatcher(
                                    tool_name=m.get("tool_name"),
                                    tool_pattern=m.get("tool_pattern"),
                                    file_pattern=m.get("file_pattern"),
                                    regex=m.get("regex"),
                                )

                            handler = HookHandler(
                                type=handler_config.get("type", "command"),
                                command=handler_config.get("command"),
                                url=handler_config.get("url"),
                                timeout=handler_config.get("timeout", 60),
                                matcher=matcher,
                                env=handler_config.get("env", {}),
                            )
                            self._hooks[event].append(handler)
                except Exception:
                    pass

    async def run_hook(self, event: Union[HookEvent, str], **context) -> Dict[str, Any]:
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
        hook_input = {"event": event.value, "cwd": self.project_dir, **context}

        results = []
        for handler in handlers:
            # Check matcher
            if handler.matcher and not handler.matcher.matches(context):
                continue

            result = await self._run_handler(handler, hook_input)
            results.append(result)

            # Stop on blocking result
            if result.decision in ("block", "deny"):
                return {
                    "success": False,
                    "decision": result.decision,
                    "reason": result.reason,
                    "additional_context": result.additional_context,
                }

        return {
            "success": True,
            "handlers_run": len(results),
        }

    async def _run_handler(
        self,
        handler: HookHandler,
        hook_input: Dict,
    ) -> HookResult:
        """Run a single hook handler."""
        if handler.type == "command":
            return await self._run_command_handler(handler, hook_input)
        elif handler.type == "url":
            return await self._run_url_handler(handler, hook_input)
        else:
            return HookResult()

    async def _run_command_handler(
        self,
        handler: HookHandler,
        hook_input: Dict,
    ) -> HookResult:
        """Run a command hook handler."""
        if not handler.command:
            return HookResult()

        try:
            # Expand variables
            command = os.path.expandvars(handler.command)
            command = command.replace("$PROJECT_DIR", self.project_dir)

            # Build environment
            env = os.environ.copy()
            env["ELO_PROJECT_DIR"] = self.project_dir
            env["ELO_HOOK_EVENT"] = hook_input.get("event", "")
            env.update(handler.env)

            # Run command with input on stdin
            process = await asyncio.create_subprocess_shell(
                command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_dir,
                env=env,
            )

            input_json = json.dumps(hook_input).encode()
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input_json), timeout=handler.timeout
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

    async def _run_url_handler(
        self,
        handler: HookHandler,
        hook_input: Dict,
    ) -> HookResult:
        """Run a URL/webhook hook handler."""
        if not handler.url:
            return HookResult()

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    handler.url,
                    json=hook_input,
                    timeout=aiohttp.ClientTimeout(total=handler.timeout),
                ) as response:
                    if response.status == 200:
                        try:
                            output = await response.json()
                            return HookResult(
                                success=True,
                                decision=output.get("decision"),
                                reason=output.get("reason"),
                                additional_context=output.get("additionalContext"),
                            )
                        except Exception:
                            return HookResult(success=True)
                    elif response.status == 403:
                        return HookResult(
                            success=False,
                            decision="block",
                            reason="Blocked by webhook",
                        )
                    else:
                        return HookResult(success=True)

        except asyncio.TimeoutError:
            return HookResult(success=False, reason="Webhook timed out")
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

    def list_all_hooks(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all configured hooks."""
        result = {}
        for event, handlers in self._hooks.items():
            result[event.value] = []
            for handler in handlers:
                info = {
                    "type": handler.type,
                    "timeout": handler.timeout,
                }
                if handler.command:
                    info["command"] = handler.command
                if handler.url:
                    info["url"] = handler.url
                if handler.matcher:
                    info["matcher"] = {
                        k: v
                        for k, v in {
                            "tool_name": handler.matcher.tool_name,
                            "tool_pattern": handler.matcher.tool_pattern,
                            "file_pattern": handler.matcher.file_pattern,
                            "regex": handler.matcher.regex,
                        }.items()
                        if v
                    }
                result[event.value].append(info)
        return result

    def has_hooks(self) -> bool:
        """Check if any hooks are configured."""
        return bool(self._hooks)
