"""
Tool Executor - Execute tools with visual feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Any, Optional
from enum import Enum
import asyncio
import sys
import time
import threading

if TYPE_CHECKING:
    from neuro.cli.tools.registry import ToolRegistry
    from neuro.cli.core.permissions import PermissionManager
    from neuro.cli.core.hooks import HooksManager
    from neuro.cli.ui.renderer import UIRenderer


class ToolStatus(Enum):
    """Tool execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    TIMEOUT = "timeout"


@dataclass
class ToolExecution:
    """Result of a tool execution."""

    name: str
    args: Dict[str, Any]
    status: ToolStatus = ToolStatus.PENDING
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0


class ToolSpinner:
    """Animated spinner for tool execution with verb display and long-run tips."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    RESET = "\033[0m"
    PURPLE = "\033[35m"
    GREEN = "\033[32m"
    RED = "\033[31m"
    YELLOW = "\033[33m"
    DIM = "\033[2m"

    # Tips shown after 30s of execution
    TIPS = [
        "Long-running commands can be cancelled with Ctrl+C",
        "Use /compact to reduce context when conversations get long",
        "Use /cost to see real token usage",
        "Try /think for deeper reasoning on complex problems",
    ]

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame = 0
        self._message = ""
        self._start_time = 0.0
        self._tip_shown = False

    def start(self, message: str):
        """Start spinner with message."""
        self._message = message
        self._running = True
        self._tip_shown = False
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def update(self, message: str):
        """Update spinner message."""
        self._message = message

    def stop(self, success: bool = True, message: str = "", rejected: bool = False):
        """Stop spinner with final status and duration."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)

        duration = time.time() - self._start_time if self._start_time else 0
        dur_str = f" {self.DIM}({duration:.1f}s){self.RESET}" if duration > 0.1 else ""

        if rejected:
            icon = f"{self.YELLOW}⊘{self.RESET}"
        elif success:
            icon = f"{self.GREEN}✓{self.RESET}"
        else:
            icon = f"{self.RED}✗{self.RESET}"

        final_msg = message or self._message
        sys.stdout.write(f"\r\033[K  {icon} {final_msg}{dur_str}\n")
        sys.stdout.flush()

    def _animate(self):
        import random

        while self._running:
            frame = self.FRAMES[self._frame % len(self.FRAMES)]
            elapsed = time.time() - self._start_time

            # Show tip after 30 seconds
            tip_line = ""
            if elapsed > 30 and not self._tip_shown:
                self._tip_shown = True
                tip = random.choice(self.TIPS)
                tip_line = f"\n  {self.DIM}💡 {tip}{self.RESET}"

            line = f"\r\033[K  {self.PURPLE}{frame}{self.RESET} {self._message}"
            if elapsed > 5:
                line += f" {self.DIM}({elapsed:.0f}s){self.RESET}"
            sys.stdout.write(line + tip_line)
            sys.stdout.flush()
            self._frame += 1
            time.sleep(0.08)


class ToolExecutor:
    """
    Executes tools with visual feedback.

    Features:
    - Animated spinners during execution
    - Permission checking
    - Pre/Post hooks
    - Parallel execution support
    """

    def __init__(
        self,
        registry: "ToolRegistry",
        permissions: Optional["PermissionManager"] = None,
        hooks: Optional["HooksManager"] = None,
        ui: Optional["UIRenderer"] = None,
    ):
        self.registry = registry
        self.permissions = permissions
        self.hooks = hooks
        self.ui = ui
        self._spinner = ToolSpinner()

    async def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        show_spinner: bool = True,
    ) -> ToolExecution:
        """Execute a tool with visual feedback."""
        execution = ToolExecution(
            name=tool_name,
            args=args,
        )
        start_time = time.time()

        # 1. Run PreToolUse hook
        if self.hooks:
            hook_result = await self.hooks.run_hook(
                "PreToolUse", tool_name=tool_name, tool_input=args
            )
            if hook_result.get("decision") == "block":
                execution.status = ToolStatus.BLOCKED
                execution.error = hook_result.get("reason", "Blocked by hook")
                return execution

        # 2. Check permissions
        if self.permissions:
            allowed = await self.permissions.check(tool_name, args)
            if not allowed:
                execution.status = ToolStatus.BLOCKED
                execution.error = "Permission denied"
                return execution

        # 3. Start spinner with active verb
        if show_spinner:
            from neuro.cli.ui.renderer import TOOL_VERBS

            verb = TOOL_VERBS.get(tool_name, "Running")
            preview = self._format_args_preview(args)
            spinner_msg = f"{verb} {tool_name}" + (f": {preview}" if preview else "")
            self._spinner.start(spinner_msg)

        execution.status = ToolStatus.RUNNING

        try:
            # 4. Get and execute tool
            tool = self.registry.get_tool(tool_name)
            if not tool:
                raise ValueError(f"Tool not found: {tool_name}")

            # Execute
            if asyncio.iscoroutinefunction(tool.func):
                result = await tool.func(**args)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tool.func(**args))

            execution.output = result
            execution.status = ToolStatus.SUCCESS

            if show_spinner:
                self._spinner.stop(success=True, message=tool_name)

        except asyncio.TimeoutError:
            execution.status = ToolStatus.TIMEOUT
            execution.error = "Timeout"
            if show_spinner:
                self._spinner.stop(success=False, message=f"{tool_name} timed out")

        except Exception as e:
            execution.status = ToolStatus.FAILED
            execution.error = str(e)
            if show_spinner:
                self._spinner.stop(success=False, message=f"{tool_name}: {e}")

        finally:
            execution.duration = time.time() - start_time

        # 5. Run PostToolUse hook
        if self.hooks:
            await self.hooks.run_hook(
                "PostToolUse",
                tool_name=tool_name,
                tool_input=args,
                tool_output=execution.output,
            )

        return execution

    async def execute_many(
        self,
        tools: list,
    ) -> list:
        """Execute multiple tools in parallel."""
        tasks = [self.execute(t["name"], t["args"]) for t in tools]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _format_args_preview(self, args: Dict[str, Any], max_len: int = 40) -> str:
        """Format args for spinner preview."""
        if not args:
            return ""
        parts = []
        for k, v in args.items():
            val_str = str(v)[:20]
            parts.append(f"{k}={val_str}")
        preview = ", ".join(parts)
        if len(preview) > max_len:
            preview = preview[:max_len] + "..."
        return preview
