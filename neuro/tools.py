"""
ToolExecutor - Robust tool execution with retry logic.

Features:
- Retry on failure
- Timeout handling
- Error recovery
- Result validation
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum


class ToolStatus(Enum):
    """Tool execution status."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class ToolCall:
    """A tool call request."""
    name: str
    args: Dict[str, Any]
    timeout: float = 30.0


@dataclass
class ToolResult:
    """Result of tool execution."""
    name: str
    status: ToolStatus
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    retries: int = 0

    @property
    def success(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "duration": self.duration,
            "retries": self.retries,
        }


class ToolExecutor:
    """
    Robust tool execution with retry logic.
    
    Features:
    - Automatic retries on transient failures
    - Timeout handling per tool
    - Error categorization and recovery
    - Result validation
    """

    def __init__(
        self,
        tools: Optional[Dict[str, Callable]] = None,
        max_retries: int = 3,
        default_timeout: float = 30.0,
        retry_delay: float = 1.0,
    ):
        self.tools = tools or {}
        self.max_retries = max_retries
        self.default_timeout = default_timeout
        self.retry_delay = retry_delay
        self._tool_registry: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        """Register a tool function."""
        self._tool_registry[name] = func

    def unregister(self, name: str):
        """Unregister a tool."""
        self._tool_registry.pop(name, None)

    def list_tools(self) -> List[str]:
        """List available tools."""
        return list(self._tool_registry.keys())

    async def execute(
        self,
        tool: str,
        args: Dict[str, Any],
        timeout: Optional[float] = None,
        retries: Optional[int] = None,
    ) -> ToolResult:
        """
        Execute a tool with retries and timeout.
        
        Args:
            tool: Tool name
            args: Tool arguments
            timeout: Override default timeout
            retries: Override max retries
            
        Returns:
            ToolResult with status and output/error
        """
        timeout = timeout or self.default_timeout
        max_retries = retries or self.max_retries
        
        # Get tool function
        tool_fn = self._tool_registry.get(tool)
        if not tool_fn:
            return ToolResult(
                name=tool,
                status=ToolStatus.ERROR,
                error=f"Unknown tool: {tool}",
            )
        
        last_error = None
        result = ToolResult(name=tool, status=ToolStatus.ERROR)
        
        for attempt in range(max_retries + 1):
            start_time = time.time()
            
            try:
                # Execute with timeout
                if asyncio.iscoroutinefunction(tool_fn):
                    output = await asyncio.wait_for(
                        tool_fn(**args),
                        timeout=timeout,
                    )
                else:
                    # Run sync function in executor
                    loop = asyncio.get_event_loop()
                    output = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: tool_fn(**args)),
                        timeout=timeout,
                    )
                
                # Validate result
                if self._validate_result(output):
                    result.status = ToolStatus.SUCCESS
                    result.output = output
                    result.duration = time.time() - start_time
                    result.retries = attempt
                    return result
                else:
                    result.error = "Invalid result format"
                    result.status = ToolStatus.ERROR
                    
            except asyncio.TimeoutError:
                result.status = ToolStatus.TIMEOUT
                result.error = f"Timeout after {timeout}s"
                result.duration = time.time() - start_time
                
            except Exception as e:
                result.status = ToolStatus.ERROR
                result.error = str(e)
                result.duration = time.time() - start_time
                last_error = e
            
            # Retry with exponential backoff
            if attempt < max_retries:
                delay = self.retry_delay * (2 ** attempt)
                await asyncio.sleep(delay)
                result.retries = attempt + 1
        
        # All retries exhausted
        if last_error:
            result.error = f"Failed after {max_retries + 1} attempts: {last_error}"
        
        return result

    async def execute_batch(
        self,
        calls: List[ToolCall],
        parallel: bool = True,
    ) -> List[ToolResult]:
        """
        Execute multiple tool calls.
        
        Args:
            calls: List of ToolCall objects
            parallel: Whether to execute in parallel
            
        Returns:
            List of ToolResult objects
        """
        if parallel and len(calls) > 1:
            # Execute in parallel
            tasks = [
                self.execute(call.name, call.args, call.timeout)
                for call in calls
            ]
            return await asyncio.gather(*tasks)
        else:
            # Execute sequentially
            results = []
            for call in calls:
                result = await self.execute(call.name, call.args, call.timeout)
                results.append(result)
            return results

    def _validate_result(self, result: Any) -> bool:
        """
        Validate tool result.
        
        Default implementation accepts any non-None result.
        Override for custom validation.
        """
        return result is not None

    def _categorize_error(self, error: Exception) -> str:
        """
        Categorize error for recovery strategies.
        
        Categories:
        - transient: Retry might help (network, timeout)
        - validation: Input error, don't retry
        - fatal: System error, stop execution
        """
        if isinstance(error, asyncio.TimeoutError):
            return "transient"
        
        error_name = type(error).__name__.lower()
        
        if any(x in error_name for x in ["timeout", "connection", "network"]):
            return "transient"
        
        if any(x in error_name for x in ["value", "type", "key", "attribute"]):
            return "validation"
        
        return "fatal"


# Default tools for NEURO
def create_default_tools() -> Dict[str, Callable]:
    """Create default tool set."""
    
    def read_file(path: str) -> str:
        """Read a file and return contents."""
        from pathlib import Path
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return p.read_text()
    
    def write_file(path: str, content: str) -> bool:
        """Write content to a file."""
        from pathlib import Path
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return True
    
    def list_files(path: str = ".") -> List[str]:
        """List files in directory."""
        from pathlib import Path
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        return [str(f) for f in p.iterdir()]
    
    def run_python(code: str) -> str:
        """Execute Python code and return output."""
        import io
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            exec(code, {"__builtins__": __builtins__})
            return sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
    
    def search_web(query: str) -> List[Dict[str, str]]:
        """Search web (placeholder - requires API)."""
        # Placeholder - would integrate with search API
        return [{"title": f"Result for: {query}", "url": "https://example.com"}]
    
    return {
        "read_file": read_file,
        "write_file": write_file,
        "list_files": list_files,
        "run_python": run_python,
        "search_web": search_web,
    }
