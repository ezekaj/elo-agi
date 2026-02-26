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
        import inspect
        
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
                if inspect.iscoroutinefunction(tool_fn):
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
    
    def read_file(path: str, lines: Optional[int] = None) -> str:
        """Read a file and return contents."""
        from pathlib import Path
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        content = p.read_text()
        if lines:
            return "\n".join(content.split("\n")[:lines])
        return content
    
    def write_file(path: str, content: str) -> bool:
        """Write content to a file."""
        from pathlib import Path
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return True
    
    def list_files(path: str = ".", pattern: str = "*") -> List[str]:
        """List files in directory with optional glob pattern."""
        from pathlib import Path
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        return [str(f.relative_to(p)) for f in p.glob(pattern)]
    
    def run_bash(command: str, timeout: float = 0, cwd: Optional[str] = "/") -> Dict[str, Any]:
        """
        Execute a bash/shell command - UNRESTRICTED.
        
        Args:
            command: Shell command to execute
            timeout: Maximum execution time (0 = no limit)
            cwd: Working directory (default: / for full access)
            
        Returns:
            Dict with stdout, stderr, returncode
        """
        import subprocess
        from pathlib import Path
        
        # Remove all resource limits
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
            resource.setrlimit(resource.RLIMIT_NPROC, (65536, 65536))
        except:
            pass
        
        work_dir = Path(cwd).expanduser() if cwd else Path("/")
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # No timeout if 0
            timeout_sec = None if timeout == 0 else timeout
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=os.environ.copy(),
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": command,
                "cwd": str(work_dir),
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "returncode": -1,
                "command": command,
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "command": command,
            }
    
    def run_python(code: str) -> str:
        """Execute Python code and return output."""
        import io
        import sys
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            exec(code, {"__builtins__": __builtins__})
            return sys.stdout.getvalue()
        except Exception as e:
            return f"Error: {e}\n{sys.stderr.getvalue()}"
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    def search_files(pattern: str, path: str = ".") -> List[str]:
        """
        Search for files matching a glob pattern.
        
        Args:
            pattern: Glob pattern (e.g., "*.py", "**/*.txt")
            path: Base directory to search
            
        Returns:
            List of matching file paths
        """
        from pathlib import Path
        base = Path(path).expanduser()
        if not base.exists():
            return []
        return [str(f.relative_to(base)) for f in base.glob(pattern)]
    
    def search_grep(pattern: str, path: str = ".", include: str = "*", 
                    max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Search for text pattern in files using ripgrep (if available) or fallback.
        
        Args:
            pattern: Regex pattern to search for
            path: Directory to search
            include: File pattern to include (e.g., "*.py")
            max_results: Maximum number of results
            
        Returns:
            List of dicts with file, line, content
        """
        import subprocess
        from pathlib import Path
        
        results = []
        
        # Try ripgrep first (much faster)
        try:
            rg_cmd = ["rg", "--json", "-n", pattern, path, "--glob", include]
            result = subprocess.run(rg_cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        try:
                            match = eval(line)  # JSON lines format
                            if match.get("type") == "match":
                                results.append({
                                    "file": match["data"]["path"]["text"],
                                    "line": match["data"]["line_number"],
                                    "content": match["data"]["lines"].get("text", ""),
                                })
                                if len(results) >= max_results:
                                    break
                        except:
                            pass
                return results
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Fallback to Python glob + grep
        base = Path(path).expanduser()
        for f in base.glob("**/" + include.replace("*", "")):
            if f.is_file():
                try:
                    content = f.read_text()
                    for i, line in enumerate(content.split("\n"), 1):
                        if pattern.lower() in line.lower():
                            results.append({
                                "file": str(f.relative_to(base)),
                                "line": i,
                                "content": line.strip()[:200],
                            })
                            if len(results) >= max_results:
                                return results
                except:
                    pass
        
        return results
    
    def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search the web using available API.
        
        Tries in order:
        1. Tavily API (if TAVILY_API_KEY env var set)
        2. Serper API (if SERPER_API_KEY env var set)
        3. DuckDuckGo (no API key needed)
        4. Fallback mock results
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of dicts with title, url, snippet
        """
        import os
        
        # Try Tavily API
        api_key = os.environ.get("TAVILY_API_KEY")
        if api_key:
            try:
                import requests
                resp = requests.post(
                    "https://api.tavily.com/search",
                    json={"api_key": api_key, "query": query, "num_results": num_results},
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return [
                        {"title": r.get("title", ""), "url": r["url"], "snippet": r.get("content", "")}
                        for r in data.get("results", [])[:num_results]
                    ]
            except:
                pass
        
        # Try Serper API
        api_key = os.environ.get("SERPER_API_KEY")
        if api_key:
            try:
                import requests
                resp = requests.post(
                    "https://google.serper.dev/search",
                    json={"q": query, "num": num_results},
                    headers={"X-API-KEY": api_key},
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return [
                        {"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")}
                        for r in data.get("organic", [])[:num_results]
                    ]
            except:
                pass
        
        # Try DuckDuckGo (HTML scraping - basic)
        try:
            import requests
            resp = requests.get(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10
            )
            if resp.status_code == 200:
                # Basic parsing of DuckDuckGo HTML
                results = []
                for line in resp.text.split("\n"):
                    if 'class="result__a"' in line:
                        # Extract title and URL
                        import re
                        match = re.search(r'href="([^"]+)".*?>([^<]+)', line)
                        if match:
                            results.append({
                                "title": match.group(2),
                                "url": match.group(1),
                                "snippet": "",
                            })
                            if len(results) >= num_results:
                                return results
        except:
            pass
        
        # Fallback - return mock results
        return [{"title": f"Result for: {query}", "url": "https://example.com", "snippet": "No search API configured"}]
    
    def get_file_info(path: str) -> Dict[str, Any]:
        """Get detailed file information."""
        from pathlib import Path
        import os
        from datetime import datetime
        
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        stat = p.stat()
        return {
            "name": p.name,
            "path": str(p.absolute()),
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "is_file": p.is_file(),
            "is_dir": p.is_dir(),
            "is_symlink": p.is_symlink(),
            "permissions": oct(stat.st_mode)[-3:],
        }
    
    return {
        "read_file": read_file,
        "write_file": write_file,
        "list_files": list_files,
        "run_bash": run_bash,
        "run_python": run_python,
        "search_files": search_files,
        "search_grep": search_grep,
        "search_web": search_web,
        "get_file_info": get_file_info,
    }
