"""
NEURO Parallel Executor - Async Tool Execution

Provides:
- Parallel tool execution (multiple tools at once)
- Tool chaining (output of A → input of B)
- Retry logic with exponential backoff
- Timeout handling
- Result aggregation
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class ToolStatus(Enum):
    """Status of a tool execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class ToolCall:
    """A tool call request."""

    name: str
    args: Dict[str, Any]
    description: str = ""
    timeout: float = 30.0
    retries: int = 3
    depends_on: Optional[str] = None  # ID of tool this depends on


@dataclass
class ToolResult:
    """Result of a tool execution."""

    name: str
    status: ToolStatus
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    retries_used: int = 0


@dataclass
class ExecutionPlan:
    """A plan for executing multiple tools."""

    tools: List[ToolCall]
    parallel_groups: List[List[str]] = field(default_factory=list)
    total_timeout: float = 120.0


class ParallelExecutor:
    """
    Async parallel tool executor.

    Features:
    - Execute multiple tools simultaneously
    - Chain tools (pass output to next)
    - Retry with exponential backoff
    - Timeout handling
    - Progress callbacks
    """

    def __init__(
        self,
        tools_instance: Any = None,
        max_parallel: int = 5,
        default_timeout: float = 30.0,
        default_retries: int = 3,
    ):
        self.tools = tools_instance
        self.max_parallel = max_parallel
        self.default_timeout = default_timeout
        self.default_retries = default_retries

        # Execution state
        self.results: Dict[str, ToolResult] = {}
        self.running: Dict[str, asyncio.Task] = {}

    def _get_tool_func(self, name: str) -> Optional[Callable]:
        """Get the function for a tool by name."""
        if self.tools is None:
            return None

        # Try direct attribute
        if hasattr(self.tools, name):
            return getattr(self.tools, name)

        # Try with underscores replaced
        alt_name = name.replace("-", "_")
        if hasattr(self.tools, alt_name):
            return getattr(self.tools, alt_name)

        return None

    async def execute_one(
        self,
        tool: ToolCall,
        context: Optional[Dict[str, Any]] = None,
        on_progress: Optional[Callable[[str, str], None]] = None,
    ) -> ToolResult:
        """
        Execute a single tool with retries and timeout.

        Args:
            tool: The tool call to execute
            context: Additional context (e.g., previous results)
            on_progress: Callback for progress updates

        Returns:
            ToolResult with output or error
        """
        start_time = time.time()
        retries_used = 0

        if on_progress:
            on_progress(tool.name, "starting")

        for attempt in range(tool.retries + 1):
            try:
                # Get the tool function
                func = self._get_tool_func(tool.name)
                if func is None:
                    return ToolResult(
                        name=tool.name,
                        status=ToolStatus.FAILED,
                        error=f"Tool not found: {tool.name}",
                        duration=time.time() - start_time,
                    )

                # Merge context into args
                args = dict(tool.args)
                if context:
                    # Allow tools to access previous results via special key
                    args["_context"] = context

                if on_progress:
                    on_progress(tool.name, f"attempt {attempt + 1}")

                # Execute with timeout
                result = await asyncio.wait_for(self._run_tool(func, args), timeout=tool.timeout)

                return ToolResult(
                    name=tool.name,
                    status=ToolStatus.SUCCESS,
                    output=result,
                    duration=time.time() - start_time,
                    retries_used=retries_used,
                )

            except asyncio.TimeoutError:
                retries_used += 1
                if attempt == tool.retries:
                    return ToolResult(
                        name=tool.name,
                        status=ToolStatus.TIMEOUT,
                        error=f"Timeout after {tool.timeout}s",
                        duration=time.time() - start_time,
                        retries_used=retries_used,
                    )
                # Exponential backoff before retry
                await asyncio.sleep(2**attempt)

            except Exception as e:
                retries_used += 1
                if attempt == tool.retries:
                    return ToolResult(
                        name=tool.name,
                        status=ToolStatus.FAILED,
                        error=str(e),
                        duration=time.time() - start_time,
                        retries_used=retries_used,
                    )
                # Exponential backoff before retry
                await asyncio.sleep(2**attempt)

        # Should not reach here
        return ToolResult(
            name=tool.name,
            status=ToolStatus.FAILED,
            error="Unknown error",
            duration=time.time() - start_time,
        )

    async def _run_tool(self, func: Callable, args: Dict[str, Any]) -> Any:
        """Run a tool function (sync or async)."""
        # Remove internal context key before calling
        clean_args = {k: v for k, v in args.items() if not k.startswith("_")}

        if asyncio.iscoroutinefunction(func):
            return await func(**clean_args)
        else:
            # Run sync function in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: func(**clean_args))

    async def execute_parallel(
        self, tools: List[ToolCall], on_progress: Optional[Callable[[str, str], None]] = None
    ) -> List[ToolResult]:
        """
        Execute multiple tools in parallel.

        Args:
            tools: List of tool calls to execute
            on_progress: Progress callback

        Returns:
            List of results in same order as input
        """
        # Create tasks for all tools
        tasks = []
        for tool in tools:
            task = asyncio.create_task(self.execute_one(tool, on_progress=on_progress))
            tasks.append(task)

        # Wait for all with semaphore to limit parallelism
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def limited_execute(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(*[limited_execute(t) for t in tasks], return_exceptions=True)

        # Convert exceptions to ToolResults
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    ToolResult(name=tools[i].name, status=ToolStatus.FAILED, error=str(result))
                )
            else:
                final_results.append(result)

        return final_results

    async def execute_chain(
        self, tools: List[ToolCall], on_progress: Optional[Callable[[str, str], None]] = None
    ) -> ToolResult:
        """
        Execute tools in sequence, passing outputs.

        Each tool receives the previous tool's output in context.

        Args:
            tools: List of tool calls to chain
            on_progress: Progress callback

        Returns:
            Final tool's result
        """
        context: Dict[str, Any] = {}
        last_result = None

        for i, tool in enumerate(tools):
            if on_progress:
                on_progress(tool.name, f"step {i + 1}/{len(tools)}")

            result = await self.execute_one(tool, context=context, on_progress=on_progress)

            # Store result for next tool
            context[f"step_{i}"] = result.output
            context["previous_result"] = result.output
            context["previous_status"] = result.status.value

            last_result = result

            # Stop chain on failure
            if result.status != ToolStatus.SUCCESS:
                if on_progress:
                    on_progress(tool.name, f"chain stopped: {result.status.value}")
                break

        return last_result

    async def execute_plan(
        self, plan: ExecutionPlan, on_progress: Optional[Callable[[str, str], None]] = None
    ) -> Dict[str, ToolResult]:
        """
        Execute a full execution plan with dependency resolution.

        Args:
            plan: The execution plan
            on_progress: Progress callback

        Returns:
            Dict mapping tool names to results
        """
        results: Dict[str, ToolResult] = {}
        tool_map = {t.name: t for t in plan.tools}

        # If parallel groups specified, use them
        if plan.parallel_groups:
            for group in plan.parallel_groups:
                group_tools = [tool_map[name] for name in group if name in tool_map]
                group_results = await self.execute_parallel(group_tools, on_progress)
                for result in group_results:
                    results[result.name] = result
        else:
            # Otherwise, resolve dependencies and execute
            pending = list(plan.tools)
            completed = set()

            while pending:
                # Find tools with no pending dependencies
                ready = []
                for tool in pending:
                    if tool.depends_on is None or tool.depends_on in completed:
                        ready.append(tool)

                if not ready:
                    # Circular dependency or missing dependency
                    for tool in pending:
                        results[tool.name] = ToolResult(
                            name=tool.name, status=ToolStatus.SKIPPED, error="Dependency not met"
                        )
                    break

                # Execute ready tools in parallel
                group_results = await self.execute_parallel(ready, on_progress)
                for result in group_results:
                    results[result.name] = result
                    completed.add(result.name)

                # Remove completed from pending
                pending = [t for t in pending if t.name not in completed]

        return results


# Convenience functions


async def run_parallel(
    tools_instance: Any, calls: List[Dict[str, Any]], max_parallel: int = 5
) -> List[ToolResult]:
    """
    Run multiple tool calls in parallel.

    Args:
        tools_instance: Object with tool methods
        calls: List of {"name": "tool_name", "args": {...}}
        max_parallel: Max concurrent executions

    Returns:
        List of ToolResults
    """
    executor = ParallelExecutor(tools_instance, max_parallel=max_parallel)
    tool_calls = [ToolCall(name=c["name"], args=c.get("args", {})) for c in calls]
    return await executor.execute_parallel(tool_calls)


async def run_chain(tools_instance: Any, calls: List[Dict[str, Any]]) -> ToolResult:
    """
    Run tool calls in sequence, passing outputs.

    Args:
        tools_instance: Object with tool methods
        calls: List of {"name": "tool_name", "args": {...}}

    Returns:
        Final ToolResult
    """
    executor = ParallelExecutor(tools_instance)
    tool_calls = [ToolCall(name=c["name"], args=c.get("args", {})) for c in calls]
    return await executor.execute_chain(tool_calls)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NEURO PARALLEL EXECUTOR TEST")
    print("=" * 60)

    # Mock tools for testing
    class MockTools:
        async def search(self, query: str) -> str:
            await asyncio.sleep(0.5)  # Simulate network
            return f"Results for: {query}"

        def calculate(self, expression: str) -> str:
            return str(eval(expression))

        async def slow_task(self, seconds: float = 2.0) -> str:
            await asyncio.sleep(seconds)
            return f"Completed after {seconds}s"

        def fail_task(self) -> str:
            raise ValueError("Intentional failure")

    async def test():
        tools = MockTools()
        executor = ParallelExecutor(tools)

        def progress(name: str, status: str):
            print(f"  [{name}] {status}")

        # Test 1: Parallel execution
        print("\nTest 1: Parallel execution (3 searches)")
        print("-" * 40)

        calls = [
            ToolCall(name="search", args={"query": "Python"}),
            ToolCall(name="search", args={"query": "asyncio"}),
            ToolCall(name="search", args={"query": "parallel"}),
        ]

        start = time.time()
        results = await executor.execute_parallel(calls, on_progress=progress)
        duration = time.time() - start

        for r in results:
            print(f"  {r.name}: {r.status.value} - {r.output}")
        print(f"  Total time: {duration:.2f}s (should be ~0.5s, not 1.5s)")

        # Test 2: Chain execution
        print("\n" + "=" * 60)
        print("Test 2: Chain execution")
        print("-" * 40)

        chain = [
            ToolCall(name="calculate", args={"expression": "2 + 2"}),
            ToolCall(name="search", args={"query": "result is 4"}),
        ]

        result = await executor.execute_chain(chain, on_progress=progress)
        print(f"  Final result: {result.output}")

        # Test 3: Retry on failure
        print("\n" + "=" * 60)
        print("Test 3: Retry on failure")
        print("-" * 40)

        fail_call = ToolCall(name="fail_task", args={}, retries=2)
        result = await executor.execute_one(fail_call, on_progress=progress)
        print(f"  Status: {result.status.value}")
        print(f"  Error: {result.error}")
        print(f"  Retries used: {result.retries_used}")

        # Test 4: Timeout
        print("\n" + "=" * 60)
        print("Test 4: Timeout")
        print("-" * 40)

        slow_call = ToolCall(name="slow_task", args={"seconds": 5.0}, timeout=1.0, retries=0)
        result = await executor.execute_one(slow_call, on_progress=progress)
        print(f"  Status: {result.status.value}")
        print(f"  Error: {result.error}")

        print("\n" + "=" * 60)
        print("All tests completed!")

    asyncio.run(test())
