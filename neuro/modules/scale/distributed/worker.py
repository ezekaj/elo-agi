"""
Worker: Parallel task execution.

Implements worker processes for distributed computing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import threading
import time
import uuid
import traceback


class WorkerStatus(Enum):
    """Worker status."""

    IDLE = "idle"
    BUSY = "busy"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class WorkResult:
    """Result from a worker execution."""

    task_id: str
    worker_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_used: int = 0  # bytes


@dataclass
class WorkerConfig:
    """Worker configuration."""

    name: Optional[str] = None
    max_concurrent: int = 1
    heartbeat_interval: float = 5.0
    max_idle_time: float = 300.0


class Worker:
    """
    A worker that executes tasks.

    Implements:
    - Task execution
    - Heartbeat reporting
    - Error handling
    - Resource monitoring
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        config: Optional[WorkerConfig] = None,
    ):
        self.id = worker_id or str(uuid.uuid4())[:8]
        self.config = config or WorkerConfig()

        self._status = WorkerStatus.IDLE
        self._current_task: Optional[str] = None
        self._task_count = 0
        self._error_count = 0

        # Threading
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Results storage
        self._results: Dict[str, WorkResult] = {}

        # Callbacks
        self._on_complete: Optional[Callable[[WorkResult], None]] = None
        self._on_heartbeat: Optional[Callable[[str, float], None]] = None

    @property
    def status(self) -> WorkerStatus:
        """Get current status."""
        return self._status

    @property
    def load(self) -> float:
        """Get current load (0-1)."""
        if self._status == WorkerStatus.BUSY:
            return 1.0
        return 0.0

    def set_callbacks(
        self,
        on_complete: Optional[Callable[[WorkResult], None]] = None,
        on_heartbeat: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        """Set callback functions."""
        self._on_complete = on_complete
        self._on_heartbeat = on_heartbeat

    def execute(
        self,
        task_id: str,
        function: Callable,
        args: tuple = (),
        kwargs: Optional[Dict] = None,
    ) -> WorkResult:
        """
        Execute a task.

        Args:
            task_id: Task identifier
            function: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            WorkResult with execution outcome
        """
        with self._lock:
            self._status = WorkerStatus.BUSY
            self._current_task = task_id

        kwargs = kwargs or {}
        start_time = time.time()

        try:
            result = function(*args, **kwargs)
            success = True
            error = None
            self._task_count += 1
        except Exception as e:
            result = None
            success = False
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self._error_count += 1

        execution_time = time.time() - start_time

        work_result = WorkResult(
            task_id=task_id,
            worker_id=self.id,
            success=success,
            result=result,
            error=error,
            execution_time=execution_time,
        )

        with self._lock:
            self._status = WorkerStatus.IDLE
            self._current_task = None
            self._results[task_id] = work_result

        # Callback
        if self._on_complete:
            self._on_complete(work_result)

        return work_result

    def get_result(self, task_id: str) -> Optional[WorkResult]:
        """Get result for a task."""
        return self._results.get(task_id)

    def pause(self) -> None:
        """Pause the worker."""
        self._status = WorkerStatus.PAUSED

    def resume(self) -> None:
        """Resume the worker."""
        if self._status == WorkerStatus.PAUSED:
            self._status = WorkerStatus.IDLE

    def stop(self) -> None:
        """Stop the worker."""
        self._stop_event.set()
        self._status = WorkerStatus.STOPPED

    def send_heartbeat(self) -> None:
        """Send heartbeat."""
        if self._on_heartbeat:
            self._on_heartbeat(self.id, self.load)

    def statistics(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "id": self.id,
            "status": self._status.value,
            "tasks_completed": self._task_count,
            "errors": self._error_count,
            "current_task": self._current_task,
        }


class WorkerPool:
    """
    Pool of workers for parallel execution.

    Implements:
    - Worker management
    - Task distribution
    - Result collection
    """

    def __init__(
        self,
        n_workers: int = 4,
        config: Optional[WorkerConfig] = None,
    ):
        self.n_workers = n_workers
        self.config = config or WorkerConfig()

        # Create workers
        self._workers: Dict[str, Worker] = {}
        for i in range(n_workers):
            worker = Worker(worker_id=f"worker_{i}", config=config)
            self._workers[worker.id] = worker

        # Task queue
        self._pending_tasks: List[Dict[str, Any]] = []

        # Results
        self._results: Dict[str, WorkResult] = {}

        # Threading
        self._lock = threading.Lock()

    def submit(
        self,
        task_id: str,
        function: Callable,
        args: tuple = (),
        kwargs: Optional[Dict] = None,
    ) -> None:
        """Submit a task to the pool."""
        with self._lock:
            self._pending_tasks.append(
                {
                    "task_id": task_id,
                    "function": function,
                    "args": args,
                    "kwargs": kwargs or {},
                }
            )

    def execute_all(self) -> List[WorkResult]:
        """
        Execute all pending tasks.

        Returns:
            List of results
        """
        results = []

        with self._lock:
            tasks = self._pending_tasks.copy()
            self._pending_tasks.clear()

        # Simple round-robin distribution
        worker_list = list(self._workers.values())
        n_workers = len(worker_list)

        for i, task in enumerate(tasks):
            worker = worker_list[i % n_workers]
            result = worker.execute(
                task["task_id"],
                task["function"],
                task["args"],
                task["kwargs"],
            )
            results.append(result)
            self._results[task["task_id"]] = result

        return results

    def map(
        self,
        function: Callable,
        items: List[Any],
    ) -> List[WorkResult]:
        """
        Map a function over items using the pool.

        Args:
            function: Function to apply
            items: Items to process

        Returns:
            List of results
        """
        # Submit all tasks
        for i, item in enumerate(items):
            self.submit(f"map_{i}", function, (item,))

        # Execute
        return self.execute_all()

    def get_result(self, task_id: str) -> Optional[WorkResult]:
        """Get result for a task."""
        return self._results.get(task_id)

    def get_worker(self, worker_id: str) -> Optional[Worker]:
        """Get a worker by ID."""
        return self._workers.get(worker_id)

    def get_idle_workers(self) -> List[Worker]:
        """Get all idle workers."""
        return [w for w in self._workers.values() if w.status == WorkerStatus.IDLE]

    def shutdown(self) -> None:
        """Shutdown all workers."""
        for worker in self._workers.values():
            worker.stop()

    def statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        worker_stats = [w.statistics() for w in self._workers.values()]

        return {
            "n_workers": self.n_workers,
            "pending_tasks": len(self._pending_tasks),
            "completed_tasks": len(self._results),
            "workers": worker_stats,
            "total_errors": sum(w.statistics()["errors"] for w in self._workers.values()),
        }
