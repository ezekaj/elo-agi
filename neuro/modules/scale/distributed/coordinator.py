"""
Coordinator: Distribute tasks across workers.

Implements task scheduling, load balancing, and fault tolerance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import threading
import queue
import time
import numpy as np


class TaskStatus(Enum):
    """Status of a distributed task."""

    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SchedulingPolicy(Enum):
    """Task scheduling policies."""

    FIFO = "fifo"  # First in, first out
    PRIORITY = "priority"  # Priority-based
    LOAD_BALANCED = "load_balanced"  # Balance across workers
    LOCALITY = "locality"  # Prefer local data


@dataclass
class CoordinatorConfig:
    """Configuration for task coordinator."""

    max_workers: int = 4
    max_queue_size: int = 1000
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.LOAD_BALANCED
    retry_limit: int = 3
    timeout_seconds: float = 60.0
    heartbeat_interval: float = 5.0


@dataclass
class DistributedTask:
    """A task to be distributed across workers."""

    id: str
    name: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    worker_id: Optional[str] = None
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0


@dataclass
class WorkerInfo:
    """Information about a worker."""

    id: str
    status: str
    load: float  # 0-1
    tasks_running: int
    tasks_completed: int
    last_heartbeat: float


class TaskQueue:
    """
    Priority queue for tasks.
    """

    def __init__(self, max_size: int = 1000):
        self._queue: List[DistributedTask] = []
        self._max_size = max_size
        self._lock = threading.Lock()

    def put(self, task: DistributedTask) -> bool:
        """Add a task to the queue."""
        with self._lock:
            if len(self._queue) >= self._max_size:
                return False

            # Insert based on priority
            inserted = False
            for i, t in enumerate(self._queue):
                if task.priority > t.priority:
                    self._queue.insert(i, task)
                    inserted = True
                    break

            if not inserted:
                self._queue.append(task)

            return True

    def get(self) -> Optional[DistributedTask]:
        """Get the highest priority task."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    def peek(self) -> Optional[DistributedTask]:
        """Peek at the highest priority task."""
        with self._lock:
            return self._queue[0] if self._queue else None

    def size(self) -> int:
        """Get queue size."""
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        """Clear the queue."""
        with self._lock:
            self._queue.clear()


class LoadBalancer:
    """
    Balance load across workers.
    """

    def __init__(self):
        self._worker_loads: Dict[str, float] = {}

    def update_load(self, worker_id: str, load: float) -> None:
        """Update worker load."""
        self._worker_loads[worker_id] = load

    def remove_worker(self, worker_id: str) -> None:
        """Remove a worker."""
        self._worker_loads.pop(worker_id, None)

    def select_worker(
        self,
        available_workers: List[str],
        task: Optional[DistributedTask] = None,
    ) -> Optional[str]:
        """Select the best worker for a task."""
        if not available_workers:
            return None

        # Select worker with lowest load
        min_load = float("inf")
        best_worker = None

        for worker_id in available_workers:
            load = self._worker_loads.get(worker_id, 0.0)
            if load < min_load:
                min_load = load
                best_worker = worker_id

        return best_worker

    def get_worker_loads(self) -> Dict[str, float]:
        """Get all worker loads."""
        return self._worker_loads.copy()


class TaskCoordinator:
    """
    Coordinate task distribution across workers.

    Implements:
    - Task scheduling
    - Load balancing
    - Dependency resolution
    - Fault tolerance
    """

    def __init__(self, config: Optional[CoordinatorConfig] = None):
        self.config = config or CoordinatorConfig()

        self._task_queue = TaskQueue(self.config.max_queue_size)
        self._load_balancer = LoadBalancer()

        # Task storage
        self._tasks: Dict[str, DistributedTask] = {}
        self._completed_tasks: Dict[str, DistributedTask] = {}

        # Worker management
        self._workers: Dict[str, WorkerInfo] = {}
        self._worker_tasks: Dict[str, Set[str]] = {}

        # Counters
        self._task_counter = 0

        # Thread safety
        self._lock = threading.Lock()

    def register_worker(
        self,
        worker_id: str,
        initial_load: float = 0.0,
    ) -> WorkerInfo:
        """Register a new worker."""
        with self._lock:
            info = WorkerInfo(
                id=worker_id,
                status="idle",
                load=initial_load,
                tasks_running=0,
                tasks_completed=0,
                last_heartbeat=time.time(),
            )
            self._workers[worker_id] = info
            self._worker_tasks[worker_id] = set()
            self._load_balancer.update_load(worker_id, initial_load)
            return info

    def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker."""
        with self._lock:
            self._workers.pop(worker_id, None)
            self._worker_tasks.pop(worker_id, None)
            self._load_balancer.remove_worker(worker_id)

    def submit_task(
        self,
        name: str,
        function: Callable,
        args: tuple = (),
        kwargs: Optional[Dict] = None,
        priority: int = 0,
        dependencies: Optional[List[str]] = None,
    ) -> DistributedTask:
        """
        Submit a task for distribution.

        Returns:
            The created task
        """
        with self._lock:
            self._task_counter += 1
            task_id = f"task_{self._task_counter}"

            task = DistributedTask(
                id=task_id,
                name=name,
                function=function,
                args=args,
                kwargs=kwargs or {},
                priority=priority,
                dependencies=dependencies or [],
            )

            self._tasks[task_id] = task

        # Queue if ready
        if self._check_dependencies(task):
            self._task_queue.put(task)

        return task

    def _check_dependencies(self, task: DistributedTask) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            if dep_id not in self._completed_tasks:
                return False
        return True

    def schedule_next(self) -> Optional[DistributedTask]:
        """
        Schedule the next task to a worker.

        Returns:
            The scheduled task, or None if no tasks/workers available
        """
        with self._lock:
            # Get available workers
            available = [
                w_id
                for w_id, info in self._workers.items()
                if info.load < 1.0 and info.status != "busy"
            ]

            if not available:
                return None

            # Get next task
            task = self._task_queue.get()
            if task is None:
                return None

            # Select worker
            worker_id = self._load_balancer.select_worker(available, task)
            if worker_id is None:
                # Put task back
                self._task_queue.put(task)
                return None

            # Assign task
            task.status = TaskStatus.SCHEDULED
            task.worker_id = worker_id
            self._worker_tasks[worker_id].add(task.id)

            # Update worker info
            worker = self._workers[worker_id]
            worker.tasks_running += 1
            worker.load = min(1.0, worker.load + 0.25)
            self._load_balancer.update_load(worker_id, worker.load)

            return task

    def start_task(self, task_id: str) -> None:
        """Mark a task as started."""
        with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.RUNNING
                task.started_at = time.time()

    def complete_task(
        self,
        task_id: str,
        result: Any,
        error: Optional[str] = None,
    ) -> None:
        """Mark a task as completed."""
        with self._lock:
            if task_id not in self._tasks:
                return

            task = self._tasks[task_id]
            task.completed_at = time.time()

            if error:
                task.status = TaskStatus.FAILED
                task.error = error

                # Retry if under limit
                if task.retry_count < self.config.retry_limit:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    self._task_queue.put(task)
            else:
                task.status = TaskStatus.COMPLETED
                task.result = result
                self._completed_tasks[task_id] = task

            # Update worker
            if task.worker_id:
                worker = self._workers.get(task.worker_id)
                if worker:
                    worker.tasks_running -= 1
                    worker.tasks_completed += 1
                    worker.load = max(0.0, worker.load - 0.25)
                    self._load_balancer.update_load(task.worker_id, worker.load)

                self._worker_tasks[task.worker_id].discard(task_id)

            # Check for newly unblocked tasks
            self._check_unblocked_tasks()

    def _check_unblocked_tasks(self) -> None:
        """Check for tasks that are now unblocked."""
        for task in self._tasks.values():
            if task.status == TaskStatus.PENDING:
                if self._check_dependencies(task):
                    self._task_queue.put(task)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]
            if task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]:
                return False

            task.status = TaskStatus.CANCELLED
            return True

    def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_result(self, task_id: str) -> Any:
        """Get the result of a completed task."""
        task = self._completed_tasks.get(task_id) or self._tasks.get(task_id)
        if task and task.status == TaskStatus.COMPLETED:
            return task.result
        return None

    def update_heartbeat(self, worker_id: str, load: float) -> None:
        """Update worker heartbeat."""
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].last_heartbeat = time.time()
                self._workers[worker_id].load = load
                self._load_balancer.update_load(worker_id, load)

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        with self._lock:
            return {
                "n_workers": len(self._workers),
                "workers": {
                    w_id: {
                        "status": info.status,
                        "load": info.load,
                        "tasks_running": info.tasks_running,
                        "tasks_completed": info.tasks_completed,
                    }
                    for w_id, info in self._workers.items()
                },
            }

    def get_task_stats(self) -> Dict[str, Any]:
        """Get task statistics."""
        with self._lock:
            status_counts = {}
            for task in self._tasks.values():
                status = task.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            return {
                "total_tasks": len(self._tasks),
                "completed_tasks": len(self._completed_tasks),
                "queue_size": self._task_queue.size(),
                "by_status": status_counts,
            }

    def shutdown(self) -> None:
        """Shutdown the coordinator."""
        with self._lock:
            self._task_queue.clear()
            for worker_id in list(self._workers.keys()):
                self.unregister_worker(worker_id)
