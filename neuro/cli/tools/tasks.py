"""
Task Tracking System - Track multi-step work with dependencies.

Provides:
- In-memory task storage with JSON persistence
- Task CRUD operations exposed as tools
- Dependency tracking (blockedBy/blocks)
- Status workflow: pending -> in_progress -> completed
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Set
from datetime import datetime
from enum import Enum
import json
import os


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELETED = "deleted"


@dataclass
class Task:
    id: str
    subject: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    active_form: str = ""
    owner: str = ""
    blocked_by: Set[str] = field(default_factory=set)
    blocks: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "description": self.description,
            "status": self.status.value,
            "active_form": self.active_form,
            "owner": self.owner,
            "blocked_by": sorted(self.blocked_by),
            "blocks": sorted(self.blocks),
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Task":
        return cls(
            id=data["id"],
            subject=data["subject"],
            description=data.get("description", ""),
            status=TaskStatus(data.get("status", "pending")),
            active_form=data.get("active_form", ""),
            owner=data.get("owner", ""),
            blocked_by=set(data.get("blocked_by", [])),
            blocks=set(data.get("blocks", [])),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    def is_blocked(self) -> bool:
        return len(self.blocked_by) > 0


class TaskManager:
    """Manages tasks with persistence."""

    TASKS_FILE = ".neuro/tasks.json"

    def __init__(self, project_dir: str = "."):
        self.project_dir = os.path.abspath(project_dir)
        self._tasks: Dict[str, Task] = {}
        self._next_id = 1
        self._load()

    def _tasks_path(self) -> str:
        return os.path.join(self.project_dir, self.TASKS_FILE)

    def _load(self):
        path = self._tasks_path()
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                for task_data in data.get("tasks", []):
                    task = Task.from_dict(task_data)
                    self._tasks[task.id] = task
                self._next_id = data.get("next_id", len(self._tasks) + 1)
            except Exception:
                pass

    def _save(self):
        path = self._tasks_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "tasks": [t.to_dict() for t in self._tasks.values() if t.status != TaskStatus.DELETED],
            "next_id": self._next_id,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def task_create(
        self,
        subject: str,
        description: str = "",
        active_form: str = "",
    ) -> str:
        """Create a new task."""
        task_id = str(self._next_id)
        self._next_id += 1

        task = Task(
            id=task_id,
            subject=subject,
            description=description,
            active_form=active_form or f"Working on: {subject}",
        )
        self._tasks[task_id] = task
        self._save()
        return json.dumps({"id": task_id, "subject": subject, "status": "pending"})

    def task_update(
        self,
        task_id: str,
        status: str = "",
        subject: str = "",
        description: str = "",
        active_form: str = "",
        owner: str = "",
        add_blocked_by: str = "",
        add_blocks: str = "",
    ) -> str:
        """Update a task."""
        task = self._tasks.get(task_id)
        if not task:
            return json.dumps({"error": f"Task {task_id} not found"})

        if status:
            if status == "deleted":
                task.status = TaskStatus.DELETED
            else:
                task.status = TaskStatus(status)

            if task.status == TaskStatus.COMPLETED:
                for other in self._tasks.values():
                    other.blocked_by.discard(task_id)

        if subject:
            task.subject = subject
        if description:
            task.description = description
        if active_form:
            task.active_form = active_form
        if owner:
            task.owner = owner
        if add_blocked_by:
            for bid in add_blocked_by.split(","):
                bid = bid.strip()
                if bid in self._tasks:
                    task.blocked_by.add(bid)
        if add_blocks:
            for bid in add_blocks.split(","):
                bid = bid.strip()
                if bid in self._tasks:
                    task.blocks.add(bid)
                    self._tasks[bid].blocked_by.add(task_id)

        task.updated_at = datetime.now().isoformat()
        self._save()
        return json.dumps(task.to_dict())

    def task_list(self) -> str:
        """List all non-deleted tasks."""
        tasks = [t for t in self._tasks.values() if t.status != TaskStatus.DELETED]
        if not tasks:
            return "No tasks."

        result = []
        for t in tasks:
            blocked = f" [blocked by: {', '.join(sorted(t.blocked_by))}]" if t.blocked_by else ""
            owner = f" @{t.owner}" if t.owner else ""
            result.append(f"[{t.id}] {t.status.value:12s} | {t.subject}{blocked}{owner}")
        return "\n".join(result)

    def task_get(self, task_id: str) -> str:
        """Get full task details."""
        task = self._tasks.get(task_id)
        if not task:
            return json.dumps({"error": f"Task {task_id} not found"})
        return json.dumps(task.to_dict(), indent=2)

    def get_active_tasks(self) -> List[Task]:
        """Get tasks that are in_progress."""
        return [t for t in self._tasks.values() if t.status == TaskStatus.IN_PROGRESS]

    def get_summary(self) -> Dict[str, int]:
        """Get task count by status."""
        summary = {"pending": 0, "in_progress": 0, "completed": 0}
        for t in self._tasks.values():
            if t.status != TaskStatus.DELETED and t.status.value in summary:
                summary[t.status.value] += 1
        return summary
