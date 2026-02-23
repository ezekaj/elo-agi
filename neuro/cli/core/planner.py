"""
Plan Mode - Read-only research and planning system.

Implements Claude Code-style plan mode:
1. Enters restricted mode (read-only tools only)
2. AI researches and explores the codebase
3. Generates a structured plan
4. User reviews and approves
5. On approval, plan is fed as context for execution
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum
import os
import uuid


class PlanStatus(Enum):
    RESEARCHING = "researching"
    DRAFT = "draft"
    APPROVED = "approved"
    EXECUTING = "executing"
    COMPLETED = "completed"
    REJECTED = "rejected"


@dataclass
class PlanStep:
    description: str
    files: List[str] = field(default_factory=list)
    tools_needed: List[str] = field(default_factory=list)
    completed: bool = False


@dataclass
class Plan:
    id: str
    task: str
    status: PlanStatus
    steps: List[PlanStep] = field(default_factory=list)
    research_notes: str = ""
    created_at: str = ""
    approved_at: str = ""
    file_path: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_markdown(self) -> str:
        lines = [
            f"# Plan: {self.task}",
            "",
            f"**Status:** {self.status.value}",
            f"**Created:** {self.created_at}",
            "",
            "## Context",
            "",
            self.research_notes or "(researching...)",
            "",
            "## Steps",
            "",
        ]
        for i, step in enumerate(self.steps, 1):
            check = "[x]" if step.completed else "[ ]"
            lines.append(f"{i}. {check} {step.description}")
            for f in step.files:
                lines.append(f"   - File: `{f}`")
        if not self.steps:
            lines.append("(to be determined)")
        return "\n".join(lines)

    def to_context(self) -> str:
        return (
            f"APPROVED PLAN FOR: {self.task}\n\n"
            f"RESEARCH:\n{self.research_notes}\n\n"
            f"STEPS:\n"
            + "\n".join(f"{i}. {s.description}" for i, s in enumerate(self.steps, 1))
        )


class PlanManager:
    PLANS_DIR = ".neuro/plans"

    PLAN_SYSTEM_PROMPT = """You are in PLAN MODE. You can ONLY use read-only tools:
- read_file: Read file contents
- list_files: List directory contents
- glob_files: Find files by pattern
- grep_content: Search file contents
- web_search: Search the web
- task_list: View existing tasks
- task_get: View task details

You CANNOT modify any files, run commands, or make changes.

Your job is to:
1. Research the codebase thoroughly
2. Understand the existing architecture and patterns
3. Identify all files that need to change
4. Write a detailed step-by-step implementation plan

Output your plan in this format:

## Research Notes
<what you found about the codebase>

## Steps
1. <step description>
   - Files: <files to modify or create>
2. <step description>
   - Files: <files to modify or create>
...

## Verification
<how to test the changes>
"""

    def __init__(self, project_dir: str = "."):
        self.project_dir = os.path.abspath(project_dir)
        self.plans_dir = os.path.join(self.project_dir, self.PLANS_DIR)
        self._current_plan: Optional[Plan] = None

    def create_plan(self, task: str) -> Plan:
        plan_id = str(uuid.uuid4())[:8]
        os.makedirs(self.plans_dir, exist_ok=True)

        plan = Plan(
            id=plan_id,
            task=task,
            status=PlanStatus.RESEARCHING,
            file_path=os.path.join(self.plans_dir, f"{plan_id}.md"),
        )
        self._current_plan = plan
        self.save_plan(plan)
        return plan

    def save_plan(self, plan: Optional[Plan] = None):
        plan = plan or self._current_plan
        if plan:
            os.makedirs(os.path.dirname(plan.file_path), exist_ok=True)
            with open(plan.file_path, "w") as f:
                f.write(plan.to_markdown())

    def update_plan_from_response(self, plan: Plan, response: str):
        """Parse AI response and update plan research_notes and steps."""
        plan.research_notes = response

        import re

        steps = []
        step_matches = re.findall(
            r"^\d+\.\s+(.+?)(?:\n\s+- Files?:\s*(.+))?$",
            response,
            re.MULTILINE,
        )
        for desc, files_str in step_matches:
            files = []
            if files_str:
                files = [f.strip().strip("`") for f in files_str.split(",")]
            steps.append(PlanStep(description=desc.strip(), files=files))

        if steps:
            plan.steps = steps

        plan.status = PlanStatus.DRAFT
        self.save_plan(plan)

    def approve_plan(self, plan: Optional[Plan] = None) -> Plan:
        plan = plan or self._current_plan
        if plan:
            plan.status = PlanStatus.APPROVED
            plan.approved_at = datetime.now().isoformat()
            self.save_plan(plan)
        return plan

    def reject_plan(self, plan: Optional[Plan] = None) -> Plan:
        plan = plan or self._current_plan
        if plan:
            plan.status = PlanStatus.REJECTED
            self.save_plan(plan)
        return plan

    def get_current(self) -> Optional[Plan]:
        return self._current_plan

    def list_plans(self, limit: int = 10) -> List[Dict]:
        """List recent plans."""
        results = []
        if os.path.exists(self.plans_dir):
            for f in sorted(os.listdir(self.plans_dir), reverse=True)[:limit]:
                if f.endswith(".md"):
                    path = os.path.join(self.plans_dir, f)
                    try:
                        with open(path) as fh:
                            first_line = fh.readline().strip()
                        task = first_line.replace("# Plan: ", "")
                        results.append({
                            "id": f.replace(".md", ""),
                            "task": task,
                            "path": path,
                        })
                    except Exception:
                        pass
        return results
