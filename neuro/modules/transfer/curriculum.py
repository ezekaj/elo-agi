"""
Curriculum: Order tasks by complexity for progressive learning.

Implements curriculum learning strategies for optimal knowledge transfer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Callable
from enum import Enum
import numpy as np


class TaskDifficulty(Enum):
    """Task difficulty levels."""

    TRIVIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


@dataclass
class Task:
    """A learning task."""

    id: str
    name: str
    domain: str
    difficulty: TaskDifficulty
    prerequisites: List[str] = field(default_factory=list)
    skills_required: List[str] = field(default_factory=list)
    skills_taught: List[str] = field(default_factory=list)
    reward_signal: Optional[Callable[[], float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPath:
    """A sequence of tasks for learning."""

    id: str
    name: str
    tasks: List[str]  # Task IDs in order
    estimated_difficulty_curve: List[float]
    skills_progression: List[str]


@dataclass
class LearnerState:
    """Current state of a learner."""

    skills_mastered: Set[str] = field(default_factory=set)
    tasks_completed: List[str] = field(default_factory=list)
    current_performance: Dict[str, float] = field(default_factory=dict)
    learning_rate: float = 1.0
    frustration_level: float = 0.0


class ProgressTracker:
    """
    Track learning progress across tasks.
    """

    def __init__(self):
        self._history: List[Dict[str, Any]] = []
        self._performance_by_task: Dict[str, List[float]] = {}
        self._performance_by_skill: Dict[str, List[float]] = {}

    def record(
        self,
        task_id: str,
        performance: float,
        skills_used: List[str],
        time_taken: float,
    ) -> None:
        """Record a task attempt."""
        self._history.append(
            {
                "task_id": task_id,
                "performance": performance,
                "skills_used": skills_used,
                "time_taken": time_taken,
            }
        )

        if task_id not in self._performance_by_task:
            self._performance_by_task[task_id] = []
        self._performance_by_task[task_id].append(performance)

        for skill in skills_used:
            if skill not in self._performance_by_skill:
                self._performance_by_skill[skill] = []
            self._performance_by_skill[skill].append(performance)

    def get_skill_mastery(self, skill: str) -> float:
        """Get estimated mastery of a skill."""
        performances = self._performance_by_skill.get(skill, [])
        if not performances:
            return 0.0

        # Weighted average with recency bias
        weights = np.exp(np.linspace(-2, 0, len(performances)))
        weights = weights / weights.sum()

        return float(np.dot(performances, weights))

    def get_task_performance(self, task_id: str) -> float:
        """Get best performance on a task."""
        performances = self._performance_by_task.get(task_id, [])
        if not performances:
            return 0.0
        return max(performances)

    def get_learning_curve(self, skill: str) -> List[float]:
        """Get learning curve for a skill."""
        return self._performance_by_skill.get(skill, [])

    def is_skill_mastered(
        self,
        skill: str,
        threshold: float = 0.8,
    ) -> bool:
        """Check if a skill is mastered."""
        return self.get_skill_mastery(skill) >= threshold


class DifficultyEstimator:
    """
    Estimate task difficulty based on features.
    """

    def __init__(self):
        self._difficulty_weights = {
            "n_prerequisites": 0.2,
            "n_skills_required": 0.3,
            "complexity": 0.3,
            "novelty": 0.2,
        }

    def estimate(
        self,
        task: Task,
        learner_state: LearnerState,
    ) -> float:
        """
        Estimate task difficulty for a learner.

        Returns:
            Difficulty score 0-1 (relative to learner's current state)
        """
        scores = {}

        # Prerequisite difficulty
        unmet = sum(1 for p in task.prerequisites if p not in learner_state.tasks_completed)
        scores["n_prerequisites"] = unmet / max(len(task.prerequisites), 1)

        # Skill gap
        missing = sum(1 for s in task.skills_required if s not in learner_state.skills_mastered)
        scores["n_skills_required"] = missing / max(len(task.skills_required), 1)

        # Base complexity from task difficulty
        scores["complexity"] = (task.difficulty.value - 1) / 4

        # Novelty (based on how many new skills it teaches)
        new_skills = sum(1 for s in task.skills_taught if s not in learner_state.skills_mastered)
        scores["novelty"] = new_skills / max(len(task.skills_taught), 1)

        # Weighted combination
        difficulty = sum(self._difficulty_weights[k] * scores[k] for k in scores)

        return min(1.0, max(0.0, difficulty))


class TaskSelector:
    """
    Select optimal next task based on curriculum.
    """

    def __init__(
        self,
        target_difficulty: float = 0.6,
        difficulty_tolerance: float = 0.2,
    ):
        self.target_difficulty = target_difficulty
        self.difficulty_tolerance = difficulty_tolerance
        self.difficulty_estimator = DifficultyEstimator()

    def select(
        self,
        available_tasks: List[Task],
        learner_state: LearnerState,
    ) -> Optional[Task]:
        """
        Select the best next task for the learner.
        """
        candidates = []

        for task in available_tasks:
            # Check prerequisites
            if not all(p in learner_state.tasks_completed for p in task.prerequisites):
                continue

            # Estimate difficulty
            difficulty = self.difficulty_estimator.estimate(task, learner_state)

            # Check if in target range
            if abs(difficulty - self.target_difficulty) <= self.difficulty_tolerance:
                score = self._score_task(task, learner_state, difficulty)
                candidates.append((task, score))

        if not candidates:
            # Fall back to easiest available task
            for task in available_tasks:
                if all(p in learner_state.tasks_completed for p in task.prerequisites):
                    difficulty = self.difficulty_estimator.estimate(task, learner_state)
                    candidates.append((task, 1.0 / (difficulty + 0.1)))

        if not candidates:
            return None

        # Return highest scoring task
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    def _score_task(
        self,
        task: Task,
        learner_state: LearnerState,
        difficulty: float,
    ) -> float:
        """Score a task for selection."""
        score = 0.0

        # Prefer tasks teaching new skills
        new_skills = sum(1 for s in task.skills_taught if s not in learner_state.skills_mastered)
        score += new_skills * 0.5

        # Prefer tasks building on existing skills
        known_skills = sum(1 for s in task.skills_required if s in learner_state.skills_mastered)
        score += known_skills * 0.3

        # Penalty for being too far from target difficulty
        score -= abs(difficulty - self.target_difficulty) * 0.2

        return score


class CurriculumLearner:
    """
    Complete curriculum learning system.

    Manages task ordering, difficulty progression, and learning paths.
    """

    def __init__(
        self,
        initial_difficulty: float = 0.3,
        difficulty_increase_rate: float = 0.1,
        mastery_threshold: float = 0.8,
    ):
        self.mastery_threshold = mastery_threshold

        self.task_selector = TaskSelector(target_difficulty=initial_difficulty)
        self.progress_tracker = ProgressTracker()
        self.difficulty_estimator = DifficultyEstimator()

        # Task registry
        self._tasks: Dict[str, Task] = {}
        self._learning_paths: Dict[str, LearningPath] = {}

        # Current learner state
        self._learner_state = LearnerState()

        # Curriculum parameters
        self._difficulty_increase_rate = difficulty_increase_rate

    def register_task(self, task: Task) -> None:
        """Register a task in the curriculum."""
        self._tasks[task.id] = task

    def register_tasks(self, tasks: List[Task]) -> None:
        """Register multiple tasks."""
        for task in tasks:
            self.register_task(task)

    def create_learning_path(
        self,
        path_id: str,
        path_name: str,
        target_skills: List[str],
    ) -> LearningPath:
        """
        Create a learning path targeting specific skills.
        """
        # Find tasks that teach target skills
        relevant_tasks = []
        for task in self._tasks.values():
            if any(s in target_skills for s in task.skills_taught):
                relevant_tasks.append(task)

        # Topological sort by prerequisites
        ordered_tasks = self._topological_sort(relevant_tasks)

        # Estimate difficulty curve
        difficulty_curve = []
        for task in ordered_tasks:
            diff = (task.difficulty.value - 1) / 4
            difficulty_curve.append(diff)

        # Collect skills progression
        skills_progression = []
        seen_skills = set()
        for task in ordered_tasks:
            for skill in task.skills_taught:
                if skill not in seen_skills:
                    skills_progression.append(skill)
                    seen_skills.add(skill)

        path = LearningPath(
            id=path_id,
            name=path_name,
            tasks=[t.id for t in ordered_tasks],
            estimated_difficulty_curve=difficulty_curve,
            skills_progression=skills_progression,
        )

        self._learning_paths[path_id] = path
        return path

    def _topological_sort(self, tasks: List[Task]) -> List[Task]:
        """Sort tasks by prerequisites."""
        task_dict = {t.id: t for t in tasks}
        visited = set()
        result = []

        def visit(task_id: str):
            if task_id in visited:
                return
            visited.add(task_id)

            task = task_dict.get(task_id)
            if task:
                for prereq in task.prerequisites:
                    if prereq in task_dict:
                        visit(prereq)
                result.append(task)

        for task in tasks:
            visit(task.id)

        return result

    def get_next_task(self) -> Optional[Task]:
        """Get the next task for the learner."""
        available = list(self._tasks.values())
        return self.task_selector.select(available, self._learner_state)

    def complete_task(
        self,
        task_id: str,
        performance: float,
        time_taken: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Record task completion.

        Returns:
            Summary of what was learned
        """
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")

        # Record in progress tracker
        self.progress_tracker.record(task_id, performance, task.skills_required, time_taken)

        # Update learner state
        self._learner_state.tasks_completed.append(task_id)
        self._learner_state.current_performance[task_id] = performance

        # Check for skill mastery
        newly_mastered = []
        for skill in task.skills_taught:
            if self.progress_tracker.is_skill_mastered(skill, self.mastery_threshold):
                if skill not in self._learner_state.skills_mastered:
                    self._learner_state.skills_mastered.add(skill)
                    newly_mastered.append(skill)

        # Update difficulty target if doing well
        if performance >= self.mastery_threshold:
            self.task_selector.target_difficulty = min(
                1.0, self.task_selector.target_difficulty + self._difficulty_increase_rate
            )
        elif performance < 0.5:
            self.task_selector.target_difficulty = max(
                0.2, self.task_selector.target_difficulty - self._difficulty_increase_rate
            )

        return {
            "task_id": task_id,
            "performance": performance,
            "newly_mastered_skills": newly_mastered,
            "total_skills_mastered": len(self._learner_state.skills_mastered),
            "current_difficulty_target": self.task_selector.target_difficulty,
        }

    def get_learner_state(self) -> LearnerState:
        """Get current learner state."""
        return self._learner_state

    def get_skill_progress(self, skill: str) -> Dict[str, Any]:
        """Get progress for a specific skill."""
        return {
            "skill": skill,
            "mastery": self.progress_tracker.get_skill_mastery(skill),
            "mastered": skill in self._learner_state.skills_mastered,
            "learning_curve": self.progress_tracker.get_learning_curve(skill),
        }

    def get_recommended_path(
        self,
        target_skills: List[str],
    ) -> Optional[LearningPath]:
        """Get a recommended learning path for target skills."""
        # Check if a suitable path exists
        for path in self._learning_paths.values():
            if all(s in path.skills_progression for s in target_skills):
                return path

        # Create a new path
        path_id = f"path_{len(self._learning_paths)}"
        return self.create_learning_path(path_id, f"Path to {target_skills}", target_skills)

    def reset(self) -> None:
        """Reset learner state."""
        self._learner_state = LearnerState()
        self.task_selector.target_difficulty = 0.3

    def statistics(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        return {
            "n_tasks": len(self._tasks),
            "n_paths": len(self._learning_paths),
            "tasks_completed": len(self._learner_state.tasks_completed),
            "skills_mastered": len(self._learner_state.skills_mastered),
            "current_difficulty": self.task_selector.target_difficulty,
        }
