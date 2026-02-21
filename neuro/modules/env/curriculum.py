"""
Developmental Curriculum: Staged learning progression.

Manages the developmental stages that allow the cognitive
system to progressively acquire capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Type
from enum import Enum
import numpy as np
import time

from .base_env import NeuroEnvironment, EnvironmentConfig, StepResult, SimplePatternEnv


class StageStatus(Enum):
    """Status of a curriculum stage."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class StageMetrics:
    """Performance metrics for a stage."""
    total_episodes: int = 0
    total_steps: int = 0
    total_reward: float = 0.0
    success_rate: float = 0.0
    avg_episode_length: float = 0.0
    avg_reward_per_episode: float = 0.0
    best_reward: float = float('-inf')
    completion_time: Optional[float] = None


@dataclass
class Stage:
    """A single developmental stage."""
    name: str
    description: str
    env_class: Type[NeuroEnvironment]
    env_config: Optional[EnvironmentConfig] = None
    duration_episodes: int = 1000
    success_threshold: float = 0.8  # Required success rate to advance
    min_episodes: int = 100  # Minimum episodes before checking advancement
    metrics: StageMetrics = field(default_factory=StageMetrics)
    status: StageStatus = StageStatus.PENDING

    def is_complete(self) -> bool:
        """Check if stage completion criteria are met."""
        if self.metrics.total_episodes < self.min_episodes:
            return False
        if self.metrics.total_episodes >= self.duration_episodes:
            return True
        if self.metrics.success_rate >= self.success_threshold:
            return True
        return False


@dataclass
class CurriculumConfig:
    """Configuration for curriculum."""
    auto_advance: bool = True  # Automatically advance when criteria met
    allow_regression: bool = False  # Allow going back to earlier stages
    save_checkpoints: bool = True
    checkpoint_interval: int = 100  # Episodes between checkpoints


class DevelopmentalCurriculum:
    """
    Manages developmental stages for the cognitive system.

    The curriculum provides:
    1. Ordered sequence of learning environments
    2. Automatic progression when criteria are met
    3. Performance tracking across stages
    4. Adaptive difficulty adjustment
    """

    def __init__(
        self,
        stages: Optional[List[Stage]] = None,
        config: Optional[CurriculumConfig] = None,
    ):
        self.config = config or CurriculumConfig()
        self._stages = stages or self._default_stages()
        self._current_stage_idx = 0
        self._current_env: Optional[NeuroEnvironment] = None
        self._start_time = time.time()

        # Statistics
        self._total_episodes = 0
        self._total_steps = 0
        self._stage_history: List[Tuple[str, StageMetrics]] = []

    def _default_stages(self) -> List[Stage]:
        """Create default developmental curriculum."""
        return [
            Stage(
                name="sensory_patterns",
                description="Learn to recognize and reproduce simple patterns",
                env_class=SimplePatternEnv,
                env_config=EnvironmentConfig(observation_dim=64, action_dim=32, max_steps=100),
                duration_episodes=1000,
                success_threshold=0.7,
            ),
            # Additional stages would use GymAdapter, TextWorld, DialogueEnvironment
            # when those environments are instantiated
        ]

    @property
    def current_stage(self) -> Stage:
        """Get current stage."""
        return self._stages[self._current_stage_idx]

    @property
    def current_env(self) -> NeuroEnvironment:
        """Get current environment."""
        if self._current_env is None:
            self._current_env = self._create_env(self.current_stage)
        return self._current_env

    def _create_env(self, stage: Stage) -> NeuroEnvironment:
        """Create environment for a stage."""
        if stage.env_config:
            return stage.env_class(stage.env_config)
        return stage.env_class()

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset current environment for new episode."""
        obs, info = self.current_env.reset()
        info['stage'] = self.current_stage.name
        info['stage_idx'] = self._current_stage_idx
        return obs, info

    def step(self, action: np.ndarray) -> StepResult:
        """Take step in current environment."""
        result = self.current_env.step(action)
        self._total_steps += 1

        # Update stage metrics
        self.current_stage.metrics.total_steps += 1

        if result.done:
            self._on_episode_complete(result)

        return result

    def _on_episode_complete(self, final_result: StepResult) -> None:
        """Handle episode completion."""
        stage = self.current_stage
        metrics = stage.metrics

        # Update metrics
        metrics.total_episodes += 1
        self._total_episodes += 1
        episode_reward = final_result.info.get('episode_reward', final_result.reward)
        metrics.total_reward += episode_reward

        if episode_reward > metrics.best_reward:
            metrics.best_reward = episode_reward

        # Update averages
        metrics.avg_reward_per_episode = metrics.total_reward / metrics.total_episodes
        metrics.avg_episode_length = metrics.total_steps / metrics.total_episodes

        # Simple success criterion: reward above threshold
        threshold = 0.5 * metrics.best_reward if metrics.best_reward > 0 else 0
        if episode_reward >= threshold:
            metrics.success_rate = (
                metrics.success_rate * (metrics.total_episodes - 1) + 1.0
            ) / metrics.total_episodes
        else:
            metrics.success_rate = (
                metrics.success_rate * (metrics.total_episodes - 1)
            ) / metrics.total_episodes

        # Check for stage advancement
        if self.config.auto_advance and stage.is_complete():
            self._advance_stage()

    def _advance_stage(self) -> bool:
        """Advance to next stage."""
        if self._current_stage_idx >= len(self._stages) - 1:
            return False  # Already at final stage

        # Record completion
        stage = self.current_stage
        stage.status = StageStatus.COMPLETED
        stage.metrics.completion_time = time.time() - self._start_time
        self._stage_history.append((stage.name, stage.metrics))

        # Advance
        self._current_stage_idx += 1
        self._stages[self._current_stage_idx].status = StageStatus.ACTIVE
        self._current_env = self._create_env(self.current_stage)

        return True

    def skip_stage(self) -> bool:
        """Skip current stage."""
        if self._current_stage_idx >= len(self._stages) - 1:
            return False

        self.current_stage.status = StageStatus.SKIPPED
        self._current_stage_idx += 1
        self._stages[self._current_stage_idx].status = StageStatus.ACTIVE
        self._current_env = self._create_env(self.current_stage)

        return True

    def regress_stage(self) -> bool:
        """Go back to previous stage."""
        if not self.config.allow_regression:
            return False
        if self._current_stage_idx <= 0:
            return False

        self._current_stage_idx -= 1
        self._stages[self._current_stage_idx].status = StageStatus.ACTIVE
        self._current_env = self._create_env(self.current_stage)

        return True

    def get_progress(self) -> Dict[str, Any]:
        """Get curriculum progress."""
        return {
            'current_stage': self.current_stage.name,
            'current_stage_idx': self._current_stage_idx,
            'total_stages': len(self._stages),
            'total_episodes': self._total_episodes,
            'total_steps': self._total_steps,
            'elapsed_time': time.time() - self._start_time,
            'stages': [
                {
                    'name': s.name,
                    'status': s.status.value,
                    'episodes': s.metrics.total_episodes,
                    'success_rate': s.metrics.success_rate,
                }
                for s in self._stages
            ],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics."""
        return {
            'progress': self.get_progress(),
            'current_stage_metrics': {
                'episodes': self.current_stage.metrics.total_episodes,
                'steps': self.current_stage.metrics.total_steps,
                'total_reward': self.current_stage.metrics.total_reward,
                'success_rate': self.current_stage.metrics.success_rate,
                'avg_reward': self.current_stage.metrics.avg_reward_per_episode,
                'best_reward': self.current_stage.metrics.best_reward,
            },
            'history': [
                {'name': name, 'completion_time': m.completion_time}
                for name, m in self._stage_history
            ],
        }

    def add_stage(self, stage: Stage, position: Optional[int] = None) -> None:
        """Add a new stage to the curriculum."""
        if position is None:
            self._stages.append(stage)
        else:
            self._stages.insert(position, stage)

    def remove_stage(self, name: str) -> bool:
        """Remove a stage by name."""
        for i, stage in enumerate(self._stages):
            if stage.name == name:
                if i == self._current_stage_idx:
                    return False  # Can't remove active stage
                self._stages.pop(i)
                if i < self._current_stage_idx:
                    self._current_stage_idx -= 1
                return True
        return False


class AdaptiveCurriculum(DevelopmentalCurriculum):
    """
    Curriculum that adapts difficulty based on performance.
    """

    def __init__(
        self,
        stages: Optional[List[Stage]] = None,
        config: Optional[CurriculumConfig] = None,
        difficulty_adjustment: float = 0.1,
    ):
        super().__init__(stages, config)
        self._difficulty_adjustment = difficulty_adjustment
        self._difficulty_level = 1.0

    def _on_episode_complete(self, final_result: StepResult) -> None:
        """Adjust difficulty based on performance."""
        super()._on_episode_complete(final_result)

        # Adjust difficulty based on recent performance
        metrics = self.current_stage.metrics
        if metrics.total_episodes % 10 == 0:  # Check every 10 episodes
            if metrics.success_rate > 0.9:
                self._difficulty_level = min(2.0, self._difficulty_level + self._difficulty_adjustment)
            elif metrics.success_rate < 0.3:
                self._difficulty_level = max(0.5, self._difficulty_level - self._difficulty_adjustment)

    @property
    def difficulty(self) -> float:
        """Current difficulty level."""
        return self._difficulty_level
