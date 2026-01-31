"""
Base Environment: Abstract interface for all environments.

Defines the contract that all environments must implement
to work with the cognitive system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import time


@dataclass
class EnvironmentConfig:
    """Configuration for environments."""
    observation_dim: int = 64
    action_dim: int = 32
    max_steps: int = 1000
    render_mode: str = "text"  # "text", "rgb_array", "none"
    seed: Optional[int] = None


@dataclass
class StepResult:
    """Result of an environment step."""
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """Check if episode is done."""
        return self.terminated or self.truncated


class NeuroEnvironment(ABC):
    """
    Abstract base class for all neuro environments.

    This interface is designed to:
    1. Accept continuous action vectors from the cognitive system
    2. Return observations compatible with sensory interface
    3. Provide rewards for active inference goal alignment
    4. Support curriculum-based development
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()
        self._step_count = 0
        self._episode_count = 0
        self._total_reward = 0.0
        self._current_observation: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(self.config.seed)

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Optional random seed

        Returns:
            Tuple of (observation, info dict)
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """
        Take an action in the environment.

        Args:
            action: Action vector from motor interface

        Returns:
            StepResult with observation, reward, done flags, info
        """
        pass

    @abstractmethod
    def render(self) -> Optional[str]:
        """
        Render the current state.

        Returns:
            String description or None
        """
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass

    def get_observation_space(self) -> Dict[str, Any]:
        """Get observation space description."""
        return {
            'dim': self.config.observation_dim,
            'dtype': 'float32',
            'low': -np.inf,
            'high': np.inf,
        }

    def get_action_space(self) -> Dict[str, Any]:
        """Get action space description."""
        return {
            'dim': self.config.action_dim,
            'dtype': 'float32',
            'low': -1.0,
            'high': 1.0,
        }

    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Normalize observation to standard format."""
        obs = np.asarray(obs, dtype=np.float32).flatten()

        if len(obs) < self.config.observation_dim:
            obs = np.pad(obs, (0, self.config.observation_dim - len(obs)))
        elif len(obs) > self.config.observation_dim:
            obs = obs[:self.config.observation_dim]

        return obs

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to environment format."""
        action = np.asarray(action, dtype=np.float32).flatten()

        if len(action) < self.config.action_dim:
            action = np.pad(action, (0, self.config.action_dim - len(action)))
        elif len(action) > self.config.action_dim:
            action = action[:self.config.action_dim]

        # Clip to valid range
        return np.clip(action, -1.0, 1.0)

    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics."""
        return {
            'step_count': self._step_count,
            'episode_count': self._episode_count,
            'total_reward': self._total_reward,
            'observation_dim': self.config.observation_dim,
            'action_dim': self.config.action_dim,
        }


class SimplePatternEnv(NeuroEnvironment):
    """
    Simple environment for basic sensory pattern learning.

    Presents patterns and rewards correct recognition.
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        super().__init__(config)
        self._patterns: List[np.ndarray] = []
        self._current_pattern_idx = 0
        self._generate_patterns()

    def _generate_patterns(self, n_patterns: int = 10) -> None:
        """Generate random patterns to recognize."""
        self._patterns = [
            self._rng.random(self.config.observation_dim).astype(np.float32)
            for _ in range(n_patterns)
        ]

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._current_pattern_idx = self._rng.integers(len(self._patterns))
        self._current_observation = self._patterns[self._current_pattern_idx].copy()
        self._episode_count += 1

        return self._current_observation, {'pattern_idx': self._current_pattern_idx}

    def step(self, action: np.ndarray) -> StepResult:
        action = self._normalize_action(action)
        self._step_count += 1

        # Reward based on similarity to current pattern
        pattern = self._patterns[self._current_pattern_idx]
        # Resize action to match pattern dimension
        if len(action) < len(pattern):
            action_resized = np.pad(action, (0, len(pattern) - len(action)))
        else:
            action_resized = action[:len(pattern)]
        similarity = np.dot(action_resized, pattern) / (
            np.linalg.norm(action_resized) * np.linalg.norm(pattern) + 1e-8
        )
        reward = float(similarity)
        self._total_reward += reward

        # Switch to new pattern
        self._current_pattern_idx = self._rng.integers(len(self._patterns))
        self._current_observation = self._patterns[self._current_pattern_idx].copy()

        terminated = self._step_count >= self.config.max_steps
        truncated = False

        return StepResult(
            observation=self._current_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={'pattern_idx': self._current_pattern_idx, 'similarity': similarity},
        )

    def render(self) -> Optional[str]:
        if self._current_observation is None:
            return "No observation"
        return f"Pattern {self._current_pattern_idx}: {self._current_observation[:5]}..."
