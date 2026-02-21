"""
Gymnasium Adapter: Wraps standard RL environments.

Converts between the neuro system's continuous action vectors
and Gymnasium's various action spaces.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

from .base_env import NeuroEnvironment, EnvironmentConfig, StepResult

# Try to import gymnasium, fall back to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYMNASIUM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYMNASIUM = True
    except ImportError:
        HAS_GYMNASIUM = False
        gym = None
        spaces = None


@dataclass
class GymConfig(EnvironmentConfig):
    """Configuration for Gym adapter."""
    env_id: str = "CartPole-v1"
    discrete_action_temp: float = 1.0  # Temperature for discrete action selection
    continuous_scale: float = 1.0  # Scale factor for continuous actions


class GymAdapter(NeuroEnvironment):
    """
    Adapter for Gymnasium/OpenAI Gym environments.

    Handles conversion between:
    - Continuous action vectors and discrete/Box action spaces
    - Various observation spaces and standard observation vectors
    """

    def __init__(
        self,
        env_id: str = "CartPole-v1",
        config: Optional[GymConfig] = None,
    ):
        if not HAS_GYMNASIUM:
            raise ImportError("gymnasium or gym package required. Install with: pip install gymnasium")

        self.gym_config = config or GymConfig(env_id=env_id)
        super().__init__(self.gym_config)

        self.env_id = env_id
        self._gym_env = gym.make(env_id)
        self._setup_spaces()

    def _setup_spaces(self) -> None:
        """Analyze gym environment spaces."""
        obs_space = self._gym_env.observation_space
        act_space = self._gym_env.action_space

        # Determine observation handling
        if isinstance(obs_space, spaces.Box):
            self._obs_type = "box"
            self._obs_shape = obs_space.shape
        elif isinstance(obs_space, spaces.Discrete):
            self._obs_type = "discrete"
            self._obs_n = obs_space.n
        else:
            self._obs_type = "other"
            self._obs_shape = None

        # Determine action handling
        if isinstance(act_space, spaces.Discrete):
            self._action_type = "discrete"
            self._action_n = act_space.n
        elif isinstance(act_space, spaces.Box):
            self._action_type = "box"
            self._action_shape = act_space.shape
            self._action_low = act_space.low
            self._action_high = act_space.high
        elif isinstance(act_space, spaces.MultiDiscrete):
            self._action_type = "multidiscrete"
            self._action_nvec = act_space.nvec
        else:
            self._action_type = "other"

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        result = self._gym_env.reset(seed=seed)

        # Handle both old and new gym API
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        self._step_count = 0
        self._episode_count += 1
        self._current_observation = self._convert_observation(obs)

        return self._current_observation, info

    def step(self, action: np.ndarray) -> StepResult:
        action = self._normalize_action(action)
        gym_action = self._convert_action(action)

        result = self._gym_env.step(gym_action)

        # Handle both old and new gym API
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        self._step_count += 1
        self._total_reward += reward
        self._current_observation = self._convert_observation(obs)

        return StepResult(
            observation=self._current_observation,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def render(self) -> Optional[str]:
        if self.config.render_mode == "text":
            return f"Gym {self.env_id} step {self._step_count}"
        elif self.config.render_mode == "rgb_array":
            frame = self._gym_env.render()
            return f"Frame shape: {frame.shape if frame is not None else 'None'}"
        return None

    def close(self) -> None:
        self._gym_env.close()

    def _convert_observation(self, obs: Any) -> np.ndarray:
        """Convert gym observation to standard format."""
        if self._obs_type == "box":
            obs = np.asarray(obs, dtype=np.float32).flatten()
        elif self._obs_type == "discrete":
            # One-hot encode discrete observation
            obs_vec = np.zeros(self._obs_n, dtype=np.float32)
            obs_vec[int(obs)] = 1.0
            obs = obs_vec
        else:
            obs = np.asarray(obs, dtype=np.float32).flatten()

        return self._normalize_observation(obs)

    def _convert_action(self, action: np.ndarray) -> Any:
        """Convert continuous action vector to gym action."""
        if self._action_type == "discrete":
            # Use action vector as logits, sample from softmax
            logits = action[:self._action_n]
            temp = self.gym_config.discrete_action_temp
            probs = self._softmax(logits / temp)
            return int(np.argmax(probs))

        elif self._action_type == "box":
            # Scale continuous action to environment range
            action = action[:np.prod(self._action_shape)]
            action = action.reshape(self._action_shape)

            # Map from [-1, 1] to [low, high]
            action = (action + 1) / 2  # [0, 1]
            action = action * (self._action_high - self._action_low) + self._action_low

            return action.astype(np.float32)

        elif self._action_type == "multidiscrete":
            actions = []
            offset = 0
            for n in self._action_nvec:
                logits = action[offset:offset + n]
                probs = self._softmax(logits)
                actions.append(int(np.argmax(probs)))
                offset += n
            return np.array(actions)

        else:
            return action

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-8)

    def get_observation_space(self) -> Dict[str, Any]:
        base = super().get_observation_space()
        base['gym_space'] = str(self._gym_env.observation_space)
        return base

    def get_action_space(self) -> Dict[str, Any]:
        base = super().get_action_space()
        base['gym_space'] = str(self._gym_env.action_space)
        base['action_type'] = self._action_type
        return base


def list_available_envs() -> List[str]:
    """List available gymnasium environments."""
    if not HAS_GYMNASIUM:
        return []

    # Common environments that work well with neuro system
    return [
        "CartPole-v1",
        "Acrobot-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
        "LunarLander-v2",
        "BipedalWalker-v3",
    ]
