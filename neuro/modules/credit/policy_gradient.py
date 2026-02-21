"""
Cross-Module Policy Gradient

Implements policy gradient methods including:
- Generalized Advantage Estimation (GAE)
- Cross-module gradient accumulation
- Variance reduction techniques
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class GAEConfig:
    """Configuration for Generalized Advantage Estimation."""
    gamma: float = 0.99
    lambda_param: float = 0.95
    normalize_advantages: bool = True
    use_td_lambda: bool = True
    value_clip: Optional[float] = None


@dataclass
class Advantage:
    """Computed advantage for a state-action pair."""
    state_index: int
    advantage: float
    value_target: float
    td_error: float
    module_id: Optional[str] = None


@dataclass
class PolicyGradientResult:
    """Result of policy gradient computation."""
    policy_loss: float
    value_loss: float
    entropy: float
    gradients: Dict[str, np.ndarray]
    advantages: List[Advantage]
    explained_variance: float


class CrossModulePolicyGradient:
    """
    Cross-module policy gradient computation.

    Supports:
    - GAE for advantage estimation
    - Per-module gradient accumulation
    - Entropy regularization
    - Value function bootstrapping
    """

    def __init__(
        self,
        config: Optional[GAEConfig] = None,
        random_seed: Optional[int] = None,
    ):
        self.config = config or GAEConfig()
        self._rng = np.random.default_rng(random_seed)

        self._accumulated_gradients: Dict[str, List[np.ndarray]] = {}
        self._module_losses: Dict[str, List[float]] = {}

        self._total_gradient_steps = 0
        self._total_trajectories = 0

    def compute_advantage(
        self,
        rewards: List[float],
        values: List[float],
        dones: Optional[List[bool]] = None,
        next_value: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute advantages using GAE.

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: Optional list of done flags
            next_value: Bootstrap value for final state

        Returns:
            Tuple of (advantages, value_targets)
        """
        n = len(rewards)
        if n == 0:
            return np.array([]), np.array([])

        if dones is None:
            dones = [False] * n

        advantages = np.zeros(n)
        value_targets = np.zeros(n)

        values_extended = list(values) + [next_value]
        gae = 0.0

        for t in reversed(range(n)):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.config.gamma * values_extended[t + 1] - values[t]
                gae = delta + self.config.gamma * self.config.lambda_param * gae

            advantages[t] = gae
            value_targets[t] = advantages[t] + values[t]

        if self.config.normalize_advantages and len(advantages) > 1:
            adv_std = np.std(advantages)
            if adv_std > 1e-8:
                advantages = (advantages - np.mean(advantages)) / adv_std

        return advantages, value_targets

    def compute_policy_loss(
        self,
        log_probs: np.ndarray,
        advantages: np.ndarray,
        old_log_probs: Optional[np.ndarray] = None,
        clip_epsilon: float = 0.2,
    ) -> float:
        """
        Compute policy loss.

        Supports both vanilla policy gradient and PPO-style clipping.

        Args:
            log_probs: Log probabilities of actions
            advantages: Advantage estimates
            old_log_probs: Optional old log probs for PPO
            clip_epsilon: Clipping parameter for PPO

        Returns:
            Policy loss value
        """
        if old_log_probs is None:
            loss = -np.mean(log_probs * advantages)
        else:
            ratio = np.exp(log_probs - old_log_probs)
            clipped_ratio = np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            loss = -np.mean(np.minimum(ratio * advantages, clipped_ratio * advantages))

        return float(loss)

    def compute_value_loss(
        self,
        values: np.ndarray,
        value_targets: np.ndarray,
        old_values: Optional[np.ndarray] = None,
        clip_range: Optional[float] = None,
    ) -> float:
        """
        Compute value function loss.

        Args:
            values: Current value predictions
            value_targets: Target values
            old_values: Optional old values for clipping
            clip_range: Optional clipping range

        Returns:
            Value loss
        """
        if old_values is not None and clip_range is not None:
            clipped_values = old_values + np.clip(
                values - old_values, -clip_range, clip_range
            )
            loss_unclipped = (values - value_targets) ** 2
            loss_clipped = (clipped_values - value_targets) ** 2
            loss = 0.5 * np.mean(np.maximum(loss_unclipped, loss_clipped))
        else:
            loss = 0.5 * np.mean((values - value_targets) ** 2)

        return float(loss)

    def compute_entropy(
        self,
        action_probs: np.ndarray,
        epsilon: float = 1e-8,
    ) -> float:
        """
        Compute entropy of action distribution.

        Args:
            action_probs: Action probability distribution
            epsilon: Small constant for numerical stability

        Returns:
            Entropy value
        """
        action_probs = np.clip(action_probs, epsilon, 1.0)
        entropy = -np.sum(action_probs * np.log(action_probs), axis=-1)
        return float(np.mean(entropy))

    def accumulate_gradients(
        self,
        trajectory: Dict[str, Any],
        module_id: str,
    ) -> Dict[str, np.ndarray]:
        """
        Accumulate gradients from a trajectory for a module.

        Args:
            trajectory: Dict with keys: states, actions, rewards, values, log_probs
            module_id: Module identifier

        Returns:
            Gradients for the module
        """
        rewards = trajectory.get("rewards", [])
        values = trajectory.get("values", [])
        log_probs = trajectory.get("log_probs", np.array([]))

        if len(rewards) == 0:
            return {}

        advantages, value_targets = self.compute_advantage(rewards, values)

        policy_grad = log_probs * advantages
        value_grad = values - value_targets

        gradients = {
            "policy": policy_grad,
            "value": value_grad,
        }

        if module_id not in self._accumulated_gradients:
            self._accumulated_gradients[module_id] = []
        self._accumulated_gradients[module_id].append(gradients)

        self._total_trajectories += 1

        return gradients

    def get_accumulated_gradients(
        self,
        module_id: str,
        average: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Get accumulated gradients for a module.

        Args:
            module_id: Module identifier
            average: Whether to average over trajectories

        Returns:
            Aggregated gradients
        """
        if module_id not in self._accumulated_gradients:
            return {}

        grads = self._accumulated_gradients[module_id]
        if not grads:
            return {}

        result = {}
        for key in grads[0].keys():
            stacked = [g[key] for g in grads if key in g]
            if average:
                result[key] = np.mean([np.mean(s) for s in stacked])
            else:
                result[key] = np.sum([np.sum(s) for s in stacked])

        return result

    def clear_gradients(self, module_id: Optional[str] = None) -> None:
        """Clear accumulated gradients."""
        if module_id is None:
            self._accumulated_gradients.clear()
        elif module_id in self._accumulated_gradients:
            del self._accumulated_gradients[module_id]

    def compute_explained_variance(
        self,
        values: np.ndarray,
        returns: np.ndarray,
    ) -> float:
        """
        Compute explained variance of value function.

        Args:
            values: Value predictions
            returns: Actual returns

        Returns:
            Explained variance (1.0 = perfect, 0.0 = no better than mean)
        """
        var_returns = np.var(returns)
        if var_returns < 1e-8:
            return 1.0

        var_unexplained = np.var(returns - values)
        return float(1.0 - var_unexplained / var_returns)

    def compute_full_result(
        self,
        trajectory: Dict[str, Any],
        module_id: str,
    ) -> PolicyGradientResult:
        """
        Compute full policy gradient result for a trajectory.

        Args:
            trajectory: Trajectory data
            module_id: Module identifier

        Returns:
            PolicyGradientResult with all computed values
        """
        rewards = trajectory.get("rewards", [])
        values = np.array(trajectory.get("values", []))
        log_probs = np.array(trajectory.get("log_probs", []))
        action_probs = trajectory.get("action_probs", None)

        if len(rewards) == 0:
            return PolicyGradientResult(
                policy_loss=0.0,
                value_loss=0.0,
                entropy=0.0,
                gradients={},
                advantages=[],
                explained_variance=0.0,
            )

        advantages_arr, value_targets = self.compute_advantage(rewards, values.tolist())

        policy_loss = self.compute_policy_loss(log_probs, advantages_arr)
        value_loss = self.compute_value_loss(values, value_targets)

        entropy = 0.0
        if action_probs is not None:
            entropy = self.compute_entropy(np.array(action_probs))

        gradients = self.accumulate_gradients(trajectory, module_id)

        returns = self._compute_returns(rewards)
        explained_var = self.compute_explained_variance(values, returns)

        advantages = [
            Advantage(
                state_index=i,
                advantage=float(advantages_arr[i]),
                value_target=float(value_targets[i]),
                td_error=float(rewards[i] + self.config.gamma * (
                    values[i + 1] if i + 1 < len(values) else 0
                ) - values[i]),
                module_id=module_id,
            )
            for i in range(len(advantages_arr))
        ]

        self._total_gradient_steps += 1

        return PolicyGradientResult(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            gradients=gradients,
            advantages=advantages,
            explained_variance=explained_var,
        )

    def _compute_returns(
        self,
        rewards: List[float],
    ) -> np.ndarray:
        """Compute discounted returns."""
        returns = np.zeros(len(rewards))
        running_return = 0.0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return

        return returns

    def statistics(self) -> Dict[str, Any]:
        """Get policy gradient statistics."""
        return {
            "total_trajectories": self._total_trajectories,
            "total_gradient_steps": self._total_gradient_steps,
            "modules_with_gradients": len(self._accumulated_gradients),
            "gradient_counts": {
                k: len(v) for k, v in self._accumulated_gradients.items()
            },
        }
