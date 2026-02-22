"""Inhibition System - Response suppression and impulse control

Neural basis: Right inferior frontal gyrus (rIFG), pre-supplementary motor area (pre-SMA)
Key functions: Stop signals, go/no-go tasks, impulse control
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class InhibitionParams:
    """Parameters for inhibition system"""

    n_units: int = 50
    stop_threshold: float = 0.6
    go_threshold: float = 0.4
    inhibition_strength: float = 0.8
    decay_rate: float = 0.1
    noise_level: float = 0.05
    stop_signal_delay: float = 200.0  # ms


class ResponseInhibitor:
    """Models response inhibition via stop signals

    Implements race model between go and stop processes
    """

    def __init__(self, params: Optional[InhibitionParams] = None):
        self.params = params or InhibitionParams()
        self.go_activation = 0.0
        self.stop_activation = 0.0
        self.response_made = False
        self.response_inhibited = False

    def reset(self):
        """Reset for new trial"""
        self.go_activation = 0.0
        self.stop_activation = 0.0
        self.response_made = False
        self.response_inhibited = False

    def update_go(self, dt: float, stimulus_strength: float = 1.0) -> float:
        """Update go process activation"""
        noise = np.random.randn() * self.params.noise_level
        self.go_activation += (stimulus_strength * dt / 100.0) + noise
        self.go_activation = np.clip(self.go_activation, 0, 1)

        if self.go_activation >= self.params.go_threshold and not self.response_inhibited:
            self.response_made = True

        return self.go_activation

    def update_stop(self, dt: float) -> float:
        """Update stop process activation"""
        noise = np.random.randn() * self.params.noise_level
        self.stop_activation += (self.params.inhibition_strength * dt / 100.0) + noise
        self.stop_activation = np.clip(self.stop_activation, 0, 1)

        # Stop signal suppresses go activation
        self.go_activation = max(0.0, self.go_activation - self.stop_activation * dt / 100.0)

        if self.stop_activation >= self.params.stop_threshold:
            self.response_inhibited = True
            self.response_made = False

        return self.stop_activation

    def run_trial(self, is_stop_trial: bool, stop_signal_delay: Optional[float] = None) -> dict:
        """Run a single go/stop trial

        Args:
            is_stop_trial: Whether this is a stop signal trial
            stop_signal_delay: Time after go signal when stop signal appears

        Returns:
            Trial results dictionary
        """
        self.reset()
        ssd = stop_signal_delay or self.params.stop_signal_delay

        dt = 1.0  # 1ms time steps
        max_time = 1000.0  # 1 second max

        reaction_time = None
        stop_time = None

        go_crossed_at = None
        for t in np.arange(0, max_time, dt):
            # Update go process
            self.update_go(dt)

            # Track when go first crossed threshold
            if self.response_made and go_crossed_at is None:
                go_crossed_at = t

            # Update stop process after delay on stop trials
            if is_stop_trial and t >= ssd:
                self.update_stop(dt)

            # Check inhibition first (stop wins ties)
            if self.response_inhibited and stop_time is None:
                stop_time = t
                break

            # Go response finalizes immediately on non-stop trials
            if self.response_made and reaction_time is None and not is_stop_trial:
                reaction_time = t
                break

            # On stop trials, go response finalizes only after motor execution delay
            if is_stop_trial and go_crossed_at is not None and reaction_time is None:
                if not self.response_inhibited and (t - go_crossed_at) >= 80.0:
                    reaction_time = t
                    break

        return {
            "is_stop_trial": is_stop_trial,
            "response_made": self.response_made,
            "response_inhibited": self.response_inhibited,
            "reaction_time": reaction_time,
            "stop_time": stop_time,
            "go_activation": self.go_activation,
            "stop_activation": self.stop_activation,
        }


class ImpulseController:
    """Models impulse control through prefrontal inhibition

    Implements tonic inhibition that can be released for appropriate actions
    """

    def __init__(self, n_actions: int = 10, params: Optional[InhibitionParams] = None):
        self.n_actions = n_actions
        self.params = params or InhibitionParams()

        # Tonic inhibition level for each action
        self.inhibition = np.ones(n_actions) * self.params.inhibition_strength
        # Action urge/activation
        self.urge = np.zeros(n_actions)
        # Which actions are allowed
        self.allowed = np.zeros(n_actions, dtype=bool)

    def set_allowed_actions(self, actions: List[int]):
        """Set which actions are currently allowed"""
        self.allowed = np.zeros(self.n_actions, dtype=bool)
        for a in actions:
            if 0 <= a < self.n_actions:
                self.allowed[a] = True

    def receive_impulse(self, action: int, strength: float):
        """Receive an impulse for an action"""
        if 0 <= action < self.n_actions:
            self.urge[action] += strength

    def update(self, dt: float = 1.0) -> np.ndarray:
        """Update impulse control state

        Returns:
            Array of executed actions (1 if executed, 0 if inhibited)
        """
        # Calculate net activation (urge - inhibition)
        net_activation = self.urge - self.inhibition

        # Add noise
        net_activation += np.random.randn(self.n_actions) * self.params.noise_level

        # Action executes if net activation > 0 AND action is allowed
        # OR if urge overwhelms inhibition (impulse failure)
        executed = np.zeros(self.n_actions)

        for i in range(self.n_actions):
            if self.allowed[i] and net_activation[i] > 0:
                executed[i] = 1
            elif self.urge[i] > self.inhibition[i] * 1.5:  # Strong impulse breaks through
                executed[i] = 1

        # Decay urge
        self.urge *= 1 - self.params.decay_rate * dt

        return executed

    def release_inhibition(self, action: int, amount: float = 0.5):
        """Voluntarily reduce inhibition for an action"""
        if 0 <= action < self.n_actions:
            self.inhibition[action] = max(0, self.inhibition[action] - amount)

    def restore_inhibition(self, action: int):
        """Restore inhibition for an action"""
        if 0 <= action < self.n_actions:
            self.inhibition[action] = self.params.inhibition_strength

    def get_control_state(self) -> dict:
        """Get current control state"""
        return {
            "inhibition": self.inhibition.copy(),
            "urge": self.urge.copy(),
            "allowed": self.allowed.copy(),
            "net_activation": self.urge - self.inhibition,
        }


class InhibitionSystem:
    """Integrated inhibition system combining response and impulse control

    Models right inferior frontal gyrus function
    """

    def __init__(self, n_actions: int = 10, params: Optional[InhibitionParams] = None):
        self.params = params or InhibitionParams()
        self.response_inhibitor = ResponseInhibitor(self.params)
        self.impulse_controller = ImpulseController(n_actions, self.params)

        # rIFG activation level
        self.rifg_activation = np.zeros(self.params.n_units)
        # pre-SMA activation
        self.presma_activation = np.zeros(self.params.n_units)

    def process_stop_signal(self) -> float:
        """Process incoming stop signal

        Returns:
            Inhibition strength sent to motor system
        """
        # rIFG responds to stop signal
        self.rifg_activation += np.random.rand(self.params.n_units) * 0.5
        self.rifg_activation = np.clip(self.rifg_activation, 0, 1)

        # pre-SMA also activates
        self.presma_activation += np.random.rand(self.params.n_units) * 0.3
        self.presma_activation = np.clip(self.presma_activation, 0, 1)

        # Combined inhibition signal
        inhibition_signal = (
            np.mean(self.rifg_activation) * 0.7 + np.mean(self.presma_activation) * 0.3
        )

        return inhibition_signal

    def run_go_nogo_trial(self, stimulus_type: str) -> dict:
        """Run a go/no-go trial

        Args:
            stimulus_type: 'go' or 'nogo'

        Returns:
            Trial results
        """
        self.response_inhibitor.reset()

        is_nogo = stimulus_type == "nogo"

        dt = 1.0
        max_time = 500.0

        for t in np.arange(0, max_time, dt):
            self.response_inhibitor.update_go(dt)

            if is_nogo:
                # Continuous inhibition on nogo trials
                self.response_inhibitor.update_stop(dt)

            if self.response_inhibitor.response_made:
                break

        correct = (not is_nogo and self.response_inhibitor.response_made) or (
            is_nogo and not self.response_inhibitor.response_made
        )

        return {
            "stimulus_type": stimulus_type,
            "response_made": self.response_inhibitor.response_made,
            "correct": correct,
            "reaction_time": t if self.response_inhibitor.response_made else None,
        }

    def update(self, dt: float = 1.0):
        """Update system state"""
        # Decay activations
        self.rifg_activation *= 1 - self.params.decay_rate * dt
        self.presma_activation *= 1 - self.params.decay_rate * dt

    def get_inhibition_strength(self) -> float:
        """Get current overall inhibition strength"""
        return np.mean(self.rifg_activation) * self.params.inhibition_strength
