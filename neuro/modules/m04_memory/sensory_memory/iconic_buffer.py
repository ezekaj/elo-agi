"""
Iconic Buffer: Visual sensory memory

Based on research showing visual sensory memory persists for ~250ms
with high capacity but rapid exponential decay.

Brain region: Visual cortex
"""

import numpy as np
import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class IconicTrace:
    """A single iconic memory trace"""

    data: np.ndarray
    timestamp: float
    initial_strength: float = 1.0


class IconicBuffer:
    """
    Visual cortex snapshot buffer - 250ms decay

    Implements the iconic memory store from Sperling's experiments.
    High capacity, very short duration, exponential decay.
    """

    def __init__(self, decay_time: float = 0.250):
        """
        Initialize iconic buffer.

        Args:
            decay_time: Time constant for decay in seconds (default 250ms)
        """
        self.decay_time = decay_time
        self._trace: Optional[IconicTrace] = None
        self._time_fn = time.time

    def capture(self, visual_input: np.ndarray) -> None:
        """
        Store a visual snapshot.

        Args:
            visual_input: Raw visual data (any shape)
        """
        self._trace = IconicTrace(data=np.copy(visual_input), timestamp=self._time_fn())

    def read(self) -> Optional[np.ndarray]:
        """
        Access the iconic trace before it decays.

        Returns:
            The visual data if still available, None if expired
        """
        if not self.is_available():
            return None
        return self._trace.data

    def read_with_strength(self) -> tuple[Optional[np.ndarray], float]:
        """
        Access the iconic trace with its current strength.

        Returns:
            Tuple of (data, strength) where strength is 0-1
        """
        if self._trace is None:
            return None, 0.0

        strength = self._compute_strength()
        if strength <= 0:
            return None, 0.0

        return self._trace.data, strength

    def _compute_strength(self) -> float:
        """Compute current trace strength based on exponential decay"""
        if self._trace is None:
            return 0.0

        elapsed = self._time_fn() - self._trace.timestamp
        # Exponential decay: strength = e^(-t/tau)
        strength = np.exp(-elapsed / self.decay_time)
        return max(0.0, strength)

    def is_available(self) -> bool:
        """Check if data is still accessible (strength > threshold)"""
        return self._compute_strength() > 0.01  # 1% threshold

    def get_remaining_time(self) -> float:
        """
        Get milliseconds until complete decay.

        Returns:
            Remaining time in ms, 0 if already expired
        """
        if self._trace is None:
            return 0.0

        elapsed = self._time_fn() - self._trace.timestamp
        remaining = max(0.0, self.decay_time * 5 - elapsed)  # 5 tau for ~99% decay
        return remaining * 1000  # Convert to ms

    def decay(self, current_time: Optional[float] = None) -> float:
        """
        Compute decay based on elapsed time.

        Args:
            current_time: Optional explicit time (for simulation)

        Returns:
            Current strength after decay (0-1)
        """
        if current_time is not None:
            old_fn = self._time_fn
            self._time_fn = lambda: current_time
            strength = self._compute_strength()
            self._time_fn = old_fn
            return strength
        return self._compute_strength()

    def clear(self) -> None:
        """Clear the buffer"""
        self._trace = None

    def set_time_function(self, time_fn) -> None:
        """Set custom time function for simulation"""
        self._time_fn = time_fn
