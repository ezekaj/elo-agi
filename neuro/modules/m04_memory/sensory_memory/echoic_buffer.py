"""
Echoic Buffer: Auditory sensory memory

Based on research showing auditory sensory memory persists for 3-4 seconds,
longer than visual to support temporal pattern processing in speech/music.

Brain region: Auditory cortex
"""

import numpy as np
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class AudioSegment:
    """A single audio segment in the echoic buffer"""

    data: np.ndarray
    timestamp: float
    duration: float  # seconds


class EchoicBuffer:
    """
    Auditory cortex buffer - 3-4 second decay

    Implements echoic memory as a rolling window of audio data.
    Longer duration than iconic to support temporal pattern processing.
    """

    def __init__(self, decay_time: float = 3.5, sample_rate: int = 44100):
        """
        Initialize echoic buffer.

        Args:
            decay_time: Maximum buffer duration in seconds (default 3.5s)
            sample_rate: Audio sample rate for duration calculations
        """
        self.decay_time = decay_time
        self.sample_rate = sample_rate
        self._segments: deque = deque()
        self._time_fn = time.time

    def capture(self, audio_chunk: np.ndarray, duration: Optional[float] = None) -> None:
        """
        Add audio chunk to the buffer.

        Args:
            audio_chunk: Audio data array
            duration: Duration in seconds (calculated from sample_rate if None)
        """
        if duration is None:
            duration = len(audio_chunk) / self.sample_rate

        segment = AudioSegment(
            data=np.copy(audio_chunk), timestamp=self._time_fn(), duration=duration
        )
        self._segments.append(segment)

        # Clean up expired segments
        self._cleanup()

    def _cleanup(self) -> None:
        """Remove expired segments from buffer"""
        current_time = self._time_fn()
        cutoff = current_time - self.decay_time

        while self._segments and self._segments[0].timestamp < cutoff:
            self._segments.popleft()

    def read_window(self, duration: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Access recent audio up to specified duration.

        Args:
            duration: How much history to retrieve (default: all available)

        Returns:
            Concatenated audio data, or None if buffer empty
        """
        self._cleanup()

        if not self._segments:
            return None

        if duration is None:
            duration = self.decay_time

        current_time = self._time_fn()
        cutoff = current_time - duration

        # Collect segments within the requested window
        segments_in_window = [seg.data for seg in self._segments if seg.timestamp >= cutoff]

        if not segments_in_window:
            return None

        return np.concatenate(segments_in_window)

    def read_with_timestamps(self) -> List[Tuple[np.ndarray, float, float]]:
        """
        Access all segments with their timestamps and strengths.

        Returns:
            List of (data, timestamp, strength) tuples
        """
        self._cleanup()

        current_time = self._time_fn()
        result = []

        for seg in self._segments:
            elapsed = current_time - seg.timestamp
            # Linear decay for echoic (simpler model)
            strength = max(0.0, 1.0 - elapsed / self.decay_time)
            result.append((seg.data, seg.timestamp, strength))

        return result

    def decay(self, current_time: Optional[float] = None) -> None:
        """
        Apply decay by removing expired segments.

        Args:
            current_time: Optional explicit time for simulation
        """
        if current_time is not None:
            old_fn = self._time_fn
            self._time_fn = lambda: current_time
            self._cleanup()
            self._time_fn = old_fn
        else:
            self._cleanup()

    def get_duration(self) -> float:
        """
        Get current buffer duration in seconds.

        Returns:
            Total duration of audio in buffer
        """
        self._cleanup()
        return sum(seg.duration for seg in self._segments)

    def is_available(self) -> bool:
        """Check if any data is available"""
        self._cleanup()
        return len(self._segments) > 0

    def get_segment_count(self) -> int:
        """Get number of segments in buffer"""
        self._cleanup()
        return len(self._segments)

    def clear(self) -> None:
        """Clear the buffer"""
        self._segments.clear()

    def set_time_function(self, time_fn) -> None:
        """Set custom time function for simulation"""
        self._time_fn = time_fn
