"""Tests for sensory memory components"""

import pytest
import numpy as np

from neuro.modules.m04_memory.sensory_memory import IconicBuffer, EchoicBuffer


class TestIconicBuffer:
    """Tests for visual sensory memory"""

    def test_capture_and_read(self):
        """Test basic capture and read"""
        buffer = IconicBuffer(decay_time=0.250)
        data = np.array([1, 2, 3, 4, 5])

        buffer.capture(data)
        result = buffer.read()

        assert result is not None
        np.testing.assert_array_equal(result, data)

    def test_decay_after_250ms(self):
        """Test that data decays after 250ms"""
        buffer = IconicBuffer(decay_time=0.250)
        data = np.array([1, 2, 3])

        # Use simulated time
        current_time = 0.0
        buffer.set_time_function(lambda: current_time)

        buffer.capture(data)
        assert buffer.is_available()

        # Advance past decay time
        current_time = 2.0  # 2 seconds later
        assert not buffer.is_available()

    def test_strength_decay(self):
        """Test exponential strength decay"""
        buffer = IconicBuffer(decay_time=0.250)
        data = np.array([1, 2, 3])

        current_time = 0.0
        buffer.set_time_function(lambda: current_time)
        buffer.capture(data)

        # Check strength at different times
        _, strength_0 = buffer.read_with_strength()
        assert strength_0 == pytest.approx(1.0, abs=0.01)

        current_time = 0.250  # One time constant
        _, strength_1 = buffer.read_with_strength()
        assert strength_1 == pytest.approx(0.368, abs=0.02)  # e^-1

    def test_clear(self):
        """Test buffer clearing"""
        buffer = IconicBuffer()
        buffer.capture(np.array([1, 2, 3]))
        buffer.clear()

        assert buffer.read() is None


class TestEchoicBuffer:
    """Tests for auditory sensory memory"""

    def test_capture_and_read(self):
        """Test basic capture and read"""
        buffer = EchoicBuffer(decay_time=3.5)
        data = np.array([0.1, 0.2, 0.3])

        buffer.capture(data, duration=0.1)
        result = buffer.read_window()

        assert result is not None
        np.testing.assert_array_equal(result, data)

    def test_rolling_window(self):
        """Test that buffer maintains rolling window"""
        buffer = EchoicBuffer(decay_time=3.5)

        current_time = 0.0
        buffer.set_time_function(lambda: current_time)

        # Add multiple chunks
        buffer.capture(np.array([1, 2]), duration=0.1)
        current_time = 1.0
        buffer.capture(np.array([3, 4]), duration=0.1)

        # Both should be available
        result = buffer.read_window()
        assert len(result) == 4

    def test_decay_after_3_5_seconds(self):
        """Test that old segments decay"""
        buffer = EchoicBuffer(decay_time=3.5)

        current_time = 0.0
        buffer.set_time_function(lambda: current_time)

        buffer.capture(np.array([1, 2]), duration=0.1)

        # Advance past decay time
        current_time = 5.0
        buffer.decay(current_time)

        assert not buffer.is_available()

    def test_partial_window(self):
        """Test reading partial time window"""
        buffer = EchoicBuffer(decay_time=3.5)

        current_time = 0.0
        buffer.set_time_function(lambda: current_time)

        buffer.capture(np.array([1, 2]), duration=0.1)
        current_time = 1.0
        buffer.capture(np.array([3, 4]), duration=0.1)
        current_time = 2.0
        buffer.capture(np.array([5, 6]), duration=0.1)

        # Read only last 1 second
        result = buffer.read_window(duration=1.5)
        # Should only include chunks from t=1.0 and t=2.0
        assert len(result) == 4  # [3, 4, 5, 6]


class TestSensoryIntegration:
    """Integration tests for sensory memory"""

    def test_visual_faster_than_auditory(self):
        """Iconic decays faster than echoic"""
        iconic = IconicBuffer(decay_time=0.250)
        echoic = EchoicBuffer(decay_time=3.5)

        current_time = 0.0

        def time_fn():
            return current_time

        iconic.set_time_function(time_fn)
        echoic.set_time_function(time_fn)

        iconic.capture(np.array([1]))
        echoic.capture(np.array([1]))

        # After 2 seconds (iconic decay_time=0.25s, so exp(-2/0.25)≈0.0003 < 0.01 threshold)
        current_time = 2.0

        assert not iconic.is_available()  # Decayed
        assert echoic.is_available()  # Still available (decay_time=3.5s)
