"""
Microphone: Audio input processing.

Implements microphone interface for auditory perception.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np
import time


class AudioFormat(Enum):
    """Audio formats."""
    FLOAT32 = "float32"
    INT16 = "int16"
    INT32 = "int32"


@dataclass
class MicrophoneConfig:
    """Configuration for microphone."""
    sample_rate: int = 16000
    channels: int = 1
    buffer_size: int = 1024
    audio_format: AudioFormat = AudioFormat.FLOAT32
    device_id: int = 0
    auto_gain: bool = True


@dataclass
class AudioBuffer:
    """A buffer of audio samples."""
    data: np.ndarray
    timestamp: float
    buffer_id: int
    sample_rate: int
    channels: int
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.data) if self.channels == 1 else self.data.shape[1]


@dataclass
class AudioEvent:
    """An detected audio event."""
    event_type: str
    start_time: float
    end_time: float
    confidence: float
    data: Optional[np.ndarray] = None


class Microphone:
    """
    Microphone interface for audio input.

    Provides:
    - Audio capture
    - Buffer management
    - Basic audio processing
    """

    def __init__(
        self,
        mic_id: str = "mic_0",
        config: Optional[MicrophoneConfig] = None,
    ):
        self.mic_id = mic_id
        self.config = config or MicrophoneConfig()

        self._is_open = False
        self._buffer_count = 0
        self._total_samples = 0

        # Callbacks
        self._on_audio: Optional[Callable[[AudioBuffer], None]] = None

        # Simulated mode
        self._simulated = True

        # Circular buffer for continuous recording
        self._ring_buffer: List[np.ndarray] = []
        self._ring_buffer_max = 100

    def open(self) -> bool:
        """Open the microphone."""
        self._is_open = True
        return True

    def close(self) -> None:
        """Close the microphone."""
        self._is_open = False

    def is_open(self) -> bool:
        """Check if microphone is open."""
        return self._is_open

    def read(self, n_samples: Optional[int] = None) -> Optional[AudioBuffer]:
        """
        Read audio samples.

        Args:
            n_samples: Number of samples to read (default: buffer_size)

        Returns:
            AudioBuffer or None if read failed
        """
        if not self._is_open:
            return None

        n_samples = n_samples or self.config.buffer_size
        self._buffer_count += 1

        # Generate or capture audio
        if self._simulated:
            data = self._generate_test_audio(n_samples)
        else:
            data = self._read_real_audio(n_samples)

        self._total_samples += n_samples

        duration = n_samples / self.config.sample_rate

        buffer = AudioBuffer(
            data=data,
            timestamp=time.time(),
            buffer_id=self._buffer_count,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            duration_seconds=duration,
        )

        # Add to ring buffer
        self._ring_buffer.append(data)
        if len(self._ring_buffer) > self._ring_buffer_max:
            self._ring_buffer.pop(0)

        # Callback
        if self._on_audio:
            self._on_audio(buffer)

        return buffer

    def _generate_test_audio(self, n_samples: int) -> np.ndarray:
        """Generate test audio signal."""
        t = np.linspace(0, n_samples / self.config.sample_rate, n_samples)

        # Mix of frequencies
        signal = (
            0.3 * np.sin(2 * np.pi * 440 * t) +   # A4
            0.2 * np.sin(2 * np.pi * 880 * t) +   # A5
            0.1 * np.sin(2 * np.pi * 220 * t) +   # A3
            0.05 * np.random.randn(n_samples)      # Noise
        )

        if self.config.channels > 1:
            signal = np.tile(signal.reshape(-1, 1), (1, self.config.channels))

        return signal.astype(np.float32)

    def _read_real_audio(self, n_samples: int) -> np.ndarray:
        """Read from real microphone (stub)."""
        return self._generate_test_audio(n_samples)

    def get_recent_audio(self, duration_seconds: float) -> np.ndarray:
        """
        Get recent audio from ring buffer.

        Args:
            duration_seconds: Duration to retrieve

        Returns:
            Audio data
        """
        n_samples = int(duration_seconds * self.config.sample_rate)
        n_buffers = (n_samples // self.config.buffer_size) + 1

        buffers = self._ring_buffer[-n_buffers:]
        if not buffers:
            return np.array([])

        combined = np.concatenate(buffers)
        return combined[-n_samples:]

    def set_callback(
        self,
        on_audio: Optional[Callable[[AudioBuffer], None]] = None,
    ) -> None:
        """Set audio callback."""
        self._on_audio = on_audio

    def statistics(self) -> Dict[str, Any]:
        """Get microphone statistics."""
        total_duration = self._total_samples / self.config.sample_rate

        return {
            "mic_id": self.mic_id,
            "is_open": self._is_open,
            "buffer_count": self._buffer_count,
            "total_samples": self._total_samples,
            "total_duration_seconds": total_duration,
            "config": {
                "sample_rate": self.config.sample_rate,
                "channels": self.config.channels,
                "buffer_size": self.config.buffer_size,
            },
        }


class AudioProcessor:
    """
    Process audio buffers for perception.

    Implements:
    - Spectral analysis
    - Feature extraction
    - Voice activity detection
    - Audio event detection
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._buffer_count = 0

    def compute_spectrum(
        self,
        audio: AudioBuffer,
        window_size: int = 512,
        hop_size: int = 256,
    ) -> np.ndarray:
        """
        Compute short-time Fourier transform.

        Args:
            audio: Audio buffer
            window_size: FFT window size
            hop_size: Hop size between windows

        Returns:
            Magnitude spectrogram
        """
        data = audio.data
        if data.ndim > 1:
            data = data[:, 0]  # Use first channel

        n_frames = (len(data) - window_size) // hop_size + 1
        spectrum = np.zeros((n_frames, window_size // 2 + 1))

        window = np.hanning(window_size)

        for i in range(n_frames):
            start = i * hop_size
            frame = data[start:start + window_size] * window
            fft = np.fft.rfft(frame)
            spectrum[i] = np.abs(fft)

        return spectrum

    def compute_mel_spectrum(
        self,
        audio: AudioBuffer,
        n_mels: int = 40,
        window_size: int = 512,
    ) -> np.ndarray:
        """
        Compute mel-frequency spectrogram.

        Args:
            audio: Audio buffer
            n_mels: Number of mel bands
            window_size: FFT window size

        Returns:
            Mel spectrogram
        """
        spectrum = self.compute_spectrum(audio, window_size)

        # Create mel filterbank
        n_fft = window_size // 2 + 1
        mel_filters = self._create_mel_filterbank(
            n_mels, n_fft, audio.sample_rate
        )

        # Apply filterbank
        mel_spectrum = spectrum @ mel_filters.T

        return mel_spectrum

    def _create_mel_filterbank(
        self,
        n_mels: int,
        n_fft: int,
        sample_rate: int,
    ) -> np.ndarray:
        """Create mel filterbank."""
        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        f_min = 0
        f_max = sample_rate / 2

        mel_min = hz_to_mel(f_min)
        mel_max = hz_to_mel(f_max)

        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = np.floor((n_fft * 2 * hz_points) / sample_rate).astype(int)

        filterbank = np.zeros((n_mels, n_fft))

        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Rising edge
            for j in range(left, center):
                if center != left:
                    filterbank[i, j] = (j - left) / (center - left)

            # Falling edge
            for j in range(center, right):
                if right != center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def extract_mfcc(
        self,
        audio: AudioBuffer,
        n_mfcc: int = 13,
        n_mels: int = 40,
    ) -> np.ndarray:
        """
        Extract MFCCs (Mel-frequency cepstral coefficients).

        Args:
            audio: Audio buffer
            n_mfcc: Number of MFCCs
            n_mels: Number of mel bands

        Returns:
            MFCC features
        """
        mel_spectrum = self.compute_mel_spectrum(audio, n_mels)

        # Log mel spectrum
        log_mel = np.log(mel_spectrum + 1e-8)

        # DCT to get MFCCs
        from scipy.fftpack import dct
        mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]

        return mfcc

    def detect_voice_activity(
        self,
        audio: AudioBuffer,
        energy_threshold: float = 0.01,
        zero_crossing_threshold: float = 0.1,
    ) -> Tuple[bool, float]:
        """
        Detect voice activity in audio.

        Args:
            audio: Audio buffer
            energy_threshold: Energy threshold for VAD
            zero_crossing_threshold: Zero crossing rate threshold

        Returns:
            Tuple of (voice_detected, confidence)
        """
        data = audio.data
        if data.ndim > 1:
            data = data[:, 0]

        # Compute energy
        energy = np.mean(data ** 2)

        # Compute zero crossing rate
        signs = np.sign(data)
        zero_crossings = np.sum(np.abs(np.diff(signs))) / (2 * len(data))

        # Voice detection heuristics
        energy_check = energy > energy_threshold
        zcr_check = zero_crossing_threshold < zero_crossings < 0.5

        voice_detected = energy_check and zcr_check
        confidence = min(1.0, energy / energy_threshold * 0.5 + (1 - abs(zero_crossings - 0.3) / 0.3) * 0.5)

        return voice_detected, float(confidence)

    def detect_events(
        self,
        audio: AudioBuffer,
        min_duration: float = 0.1,
    ) -> List[AudioEvent]:
        """
        Detect audio events (sound onsets).

        Args:
            audio: Audio buffer
            min_duration: Minimum event duration

        Returns:
            List of detected events
        """
        data = audio.data
        if data.ndim > 1:
            data = data[:, 0]

        # Simple onset detection using spectral flux
        spectrum = self.compute_spectrum(audio, window_size=256, hop_size=128)

        # Spectral flux
        flux = np.sum(np.maximum(np.diff(spectrum, axis=0), 0), axis=1)

        # Threshold
        threshold = np.mean(flux) + np.std(flux)
        onsets = flux > threshold

        # Find event boundaries
        events = []
        in_event = False
        event_start = 0

        hop_duration = 128 / audio.sample_rate

        for i, is_onset in enumerate(onsets):
            t = i * hop_duration

            if is_onset and not in_event:
                in_event = True
                event_start = t
            elif not is_onset and in_event:
                in_event = False
                duration = t - event_start
                if duration >= min_duration:
                    events.append(AudioEvent(
                        event_type="sound",
                        start_time=event_start,
                        end_time=t,
                        confidence=float(flux[i - 1] / threshold),
                    ))

        return events

    def compute_pitch(
        self,
        audio: AudioBuffer,
        min_freq: float = 50.0,
        max_freq: float = 500.0,
    ) -> Optional[float]:
        """
        Estimate fundamental frequency using autocorrelation.

        Args:
            audio: Audio buffer
            min_freq: Minimum frequency
            max_freq: Maximum frequency

        Returns:
            Estimated pitch in Hz, or None
        """
        data = audio.data
        if data.ndim > 1:
            data = data[:, 0]

        # Autocorrelation
        n = len(data)
        corr = np.correlate(data, data, mode='full')[n - 1:]

        # Find peaks in valid range
        min_lag = int(audio.sample_rate / max_freq)
        max_lag = int(audio.sample_rate / min_freq)

        corr_segment = corr[min_lag:max_lag]
        if len(corr_segment) == 0:
            return None

        # Find first significant peak
        peak_idx = np.argmax(corr_segment)
        lag = min_lag + peak_idx

        if corr[lag] < 0.3 * corr[0]:
            return None  # No clear pitch

        pitch = audio.sample_rate / lag
        return float(pitch)

    def statistics(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "sample_rate": self.sample_rate,
            "buffers_processed": self._buffer_count,
        }
