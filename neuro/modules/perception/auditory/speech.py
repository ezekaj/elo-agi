"""
Speech: Speech and phoneme recognition.

Implements phoneme feature extraction, formant tracking,
and basic speech recognition components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np


class PhonemeCategory(Enum):
    """Categories of phonemes."""

    VOWEL = "vowel"
    STOP = "stop"
    FRICATIVE = "fricative"
    NASAL = "nasal"
    APPROXIMANT = "approximant"
    AFFRICATE = "affricate"
    SILENCE = "silence"


@dataclass
class PhonemeFeatures:
    """Acoustic features of a phoneme."""

    category: PhonemeCategory
    voiced: bool
    formants: List[float]  # F1, F2, F3
    duration: float
    energy: float
    spectral_centroid: float


@dataclass
class PhonemeDetection:
    """Detected phoneme with timing."""

    phoneme: str
    features: PhonemeFeatures
    start_time: float
    end_time: float
    confidence: float


@dataclass
class SpeechOutput:
    """Output from speech processing."""

    phonemes: List[PhonemeDetection]
    formant_tracks: np.ndarray  # (3, n_frames) F1, F2, F3
    voicing: np.ndarray  # (n_frames,) voicing probability
    energy: np.ndarray  # (n_frames,) energy contour
    pitch: np.ndarray  # (n_frames,) F0 estimate
    n_frames: int


class FormantTracker:
    """
    Track formant frequencies from spectral data.
    """

    def __init__(
        self,
        n_formants: int = 3,
        sample_rate: int = 16000,
    ):
        self.n_formants = n_formants
        self.sample_rate = sample_rate

        # Expected formant ranges (Hz)
        self.formant_ranges = [
            (200, 1000),  # F1
            (800, 2500),  # F2
            (1500, 3500),  # F3
        ]

    def track(self, spectrum: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        """
        Extract formants from spectrum.

        Args:
            spectrum: Power spectrum (n_freq,) or (n_freq, n_frames)
            frequencies: Frequency axis (n_freq,)

        Returns:
            Formant frequencies (n_formants,) or (n_formants, n_frames)
        """
        if spectrum.ndim == 1:
            return self._track_frame(spectrum, frequencies)

        n_frames = spectrum.shape[1]
        formants = np.zeros((self.n_formants, n_frames))

        for i in range(n_frames):
            formants[:, i] = self._track_frame(spectrum[:, i], frequencies)

        return formants

    def _track_frame(
        self,
        spectrum: np.ndarray,
        frequencies: np.ndarray,
    ) -> np.ndarray:
        """Track formants for a single frame."""
        formants = np.zeros(self.n_formants)

        # Find peaks in spectrum
        from scipy.signal import find_peaks

        peaks, properties = find_peaks(spectrum, height=spectrum.max() * 0.1)
        peak_freqs = frequencies[peaks]
        peak_heights = spectrum[peaks]

        # Assign peaks to formants based on expected ranges
        for fi, (low, high) in enumerate(self.formant_ranges):
            # Find peaks in this range
            in_range = (peak_freqs >= low) & (peak_freqs <= high)
            if np.any(in_range):
                range_peaks = peaks[in_range]
                range_heights = spectrum[range_peaks]
                best_peak = range_peaks[np.argmax(range_heights)]
                formants[fi] = frequencies[best_peak]
            else:
                # Default to center of range
                formants[fi] = (low + high) / 2

        return formants


class VoicingDetector:
    """
    Detect voiced vs unvoiced speech.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 256,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size

        # Pitch range for voiced speech
        self.min_pitch = 75  # Hz
        self.max_pitch = 500  # Hz

    def detect(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect voicing and estimate pitch.

        Returns:
            voicing: (n_frames,) voicing probability
            pitch: (n_frames,) F0 estimate
        """
        n_samples = len(signal)
        hop_size = self.frame_size // 2
        n_frames = (n_samples - self.frame_size) // hop_size + 1

        voicing = np.zeros(n_frames)
        pitch = np.zeros(n_frames)

        # Min/max lag for pitch search
        min_lag = int(self.sample_rate / self.max_pitch)
        max_lag = int(self.sample_rate / self.min_pitch)

        for i in range(n_frames):
            start = i * hop_size
            end = start + self.frame_size
            frame = signal[start:end]

            # Autocorrelation for pitch
            if len(frame) < max_lag:
                continue

            autocorr = np.correlate(frame, frame, mode="full")
            autocorr = autocorr[len(autocorr) // 2 :]

            # Find peak in pitch range
            search_region = autocorr[min_lag:max_lag]
            if len(search_region) > 0 and np.max(search_region) > 0:
                peak_lag = np.argmax(search_region) + min_lag
                peak_val = autocorr[peak_lag]

                # Voicing = normalized autocorrelation peak
                voicing[i] = peak_val / (autocorr[0] + 1e-8)

                # Pitch from lag
                if voicing[i] > 0.3:
                    pitch[i] = self.sample_rate / peak_lag
                else:
                    pitch[i] = 0

        return voicing, pitch


class PhonemeRecognizer:
    """
    Recognize phonemes from acoustic features.
    """

    def __init__(self):
        # Phoneme templates (simplified)
        self._phoneme_templates = self._create_templates()

    def _create_templates(self) -> Dict[str, PhonemeFeatures]:
        """Create phoneme templates based on typical features."""
        templates = {}

        # Vowels (voiced, distinct formants)
        vowels = {
            "AA": ([700, 1100, 2500], True),  # "father"
            "AE": ([700, 1800, 2500], True),  # "cat"
            "AH": ([600, 1200, 2500], True),  # "but"
            "EH": ([500, 1800, 2500], True),  # "bed"
            "IY": ([300, 2300, 3000], True),  # "beet"
            "IH": ([400, 1900, 2500], True),  # "bit"
            "UW": ([300, 800, 2300], True),  # "boot"
            "UH": ([400, 1100, 2300], True),  # "book"
            "OW": ([500, 800, 2500], True),  # "boat"
        }

        for phone, (formants, voiced) in vowels.items():
            templates[phone] = PhonemeFeatures(
                category=PhonemeCategory.VOWEL,
                voiced=voiced,
                formants=formants,
                duration=0.1,
                energy=0.8,
                spectral_centroid=sum(formants) / len(formants),
            )

        # Stops
        stops = {
            "P": (False, 0.02),
            "B": (True, 0.02),
            "T": (False, 0.02),
            "D": (True, 0.02),
            "K": (False, 0.02),
            "G": (True, 0.02),
        }

        for phone, (voiced, dur) in stops.items():
            templates[phone] = PhonemeFeatures(
                category=PhonemeCategory.STOP,
                voiced=voiced,
                formants=[0, 0, 0],
                duration=dur,
                energy=0.3 if not voiced else 0.5,
                spectral_centroid=2000 if phone in ["T", "D"] else 1500,
            )

        # Fricatives
        fricatives = {
            "S": (False, 4500),
            "Z": (True, 4500),
            "SH": (False, 3500),
            "F": (False, 5000),
            "V": (True, 5000),
            "TH": (False, 6000),
        }

        for phone, (voiced, centroid) in fricatives.items():
            templates[phone] = PhonemeFeatures(
                category=PhonemeCategory.FRICATIVE,
                voiced=voiced,
                formants=[0, 0, 0],
                duration=0.08,
                energy=0.4,
                spectral_centroid=centroid,
            )

        # Nasals
        nasals = {
            "M": [300, 1100, 2500],
            "N": [300, 1500, 2500],
            "NG": [300, 1900, 2500],
        }

        for phone, formants in nasals.items():
            templates[phone] = PhonemeFeatures(
                category=PhonemeCategory.NASAL,
                voiced=True,
                formants=formants,
                duration=0.06,
                energy=0.6,
                spectral_centroid=sum(formants) / len(formants),
            )

        return templates

    def recognize(
        self,
        formants: np.ndarray,
        voicing: np.ndarray,
        energy: np.ndarray,
        frame_rate: float,
    ) -> List[PhonemeDetection]:
        """
        Recognize phonemes from features.

        Args:
            formants: (3, n_frames) formant tracks
            voicing: (n_frames,) voicing probability
            energy: (n_frames,) energy contour
            frame_rate: Frames per second

        Returns:
            List of detected phonemes
        """
        n_frames = formants.shape[1]
        detections = []

        # Segment into regions (simplified: fixed windows)
        window_size = int(0.05 * frame_rate)  # 50ms windows
        hop = window_size // 2

        for start in range(0, n_frames - window_size, hop):
            end = start + window_size

            # Extract features for this segment
            seg_formants = formants[:, start:end].mean(axis=1)
            seg_voicing = voicing[start:end].mean()
            seg_energy = energy[start:end].mean()

            # Skip silence
            if seg_energy < 0.1:
                continue

            # Find best matching phoneme
            best_phone = None
            best_score = float("inf")

            for phone, template in self._phoneme_templates.items():
                score = self._match_score(seg_formants, seg_voicing, seg_energy, template)
                if score < best_score:
                    best_score = score
                    best_phone = phone

            if best_phone and best_score < 1000:
                detections.append(
                    PhonemeDetection(
                        phoneme=best_phone,
                        features=self._phoneme_templates[best_phone],
                        start_time=start / frame_rate,
                        end_time=end / frame_rate,
                        confidence=1.0 / (1.0 + best_score),
                    )
                )

        # Merge consecutive identical phonemes
        detections = self._merge_detections(detections)

        return detections

    def _match_score(
        self,
        formants: np.ndarray,
        voicing: float,
        energy: float,
        template: PhonemeFeatures,
    ) -> float:
        """Compute matching score (lower is better)."""
        score = 0.0

        # Formant distance (for vowels and nasals)
        if template.category in [PhonemeCategory.VOWEL, PhonemeCategory.NASAL]:
            for i, (f, tf) in enumerate(zip(formants, template.formants)):
                weight = 1.0 / (i + 1)  # F1 most important
                score += weight * abs(f - tf)

        # Voicing mismatch
        expected_voicing = 1.0 if template.voiced else 0.0
        score += 500 * abs(voicing - expected_voicing)

        # Energy mismatch
        score += 200 * abs(energy - template.energy)

        return score

    def _merge_detections(
        self,
        detections: List[PhonemeDetection],
    ) -> List[PhonemeDetection]:
        """Merge consecutive identical phonemes."""
        if not detections:
            return detections

        merged = [detections[0]]

        for det in detections[1:]:
            if det.phoneme == merged[-1].phoneme:
                # Extend previous detection
                merged[-1] = PhonemeDetection(
                    phoneme=det.phoneme,
                    features=det.features,
                    start_time=merged[-1].start_time,
                    end_time=det.end_time,
                    confidence=(merged[-1].confidence + det.confidence) / 2,
                )
            else:
                merged.append(det)

        return merged


class SpeechProcessor:
    """
    Complete speech processing pipeline.

    Combines formant tracking, voicing detection, and phoneme recognition.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 256,
        hop_size: int = 128,
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size

        self.formant_tracker = FormantTracker(sample_rate=sample_rate)
        self.voicing_detector = VoicingDetector(
            sample_rate=sample_rate,
            frame_size=frame_size,
        )
        self.phoneme_recognizer = PhonemeRecognizer()

    def process(self, a1_output) -> SpeechOutput:
        """
        Process A1 output for speech recognition.

        Args:
            a1_output: A1Output from primary auditory cortex

        Returns:
            SpeechOutput with phonemes and prosodic features
        """
        # Get tonotopic representation
        tonotopic = a1_output.tonotopic_map
        n_freq, n_time = tonotopic.shape

        # Create frequency axis (approximate)
        frequencies = np.logspace(np.log10(80), np.log10(8000), n_freq)

        # Track formants
        formant_tracks = self.formant_tracker.track(tonotopic, frequencies)

        # Compute energy contour
        energy = np.mean(tonotopic, axis=0)
        energy = energy / (energy.max() + 1e-8)

        # Voicing detection (from onset map as proxy)
        voicing = 1.0 - (a1_output.onset_map.mean(axis=0) / (a1_output.onset_map.max() + 1e-8))

        # Pitch (from rate responses - lower rates correlate with pitch)
        pitch = a1_output.rate_map.mean(axis=0) * 200  # Scale to Hz

        # Frame rate
        frame_rate = n_time  # Approximate

        # Recognize phonemes
        phonemes = self.phoneme_recognizer.recognize(formant_tracks, voicing, energy, frame_rate)

        return SpeechOutput(
            phonemes=phonemes,
            formant_tracks=formant_tracks,
            voicing=voicing,
            energy=energy,
            pitch=pitch,
            n_frames=n_time,
        )

    def get_transcript(self, speech_output: SpeechOutput) -> str:
        """Get text transcript from phoneme detections."""
        return " ".join([p.phoneme for p in speech_output.phonemes])

    def statistics(self) -> Dict[str, Any]:
        """Get speech processor statistics."""
        return {
            "sample_rate": self.sample_rate,
            "frame_size": self.frame_size,
            "n_phoneme_templates": len(self.phoneme_recognizer._phoneme_templates),
        }
