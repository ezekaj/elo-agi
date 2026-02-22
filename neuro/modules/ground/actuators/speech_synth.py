"""
Speech Synthesizer: Voice output generation.

Implements text-to-speech synthesis for verbal communication.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np


class Voice(Enum):
    """Available voices."""

    MALE_1 = "male_1"
    MALE_2 = "male_2"
    FEMALE_1 = "female_1"
    FEMALE_2 = "female_2"
    NEUTRAL = "neutral"


class EmotionType(Enum):
    """Emotion types for prosody."""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEARFUL = "fearful"


@dataclass
class SpeechConfig:
    """Configuration for speech synthesis."""

    voice: Voice = Voice.NEUTRAL
    sample_rate: int = 22050
    pitch_hz: float = 150.0  # Base pitch
    speaking_rate: float = 1.0
    volume: float = 1.0


@dataclass
class Phoneme:
    """A single phoneme."""

    symbol: str
    duration_ms: float
    pitch_contour: Optional[np.ndarray] = None


@dataclass
class Utterance:
    """A synthesized utterance."""

    text: str
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    phonemes: List[Phoneme] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProsodyParams:
    """Prosody parameters for speech."""

    pitch_shift: float = 0.0  # Semitones
    pitch_range: float = 1.0  # Multiplier
    duration_scale: float = 1.0
    energy_scale: float = 1.0
    emotion: EmotionType = EmotionType.NEUTRAL


class SpeechSynthesizer:
    """
    Text-to-speech synthesizer.

    Implements:
    - Text to phoneme conversion
    - Phoneme to audio synthesis
    - Prosody control
    - Voice selection
    """

    def __init__(
        self,
        synth_id: str = "synth_0",
        config: Optional[SpeechConfig] = None,
    ):
        self.synth_id = synth_id
        self.config = config or SpeechConfig()

        # Phoneme dictionary (simplified)
        self._phoneme_dict = self._build_phoneme_dict()

        # Utterance history
        self._utterance_history: List[Utterance] = []
        self._max_history = 100

        # Current state
        self._is_speaking = False

    def _build_phoneme_dict(self) -> Dict[str, List[str]]:
        """Build simple grapheme-to-phoneme dictionary."""
        return {
            "a": ["AH"],
            "e": ["EH"],
            "i": ["IH"],
            "o": ["OW"],
            "u": ["UH"],
            "b": ["B"],
            "c": ["K"],
            "d": ["D"],
            "f": ["F"],
            "g": ["G"],
            "h": ["HH"],
            "j": ["JH"],
            "k": ["K"],
            "l": ["L"],
            "m": ["M"],
            "n": ["N"],
            "p": ["P"],
            "r": ["R"],
            "s": ["S"],
            "t": ["T"],
            "v": ["V"],
            "w": ["W"],
            "x": ["K", "S"],
            "y": ["Y"],
            "z": ["Z"],
            " ": ["SIL"],
            ".": ["SIL"],
            ",": ["SIL"],
        }

    def synthesize(
        self,
        text: str,
        prosody: Optional[ProsodyParams] = None,
    ) -> Utterance:
        """
        Synthesize speech from text.

        Args:
            text: Input text
            prosody: Prosody parameters

        Returns:
            Synthesized utterance
        """
        prosody = prosody or ProsodyParams()
        self._is_speaking = True

        # Convert text to phonemes
        phonemes = self._text_to_phonemes(text)

        # Apply prosody
        phonemes = self._apply_prosody(phonemes, prosody)

        # Synthesize audio
        audio = self._phonemes_to_audio(phonemes, prosody)

        duration = len(audio) / self.config.sample_rate

        utterance = Utterance(
            text=text,
            audio=audio,
            sample_rate=self.config.sample_rate,
            duration_seconds=duration,
            phonemes=phonemes,
            metadata={
                "voice": self.config.voice.value,
                "prosody": {
                    "pitch_shift": prosody.pitch_shift,
                    "duration_scale": prosody.duration_scale,
                    "emotion": prosody.emotion.value,
                },
            },
        )

        # Store history
        self._utterance_history.append(utterance)
        if len(self._utterance_history) > self._max_history:
            self._utterance_history.pop(0)

        self._is_speaking = False
        return utterance

    def _text_to_phonemes(self, text: str) -> List[Phoneme]:
        """Convert text to phoneme sequence."""
        text = text.lower()
        phonemes = []

        for char in text:
            if char in self._phoneme_dict:
                for p_symbol in self._phoneme_dict[char]:
                    duration = 50.0 if p_symbol == "SIL" else 80.0
                    phonemes.append(
                        Phoneme(
                            symbol=p_symbol,
                            duration_ms=duration,
                        )
                    )
            else:
                # Unknown character - skip
                pass

        return phonemes

    def _apply_prosody(
        self,
        phonemes: List[Phoneme],
        prosody: ProsodyParams,
    ) -> List[Phoneme]:
        """Apply prosody modifications to phonemes."""
        modified = []

        for p in phonemes:
            new_p = Phoneme(
                symbol=p.symbol,
                duration_ms=p.duration_ms * prosody.duration_scale,
            )

            # Generate pitch contour
            n_samples = int(new_p.duration_ms * self.config.sample_rate / 1000)
            base_pitch = self.config.pitch_hz * (2 ** (prosody.pitch_shift / 12))

            # Emotion-based contour
            if prosody.emotion == EmotionType.HAPPY:
                contour = base_pitch * (1 + 0.1 * np.sin(np.linspace(0, 2 * np.pi, n_samples)))
            elif prosody.emotion == EmotionType.SAD:
                contour = base_pitch * np.linspace(1.0, 0.9, n_samples)
            elif prosody.emotion == EmotionType.ANGRY:
                contour = base_pitch * (1.1 + 0.05 * np.random.randn(n_samples))
            else:
                contour = base_pitch * np.ones(n_samples)

            new_p.pitch_contour = contour * prosody.pitch_range

            modified.append(new_p)

        return modified

    def _phonemes_to_audio(
        self,
        phonemes: List[Phoneme],
        prosody: ProsodyParams,
    ) -> np.ndarray:
        """Synthesize audio from phonemes."""
        audio_segments = []

        for phoneme in phonemes:
            duration_samples = int(phoneme.duration_ms * self.config.sample_rate / 1000)

            if phoneme.symbol == "SIL":
                # Silence
                segment = np.zeros(duration_samples)
            else:
                # Generate voiced/unvoiced sound
                segment = self._generate_phoneme_audio(phoneme, duration_samples, prosody)

            audio_segments.append(segment)

        if not audio_segments:
            return np.array([])

        audio = np.concatenate(audio_segments)
        audio = audio * self.config.volume * prosody.energy_scale

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9

        return audio.astype(np.float32)

    def _generate_phoneme_audio(
        self,
        phoneme: Phoneme,
        n_samples: int,
        prosody: ProsodyParams,
    ) -> np.ndarray:
        """Generate audio for a single phoneme."""
        t = np.linspace(0, phoneme.duration_ms / 1000, n_samples)

        # Determine if voiced or unvoiced
        voiced_phonemes = ["AH", "EH", "IH", "OW", "UH", "M", "N", "L", "R", "W", "Y"]
        is_voiced = phoneme.symbol in voiced_phonemes

        if is_voiced:
            # Use pitch contour
            if phoneme.pitch_contour is not None:
                pitch = phoneme.pitch_contour
            else:
                pitch = self.config.pitch_hz * np.ones(n_samples)

            # Generate harmonics
            phase = np.cumsum(2 * np.pi * pitch / self.config.sample_rate)
            audio = 0.5 * np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)

            # Apply formant envelope (simplified)
            envelope = np.exp(-t * 5) * (1 - np.exp(-t * 50))
            audio = audio * envelope

        else:
            # Unvoiced - noise-based
            audio = np.random.randn(n_samples) * 0.3

            # Shape noise based on phoneme
            if phoneme.symbol in ["S", "F", "HH"]:
                # High-frequency noise
                audio = np.convolve(audio, [0.1, 0.2, 0.4, 0.2, 0.1], mode="same")
            elif phoneme.symbol in ["SH", "CH"]:
                # Broader noise
                audio = np.convolve(audio, np.ones(10) / 10, mode="same")

            envelope = np.exp(-t * 3) * (1 - np.exp(-t * 30))
            audio = audio * envelope

        return audio

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._is_speaking

    def stop(self) -> None:
        """Stop current synthesis."""
        self._is_speaking = False

    def get_recent_utterances(self, n: int = 5) -> List[Utterance]:
        """Get recent utterances."""
        return self._utterance_history[-n:]

    def statistics(self) -> Dict[str, Any]:
        """Get synthesizer statistics."""
        total_duration = sum(u.duration_seconds for u in self._utterance_history)

        return {
            "synth_id": self.synth_id,
            "voice": self.config.voice.value,
            "utterances_generated": len(self._utterance_history),
            "total_duration_seconds": total_duration,
            "is_speaking": self._is_speaking,
        }


class ProsodyController:
    """
    Control prosody for expressive speech.

    Implements:
    - Emotion-based prosody
    - Emphasis and stress
    - Intonation patterns
    """

    def __init__(self):
        # Emotion presets
        self._emotion_presets = {
            EmotionType.NEUTRAL: ProsodyParams(
                pitch_shift=0.0,
                pitch_range=1.0,
                duration_scale=1.0,
                energy_scale=1.0,
                emotion=EmotionType.NEUTRAL,
            ),
            EmotionType.HAPPY: ProsodyParams(
                pitch_shift=2.0,
                pitch_range=1.3,
                duration_scale=0.9,
                energy_scale=1.2,
                emotion=EmotionType.HAPPY,
            ),
            EmotionType.SAD: ProsodyParams(
                pitch_shift=-2.0,
                pitch_range=0.7,
                duration_scale=1.2,
                energy_scale=0.8,
                emotion=EmotionType.SAD,
            ),
            EmotionType.ANGRY: ProsodyParams(
                pitch_shift=1.0,
                pitch_range=1.5,
                duration_scale=0.85,
                energy_scale=1.4,
                emotion=EmotionType.ANGRY,
            ),
            EmotionType.SURPRISED: ProsodyParams(
                pitch_shift=4.0,
                pitch_range=1.4,
                duration_scale=0.8,
                energy_scale=1.3,
                emotion=EmotionType.SURPRISED,
            ),
            EmotionType.FEARFUL: ProsodyParams(
                pitch_shift=3.0,
                pitch_range=1.6,
                duration_scale=1.1,
                energy_scale=0.9,
                emotion=EmotionType.FEARFUL,
            ),
        }

    def get_emotion_prosody(self, emotion: EmotionType) -> ProsodyParams:
        """Get prosody parameters for an emotion."""
        return self._emotion_presets.get(emotion, self._emotion_presets[EmotionType.NEUTRAL])

    def blend_emotions(
        self,
        emotion1: EmotionType,
        emotion2: EmotionType,
        weight: float = 0.5,
    ) -> ProsodyParams:
        """
        Blend two emotion prosodies.

        Args:
            emotion1: First emotion
            emotion2: Second emotion
            weight: Blend weight (0 = emotion1, 1 = emotion2)

        Returns:
            Blended prosody parameters
        """
        p1 = self._emotion_presets[emotion1]
        p2 = self._emotion_presets[emotion2]

        return ProsodyParams(
            pitch_shift=p1.pitch_shift * (1 - weight) + p2.pitch_shift * weight,
            pitch_range=p1.pitch_range * (1 - weight) + p2.pitch_range * weight,
            duration_scale=p1.duration_scale * (1 - weight) + p2.duration_scale * weight,
            energy_scale=p1.energy_scale * (1 - weight) + p2.energy_scale * weight,
            emotion=emotion1 if weight < 0.5 else emotion2,
        )

    def apply_emphasis(
        self,
        prosody: ProsodyParams,
        emphasis_level: float = 1.5,
    ) -> ProsodyParams:
        """
        Apply emphasis to prosody.

        Args:
            prosody: Base prosody
            emphasis_level: Emphasis multiplier

        Returns:
            Emphasized prosody
        """
        return ProsodyParams(
            pitch_shift=prosody.pitch_shift + 1.0,
            pitch_range=prosody.pitch_range * emphasis_level,
            duration_scale=prosody.duration_scale * 1.1,
            energy_scale=prosody.energy_scale * emphasis_level,
            emotion=prosody.emotion,
        )

    def create_question_intonation(
        self,
        base_prosody: ProsodyParams,
    ) -> ProsodyParams:
        """
        Create question intonation (rising pitch at end).

        Args:
            base_prosody: Base prosody parameters

        Returns:
            Question prosody
        """
        return ProsodyParams(
            pitch_shift=base_prosody.pitch_shift + 3.0,
            pitch_range=base_prosody.pitch_range * 1.2,
            duration_scale=base_prosody.duration_scale * 1.1,
            energy_scale=base_prosody.energy_scale,
            emotion=base_prosody.emotion,
        )

    def analyze_text_sentiment(self, text: str) -> EmotionType:
        """
        Simple sentiment analysis for prosody selection.

        Args:
            text: Input text

        Returns:
            Detected emotion type
        """
        text_lower = text.lower()

        # Simple keyword matching
        happy_words = ["happy", "great", "wonderful", "excellent", "joy", "love", "excited"]
        sad_words = ["sad", "sorry", "unfortunate", "miss", "lonely", "depressed"]
        angry_words = ["angry", "furious", "hate", "annoying", "frustrated"]
        surprised_words = ["wow", "amazing", "incredible", "unexpected", "shocked"]
        fearful_words = ["scared", "afraid", "worried", "nervous", "anxious"]

        for word in happy_words:
            if word in text_lower:
                return EmotionType.HAPPY

        for word in sad_words:
            if word in text_lower:
                return EmotionType.SAD

        for word in angry_words:
            if word in text_lower:
                return EmotionType.ANGRY

        for word in surprised_words:
            if word in text_lower:
                return EmotionType.SURPRISED

        for word in fearful_words:
            if word in text_lower:
                return EmotionType.FEARFUL

        # Check for question
        if "?" in text:
            return EmotionType.SURPRISED

        return EmotionType.NEUTRAL

    def statistics(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            "n_emotion_presets": len(self._emotion_presets),
            "available_emotions": [e.value for e in self._emotion_presets.keys()],
        }
