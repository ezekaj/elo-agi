"""
Auditory Imagery - Inner Hearing

Located in temporal cortex, auditory imagery allows:
- "Hearing" sounds in the mind
- Mental rehearsal of music
- Inner speech and verbal thought
- Sound manipulation and composition
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class AuditoryProperty(Enum):
    """Properties of auditory images"""

    PITCH = "pitch"
    VOLUME = "volume"
    TIMBRE = "timbre"
    DURATION = "duration"
    RHYTHM = "rhythm"
    TEMPO = "tempo"
    LOCATION = "location"  # Spatial position of sound


@dataclass
class AuditoryImage:
    """A mental auditory image"""

    id: str
    description: str
    properties: Dict[AuditoryProperty, Any]
    vividness: float
    duration: float  # seconds
    is_speech: bool = False
    is_music: bool = False
    sequence: List["AuditoryImage"] = field(default_factory=list)


class AuditoryImagery:
    """
    Auditory imagery system - inner hearing.

    Located in temporal cortex, enables:
    - Mental "hearing" of sounds
    - Inner speech
    - Musical imagination
    - Sound manipulation
    """

    def __init__(self, default_vividness: float = 0.6):
        self.sounds: Dict[str, AuditoryImage] = {}
        self.default_vividness = default_vividness
        self._inner_voice_active = False

    def create_sound(
        self,
        sound_id: str,
        description: str,
        properties: Optional[Dict[AuditoryProperty, Any]] = None,
        duration: float = 1.0,
        is_speech: bool = False,
        is_music: bool = False,
    ) -> AuditoryImage:
        """Create a mental sound image"""
        sound = AuditoryImage(
            id=sound_id,
            description=description,
            properties=properties or {},
            vividness=self.default_vividness,
            duration=duration,
            is_speech=is_speech,
            is_music=is_music,
        )

        self.sounds[sound_id] = sound
        return sound

    def inner_speech(self, text: str) -> AuditoryImage:
        """
        Generate inner speech - "hearing" words in your head.

        Inner speech is crucial for verbal thinking and creativity.
        """
        self._inner_voice_active = True

        # Estimate duration from text length
        words = len(text.split())
        duration = words * 0.3  # ~200 words per minute

        speech = self.create_sound(
            f"speech_{len(self.sounds)}",
            text,
            properties={
                AuditoryProperty.VOLUME: 0.5,  # Inner speech is quieter
                AuditoryProperty.PITCH: "normal",
            },
            duration=duration,
            is_speech=True,
        )

        return speech

    def imagine_music(
        self, description: str, tempo: int = 120, key: str = "C major"
    ) -> AuditoryImage:
        """
        Imagine music - musical mental imagery.

        Musicians often have vivid musical imagery.
        """
        music = self.create_sound(
            f"music_{len(self.sounds)}",
            description,
            properties={
                AuditoryProperty.TEMPO: tempo,
                AuditoryProperty.TIMBRE: "melodic",
                AuditoryProperty.RHYTHM: "structured",
            },
            duration=10.0,  # Default segment
            is_music=True,
        )

        return music

    def imagine_sound(self, description: str) -> AuditoryImage:
        """
        Imagine a general sound from description.
        """
        properties = self._extract_properties(description)

        return self.create_sound(
            f"sound_{len(self.sounds)}", description, properties=properties, duration=2.0
        )

    def _extract_properties(self, description: str) -> Dict[AuditoryProperty, Any]:
        """Extract auditory properties from description"""
        properties = {}
        desc_lower = description.lower()

        # Volume
        if "loud" in desc_lower:
            properties[AuditoryProperty.VOLUME] = "loud"
        elif "quiet" in desc_lower or "soft" in desc_lower:
            properties[AuditoryProperty.VOLUME] = "quiet"

        # Pitch
        if "high" in desc_lower:
            properties[AuditoryProperty.PITCH] = "high"
        elif "low" in desc_lower or "deep" in desc_lower:
            properties[AuditoryProperty.PITCH] = "low"

        return properties

    def change_pitch(self, sound_id: str, semitones: int) -> AuditoryImage:
        """Mentally transpose a sound"""
        if sound_id not in self.sounds:
            raise ValueError(f"Sound {sound_id} not found")

        original = self.sounds[sound_id]
        transposed_id = f"{sound_id}_pitch_{semitones}"

        new_props = original.properties.copy()
        current_pitch = new_props.get(AuditoryProperty.PITCH, "normal")

        if semitones > 0:
            new_props[AuditoryProperty.PITCH] = "higher"
        else:
            new_props[AuditoryProperty.PITCH] = "lower"

        transposed = AuditoryImage(
            id=transposed_id,
            description=f"{original.description} (transposed {semitones} semitones)",
            properties=new_props,
            vividness=original.vividness * 0.9,
            duration=original.duration,
            is_speech=original.is_speech,
            is_music=original.is_music,
        )

        self.sounds[transposed_id] = transposed
        return transposed

    def change_tempo(self, sound_id: str, factor: float) -> AuditoryImage:
        """Change the tempo/speed of a sound"""
        if sound_id not in self.sounds:
            raise ValueError(f"Sound {sound_id} not found")

        original = self.sounds[sound_id]
        new_id = f"{sound_id}_tempo_{factor}"

        new_props = original.properties.copy()
        current_tempo = new_props.get(AuditoryProperty.TEMPO, 120)
        if isinstance(current_tempo, (int, float)):
            new_props[AuditoryProperty.TEMPO] = current_tempo * factor

        modified = AuditoryImage(
            id=new_id,
            description=f"{original.description} (tempo x{factor})",
            properties=new_props,
            vividness=original.vividness * 0.9,
            duration=original.duration / factor,
            is_speech=original.is_speech,
            is_music=original.is_music,
        )

        self.sounds[new_id] = modified
        return modified

    def sequence_sounds(self, sound_ids: List[str]) -> AuditoryImage:
        """
        Create a sequence of sounds.

        Mental composition - arranging sounds in time.
        """
        sounds = [self.sounds[sid] for sid in sound_ids if sid in self.sounds]

        if not sounds:
            raise ValueError("No valid sounds to sequence")

        total_duration = sum(s.duration for s in sounds)
        seq_id = f"sequence_{len(self.sounds)}"

        sequence = AuditoryImage(
            id=seq_id,
            description=f"Sequence: {' -> '.join(s.description for s in sounds)}",
            properties={AuditoryProperty.DURATION: total_duration},
            vividness=np.mean([s.vividness for s in sounds]),
            duration=total_duration,
            sequence=sounds,
        )

        self.sounds[seq_id] = sequence
        return sequence

    def layer_sounds(self, sound_ids: List[str]) -> AuditoryImage:
        """
        Layer multiple sounds together.

        Like imagining multiple instruments playing simultaneously.
        """
        sounds = [self.sounds[sid] for sid in sound_ids if sid in self.sounds]

        if not sounds:
            raise ValueError("No valid sounds to layer")

        max_duration = max(s.duration for s in sounds)
        layer_id = f"layer_{len(self.sounds)}"

        layered = AuditoryImage(
            id=layer_id,
            description=f"Layered: {' + '.join(s.description for s in sounds)}",
            properties={AuditoryProperty.DURATION: max_duration},
            vividness=np.mean([s.vividness for s in sounds]) * 0.8,  # Complexity cost
            duration=max_duration,
            sequence=sounds,
        )

        self.sounds[layer_id] = layered
        return layered

    def get_vividness(self, sound_id: str) -> float:
        """Get vividness of a sound"""
        if sound_id in self.sounds:
            return self.sounds[sound_id].vividness
        return 0.0

    def replay(self, sound_id: str) -> Optional[AuditoryImage]:
        """
        Mentally replay a sound.

        Rehearsal strengthens the representation.
        """
        if sound_id not in self.sounds:
            return None

        sound = self.sounds[sound_id]
        sound.vividness = min(1.0, sound.vividness + 0.1)
        return sound
