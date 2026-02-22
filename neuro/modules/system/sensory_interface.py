"""
Sensory Interface: Processes incoming sensory data.

Adapts various input types to the internal representation
used by the cognitive modules.

Supports:
- Vector input (numpy arrays)
- Text input (converted to embeddings)
- Image input (placeholder for future)
- Multimodal input (combined)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import numpy as np
import time

from .config import SystemConfig


class InputType(Enum):
    """Types of sensory input."""

    VECTOR = "vector"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


@dataclass
class SensoryInput:
    """A single sensory input."""

    input_type: InputType
    raw_data: Any
    processed: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_processed(self) -> bool:
        """Check if input has been processed."""
        return self.processed is not None


@dataclass
class SensoryBuffer:
    """Buffer for recent sensory inputs."""

    capacity: int = 10
    inputs: List[SensoryInput] = field(default_factory=list)

    def add(self, inp: SensoryInput) -> None:
        """Add input to buffer."""
        self.inputs.append(inp)
        if len(self.inputs) > self.capacity:
            self.inputs.pop(0)

    def get_recent(self, n: int = 1) -> List[SensoryInput]:
        """Get n most recent inputs."""
        return self.inputs[-n:]

    def clear(self) -> None:
        """Clear buffer."""
        self.inputs = []


class SensoryInterface:
    """
    Processes sensory inputs for the cognitive system.

    The interface:
    1. Accepts various input types (vector, text, image, etc.)
    2. Converts to internal representation (numpy arrays)
    3. Normalizes to standard dimensions
    4. Maintains a sensory buffer
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.buffer = SensoryBuffer(capacity=10)

        # Processing state
        self._last_input: Optional[SensoryInput] = None
        self._input_count = 0

    def process(self, data: Any, input_type: Optional[InputType] = None) -> SensoryInput:
        """
        Process incoming sensory data.

        Args:
            data: Raw input data
            input_type: Type of input (auto-detected if None)

        Returns:
            Processed SensoryInput
        """
        # Auto-detect type
        if input_type is None:
            input_type = self._detect_type(data)

        # Create input object
        sensory_input = SensoryInput(
            input_type=input_type,
            raw_data=data,
        )

        # Process based on type
        if input_type == InputType.VECTOR:
            sensory_input.processed = self._process_vector(data)
        elif input_type == InputType.TEXT:
            sensory_input.processed = self._process_text(data)
        elif input_type == InputType.IMAGE:
            sensory_input.processed = self._process_image(data)
        elif input_type == InputType.AUDIO:
            sensory_input.processed = self._process_audio(data)
        elif input_type == InputType.MULTIMODAL:
            sensory_input.processed = self._process_multimodal(data)
        else:
            # Fallback
            sensory_input.processed = self._to_vector(data)

        # Add to buffer
        self.buffer.add(sensory_input)
        self._last_input = sensory_input
        self._input_count += 1

        return sensory_input

    def _detect_type(self, data: Any) -> InputType:
        """Auto-detect input type."""
        if isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                return InputType.VECTOR
            elif len(data.shape) == 3:
                return InputType.IMAGE
            elif len(data.shape) == 2:
                # Could be image or audio spectrogram
                return InputType.IMAGE

        elif isinstance(data, str):
            return InputType.TEXT

        elif isinstance(data, (list, tuple)):
            if all(isinstance(x, (int, float)) for x in data):
                return InputType.VECTOR
            else:
                return InputType.MULTIMODAL

        elif isinstance(data, dict):
            return InputType.MULTIMODAL

        return InputType.VECTOR

    def _process_vector(self, data: Any) -> np.ndarray:
        """Process vector input."""
        # Convert to numpy
        if isinstance(data, np.ndarray):
            vec = data.flatten()
        elif isinstance(data, (list, tuple)):
            vec = np.array(data, dtype=np.float32).flatten()
        else:
            vec = np.array([float(data)])

        # Resize to standard dimension
        vec = self._resize(vec, self.config.input_dim)

        # Normalize if configured
        if self.config.input_normalization:
            vec = self._normalize(vec)

        return vec

    def _process_text(self, data: str) -> np.ndarray:
        """Process text input."""
        # Simple character-level embedding
        # In production, would use sentence transformers or similar
        vec = np.zeros(self.config.input_dim, dtype=np.float32)

        for i, char in enumerate(data[: self.config.input_dim]):
            vec[i % self.config.input_dim] += ord(char) / 256.0

        # Normalize
        vec = self._normalize(vec)

        return vec

    def _process_image(self, data: np.ndarray) -> np.ndarray:
        """Process image input."""
        # Flatten and resize
        if isinstance(data, np.ndarray):
            vec = data.flatten().astype(np.float32)
        else:
            vec = np.zeros(self.config.input_dim, dtype=np.float32)

        vec = self._resize(vec, self.config.input_dim)

        # Normalize to 0-1
        if vec.max() > 1.0:
            vec = vec / 255.0

        return vec

    def _process_audio(self, data: Any) -> np.ndarray:
        """Process audio input."""
        # Placeholder - in production would use audio features
        if isinstance(data, np.ndarray):
            vec = data.flatten().astype(np.float32)
        else:
            vec = np.zeros(self.config.input_dim, dtype=np.float32)

        vec = self._resize(vec, self.config.input_dim)
        vec = self._normalize(vec)

        return vec

    def _process_multimodal(self, data: Any) -> np.ndarray:
        """Process multimodal input."""
        vec = np.zeros(self.config.input_dim, dtype=np.float32)

        if isinstance(data, dict):
            # Process each modality and concatenate
            modalities = []
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    modalities.append(value.flatten())
                elif isinstance(value, str):
                    modalities.append(self._process_text(value))
                elif isinstance(value, (list, tuple)):
                    modalities.append(np.array(value, dtype=np.float32))

            if modalities:
                combined = np.concatenate(modalities)
                vec = self._resize(combined, self.config.input_dim)

        elif isinstance(data, (list, tuple)):
            combined = []
            for item in data:
                if isinstance(item, np.ndarray):
                    combined.extend(item.flatten())
                elif isinstance(item, (int, float)):
                    combined.append(float(item))
            if combined:
                vec = self._resize(np.array(combined, dtype=np.float32), self.config.input_dim)

        vec = self._normalize(vec)
        return vec

    def _to_vector(self, data: Any) -> np.ndarray:
        """Convert any data to vector."""
        try:
            if isinstance(data, np.ndarray):
                vec = data.flatten().astype(np.float32)
            elif isinstance(data, (list, tuple)):
                vec = np.array(data, dtype=np.float32).flatten()
            elif isinstance(data, (int, float)):
                vec = np.array([float(data)], dtype=np.float32)
            elif isinstance(data, str):
                return self._process_text(data)
            else:
                vec = np.zeros(self.config.input_dim, dtype=np.float32)

            return self._resize(vec, self.config.input_dim)

        except Exception:
            return np.zeros(self.config.input_dim, dtype=np.float32)

    def _resize(self, vec: np.ndarray, target_dim: int) -> np.ndarray:
        """Resize vector to target dimension."""
        if len(vec) == target_dim:
            return vec
        elif len(vec) < target_dim:
            # Pad with zeros
            return np.pad(vec, (0, target_dim - len(vec)))
        else:
            # Truncate
            return vec[:target_dim]

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return vec

    def get_last_input(self) -> Optional[np.ndarray]:
        """Get most recent processed input."""
        if self._last_input and self._last_input.processed is not None:
            return self._last_input.processed
        return None

    def get_combined_input(self, n: int = 3) -> np.ndarray:
        """Get combined representation of recent inputs."""
        recent = self.buffer.get_recent(n)
        if not recent:
            return np.zeros(self.config.input_dim, dtype=np.float32)

        # Average recent inputs
        processed = [inp.processed for inp in recent if inp.processed is not None]
        if processed:
            return np.mean(processed, axis=0)
        return np.zeros(self.config.input_dim, dtype=np.float32)

    def get_statistics(self) -> Dict[str, Any]:
        """Get interface statistics."""
        return {
            "input_count": self._input_count,
            "buffer_size": len(self.buffer.inputs),
            "input_dim": self.config.input_dim,
            "normalization": self.config.input_normalization,
        }

    def reset(self) -> None:
        """Reset interface state."""
        self.buffer.clear()
        self._last_input = None
        self._input_count = 0
