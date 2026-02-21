"""
Interface: Connect perception to neuro-system sensory components.

Provides the bridge between raw sensory input and the cognitive architecture.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np

from .visual.retina import Retina, RetinaOutput
from .visual.v1_v2 import V1Processor, V2Processor, V1Output, V2Output
from .visual.v4_it import V4Processor, ITProcessor, V4Output, ITOutput
from .visual.dorsal_ventral import (
    DorsalStream, VentralStream, VisualPathways,
    DorsalOutput, VentralOutput
)
from .auditory.cochlea import Cochlea, CochleaOutput
from .auditory.a1 import A1Processor, A1Output
from .auditory.speech import SpeechProcessor, SpeechOutput
from .multimodal.binding import (
    CrossModalBinder, BindingOutput, BoundPercept,
    Modality, ModalityInput
)
from .multimodal.attention import SelectiveAttention, AttentionOutput


@dataclass
class VisualPercept:
    """Complete visual perception output."""
    retina: RetinaOutput
    v1: V1Output
    v2: V2Output
    v4: V4Output
    it: ITOutput
    dorsal: DorsalOutput
    ventral: VentralOutput

    @property
    def objects(self) -> List[str]:
        """Get detected object identities."""
        return self.ventral.object_identities

    @property
    def affordances(self) -> Dict[str, float]:
        """Get action affordances."""
        return self.dorsal.action_affordances


@dataclass
class AuditoryPercept:
    """Complete auditory perception output."""
    cochlea: CochleaOutput
    a1: A1Output
    speech: Optional[SpeechOutput] = None

    @property
    def phonemes(self) -> List[str]:
        """Get detected phonemes."""
        if self.speech is None:
            return []
        return [p.phoneme for p in self.speech.phonemes]


@dataclass
class MultimodalPercept:
    """Integrated multimodal perception."""
    visual: Optional[VisualPercept] = None
    auditory: Optional[AuditoryPercept] = None
    binding: Optional[BindingOutput] = None
    attention: Optional[AttentionOutput] = None

    @property
    def attended_objects(self) -> List[BoundPercept]:
        """Get attended bound percepts."""
        if self.attention is None:
            return []
        return self.attention.attended_percepts


class VisualPipeline:
    """
    Complete visual processing pipeline.

    Retina → V1 → V2 → V4/IT → Dorsal/Ventral
    """

    def __init__(
        self,
        n_orientations: int = 8,
        n_frequencies: int = 4,
        n_categories: int = 20,
    ):
        self.retina = Retina()
        self.v1 = V1Processor(
            n_orientations=n_orientations,
            n_frequencies=n_frequencies,
        )
        self.v2 = V2Processor()
        self.v4 = V4Processor()
        self.it = ITProcessor(n_categories=n_categories)
        self.pathways = VisualPathways()

    def process(self, image: np.ndarray) -> VisualPercept:
        """
        Process image through complete visual pipeline.

        Args:
            image: Input image (H, W) or (H, W, 3)

        Returns:
            VisualPercept with all processing stages
        """
        # Early vision
        retina_out = self.retina.process(image)
        v1_out = self.v1.process(retina_out)
        v2_out = self.v2.process(v1_out)

        # Higher vision
        v4_out = self.v4.process(v2_out)
        it_out = self.it.process(v4_out)

        # Dual streams
        dorsal_out, ventral_out = self.pathways.process(
            v1_out, v2_out, v4_out, it_out
        )

        return VisualPercept(
            retina=retina_out,
            v1=v1_out,
            v2=v2_out,
            v4=v4_out,
            it=it_out,
            dorsal=dorsal_out,
            ventral=ventral_out,
        )

    def get_features(self, percept: VisualPercept) -> np.ndarray:
        """Extract feature vector from visual percept."""
        features = []

        # IT embedding (object identity)
        if percept.it.detected_objects:
            features.append(percept.it.detected_objects[0].embedding)
        else:
            features.append(np.zeros(128))

        # Affordances
        affordance_vec = np.array(list(percept.dorsal.action_affordances.values()))
        features.append(affordance_vec)

        # Combine and normalize
        combined = np.concatenate(features)
        return combined / (np.linalg.norm(combined) + 1e-8)

    def statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "retina": self.retina.statistics(),
            "v1": self.v1.statistics(),
            "v2": self.v2.statistics(),
            "v4": self.v4.statistics(),
            "it": self.it.statistics(),
        }


class AuditoryPipeline:
    """
    Complete auditory processing pipeline.

    Cochlea → A1 → Speech
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_channels: int = 64,
        process_speech: bool = True,
    ):
        self.sample_rate = sample_rate
        self.process_speech = process_speech

        self.cochlea = Cochlea(
            sample_rate=sample_rate,
            n_channels=n_channels,
        )
        self.a1 = A1Processor()
        self.speech = SpeechProcessor(sample_rate=sample_rate) if process_speech else None

    def process(self, audio: np.ndarray) -> AuditoryPercept:
        """
        Process audio through complete auditory pipeline.

        Args:
            audio: Input audio (n_samples,)

        Returns:
            AuditoryPercept with all processing stages
        """
        # Early audition
        cochlea_out = self.cochlea.process(audio)
        a1_out = self.a1.process(cochlea_out)

        # Speech processing
        speech_out = None
        if self.speech is not None:
            speech_out = self.speech.process(a1_out)

        return AuditoryPercept(
            cochlea=cochlea_out,
            a1=a1_out,
            speech=speech_out,
        )

    def get_features(self, percept: AuditoryPercept) -> np.ndarray:
        """Extract feature vector from auditory percept."""
        # Use A1 modulation spectrum
        mod_spec = self.a1.get_modulation_spectrum(percept.a1)

        features = np.concatenate([
            mod_spec["rate_spectrum"],
            mod_spec["scale_spectrum"],
        ])

        return features / (np.linalg.norm(features) + 1e-8)

    def statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            "cochlea": self.cochlea.statistics(),
            "a1": self.a1.statistics(),
        }
        if self.speech:
            stats["speech"] = self.speech.statistics()
        return stats


class PerceptionSystem:
    """
    Complete multimodal perception system.

    Integrates visual and auditory processing with
    cross-modal binding and attention.
    """

    def __init__(
        self,
        visual_config: Optional[Dict[str, Any]] = None,
        auditory_config: Optional[Dict[str, Any]] = None,
        binding_window: float = 0.1,
        attention_capacity: int = 4,
    ):
        visual_config = visual_config or {}
        auditory_config = auditory_config or {}

        # Processing pipelines
        self.visual = VisualPipeline(**visual_config)
        self.auditory = AuditoryPipeline(**auditory_config)

        # Integration
        self.binder = CrossModalBinder(binding_window=binding_window)
        self.attention = SelectiveAttention(capacity=attention_capacity)

        # State
        self._current_time: float = 0.0

    def process_visual(self, image: np.ndarray) -> VisualPercept:
        """Process visual input only."""
        return self.visual.process(image)

    def process_auditory(self, audio: np.ndarray) -> AuditoryPercept:
        """Process auditory input only."""
        return self.auditory.process(audio)

    def process(
        self,
        visual_input: Optional[np.ndarray] = None,
        auditory_input: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
    ) -> MultimodalPercept:
        """
        Process multimodal input.

        Args:
            visual_input: Image (H, W) or (H, W, 3)
            auditory_input: Audio (n_samples,)
            timestamp: Current time (auto-increments if None)

        Returns:
            MultimodalPercept with integrated perception
        """
        if timestamp is None:
            timestamp = self._current_time
            self._current_time += 0.1

        # Process individual modalities
        visual_percept = None
        auditory_percept = None
        modality_inputs = []

        if visual_input is not None:
            visual_percept = self.visual.process(visual_input)

            # Create modality input for binding
            visual_features = self.visual.get_features(visual_percept)
            modality_inputs.append(ModalityInput(
                modality=Modality.VISUAL,
                features=visual_features,
                spatial_location=(0.0, 0.0, 1.0),  # Center, medium depth
                timestamp=timestamp,
                confidence=1.0,
            ))

        if auditory_input is not None:
            auditory_percept = self.auditory.process(auditory_input)

            # Create modality input for binding
            auditory_features = self.auditory.get_features(auditory_percept)
            modality_inputs.append(ModalityInput(
                modality=Modality.AUDITORY,
                features=auditory_features,
                spatial_location=None,  # No spatial location for audio
                timestamp=timestamp,
                confidence=1.0,
            ))

        # Cross-modal binding
        binding_output = None
        if modality_inputs:
            binding_output = self.binder.bind(modality_inputs)

        # Attention
        attention_output = None
        if binding_output and binding_output.percepts:
            # Get visual feature map for saliency if available
            visual_map = None
            if visual_percept is not None:
                visual_map = visual_percept.v1.edge_map

            auditory_map = None
            if auditory_percept is not None:
                auditory_map = auditory_percept.a1.tonotopic_map

            attention_output = self.attention.attend(
                inputs=modality_inputs,
                percepts=binding_output.percepts,
                visual_input=visual_map,
                auditory_input=auditory_map,
            )

        return MultimodalPercept(
            visual=visual_percept,
            auditory=auditory_percept,
            binding=binding_output,
            attention=attention_output,
        )

    def set_attention_goal(
        self,
        template: Optional[np.ndarray] = None,
        modality: Optional[Modality] = None,
        location: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Set top-down attentional goal."""
        self.attention.set_goal(template, modality, location)

    def reset(self) -> None:
        """Reset perception system state."""
        self._current_time = 0.0
        self.visual.retina._previous_input = None
        self.visual.pathways.dorsal._previous_frame = None

    def statistics(self) -> Dict[str, Any]:
        """Get perception system statistics."""
        return {
            "visual": self.visual.statistics(),
            "auditory": self.auditory.statistics(),
            "binding": self.binder.statistics(),
            "attention": self.attention.statistics(),
        }


# Convenience function for creating perception system
def create_perception_system(
    visual: bool = True,
    auditory: bool = True,
    **kwargs,
) -> PerceptionSystem:
    """
    Create a configured perception system.

    Args:
        visual: Enable visual processing
        auditory: Enable auditory processing
        **kwargs: Additional configuration

    Returns:
        Configured PerceptionSystem
    """
    config = {}

    if not visual:
        config["visual_config"] = {"n_orientations": 4, "n_frequencies": 2}

    if not auditory:
        config["auditory_config"] = {"n_channels": 32, "process_speech": False}

    config.update(kwargs)

    return PerceptionSystem(**config)
