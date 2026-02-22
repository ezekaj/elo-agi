"""
Attention: Selective attention mechanisms for perception.

Implements bottom-up (saliency) and top-down (goal-directed)
attention across modalities.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np

from .binding import Modality, ModalityInput, BoundPercept


class AttentionType(Enum):
    """Types of attention."""

    BOTTOM_UP = "bottom_up"  # Stimulus-driven
    TOP_DOWN = "top_down"  # Goal-directed
    EXOGENOUS = "exogenous"  # Involuntary capture
    ENDOGENOUS = "endogenous"  # Voluntary control


@dataclass
class AttentionFocus:
    """Current focus of attention."""

    location: Optional[Tuple[float, float, float]]
    modality: Optional[Modality]
    feature_template: Optional[np.ndarray]
    strength: float
    duration: float  # How long attended


@dataclass
class SaliencyMap:
    """Saliency map for a modality."""

    modality: Modality
    map: np.ndarray  # Spatial saliency
    peak_location: Tuple[int, int]
    peak_value: float


@dataclass
class AttentionOutput:
    """Output from attention system."""

    focus: AttentionFocus
    saliency_maps: Dict[Modality, SaliencyMap]
    attended_percepts: List[BoundPercept]
    suppressed_percepts: List[BoundPercept]
    attention_weights: np.ndarray  # Per-input attention weights


class SaliencyComputer:
    """
    Compute bottom-up saliency from sensory input.

    Implements Itti-Koch style saliency computation.
    """

    def __init__(
        self,
        n_scales: int = 4,
    ):
        self.n_scales = n_scales

    def compute_visual_saliency(
        self,
        intensity: np.ndarray,
        color: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute visual saliency map.

        Args:
            intensity: Intensity/luminance map
            color: Color contrast map (optional)
            orientation: Orientation energy map (optional)

        Returns:
            Saliency map
        """

        h, w = intensity.shape[:2]
        saliency = np.zeros((h, w))

        # Intensity channel
        intensity_sal = self._center_surround_contrast(intensity)
        saliency += intensity_sal

        # Color channel (if available)
        if color is not None:
            color_sal = self._center_surround_contrast(color)
            saliency += color_sal

        # Orientation channel (if available)
        if orientation is not None:
            ori_sal = self._center_surround_contrast(orientation)
            saliency += ori_sal

        # Normalize
        saliency = saliency / (saliency.max() + 1e-8)

        return saliency

    def _center_surround_contrast(
        self,
        feature_map: np.ndarray,
    ) -> np.ndarray:
        """Compute center-surround contrast across scales."""
        from scipy.ndimage import gaussian_filter

        h, w = feature_map.shape[:2]
        contrast = np.zeros((h, w))

        for scale in range(1, self.n_scales + 1):
            center_sigma = 2**scale
            surround_sigma = 2 ** (scale + 1)

            center = gaussian_filter(feature_map, center_sigma)
            surround = gaussian_filter(feature_map, surround_sigma)

            # Absolute difference
            scale_contrast = np.abs(center - surround)
            contrast += scale_contrast / self.n_scales

        return contrast

    def compute_auditory_saliency(
        self,
        spectrogram: np.ndarray,
        onset_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute auditory saliency.

        Args:
            spectrogram: (n_freq, n_time) spectral representation
            onset_map: Onset detection map (optional)

        Returns:
            Saliency map (n_freq, n_time)
        """

        n_freq, n_time = spectrogram.shape
        saliency = np.zeros((n_freq, n_time))

        # Spectral contrast
        spectral_sal = self._center_surround_contrast(spectrogram)
        saliency += spectral_sal

        # Temporal contrast (sudden changes)
        temporal_diff = np.abs(np.diff(spectrogram, axis=1, prepend=0))
        saliency += temporal_diff / (temporal_diff.max() + 1e-8)

        # Onset bonus
        if onset_map is not None:
            saliency += onset_map / (onset_map.max() + 1e-8)

        # Normalize
        saliency = saliency / (saliency.max() + 1e-8)

        return saliency


class TopDownController:
    """
    Top-down attentional control.

    Implements goal-directed attention based on task demands.
    """

    def __init__(
        self,
        feature_dim: int = 256,
    ):
        self.feature_dim = feature_dim

        # Current goal/template
        self._goal_template: Optional[np.ndarray] = None
        self._goal_modality: Optional[Modality] = None
        self._goal_location: Optional[Tuple[float, float, float]] = None

    def set_goal(
        self,
        template: Optional[np.ndarray] = None,
        modality: Optional[Modality] = None,
        location: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Set attentional goal."""
        if template is not None:
            # Normalize template
            template = template / (np.linalg.norm(template) + 1e-8)
        self._goal_template = template
        self._goal_modality = modality
        self._goal_location = location

    def compute_goal_relevance(
        self,
        inputs: List[ModalityInput],
    ) -> np.ndarray:
        """
        Compute relevance of each input to current goal.

        Returns:
            (n_inputs,) relevance scores
        """
        n = len(inputs)
        relevance = np.ones(n)

        for i, inp in enumerate(inputs):
            score = 1.0

            # Modality match
            if self._goal_modality is not None:
                if inp.modality == self._goal_modality:
                    score *= 2.0
                else:
                    score *= 0.5

            # Feature template match
            if self._goal_template is not None:
                # Project input features to template space
                features = inp.features
                if len(features) < len(self._goal_template):
                    features = np.pad(features, (0, len(self._goal_template) - len(features)))
                elif len(features) > len(self._goal_template):
                    features = features[: len(self._goal_template)]

                features = features / (np.linalg.norm(features) + 1e-8)
                similarity = np.dot(features, self._goal_template)
                score *= max(0, similarity + 1) / 2  # Map [-1,1] to [0,1]

            # Location match
            if self._goal_location is not None and inp.spatial_location is not None:
                dist = np.linalg.norm(
                    np.array(inp.spatial_location) - np.array(self._goal_location)
                )
                score *= np.exp(-dist)

            relevance[i] = score

        return relevance


class AttentionGate:
    """
    Gate information flow based on attention.
    """

    def __init__(
        self,
        capacity: int = 4,  # Attention capacity (number of items)
        decay_rate: float = 0.1,
    ):
        self.capacity = capacity
        self.decay_rate = decay_rate

        # Current attention state
        self._attention_strengths: Dict[str, float] = {}

    def gate(
        self,
        percepts: List[BoundPercept],
        attention_scores: np.ndarray,
    ) -> Tuple[List[BoundPercept], List[BoundPercept]]:
        """
        Gate percepts based on attention scores.

        Returns:
            (attended, suppressed) percepts
        """
        # Sort by attention score
        sorted_indices = np.argsort(attention_scores)[::-1]

        # Select top-k based on capacity
        attended = []
        suppressed = []

        for i, idx in enumerate(sorted_indices):
            if i < self.capacity and attention_scores[idx] > 0.1:
                attended.append(percepts[idx])
            else:
                suppressed.append(percepts[idx])

        return attended, suppressed

    def update_strengths(self, percept_ids: List[str], boosts: List[float]) -> None:
        """Update attention strengths for percepts."""
        # Decay existing
        for pid in list(self._attention_strengths.keys()):
            self._attention_strengths[pid] *= 1 - self.decay_rate
            if self._attention_strengths[pid] < 0.01:
                del self._attention_strengths[pid]

        # Boost attended
        for pid, boost in zip(percept_ids, boosts):
            if pid in self._attention_strengths:
                self._attention_strengths[pid] += boost
            else:
                self._attention_strengths[pid] = boost

    def get_strength(self, percept_id: str) -> float:
        """Get current attention strength for a percept."""
        return self._attention_strengths.get(percept_id, 0.0)


class SelectiveAttention:
    """
    Complete selective attention system.

    Combines bottom-up saliency with top-down control.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        capacity: int = 4,
        saliency_weight: float = 0.4,
        goal_weight: float = 0.6,
    ):
        self.saliency_weight = saliency_weight
        self.goal_weight = goal_weight

        self.saliency_computer = SaliencyComputer()
        self.top_down = TopDownController(feature_dim=feature_dim)
        self.gate = AttentionGate(capacity=capacity)

        # Current focus
        self._current_focus: Optional[AttentionFocus] = None
        self._focus_duration: float = 0.0

    def set_goal(
        self,
        template: Optional[np.ndarray] = None,
        modality: Optional[Modality] = None,
        location: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Set attentional goal for top-down control."""
        self.top_down.set_goal(template, modality, location)

    def attend(
        self,
        inputs: List[ModalityInput],
        percepts: List[BoundPercept],
        visual_input: Optional[np.ndarray] = None,
        auditory_input: Optional[np.ndarray] = None,
    ) -> AttentionOutput:
        """
        Apply selective attention.

        Args:
            inputs: Raw modality inputs
            percepts: Bound percepts from binding
            visual_input: Visual feature map (optional)
            auditory_input: Auditory spectrogram (optional)

        Returns:
            AttentionOutput with attended/suppressed percepts
        """
        # Compute saliency maps
        saliency_maps = {}

        if visual_input is not None:
            vis_saliency = self.saliency_computer.compute_visual_saliency(visual_input)
            peak = np.unravel_index(np.argmax(vis_saliency), vis_saliency.shape)
            saliency_maps[Modality.VISUAL] = SaliencyMap(
                modality=Modality.VISUAL,
                map=vis_saliency,
                peak_location=peak,
                peak_value=float(vis_saliency.max()),
            )

        if auditory_input is not None:
            aud_saliency = self.saliency_computer.compute_auditory_saliency(auditory_input)
            peak = np.unravel_index(np.argmax(aud_saliency), aud_saliency.shape)
            saliency_maps[Modality.AUDITORY] = SaliencyMap(
                modality=Modality.AUDITORY,
                map=aud_saliency,
                peak_location=peak,
                peak_value=float(aud_saliency.max()),
            )

        # Compute attention scores for percepts
        n_percepts = len(percepts)
        if n_percepts == 0:
            return AttentionOutput(
                focus=self._current_focus
                or AttentionFocus(
                    location=None, modality=None, feature_template=None, strength=0.0, duration=0.0
                ),
                saliency_maps=saliency_maps,
                attended_percepts=[],
                suppressed_percepts=[],
                attention_weights=np.array([]),
            )

        # Bottom-up scores from binding strength
        saliency_scores = np.array([p.binding_strength for p in percepts])

        # Top-down scores from goal relevance
        # Create ModalityInputs from percepts for goal relevance computation
        percept_inputs = []
        for p in percepts:
            if p.component_inputs:
                percept_inputs.append(p.component_inputs[0])
            else:
                percept_inputs.append(
                    ModalityInput(
                        modality=list(p.modalities)[0] if p.modalities else Modality.VISUAL,
                        features=p.unified_features,
                        spatial_location=p.spatial_location,
                    )
                )

        goal_scores = self.top_down.compute_goal_relevance(percept_inputs)

        # Combined attention score
        attention_scores = self.saliency_weight * saliency_scores + self.goal_weight * goal_scores

        # Normalize
        attention_scores = attention_scores / (attention_scores.max() + 1e-8)

        # Gate percepts
        attended, suppressed = self.gate.gate(percepts, attention_scores)

        # Update focus
        if attended:
            best_percept = attended[0]
            self._current_focus = AttentionFocus(
                location=best_percept.spatial_location,
                modality=list(best_percept.modalities)[0] if best_percept.modalities else None,
                feature_template=best_percept.unified_features,
                strength=float(attention_scores.max()),
                duration=self._focus_duration,
            )
            self._focus_duration += 0.1  # Increment focus duration
        else:
            self._focus_duration = 0.0

        return AttentionOutput(
            focus=self._current_focus
            or AttentionFocus(
                location=None, modality=None, feature_template=None, strength=0.0, duration=0.0
            ),
            saliency_maps=saliency_maps,
            attended_percepts=attended,
            suppressed_percepts=suppressed,
            attention_weights=attention_scores,
        )

    def shift_attention(
        self,
        new_location: Optional[Tuple[float, float, float]] = None,
        new_modality: Optional[Modality] = None,
    ) -> None:
        """Voluntarily shift attention."""
        self._focus_duration = 0.0

        if new_location is not None:
            self.top_down.set_goal(location=new_location)

        if new_modality is not None:
            self.top_down.set_goal(modality=new_modality)

    def statistics(self) -> Dict[str, Any]:
        """Get attention statistics."""
        return {
            "capacity": self.gate.capacity,
            "saliency_weight": self.saliency_weight,
            "goal_weight": self.goal_weight,
            "current_focus_duration": self._focus_duration,
            "n_tracked_percepts": len(self.gate._attention_strengths),
        }
