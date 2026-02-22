"""
Multimodal Integration - Parietal Lobe Simulation

Binds information across sensory modalities (visual, auditory, tactile).
Handles spatial reference frame alignment and temporal synchronization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class Modality(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    TACTILE = "tactile"
    PROPRIOCEPTIVE = "proprioceptive"


@dataclass
class SensoryInput:
    """Input from a single sensory modality"""

    modality: Modality
    timestamp: float
    location: Optional[Tuple[float, float, float]] = None
    features: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reference_frame: str = "egocentric"


@dataclass
class UnifiedPercept:
    """Coherent multimodal representation"""

    timestamp: float
    location: Tuple[float, float, float]
    modalities: Dict[Modality, SensoryInput]
    binding_strength: float
    properties: Dict[str, Any] = field(default_factory=dict)
    conflicts: List[str] = field(default_factory=list)

    def has_modality(self, modality: Modality) -> bool:
        return modality in self.modalities

    def get_feature(self, feature_name: str) -> Optional[Any]:
        """Get a feature from any modality that has it"""
        for inp in self.modalities.values():
            if feature_name in inp.features:
                return inp.features[feature_name]
        return self.properties.get(feature_name)


class ReferenceFrameTransformer:
    """Transforms between different spatial reference frames"""

    def __init__(self):
        self.head_position = np.array([0.0, 0.0, 0.0])
        self.head_orientation = np.eye(3)
        self.body_position = np.array([0.0, 0.0, 0.0])

    def egocentric_to_allocentric(
        self, point: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Convert from head-centered to world coordinates"""
        p = np.array(point)
        world_p = self.head_orientation @ p + self.head_position
        return tuple(world_p)

    def allocentric_to_egocentric(
        self, point: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Convert from world to head-centered coordinates"""
        p = np.array(point)
        ego_p = self.head_orientation.T @ (p - self.head_position)
        return tuple(ego_p)

    def update_head_pose(self, position: Tuple[float, float, float], orientation: np.ndarray):
        """Update head position and orientation"""
        self.head_position = np.array(position)
        self.head_orientation = orientation


class TemporalSynchronizer:
    """Synchronizes inputs across different temporal delays"""

    def __init__(
        self,
        visual_delay: float = 0.050,
        auditory_delay: float = 0.020,
        tactile_delay: float = 0.030,
    ):
        self.delays = {
            Modality.VISUAL: visual_delay,
            Modality.AUDITORY: auditory_delay,
            Modality.TACTILE: tactile_delay,
            Modality.PROPRIOCEPTIVE: 0.010,
        }
        self.buffer: Dict[Modality, List[SensoryInput]] = {m: [] for m in Modality}
        self.buffer_duration = 0.200

    def add_input(self, inp: SensoryInput):
        """Add input to temporal buffer"""
        self.buffer[inp.modality].append(inp)

        cutoff = inp.timestamp - self.buffer_duration
        self.buffer[inp.modality] = [i for i in self.buffer[inp.modality] if i.timestamp > cutoff]

    def get_synchronized(
        self, target_time: float, tolerance: float = 0.050
    ) -> Dict[Modality, Optional[SensoryInput]]:
        """Get inputs synchronized to target time"""
        result = {}

        for modality in Modality:
            delay = self.delays[modality]
            adjusted_time = target_time - delay

            best_match = None
            best_diff = float("inf")

            for inp in self.buffer[modality]:
                diff = abs(inp.timestamp - adjusted_time)
                if diff < best_diff and diff < tolerance:
                    best_diff = diff
                    best_match = inp

            result[modality] = best_match

        return result


class MultimodalIntegrator:
    """
    Integrates information across sensory modalities.
    Simulates parietal lobe multimodal integration.
    """

    def __init__(self, spatial_tolerance: float = 0.1, temporal_tolerance: float = 0.050):
        self.spatial_tolerance = spatial_tolerance
        self.temporal_tolerance = temporal_tolerance
        self.reference_transformer = ReferenceFrameTransformer()
        self.temporal_sync = TemporalSynchronizer()

        self.reliability = {
            Modality.VISUAL: 0.8,
            Modality.AUDITORY: 0.6,
            Modality.TACTILE: 0.9,
            Modality.PROPRIOCEPTIVE: 0.95,
        }

    def process_input(self, inp: SensoryInput):
        """Process a new sensory input"""
        self.temporal_sync.add_input(inp)

    def bind_visual_audio(
        self, visual: SensoryInput, audio: SensoryInput
    ) -> Optional[UnifiedPercept]:
        """Bind visual and auditory inputs (ventriloquist effect possible)"""
        if visual.location is None or audio.location is None:
            return None

        v_loc = np.array(visual.location)
        a_loc = np.array(audio.location)

        if visual.reference_frame != audio.reference_frame:
            if visual.reference_frame == "egocentric":
                v_loc = np.array(self.reference_transformer.egocentric_to_allocentric(tuple(v_loc)))
            else:
                a_loc = np.array(self.reference_transformer.egocentric_to_allocentric(tuple(a_loc)))

        spatial_diff = np.linalg.norm(v_loc - a_loc)

        if spatial_diff > self.spatial_tolerance * 10:
            return None

        v_weight = self.reliability[Modality.VISUAL] * visual.confidence
        a_weight = self.reliability[Modality.AUDITORY] * audio.confidence
        total_weight = v_weight + a_weight

        integrated_loc = (v_weight * v_loc + a_weight * a_loc) / total_weight

        binding_strength = np.exp(-spatial_diff / self.spatial_tolerance)
        binding_strength *= min(visual.confidence, audio.confidence)

        conflicts = []
        if spatial_diff > self.spatial_tolerance:
            conflicts.append(f"spatial_mismatch: {spatial_diff:.3f}")

        return UnifiedPercept(
            timestamp=max(visual.timestamp, audio.timestamp),
            location=tuple(integrated_loc),
            modalities={Modality.VISUAL: visual, Modality.AUDITORY: audio},
            binding_strength=binding_strength,
            properties={
                "spatial_discrepancy": spatial_diff,
                "visual_weight": v_weight / total_weight,
                "audio_weight": a_weight / total_weight,
            },
            conflicts=conflicts,
        )

    def bind_visual_tactile(
        self, visual: SensoryInput, tactile: SensoryInput
    ) -> Optional[UnifiedPercept]:
        """Bind visual and tactile inputs for object recognition"""
        if visual.location is None or tactile.location is None:
            return None

        v_loc = np.array(visual.location)
        t_loc = np.array(tactile.location)

        spatial_diff = np.linalg.norm(v_loc - t_loc)

        if spatial_diff > self.spatial_tolerance * 5:
            return None

        v_weight = self.reliability[Modality.VISUAL] * visual.confidence
        t_weight = self.reliability[Modality.TACTILE] * tactile.confidence
        total_weight = v_weight + t_weight

        integrated_loc = (v_weight * v_loc + t_weight * t_loc) / total_weight

        binding_strength = np.exp(-spatial_diff / self.spatial_tolerance)

        combined_features = {}
        combined_features.update(visual.features)
        combined_features.update(tactile.features)

        return UnifiedPercept(
            timestamp=max(visual.timestamp, tactile.timestamp),
            location=tuple(integrated_loc),
            modalities={Modality.VISUAL: visual, Modality.TACTILE: tactile},
            binding_strength=binding_strength,
            properties=combined_features,
            conflicts=[],
        )

    def resolve_conflict(self, inputs: Dict[Modality, SensoryInput]) -> Tuple[Any, str]:
        """Resolve conflicts when modalities disagree"""
        if not inputs:
            return None, "no_inputs"

        weighted_estimates = []
        total_weight = 0

        for modality, inp in inputs.items():
            if inp is None or inp.location is None:
                continue

            weight = self.reliability[modality] * inp.confidence
            weighted_estimates.append((np.array(inp.location), weight))
            total_weight += weight

        if not weighted_estimates:
            return None, "no_valid_inputs"

        integrated = sum(loc * w for loc, w in weighted_estimates) / total_weight

        locations = [est[0] for est in weighted_estimates]
        if len(locations) > 1:
            variance = np.mean([np.linalg.norm(loc - integrated) ** 2 for loc in locations])
            if variance > self.spatial_tolerance**2:
                return tuple(integrated), "conflict_resolved_weighted"

        return tuple(integrated), "consistent"

    def create_unified_percept(
        self, inputs: Dict[Modality, SensoryInput]
    ) -> Optional[UnifiedPercept]:
        """Create a coherent representation from all available modalities"""
        valid_inputs = {m: i for m, i in inputs.items() if i is not None}

        if not valid_inputs:
            return None

        location, resolution_status = self.resolve_conflict(valid_inputs)

        if location is None:
            return None

        confidences = [i.confidence for i in valid_inputs.values()]
        binding_strength = np.mean(confidences)

        if len(valid_inputs) > 1:
            binding_strength *= 1 + 0.1 * (len(valid_inputs) - 1)

        conflicts = []
        if "conflict" in resolution_status:
            conflicts.append(resolution_status)

        combined_properties = {}
        for inp in valid_inputs.values():
            combined_properties.update(inp.features)

        timestamps = [i.timestamp for i in valid_inputs.values()]

        return UnifiedPercept(
            timestamp=max(timestamps),
            location=location,
            modalities=valid_inputs,
            binding_strength=min(binding_strength, 1.0),
            properties=combined_properties,
            conflicts=conflicts,
        )

    def integrate_current(self, target_time: float) -> Optional[UnifiedPercept]:
        """Integrate all currently available synchronized inputs"""
        synced = self.temporal_sync.get_synchronized(target_time, self.temporal_tolerance)
        return self.create_unified_percept(synced)
