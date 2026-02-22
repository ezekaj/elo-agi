"""
Dorsal and Ventral Streams: "Where" and "What" pathways.

Implements the dual-stream model of visual processing.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np


@dataclass
class MotionVector:
    """Motion information at a location."""

    dx: float
    dy: float
    magnitude: float
    direction: float  # In radians


@dataclass
class SpatialInfo:
    """Spatial information about objects."""

    location: Tuple[float, float]
    depth: float
    size: Tuple[float, float]
    velocity: Optional[MotionVector] = None


@dataclass
class DorsalOutput:
    """Output from dorsal (where/how) stream."""

    motion_map: np.ndarray  # Optical flow
    depth_map: np.ndarray  # Depth estimation
    spatial_map: np.ndarray  # Spatial locations
    action_affordances: Dict[str, float]  # Possible actions
    egocentric_coords: np.ndarray  # Self-relative coordinates


@dataclass
class VentralOutput:
    """Output from ventral (what) stream."""

    object_map: np.ndarray  # Object identity map
    category_map: np.ndarray  # Category assignments
    feature_map: np.ndarray  # Detailed features
    object_identities: List[str]  # Recognized identities


class DorsalStream:
    """
    Dorsal visual stream processing ("where"/"how" pathway).

    Implements:
    - Motion processing (MT/V5)
    - Depth from motion
    - Spatial localization
    - Action affordances
    - Egocentric coordinates
    """

    def __init__(
        self,
        motion_window: int = 3,
        depth_scale: float = 1.0,
    ):
        self.motion_window = motion_window
        self.depth_scale = depth_scale

        # Previous frame for motion estimation
        self._previous_frame: Optional[np.ndarray] = None

        # Action affordance templates
        self._affordances = {
            "grasp": self._grasp_affordance,
            "reach": self._reach_affordance,
            "avoid": self._avoid_affordance,
            "track": self._track_affordance,
        }

    def process(
        self,
        v1_output,
        v2_output,
    ) -> DorsalOutput:
        """
        Process through dorsal stream.

        Args:
            v1_output: V1Output with edge/motion features
            v2_output: V2Output with depth cues

        Returns:
            DorsalOutput with spatial/motion information
        """
        h, w = v1_output.size

        # Motion estimation
        motion_map = self._estimate_motion(v1_output.edge_map)

        # Depth estimation
        depth_map = self._estimate_depth(v2_output.depth_cues, motion_map)

        # Spatial map (2D location + depth)
        spatial_map = np.stack(
            [
                np.arange(w)[None, :].repeat(h, axis=0),  # x
                np.arange(h)[:, None].repeat(w, axis=1),  # y
                depth_map,
            ],
            axis=-1,
        ).astype(float)

        # Egocentric coordinates (relative to center)
        egocentric = self._to_egocentric(spatial_map)

        # Action affordances
        affordances = self._compute_affordances(motion_map, depth_map, v1_output.edge_map)

        return DorsalOutput(
            motion_map=motion_map,
            depth_map=depth_map,
            spatial_map=spatial_map,
            action_affordances=affordances,
            egocentric_coords=egocentric,
        )

    def _estimate_motion(self, edge_map: np.ndarray) -> np.ndarray:
        """
        Estimate optical flow using block matching.
        """
        h, w = edge_map.shape

        if self._previous_frame is None:
            self._previous_frame = edge_map.copy()
            return np.zeros((h, w, 2))

        # Simple block matching for motion
        motion = np.zeros((h, w, 2))
        block_size = self.motion_window

        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = edge_map[y : y + block_size, x : x + block_size]

                # Search in neighborhood
                best_match = (0, 0)
                best_score = float("inf")

                search_range = 3
                for dy in range(-search_range, search_range + 1):
                    for dx in range(-search_range, search_range + 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h - block_size and 0 <= nx < w - block_size:
                            prev_block = self._previous_frame[
                                ny : ny + block_size, nx : nx + block_size
                            ]
                            score = np.sum(np.abs(block - prev_block))
                            if score < best_score:
                                best_score = score
                                best_match = (dx, dy)

                motion[y : y + block_size, x : x + block_size, 0] = best_match[0]
                motion[y : y + block_size, x : x + block_size, 1] = best_match[1]

        self._previous_frame = edge_map.copy()
        return motion

    def _estimate_depth(
        self,
        monocular_cues: np.ndarray,
        motion: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate depth from monocular cues and motion.
        """
        # Motion parallax: faster motion = closer
        motion_magnitude = np.sqrt(motion[:, :, 0] ** 2 + motion[:, :, 1] ** 2)
        motion_depth = 1.0 / (motion_magnitude + 0.1)
        motion_depth = motion_depth / motion_depth.max()

        # Combine with monocular cues
        depth = 0.5 * monocular_cues + 0.5 * motion_depth
        depth = depth * self.depth_scale

        return depth

    def _to_egocentric(self, spatial_map: np.ndarray) -> np.ndarray:
        """
        Convert to egocentric (self-centered) coordinates.
        """
        h, w, _ = spatial_map.shape
        center = np.array([w // 2, h // 2, 0])

        egocentric = spatial_map.copy()
        egocentric[:, :, 0] -= center[0]
        egocentric[:, :, 1] -= center[1]

        return egocentric

    def _compute_affordances(
        self,
        motion: np.ndarray,
        depth: np.ndarray,
        edges: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute action affordances from visual information.
        """
        affordances = {}
        for name, func in self._affordances.items():
            affordances[name] = func(motion, depth, edges)
        return affordances

    def _grasp_affordance(
        self,
        motion: np.ndarray,
        depth: np.ndarray,
        edges: np.ndarray,
    ) -> float:
        """Estimate graspability of objects in view."""
        # Nearby, stationary objects with edges are graspable
        near = depth < 0.5
        stationary = np.sqrt(motion[:, :, 0] ** 2 + motion[:, :, 1] ** 2) < 0.5
        has_form = edges > edges.mean()

        graspable = near & stationary & has_form
        return float(graspable.sum()) / graspable.size

    def _reach_affordance(
        self,
        motion: np.ndarray,
        depth: np.ndarray,
        edges: np.ndarray,
    ) -> float:
        """Estimate reachability."""
        # Objects at medium depth
        reachable = (depth > 0.3) & (depth < 0.7)
        return float(reachable.sum()) / reachable.size

    def _avoid_affordance(
        self,
        motion: np.ndarray,
        depth: np.ndarray,
        edges: np.ndarray,
    ) -> float:
        """Estimate need for avoidance (approaching objects)."""
        # Objects moving towards viewer (increasing size/decreasing depth)
        motion_magnitude = np.sqrt(motion[:, :, 0] ** 2 + motion[:, :, 1] ** 2)
        approaching = (motion_magnitude > 1.0) & (depth < 0.5)
        return float(approaching.sum()) / approaching.size

    def _track_affordance(
        self,
        motion: np.ndarray,
        depth: np.ndarray,
        edges: np.ndarray,
    ) -> float:
        """Estimate trackability of moving objects."""
        motion_magnitude = np.sqrt(motion[:, :, 0] ** 2 + motion[:, :, 1] ** 2)
        moving = motion_magnitude > 0.5
        return float(moving.sum()) / moving.size


class VentralStream:
    """
    Ventral visual stream processing ("what" pathway).

    Implements:
    - Object recognition
    - Category assignment
    - Feature extraction
    - Identity maintenance
    """

    def __init__(
        self,
        embedding_dim: int = 128,
    ):
        self.embedding_dim = embedding_dim

    def process(
        self,
        v4_output,
        it_output,
    ) -> VentralOutput:
        """
        Process through ventral stream.

        Args:
            v4_output: V4Output with shape features
            it_output: ITOutput with object representations

        Returns:
            VentralOutput with object identities
        """
        h, w = v4_output.size

        # Object map from IT responses
        if len(it_output.object_responses) > 0:
            n_categories = len(it_output.object_responses)
            object_map = np.zeros((h, w, n_categories))

            # Spread object identity based on shape saliency
            saliency = v4_output.shape_map / (v4_output.shape_map.max() + 1e-8)
            for i in range(n_categories):
                object_map[:, :, i] = saliency * it_output.object_responses[i]
        else:
            object_map = np.zeros((h, w, 1))

        # Category map (argmax of object responses)
        category_map = np.argmax(object_map, axis=2)

        # Feature map
        feature_map = np.stack(
            [
                v4_output.shape_map,
                v4_output.curvature_map,
                v4_output.color_shape,
            ],
            axis=-1,
        )

        # Collect recognized identities
        identities = [obj.identity for obj in it_output.detected_objects]

        return VentralOutput(
            object_map=object_map,
            category_map=category_map,
            feature_map=feature_map,
            object_identities=identities,
        )


class VisualPathways:
    """
    Combined dorsal and ventral visual pathways.

    Integrates "what" and "where" information for unified perception.
    """

    def __init__(self):
        self.dorsal = DorsalStream()
        self.ventral = VentralStream()

    def process(
        self,
        v1_output,
        v2_output,
        v4_output,
        it_output,
    ) -> Tuple[DorsalOutput, VentralOutput]:
        """
        Process through both pathways.

        Returns:
            Tuple of (DorsalOutput, VentralOutput)
        """
        dorsal_out = self.dorsal.process(v1_output, v2_output)
        ventral_out = self.ventral.process(v4_output, it_output)

        return dorsal_out, ventral_out

    def get_unified_representation(
        self,
        dorsal_output: DorsalOutput,
        ventral_output: VentralOutput,
    ) -> Dict[str, Any]:
        """
        Create unified representation combining both streams.
        """
        return {
            "objects": [
                {
                    "identity": identity,
                    "depth": float(dorsal_output.depth_map.mean()),
                    "motion": float(
                        np.sqrt(
                            dorsal_output.motion_map[:, :, 0].mean() ** 2
                            + dorsal_output.motion_map[:, :, 1].mean() ** 2
                        )
                    ),
                }
                for identity in ventral_output.object_identities
            ],
            "affordances": dorsal_output.action_affordances,
            "n_categories_detected": len(ventral_output.object_identities),
        }

    def statistics(self) -> Dict[str, Any]:
        """Get pathway statistics."""
        return {
            "dorsal_motion_window": self.dorsal.motion_window,
            "ventral_embedding_dim": self.ventral.embedding_dim,
        }
