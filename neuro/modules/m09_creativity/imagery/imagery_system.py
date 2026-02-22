"""
Integrated Imagery System

Combines all imagery modalities (visual, auditory, motor, tactile)
for rich multimodal mental simulation.

Creative imagination often involves multiple modalities working together.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .visual_imagery import VisualImagery, VisualImage
from .auditory_imagery import AuditoryImagery, AuditoryImage
from .motor_imagery import MotorImagery, MotorImage
from .tactile_imagery import TactileImagery, TactileImage


@dataclass
class MultimodalImage:
    """An image that spans multiple sensory modalities"""

    id: str
    description: str
    visual: Optional[VisualImage] = None
    auditory: Optional[AuditoryImage] = None
    motor: Optional[MotorImage] = None
    tactile: Optional[TactileImage] = None
    overall_vividness: float = 0.5
    coherence: float = 0.5  # How well modalities fit together


class ImagerySystem:
    """
    Integrated imagery system - multimodal mental simulation.

    Combines:
    - Visual: Seeing in mind's eye
    - Auditory: Inner hearing
    - Motor: Action simulation
    - Tactile: Touch sensation

    Creative imagination leverages all modalities for rich simulation.
    """

    def __init__(self):
        self.visual = VisualImagery()
        self.auditory = AuditoryImagery()
        self.motor = MotorImagery()
        self.tactile = TactileImagery()

        self.multimodal_images: Dict[str, MultimodalImage] = {}

    def create_multimodal_image(
        self,
        image_id: str,
        description: str,
        include_visual: bool = True,
        include_auditory: bool = False,
        include_motor: bool = False,
        include_tactile: bool = False,
    ) -> MultimodalImage:
        """
        Create an image with multiple sensory modalities.
        """
        visual_img = None
        auditory_img = None
        motor_img = None
        tactile_img = None

        if include_visual:
            visual_img = self.visual.visualize(description)

        if include_auditory:
            auditory_img = self.auditory.imagine_sound(description)

        if include_motor:
            motor_img = self.motor.simulate_movement(description)

        if include_tactile:
            tactile_img = self.tactile.feel_texture(description)

        # Compute overall vividness
        vividness_scores = []
        if visual_img:
            vividness_scores.append(visual_img.vividness)
        if auditory_img:
            vividness_scores.append(auditory_img.vividness)
        if motor_img:
            vividness_scores.append(motor_img.vividness)
        if tactile_img:
            vividness_scores.append(tactile_img.vividness)

        overall_vividness = np.mean(vividness_scores) if vividness_scores else 0.0

        multimodal = MultimodalImage(
            id=image_id,
            description=description,
            visual=visual_img,
            auditory=auditory_img,
            motor=motor_img,
            tactile=tactile_img,
            overall_vividness=overall_vividness,
            coherence=self._compute_coherence(visual_img, auditory_img, motor_img, tactile_img),
        )

        self.multimodal_images[image_id] = multimodal
        return multimodal

    def _compute_coherence(
        self,
        visual: Optional[VisualImage],
        auditory: Optional[AuditoryImage],
        motor: Optional[MotorImage],
        tactile: Optional[TactileImage],
    ) -> float:
        """
        Compute how coherently the modalities fit together.

        High coherence = modalities describe the same experience
        Low coherence = conflicting or unrelated modalities
        """
        # Count modalities present
        count = sum(1 for x in [visual, auditory, motor, tactile] if x is not None)

        if count <= 1:
            return 1.0  # Single modality is perfectly coherent

        # More modalities = harder to maintain coherence
        base_coherence = 1.0 - (count - 1) * 0.1

        return base_coherence

    def imagine_scene(self, scene_description: str) -> MultimodalImage:
        """
        Create a rich multimodal scene imagination.

        Full scene imagination includes sights, sounds, and feelings.
        """
        return self.create_multimodal_image(
            f"scene_{len(self.multimodal_images)}",
            scene_description,
            include_visual=True,
            include_auditory=True,
            include_tactile=True,
        )

    def imagine_action(self, action_description: str) -> MultimodalImage:
        """
        Imagine performing an action with full sensory simulation.
        """
        return self.create_multimodal_image(
            f"action_{len(self.multimodal_images)}",
            action_description,
            include_visual=True,
            include_motor=True,
            include_tactile=True,
        )

    def simulate_experience(self, experience: str, modalities: List[str]) -> MultimodalImage:
        """
        Simulate a full experience with specified modalities.
        """
        return self.create_multimodal_image(
            f"experience_{len(self.multimodal_images)}",
            experience,
            include_visual="visual" in modalities,
            include_auditory="auditory" in modalities,
            include_motor="motor" in modalities,
            include_tactile="tactile" in modalities,
        )

    def get_all_associations(self, image_id: str) -> List[str]:
        """
        Get all semantic associations from all modalities.

        Multimodal imagery provides richer associations.
        """
        if image_id not in self.multimodal_images:
            return []

        image = self.multimodal_images[image_id]
        associations = []

        if image.tactile:
            associations.extend(image.tactile.semantic_associations)

        # Visual and auditory properties can also evoke associations
        if image.visual:
            # Colors evoke associations
            color = image.visual.properties.get("color")
            if color:
                color_associations = {
                    "red": ["passion", "energy", "danger"],
                    "blue": ["calm", "trust", "depth"],
                    "green": ["nature", "growth", "harmony"],
                    "yellow": ["happiness", "energy", "warmth"],
                    "black": ["mystery", "elegance", "power"],
                    "white": ["purity", "clarity", "simplicity"],
                }
                associations.extend(color_associations.get(color, []))

        return list(set(associations))

    def transform_multimodal(self, image_id: str, transformation: str) -> MultimodalImage:
        """
        Apply transformation across all modalities.
        """
        if image_id not in self.multimodal_images:
            raise ValueError(f"Image {image_id} not found")

        original = self.multimodal_images[image_id]
        new_id = f"{image_id}_{transformation}"

        new_visual = None
        new_auditory = None
        new_motor = None
        new_tactile = None

        if original.visual:
            if transformation == "rotate":
                new_visual = self.visual.rotate(original.visual.id, 90)
            elif transformation == "scale_up":
                new_visual = self.visual.scale(original.visual.id, 1.5)
            elif transformation == "scale_down":
                new_visual = self.visual.scale(original.visual.id, 0.5)
            else:
                new_visual = original.visual

        if original.auditory:
            if transformation == "speed_up":
                new_auditory = self.auditory.change_tempo(original.auditory.id, 1.5)
            elif transformation == "slow_down":
                new_auditory = self.auditory.change_tempo(original.auditory.id, 0.5)
            else:
                new_auditory = original.auditory

        if original.motor:
            if transformation == "slow_down":
                new_motor = self.motor.slow_motion(original.motor.id, 0.5)
            elif transformation == "mirror":
                new_motor = self.motor.mirror(original.motor.id)
            else:
                new_motor = original.motor

        new_tactile = original.tactile  # Tactile doesn't transform as easily

        transformed = MultimodalImage(
            id=new_id,
            description=f"{original.description} ({transformation})",
            visual=new_visual,
            auditory=new_auditory,
            motor=new_motor,
            tactile=new_tactile,
            overall_vividness=original.overall_vividness * 0.9,
            coherence=original.coherence * 0.95,
        )

        self.multimodal_images[new_id] = transformed
        return transformed

    def blend_images(
        self, image1_id: str, image2_id: str, blend_factor: float = 0.5
    ) -> MultimodalImage:
        """
        Blend two multimodal images together.

        Creative blending of different experiences.
        """
        if image1_id not in self.multimodal_images or image2_id not in self.multimodal_images:
            raise ValueError("Both images must exist")

        img1 = self.multimodal_images[image1_id]
        img2 = self.multimodal_images[image2_id]

        blend_id = f"blend_{image1_id}_{image2_id}"

        # For visual, use morph if both present
        blended_visual = None
        if img1.visual and img2.visual:
            blended_visual = self.visual.morph(img1.visual.id, img2.visual.id, blend_factor)
        elif img1.visual:
            blended_visual = img1.visual
        elif img2.visual:
            blended_visual = img2.visual

        # For auditory, layer
        blended_auditory = None
        if img1.auditory and img2.auditory:
            blended_auditory = self.auditory.layer_sounds([img1.auditory.id, img2.auditory.id])
        elif img1.auditory:
            blended_auditory = img1.auditory
        elif img2.auditory:
            blended_auditory = img2.auditory

        # For motor, sequence
        blended_motor = None
        if img1.motor and img2.motor:
            blended_motor = self.motor.sequence_actions([img1.motor.id, img2.motor.id])
        elif img1.motor:
            blended_motor = img1.motor
        elif img2.motor:
            blended_motor = img2.motor

        # For tactile, combine
        blended_tactile = None
        if img1.tactile and img2.tactile:
            blended_tactile = self.tactile.combine_sensations([img1.tactile.id, img2.tactile.id])
        elif img1.tactile:
            blended_tactile = img1.tactile
        elif img2.tactile:
            blended_tactile = img2.tactile

        blended = MultimodalImage(
            id=blend_id,
            description=f"Blend of {img1.description} and {img2.description}",
            visual=blended_visual,
            auditory=blended_auditory,
            motor=blended_motor,
            tactile=blended_tactile,
            overall_vividness=(img1.overall_vividness + img2.overall_vividness) / 2 * 0.9,
            coherence=0.6,  # Blends have lower coherence
        )

        self.multimodal_images[blend_id] = blended
        return blended

    def get_vividness_by_modality(self, image_id: str) -> Dict[str, float]:
        """Get vividness breakdown by modality"""
        if image_id not in self.multimodal_images:
            return {}

        image = self.multimodal_images[image_id]
        result = {}

        if image.visual:
            result["visual"] = image.visual.vividness
        if image.auditory:
            result["auditory"] = image.auditory.vividness
        if image.motor:
            result["motor"] = image.motor.vividness
        if image.tactile:
            result["tactile"] = image.tactile.vividness

        return result
