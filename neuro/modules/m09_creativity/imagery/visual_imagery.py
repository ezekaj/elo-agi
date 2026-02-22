"""
Visual Imagery - "Seeing" with the Mind's Eye

Located in occipital cortex, visual imagery allows:
- Mental visualization of objects and scenes
- Manipulation of mental images (rotation, transformation)
- Creative visualization of novel concepts

Visual imagery is the most studied form of mental imagery.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class VisualProperty(Enum):
    """Properties of visual images"""

    COLOR = "color"
    SHAPE = "shape"
    SIZE = "size"
    POSITION = "position"
    ORIENTATION = "orientation"
    TEXTURE = "texture"
    BRIGHTNESS = "brightness"
    MOTION = "motion"


@dataclass
class VisualImage:
    """A mental visual image"""

    id: str
    description: str
    properties: Dict[VisualProperty, Any]
    vividness: float  # 0-1, how clear/vivid the image is
    stability: float  # 0-1, how well it holds in mind
    components: List["VisualImage"] = field(default_factory=list)
    spatial_relations: Dict[str, Tuple[str, str]] = field(default_factory=dict)


@dataclass
class VisualTransformation:
    """A transformation applied to a visual image"""

    transformation_type: str  # "rotate", "scale", "translate", "morph"
    parameters: Dict[str, Any]
    difficulty: float  # How hard this transformation is


class VisualImagery:
    """
    Visual imagery system - mind's eye.

    Located in occipital cortex, enables:
    - Creating mental images
    - Manipulating images (rotation, scaling)
    - Inspecting images for details
    - Combining images into scenes
    """

    def __init__(self, default_vividness: float = 0.7, decay_rate: float = 0.1):
        self.images: Dict[str, VisualImage] = {}
        self.default_vividness = default_vividness
        self.decay_rate = decay_rate
        self._active_image: Optional[str] = None

    def create_image(
        self,
        image_id: str,
        description: str,
        properties: Optional[Dict[VisualProperty, Any]] = None,
        vividness: Optional[float] = None,
    ) -> VisualImage:
        """
        Create a new mental visual image.

        This is "seeing" something in the mind's eye.
        """
        image = VisualImage(
            id=image_id,
            description=description,
            properties=properties or {},
            vividness=vividness or self.default_vividness,
            stability=0.8,
        )

        self.images[image_id] = image
        self._active_image = image_id

        return image

    def visualize(self, description: str) -> VisualImage:
        """
        Create image from verbal description.

        Converts language to mental image.
        """
        # Extract properties from description (simplified)
        properties = self._extract_properties(description)

        image_id = f"viz_{len(self.images)}"
        return self.create_image(image_id, description, properties)

    def _extract_properties(self, description: str) -> Dict[VisualProperty, Any]:
        """Extract visual properties from description"""
        properties = {}

        # Simple keyword extraction
        desc_lower = description.lower()

        # Colors
        colors = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"]
        for color in colors:
            if color in desc_lower:
                properties[VisualProperty.COLOR] = color
                break

        # Sizes
        if "large" in desc_lower or "big" in desc_lower:
            properties[VisualProperty.SIZE] = "large"
        elif "small" in desc_lower or "tiny" in desc_lower:
            properties[VisualProperty.SIZE] = "small"

        # Shapes
        shapes = ["circle", "square", "triangle", "rectangle", "sphere", "cube"]
        for shape in shapes:
            if shape in desc_lower:
                properties[VisualProperty.SHAPE] = shape
                break

        return properties

    def rotate(self, image_id: str, angle: float, axis: str = "z") -> VisualImage:
        """
        Mentally rotate an image.

        Mental rotation is a well-studied imagery operation.
        Rotation time is proportional to angle (Shepard & Metzler).
        """
        if image_id not in self.images:
            raise ValueError(f"Image {image_id} not found")

        image = self.images[image_id]

        # Create rotated version
        rotated_id = f"{image_id}_rot_{angle}"

        new_properties = image.properties.copy()
        current_orientation = new_properties.get(VisualProperty.ORIENTATION, 0)
        new_properties[VisualProperty.ORIENTATION] = (current_orientation + angle) % 360

        # Vividness decreases with larger rotations
        rotation_cost = abs(angle) / 180.0 * 0.2
        new_vividness = max(0.3, image.vividness - rotation_cost)

        rotated = VisualImage(
            id=rotated_id,
            description=f"{image.description} (rotated {angle}°)",
            properties=new_properties,
            vividness=new_vividness,
            stability=image.stability * 0.9,
        )

        self.images[rotated_id] = rotated
        return rotated

    def scale(self, image_id: str, factor: float) -> VisualImage:
        """Mentally resize an image"""
        if image_id not in self.images:
            raise ValueError(f"Image {image_id} not found")

        image = self.images[image_id]
        scaled_id = f"{image_id}_scale_{factor}"

        new_properties = image.properties.copy()
        new_properties.get(VisualProperty.SIZE, "medium")

        if factor > 1.5:
            new_properties[VisualProperty.SIZE] = "large"
        elif factor < 0.5:
            new_properties[VisualProperty.SIZE] = "small"

        scaled = VisualImage(
            id=scaled_id,
            description=f"{image.description} (scaled {factor}x)",
            properties=new_properties,
            vividness=image.vividness * 0.95,
            stability=image.stability * 0.95,
        )

        self.images[scaled_id] = scaled
        return scaled

    def combine(
        self, image_ids: List[str], spatial_arrangement: Optional[Dict[str, Tuple[str, str]]] = None
    ) -> VisualImage:
        """
        Combine multiple images into a scene.

        Creative visualization often involves combining
        disparate elements into novel scenes.
        """
        components = []
        for img_id in image_ids:
            if img_id in self.images:
                components.append(self.images[img_id])

        if not components:
            raise ValueError("No valid images to combine")

        combined_id = f"combined_{'_'.join(image_ids)}"
        descriptions = [c.description for c in components]

        # Combined vividness is average of components
        avg_vividness = np.mean([c.vividness for c in components])

        combined = VisualImage(
            id=combined_id,
            description=f"Scene with: {', '.join(descriptions)}",
            properties={},
            vividness=avg_vividness * 0.9,  # Slight loss from complexity
            stability=0.7,  # Complex images less stable
            components=components,
            spatial_relations=spatial_arrangement or {},
        )

        self.images[combined_id] = combined
        return combined

    def morph(self, image1_id: str, image2_id: str, blend_factor: float = 0.5) -> VisualImage:
        """
        Morph between two images.

        Creates intermediate image - useful for creative blending.
        """
        if image1_id not in self.images or image2_id not in self.images:
            raise ValueError("Both images must exist")

        img1 = self.images[image1_id]
        img2 = self.images[image2_id]

        morphed_id = f"morph_{image1_id}_{image2_id}_{blend_factor}"

        # Blend properties
        blended_props = {}
        all_props = set(img1.properties.keys()) | set(img2.properties.keys())

        for prop in all_props:
            v1 = img1.properties.get(prop)
            v2 = img2.properties.get(prop)

            if v1 is not None and v2 is not None:
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    blended_props[prop] = v1 * (1 - blend_factor) + v2 * blend_factor
                else:
                    blended_props[prop] = v1 if blend_factor < 0.5 else v2
            elif v1 is not None:
                blended_props[prop] = v1
            else:
                blended_props[prop] = v2

        morphed = VisualImage(
            id=morphed_id,
            description=f"Morph of {img1.description} and {img2.description}",
            properties=blended_props,
            vividness=min(img1.vividness, img2.vividness) * 0.8,
            stability=0.6,  # Morphed images are unstable
        )

        self.images[morphed_id] = morphed
        return morphed

    def inspect(
        self, image_id: str, focus_property: Optional[VisualProperty] = None
    ) -> Dict[str, Any]:
        """
        Inspect an image for details.

        Mental inspection - "looking" at parts of the mental image.
        """
        if image_id not in self.images:
            return {"error": "Image not found"}

        image = self.images[image_id]

        result = {
            "description": image.description,
            "vividness": image.vividness,
            "stability": image.stability,
            "properties": {p.value: v for p, v in image.properties.items()},
        }

        if focus_property and focus_property in image.properties:
            result["focused_property"] = {
                "property": focus_property.value,
                "value": image.properties[focus_property],
                "detail_level": "high" if image.vividness > 0.7 else "low",
            }

        if image.components:
            result["components"] = [c.description for c in image.components]

        return result

    def get_vividness(self, image_id: str) -> float:
        """Get vividness of an image"""
        if image_id in self.images:
            return self.images[image_id].vividness
        return 0.0

    def decay_inactive(self):
        """Apply decay to images not actively maintained"""
        for image in self.images.values():
            if image.id != self._active_image:
                image.vividness = max(0.1, image.vividness - self.decay_rate)
                image.stability = max(0.1, image.stability - self.decay_rate)

    def focus_on(self, image_id: str):
        """Focus attention on specific image"""
        if image_id in self.images:
            self._active_image = image_id
            # Focusing strengthens the image
            self.images[image_id].vividness = min(1.0, self.images[image_id].vividness + 0.1)
            self.images[image_id].stability = min(1.0, self.images[image_id].stability + 0.1)
