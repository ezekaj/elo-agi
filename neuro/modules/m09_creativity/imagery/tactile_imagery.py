"""
Tactile Imagery - Touch Simulation

Located in somatosensory cortex, tactile imagery allows:
- "Feeling" textures mentally
- Simulating touch sensations
- Imagining physical contact

New Finding (2025): Touch imagery vividness correlates positively
with creative performance. Touch imagery facilitates creative writing
through semantic integration and reorganization.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class TactileProperty(Enum):
    """Properties of tactile sensations"""

    TEXTURE = "texture"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    VIBRATION = "vibration"
    LOCATION = "location"
    WETNESS = "wetness"
    PAIN = "pain"  # Including pain as a tactile property


@dataclass
class TactileImage:
    """A mental tactile image - simulated touch sensation"""

    id: str
    description: str
    properties: Dict[TactileProperty, Any]
    vividness: float
    intensity: float  # Strength of sensation
    body_location: str  # Where on body
    semantic_associations: List[str] = field(default_factory=list)


class TactileImagery:
    """
    Tactile imagery system - touch simulation.

    Located in somatosensory cortex, enables:
    - Mental simulation of touch
    - Texture imagination
    - Temperature sensation
    - Physical contact simulation

    KEY FINDING: Touch imagery facilitates creativity through
    semantic integration - connecting concepts through physical experience.
    """

    def __init__(self, default_vividness: float = 0.5):
        self.sensations: Dict[str, TactileImage] = {}
        self.default_vividness = default_vividness

    def imagine_touch(
        self,
        touch_id: str,
        description: str,
        properties: Optional[Dict[TactileProperty, Any]] = None,
        body_location: str = "hand",
        intensity: float = 0.5,
    ) -> TactileImage:
        """
        Imagine a tactile sensation.

        Create mental experience of touch.
        """
        tactile = TactileImage(
            id=touch_id,
            description=description,
            properties=properties or {},
            vividness=self.default_vividness,
            intensity=intensity,
            body_location=body_location,
            semantic_associations=[],
        )

        self.sensations[touch_id] = tactile
        return tactile

    def feel_texture(self, texture_description: str) -> TactileImage:
        """
        Imagine feeling a texture.

        Textures are rich with semantic associations.
        """
        properties = self._extract_texture_properties(texture_description)
        associations = self._get_texture_associations(texture_description)

        texture_id = f"texture_{len(self.sensations)}"

        tactile = TactileImage(
            id=texture_id,
            description=f"Feeling {texture_description}",
            properties=properties,
            vividness=self.default_vividness,
            intensity=0.6,
            body_location="fingertips",
            semantic_associations=associations,
        )

        self.sensations[texture_id] = tactile
        return tactile

    def _extract_texture_properties(self, description: str) -> Dict[TactileProperty, Any]:
        """Extract tactile properties from texture description"""
        properties = {}
        desc_lower = description.lower()

        # Texture types
        if "smooth" in desc_lower:
            properties[TactileProperty.TEXTURE] = "smooth"
        elif "rough" in desc_lower:
            properties[TactileProperty.TEXTURE] = "rough"
        elif "soft" in desc_lower:
            properties[TactileProperty.TEXTURE] = "soft"
        elif "hard" in desc_lower:
            properties[TactileProperty.TEXTURE] = "hard"
        elif "fuzzy" in desc_lower:
            properties[TactileProperty.TEXTURE] = "fuzzy"
        elif "slimy" in desc_lower:
            properties[TactileProperty.TEXTURE] = "slimy"
        elif "grainy" in desc_lower or "sandy" in desc_lower:
            properties[TactileProperty.TEXTURE] = "grainy"

        # Temperature
        if "hot" in desc_lower or "warm" in desc_lower:
            properties[TactileProperty.TEMPERATURE] = "warm"
        elif "cold" in desc_lower or "cool" in desc_lower:
            properties[TactileProperty.TEMPERATURE] = "cold"

        # Wetness
        if "wet" in desc_lower or "damp" in desc_lower:
            properties[TactileProperty.WETNESS] = "wet"
        elif "dry" in desc_lower:
            properties[TactileProperty.WETNESS] = "dry"

        return properties

    def _get_texture_associations(self, description: str) -> List[str]:
        """
        Get semantic associations for a texture.

        This is key for creativity - textures evoke concepts.
        """
        associations = []
        desc_lower = description.lower()

        # Texture-concept associations
        texture_concepts = {
            "smooth": ["elegance", "calm", "flow", "polished"],
            "rough": ["nature", "raw", "authentic", "challenge"],
            "soft": ["comfort", "safety", "gentle", "nurture"],
            "hard": ["strength", "durability", "resistance", "solid"],
            "warm": ["comfort", "life", "energy", "passion"],
            "cold": ["fresh", "crisp", "distant", "clarity"],
            "wet": ["life", "growth", "change", "fluid"],
            "fuzzy": ["cozy", "childhood", "comfort", "softness"],
        }

        for texture, concepts in texture_concepts.items():
            if texture in desc_lower:
                associations.extend(concepts)

        return list(set(associations))

    def feel_temperature(self, temperature: str, body_location: str = "skin") -> TactileImage:
        """
        Imagine a temperature sensation.
        """
        temp_id = f"temp_{len(self.sensations)}"

        properties = {TactileProperty.TEMPERATURE: temperature}

        # Temperature associations
        associations = []
        if temperature in ["hot", "warm"]:
            associations = ["comfort", "energy", "passion", "sun"]
        elif temperature in ["cold", "cool"]:
            associations = ["fresh", "alert", "crisp", "clarity"]

        tactile = TactileImage(
            id=temp_id,
            description=f"Feeling {temperature}",
            properties=properties,
            vividness=self.default_vividness,
            intensity=0.6,
            body_location=body_location,
            semantic_associations=associations,
        )

        self.sensations[temp_id] = tactile
        return tactile

    def feel_pressure(self, pressure_level: str, body_location: str = "hand") -> TactileImage:
        """
        Imagine pressure sensation.
        """
        pressure_id = f"pressure_{len(self.sensations)}"

        properties = {TactileProperty.PRESSURE: pressure_level}

        associations = []
        if pressure_level in ["heavy", "strong"]:
            associations = ["weight", "importance", "gravity", "burden"]
        elif pressure_level in ["light", "gentle"]:
            associations = ["ease", "freedom", "lightness", "subtle"]

        tactile = TactileImage(
            id=pressure_id,
            description=f"Feeling {pressure_level} pressure",
            properties=properties,
            vividness=self.default_vividness,
            intensity=0.5 if pressure_level in ["light", "gentle"] else 0.8,
            body_location=body_location,
            semantic_associations=associations,
        )

        self.sensations[pressure_id] = tactile
        return tactile

    def combine_sensations(self, sensation_ids: List[str]) -> TactileImage:
        """
        Combine multiple tactile sensations.

        Complex touch experiences involve multiple properties.
        """
        sensations = [self.sensations[sid] for sid in sensation_ids if sid in self.sensations]

        if not sensations:
            raise ValueError("No valid sensations to combine")

        combined_id = f"combined_{len(self.sensations)}"

        # Merge properties
        merged_props = {}
        for s in sensations:
            merged_props.update(s.properties)

        # Merge associations
        all_associations = list(set(assoc for s in sensations for assoc in s.semantic_associations))

        combined = TactileImage(
            id=combined_id,
            description=f"Combined: {', '.join(s.description for s in sensations)}",
            properties=merged_props,
            vividness=np.mean([s.vividness for s in sensations]),
            intensity=np.mean([s.intensity for s in sensations]),
            body_location=sensations[0].body_location,
            semantic_associations=all_associations,
        )

        self.sensations[combined_id] = combined
        return combined

    def get_creative_associations(self, sensation_id: str) -> List[str]:
        """
        Get semantic associations for creative use.

        This is the key creative function of tactile imagery -
        bridging physical sensation to abstract concepts.
        """
        if sensation_id not in self.sensations:
            return []

        return self.sensations[sensation_id].semantic_associations

    def enhance_with_touch(self, concept: str) -> TactileImage:
        """
        Enhance a concept with tactile qualities.

        Used in creative writing - grounding abstract ideas in physical sensation.
        """
        # Find appropriate tactile qualities for concept
        concept_lower = concept.lower()

        if any(word in concept_lower for word in ["love", "comfort", "safe"]):
            return self.feel_texture("soft warm blanket")
        elif any(word in concept_lower for word in ["fear", "cold", "alone"]):
            return self.feel_texture("cold hard stone")
        elif any(word in concept_lower for word in ["excitement", "energy"]):
            return self.feel_texture("buzzing vibration")
        elif any(word in concept_lower for word in ["peace", "calm"]):
            return self.feel_texture("smooth cool water")
        else:
            return self.feel_texture("neutral surface")

    def get_vividness(self, sensation_id: str) -> float:
        """Get vividness of a tactile sensation"""
        if sensation_id in self.sensations:
            return self.sensations[sensation_id].vividness
        return 0.0
