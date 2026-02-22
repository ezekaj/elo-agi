"""
Motor Imagery - Action Simulation

Located in premotor cortex, motor imagery allows:
- Mental rehearsal of actions
- Simulation of movements
- Planning of motor sequences
- "Feeling" of actions without executing them

Motor imagery is used by athletes, musicians, and anyone
planning physical actions.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class MotorProperty(Enum):
    """Properties of motor images"""

    FORCE = "force"
    SPEED = "speed"
    PRECISION = "precision"
    DURATION = "duration"
    BODY_PART = "body_part"
    DIRECTION = "direction"
    COORDINATION = "coordination"


@dataclass
class MotorImage:
    """A mental motor image - simulated action"""

    id: str
    action: str
    properties: Dict[MotorProperty, Any]
    vividness: float
    duration: float  # seconds
    body_parts: List[str] = field(default_factory=list)
    sequence: List["MotorImage"] = field(default_factory=list)
    kinesthetic_feel: float = 0.5  # How much you "feel" the movement


class MotorImagery:
    """
    Motor imagery system - action simulation.

    Located in premotor cortex, enables:
    - Mental rehearsal of movements
    - Action planning
    - Kinesthetic simulation
    - Motor sequence composition
    """

    def __init__(self, default_vividness: float = 0.6):
        self.actions: Dict[str, MotorImage] = {}
        self.default_vividness = default_vividness

    def imagine_action(
        self,
        action_id: str,
        action: str,
        body_parts: Optional[List[str]] = None,
        properties: Optional[Dict[MotorProperty, Any]] = None,
        duration: float = 1.0,
    ) -> MotorImage:
        """
        Imagine performing an action.

        Mental rehearsal of movement.
        """
        motor_image = MotorImage(
            id=action_id,
            action=action,
            properties=properties or {},
            vividness=self.default_vividness,
            duration=duration,
            body_parts=body_parts or [],
            kinesthetic_feel=0.5,
        )

        self.actions[action_id] = motor_image
        return motor_image

    def simulate_movement(self, description: str) -> MotorImage:
        """
        Simulate a movement from verbal description.
        """
        properties = self._extract_properties(description)
        body_parts = self._extract_body_parts(description)

        action_id = f"movement_{len(self.actions)}"
        return self.imagine_action(
            action_id, description, body_parts=body_parts, properties=properties
        )

    def _extract_properties(self, description: str) -> Dict[MotorProperty, Any]:
        """Extract motor properties from description"""
        properties = {}
        desc_lower = description.lower()

        # Speed
        if "fast" in desc_lower or "quick" in desc_lower:
            properties[MotorProperty.SPEED] = "fast"
        elif "slow" in desc_lower or "careful" in desc_lower:
            properties[MotorProperty.SPEED] = "slow"

        # Force
        if "hard" in desc_lower or "strong" in desc_lower:
            properties[MotorProperty.FORCE] = "high"
        elif "gentle" in desc_lower or "soft" in desc_lower:
            properties[MotorProperty.FORCE] = "low"

        # Precision
        if "precise" in desc_lower or "accurate" in desc_lower:
            properties[MotorProperty.PRECISION] = "high"

        return properties

    def _extract_body_parts(self, description: str) -> List[str]:
        """Extract body parts from description"""
        parts = []
        desc_lower = description.lower()

        body_part_keywords = [
            "hand",
            "hands",
            "arm",
            "arms",
            "leg",
            "legs",
            "foot",
            "feet",
            "finger",
            "fingers",
            "head",
            "body",
            "whole body",
        ]

        for part in body_part_keywords:
            if part in desc_lower:
                parts.append(part)

        return parts if parts else ["whole body"]

    def rehearse(self, action_id: str, repetitions: int = 3) -> MotorImage:
        """
        Mentally rehearse an action.

        Repeated mental rehearsal strengthens motor representations.
        """
        if action_id not in self.actions:
            raise ValueError(f"Action {action_id} not found")

        action = self.actions[action_id]

        # Rehearsal improves vividness and kinesthetic feel
        for _ in range(repetitions):
            action.vividness = min(1.0, action.vividness + 0.05)
            action.kinesthetic_feel = min(1.0, action.kinesthetic_feel + 0.03)

        return action

    def slow_motion(self, action_id: str, factor: float = 0.5) -> MotorImage:
        """
        Imagine action in slow motion.

        Slowing down helps analyze movement details.
        """
        if action_id not in self.actions:
            raise ValueError(f"Action {action_id} not found")

        original = self.actions[action_id]
        slow_id = f"{action_id}_slow_{factor}"

        new_props = original.properties.copy()
        new_props[MotorProperty.SPEED] = "slow"

        slow = MotorImage(
            id=slow_id,
            action=f"{original.action} (slow motion)",
            properties=new_props,
            vividness=original.vividness,
            duration=original.duration / factor,
            body_parts=original.body_parts,
            kinesthetic_feel=original.kinesthetic_feel * 1.1,  # Better feel in slow-mo
        )

        self.actions[slow_id] = slow
        return slow

    def sequence_actions(self, action_ids: List[str]) -> MotorImage:
        """
        Create a sequence of actions.

        Motor planning - chaining actions together.
        """
        actions = [self.actions[aid] for aid in action_ids if aid in self.actions]

        if not actions:
            raise ValueError("No valid actions to sequence")

        total_duration = sum(a.duration for a in actions)
        all_body_parts = list(set(p for a in actions for p in a.body_parts))

        seq_id = f"sequence_{len(self.actions)}"

        sequence = MotorImage(
            id=seq_id,
            action=f"Sequence: {' -> '.join(a.action for a in actions)}",
            properties={
                MotorProperty.DURATION: total_duration,
                MotorProperty.COORDINATION: "sequential",
            },
            vividness=np.mean([a.vividness for a in actions]),
            duration=total_duration,
            body_parts=all_body_parts,
            sequence=actions,
            kinesthetic_feel=np.mean([a.kinesthetic_feel for a in actions]),
        )

        self.actions[seq_id] = sequence
        return sequence

    def modify_force(self, action_id: str, force_level: str) -> MotorImage:
        """
        Modify the force of an action.

        Imagine doing the same action with different force.
        """
        if action_id not in self.actions:
            raise ValueError(f"Action {action_id} not found")

        original = self.actions[action_id]
        new_id = f"{action_id}_force_{force_level}"

        new_props = original.properties.copy()
        new_props[MotorProperty.FORCE] = force_level

        modified = MotorImage(
            id=new_id,
            action=f"{original.action} ({force_level} force)",
            properties=new_props,
            vividness=original.vividness,
            duration=original.duration,
            body_parts=original.body_parts,
            kinesthetic_feel=original.kinesthetic_feel,
        )

        self.actions[new_id] = modified
        return modified

    def mirror(self, action_id: str) -> MotorImage:
        """
        Create mirror image of action (e.g., left hand instead of right).
        """
        if action_id not in self.actions:
            raise ValueError(f"Action {action_id} not found")

        original = self.actions[action_id]
        mirror_id = f"{action_id}_mirror"

        # Mirror body parts
        mirrored_parts = []
        for part in original.body_parts:
            if "left" in part:
                mirrored_parts.append(part.replace("left", "right"))
            elif "right" in part:
                mirrored_parts.append(part.replace("right", "left"))
            else:
                mirrored_parts.append(part)

        mirrored = MotorImage(
            id=mirror_id,
            action=f"{original.action} (mirrored)",
            properties=original.properties.copy(),
            vividness=original.vividness * 0.9,  # Slight loss
            duration=original.duration,
            body_parts=mirrored_parts,
            kinesthetic_feel=original.kinesthetic_feel * 0.9,
        )

        self.actions[mirror_id] = mirrored
        return mirrored

    def get_kinesthetic_feel(self, action_id: str) -> float:
        """Get kinesthetic vividness of an action"""
        if action_id in self.actions:
            return self.actions[action_id].kinesthetic_feel
        return 0.0

    def get_vividness(self, action_id: str) -> float:
        """Get vividness of an action"""
        if action_id in self.actions:
            return self.actions[action_id].vividness
        return 0.0
