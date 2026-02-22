"""
Common Sense Reasoning: Physics, social, and temporal reasoning.

Implements intuitive reasoning about the physical and social world.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum


class PhysicsProperty(Enum):
    """Physical properties of objects."""

    SOLID = "solid"
    LIQUID = "liquid"
    GAS = "gas"
    HEAVY = "heavy"
    LIGHT = "light"
    BREAKABLE = "breakable"
    FLEXIBLE = "flexible"
    HOT = "hot"
    COLD = "cold"
    SHARP = "sharp"
    SOFT = "soft"


class SocialRelation(Enum):
    """Social relations between agents."""

    FRIEND = "friend"
    FAMILY = "family"
    COLLEAGUE = "colleague"
    STRANGER = "stranger"
    AUTHORITY = "authority"
    SUBORDINATE = "subordinate"


@dataclass
class PhysicsRule:
    """A physics rule for reasoning."""

    name: str
    condition: str  # Simplified condition description
    effect: str  # Simplified effect description
    confidence: float = 1.0


@dataclass
class SocialNorm:
    """A social norm or expectation."""

    name: str
    context: str
    expected_behavior: str
    violation_consequence: str
    strength: float = 1.0  # How strong is this norm


@dataclass
class TemporalRelation:
    """A temporal relation between events."""

    event1: str
    event2: str
    relation: str  # before, after, during, overlaps, meets
    duration: Optional[float] = None


class CommonSenseKB:
    """
    Common sense knowledge base.

    Stores general world knowledge that humans take for granted.
    """

    def __init__(self):
        # Object properties
        self._object_properties: Dict[str, Set[PhysicsProperty]] = {}

        # Default properties by category
        self._category_defaults: Dict[str, Set[PhysicsProperty]] = {
            "glass": {PhysicsProperty.SOLID, PhysicsProperty.BREAKABLE},
            "water": {PhysicsProperty.LIQUID, PhysicsProperty.COLD},
            "metal": {PhysicsProperty.SOLID, PhysicsProperty.HEAVY},
            "wood": {PhysicsProperty.SOLID},
            "fire": {PhysicsProperty.HOT},
            "ice": {PhysicsProperty.SOLID, PhysicsProperty.COLD},
            "rock": {PhysicsProperty.SOLID, PhysicsProperty.HEAVY},
            "feather": {PhysicsProperty.LIGHT, PhysicsProperty.SOFT},
            "knife": {PhysicsProperty.SOLID, PhysicsProperty.SHARP},
        }

        # Typical object locations
        self._typical_locations: Dict[str, List[str]] = {
            "kitchen": ["stove", "refrigerator", "sink", "knife", "plate"],
            "bedroom": ["bed", "pillow", "lamp", "closet"],
            "bathroom": ["toilet", "shower", "sink", "mirror"],
            "office": ["desk", "computer", "chair", "printer"],
            "garage": ["car", "tools", "workbench"],
        }

        # Typical actions for objects
        self._typical_actions: Dict[str, List[str]] = {
            "knife": ["cut", "slice", "chop"],
            "cup": ["drink", "fill", "pour"],
            "book": ["read", "open", "close"],
            "door": ["open", "close", "lock"],
            "car": ["drive", "park", "start"],
            "phone": ["call", "text", "charge"],
        }

        # Social roles
        self._social_roles: Dict[str, Dict[str, Any]] = {
            "doctor": {"authority": True, "trust": 0.9, "expertise": ["health"]},
            "teacher": {"authority": True, "trust": 0.8, "expertise": ["education"]},
            "friend": {"authority": False, "trust": 0.7, "expertise": []},
            "stranger": {"authority": False, "trust": 0.2, "expertise": []},
        }

    def get_properties(self, obj: str) -> Set[PhysicsProperty]:
        """Get properties of an object."""
        if obj in self._object_properties:
            return self._object_properties[obj]

        # Check category defaults
        for category, props in self._category_defaults.items():
            if category in obj.lower():
                return props

        return set()

    def set_properties(
        self,
        obj: str,
        properties: Set[PhysicsProperty],
    ) -> None:
        """Set properties for an object."""
        self._object_properties[obj] = properties

    def get_typical_location(self, obj: str) -> Optional[str]:
        """Get typical location for an object."""
        for location, objects in self._typical_locations.items():
            if obj in objects:
                return location
        return None

    def get_objects_in_location(self, location: str) -> List[str]:
        """Get typical objects in a location."""
        return self._typical_locations.get(location, [])

    def get_typical_actions(self, obj: str) -> List[str]:
        """Get typical actions for an object."""
        return self._typical_actions.get(obj, [])

    def get_social_role(self, role: str) -> Dict[str, Any]:
        """Get information about a social role."""
        return self._social_roles.get(role, {})


class PhysicsReasoner:
    """
    Intuitive physics reasoning.

    Reasons about physical properties, causation, and mechanics.
    """

    def __init__(self, kb: Optional[CommonSenseKB] = None):
        self.kb = kb or CommonSenseKB()

        # Physics rules
        self._rules: List[PhysicsRule] = [
            PhysicsRule(
                "gravity",
                "object is unsupported",
                "object falls down",
            ),
            PhysicsRule(
                "breakable",
                "breakable object falls on hard surface",
                "object breaks",
            ),
            PhysicsRule(
                "liquid_flow",
                "container with liquid is tilted",
                "liquid flows out",
            ),
            PhysicsRule(
                "heat_transfer",
                "hot object contacts cold object",
                "heat transfers from hot to cold",
            ),
            PhysicsRule(
                "melting",
                "ice is exposed to heat",
                "ice melts into water",
            ),
            PhysicsRule(
                "freezing",
                "water is exposed to cold below freezing",
                "water freezes into ice",
            ),
            PhysicsRule(
                "burning",
                "flammable object contacts fire",
                "object burns",
            ),
            PhysicsRule(
                "floating",
                "light object is in water",
                "object floats",
            ),
            PhysicsRule(
                "sinking",
                "heavy object is in water",
                "object sinks",
            ),
            PhysicsRule(
                "cutting",
                "sharp object contacts soft object with force",
                "soft object is cut",
            ),
        ]

    def predict_effect(
        self,
        obj: str,
        situation: str,
    ) -> List[Tuple[str, float]]:
        """Predict effects of a situation on an object."""
        effects = []
        properties = self.kb.get_properties(obj)

        for rule in self._rules:
            # Simple keyword matching for demonstration
            condition_match = False

            if "falls" in situation.lower() and "unsupported" in rule.condition:
                condition_match = True
            elif "falls" in situation.lower() and PhysicsProperty.BREAKABLE in properties:
                if "breakable" in rule.condition:
                    condition_match = True
            elif "heat" in situation.lower() and "heat" in rule.condition:
                condition_match = True
            elif "fire" in situation.lower() and "fire" in rule.condition:
                condition_match = True
            elif "water" in situation.lower():
                if PhysicsProperty.HEAVY in properties and "heavy" in rule.condition:
                    condition_match = True
                elif PhysicsProperty.LIGHT in properties and "light" in rule.condition:
                    condition_match = True

            if condition_match:
                effects.append((rule.effect, rule.confidence))

        return effects

    def will_break(self, obj: str, action: str) -> float:
        """Estimate probability of object breaking."""
        properties = self.kb.get_properties(obj)

        if PhysicsProperty.BREAKABLE not in properties:
            return 0.0

        # Risk factors
        risk = 0.5  # Base risk for breakable

        if "drop" in action.lower() or "fall" in action.lower():
            risk += 0.4
        if "throw" in action.lower():
            risk += 0.3
        if "hit" in action.lower() or "strike" in action.lower():
            risk += 0.3

        return min(1.0, risk)

    def can_contain(self, container: str, contained: str) -> bool:
        """Check if container can hold contained object."""
        container_props = self.kb.get_properties(container)
        contained_props = self.kb.get_properties(contained)

        # Liquids need solid containers
        if PhysicsProperty.LIQUID in contained_props:
            return PhysicsProperty.SOLID in container_props

        return True  # Simplified

    def get_applicable_rules(self, situation: str) -> List[PhysicsRule]:
        """Get physics rules applicable to a situation."""
        applicable = []

        for rule in self._rules:
            # Check if rule condition keywords are in situation
            keywords = rule.condition.lower().split()
            if any(kw in situation.lower() for kw in keywords):
                applicable.append(rule)

        return applicable


class SocialReasoner:
    """
    Social reasoning and theory of mind.

    Reasons about social situations, norms, and mental states.
    """

    def __init__(self, kb: Optional[CommonSenseKB] = None):
        self.kb = kb or CommonSenseKB()

        # Social norms
        self._norms: List[SocialNorm] = [
            SocialNorm(
                "greeting",
                "meeting someone",
                "say hello or greet",
                "perceived as rude",
                strength=0.9,
            ),
            SocialNorm(
                "thanking",
                "receiving help or gift",
                "express gratitude",
                "perceived as ungrateful",
                strength=0.95,
            ),
            SocialNorm(
                "personal_space",
                "interacting with strangers",
                "maintain appropriate distance",
                "discomfort and distrust",
                strength=0.8,
            ),
            SocialNorm(
                "turn_taking",
                "conversation",
                "wait for others to finish speaking",
                "perceived as rude or aggressive",
                strength=0.7,
            ),
            SocialNorm(
                "honesty",
                "general interaction",
                "tell the truth",
                "loss of trust if discovered",
                strength=0.85,
            ),
            SocialNorm(
                "reciprocity",
                "receiving favor",
                "return favor when possible",
                "relationship strain",
                strength=0.7,
            ),
        ]

    def get_applicable_norms(self, context: str) -> List[SocialNorm]:
        """Get social norms applicable to a context."""
        applicable = []

        for norm in self._norms:
            if norm.context.lower() in context.lower():
                applicable.append(norm)
            elif any(word in context.lower() for word in norm.context.split()):
                applicable.append(norm)

        return applicable

    def predict_reaction(
        self,
        action: str,
        relationship: SocialRelation,
    ) -> Tuple[str, float]:
        """Predict social reaction to an action."""
        # Check norm violations
        for norm in self._norms:
            # Simple check if action violates norm
            if "help" in action.lower() and "thanking" in norm.name:
                if "thank" not in action.lower():
                    return (norm.violation_consequence, norm.strength)

        # Default positive reaction
        return ("neutral or positive", 0.5)

    def infer_goal(
        self,
        observed_actions: List[str],
    ) -> List[Tuple[str, float]]:
        """Infer likely goals from observed actions."""
        goals = []

        for action in observed_actions:
            action_lower = action.lower()

            if "eat" in action_lower or "cook" in action_lower:
                goals.append(("satisfy hunger", 0.8))
            if "work" in action_lower or "study" in action_lower:
                goals.append(("achieve task", 0.7))
            if "talk" in action_lower or "meet" in action_lower:
                goals.append(("social connection", 0.6))
            if "buy" in action_lower or "shop" in action_lower:
                goals.append(("acquire item", 0.7))
            if "help" in action_lower:
                goals.append(("assist others", 0.7))

        return goals

    def infer_emotion(
        self,
        situation: str,
        outcome: str,
    ) -> List[Tuple[str, float]]:
        """Infer likely emotional state."""
        emotions = []

        if "success" in outcome.lower() or "win" in outcome.lower():
            emotions.append(("happy", 0.8))
            emotions.append(("proud", 0.6))
        elif "fail" in outcome.lower() or "lose" in outcome.lower():
            emotions.append(("sad", 0.7))
            emotions.append(("frustrated", 0.6))
        elif "surprise" in situation.lower():
            emotions.append(("surprised", 0.8))
        elif "threat" in situation.lower() or "danger" in situation.lower():
            emotions.append(("afraid", 0.8))
        elif "unfair" in situation.lower():
            emotions.append(("angry", 0.7))

        return emotions

    def trust_level(
        self,
        source_role: str,
        topic: str,
    ) -> float:
        """Estimate trust level for information from source about topic."""
        role_info = self.kb.get_social_role(source_role)

        if not role_info:
            return 0.3  # Unknown role

        base_trust = role_info.get("trust", 0.5)
        expertise = role_info.get("expertise", [])

        # Boost trust if topic matches expertise
        if any(exp in topic.lower() for exp in expertise):
            base_trust = min(1.0, base_trust + 0.2)

        return base_trust


class TemporalReasoner:
    """
    Temporal reasoning about events and time.

    Reasons about sequences, durations, and temporal relations.
    """

    def __init__(self):
        # Typical durations (in minutes)
        self._typical_durations: Dict[str, Tuple[float, float]] = {
            "eating_meal": (15, 60),
            "cooking": (20, 120),
            "showering": (5, 20),
            "sleeping": (300, 600),  # 5-10 hours
            "working": (240, 600),  # 4-10 hours
            "meeting": (30, 120),
            "phone_call": (2, 30),
            "driving_commute": (15, 90),
            "exercise": (20, 90),
            "reading": (15, 120),
        }

        # Event sequences (typical orderings)
        self._typical_sequences: Dict[str, List[str]] = {
            "morning_routine": ["wake_up", "shower", "dress", "eat_breakfast", "leave_home"],
            "cooking": ["prepare_ingredients", "cook", "serve", "eat"],
            "shopping": ["travel_to_store", "browse", "select_items", "pay", "leave"],
            "meeting": ["arrive", "greet", "discuss", "conclude", "leave"],
        }

    def get_typical_duration(
        self,
        activity: str,
    ) -> Optional[Tuple[float, float]]:
        """Get typical duration range for an activity (min, max in minutes)."""
        for key, duration in self._typical_durations.items():
            if key in activity.lower() or activity.lower() in key:
                return duration
        return None

    def get_typical_sequence(self, activity_type: str) -> List[str]:
        """Get typical sequence of events for an activity type."""
        for key, sequence in self._typical_sequences.items():
            if key in activity_type.lower():
                return sequence
        return []

    def order_events(
        self,
        events: List[str],
        context: Optional[str] = None,
    ) -> List[str]:
        """Order events in likely temporal sequence."""
        if context:
            template = self.get_typical_sequence(context)
            if template:
                # Order events according to template
                ordered = []
                remaining = list(events)

                for step in template:
                    for event in remaining:
                        if step in event.lower() or event.lower() in step:
                            ordered.append(event)
                            remaining.remove(event)
                            break

                ordered.extend(remaining)
                return ordered

        return events  # Return original order if no template

    def infer_relation(
        self,
        event1: str,
        event2: str,
    ) -> Optional[str]:
        """Infer temporal relation between events."""
        # Check typical sequences
        for sequence in self._typical_sequences.values():
            idx1 = -1
            idx2 = -1

            for i, step in enumerate(sequence):
                if step in event1.lower() or event1.lower() in step:
                    idx1 = i
                if step in event2.lower() or event2.lower() in step:
                    idx2 = i

            if idx1 >= 0 and idx2 >= 0:
                if idx1 < idx2:
                    return "before"
                elif idx1 > idx2:
                    return "after"
                else:
                    return "simultaneous"

        return None

    def is_plausible_duration(
        self,
        activity: str,
        duration: float,
    ) -> bool:
        """Check if duration is plausible for activity."""
        typical = self.get_typical_duration(activity)
        if typical is None:
            return True  # Unknown activity, assume plausible

        min_dur, max_dur = typical
        # Allow some flexibility
        return min_dur * 0.5 <= duration <= max_dur * 2


class CommonSenseReasoner:
    """
    Unified common sense reasoning system.

    Combines physics, social, and temporal reasoning.
    """

    def __init__(self):
        self.kb = CommonSenseKB()
        self.physics = PhysicsReasoner(self.kb)
        self.social = SocialReasoner(self.kb)
        self.temporal = TemporalReasoner()

    def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform common sense reasoning on a query."""
        result = {
            "query": query,
            "physics_effects": [],
            "social_norms": [],
            "temporal_info": None,
        }

        # Physics reasoning
        if context and "object" in context:
            effects = self.physics.predict_effect(context["object"], query)
            result["physics_effects"] = effects

        # Social reasoning
        norms = self.social.get_applicable_norms(query)
        result["social_norms"] = [{"name": n.name, "expected": n.expected_behavior} for n in norms]

        # Temporal reasoning
        if context and "activity" in context:
            duration = self.temporal.get_typical_duration(context["activity"])
            if duration:
                result["temporal_info"] = {
                    "typical_duration_min": duration[0],
                    "typical_duration_max": duration[1],
                }

        return result

    def is_plausible(
        self,
        statement: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, float, str]:
        """Check if a statement is plausible."""
        # Simple plausibility checks
        statement_lower = statement.lower()

        # Physics violations
        if "fly" in statement_lower and "human" in statement_lower:
            if "airplane" not in statement_lower:
                return False, 0.1, "Humans cannot fly without equipment"

        if "underwater" in statement_lower and "breathe" in statement_lower:
            if "fish" not in statement_lower and "scuba" not in statement_lower:
                return False, 0.1, "Most creatures cannot breathe underwater"

        # Temporal violations
        if "before born" in statement_lower or "before birth" in statement_lower:
            if "remember" in statement_lower:
                return False, 0.05, "Cannot remember events before birth"

        return True, 0.7, "No obvious violations detected"

    def fill_gaps(
        self,
        partial_story: List[str],
    ) -> List[str]:
        """Fill in missing common sense details in a story."""
        filled = []

        for i, event in enumerate(partial_story):
            filled.append(event)

            # Check if we need to add implicit events
            if i < len(partial_story) - 1:
                next_event = partial_story[i + 1]

                # Travel inference
                if "at home" in event.lower() and "at work" in next_event.lower():
                    filled.append("(implicit: traveled to work)")

                # Preparation inference
                if "eat" in next_event.lower() and "cook" not in event.lower():
                    filled.append("(implicit: prepared or obtained food)")

        return filled

    def statistics(self) -> Dict[str, Any]:
        """Get reasoner statistics."""
        return {
            "physics_rules": len(self.physics._rules),
            "social_norms": len(self.social._norms),
            "typical_durations": len(self.temporal._typical_durations),
            "typical_sequences": len(self.temporal._typical_sequences),
        }
