"""
Theory of Mind - Social Inference

Reasoning about others' mental states: beliefs, desires, intentions.
Implements recursive mentalizing and false belief reasoning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class MentalStateType(Enum):
    BELIEF = "belief"
    DESIRE = "desire"
    INTENTION = "intention"
    EMOTION = "emotion"
    KNOWLEDGE = "knowledge"


@dataclass
class Belief:
    """A belief about the world"""
    belief_id: str
    content: str
    subject: Optional[str] = None
    confidence: float = 1.0
    source: str = "observation"
    is_true: Optional[bool] = None


@dataclass
class Desire:
    """A desire/goal"""
    desire_id: str
    content: str
    intensity: float = 0.5
    achieved: bool = False


@dataclass
class Intention:
    """An intention to act"""
    intention_id: str
    action: str
    target: Optional[str] = None
    conditions: List[str] = field(default_factory=list)
    commitment: float = 0.5


@dataclass
class MentalStateModel:
    """Model of another agent's mental states"""
    agent_id: str
    beliefs: Dict[str, Belief] = field(default_factory=dict)
    desires: Dict[str, Desire] = field(default_factory=dict)
    intentions: Dict[str, Intention] = field(default_factory=dict)
    emotions: Dict[str, float] = field(default_factory=dict)
    knowledge: Set[str] = field(default_factory=set)

    last_updated: float = 0.0
    confidence: float = 0.5

    def add_belief(self, belief: Belief):
        self.beliefs[belief.belief_id] = belief

    def add_desire(self, desire: Desire):
        self.desires[desire.desire_id] = desire

    def add_intention(self, intention: Intention):
        self.intentions[intention.intention_id] = intention

    def update_emotion(self, emotion: str, value: float):
        self.emotions[emotion] = np.clip(value, -1.0, 1.0)

    def knows(self, fact: str) -> bool:
        return fact in self.knowledge


@dataclass
class Observation:
    """An observation of an agent's behavior"""
    observation_id: str
    agent_id: str
    action: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


class TheoryOfMind:
    """
    Model other agents' beliefs, desires, and intentions.
    Implements mentalizing and social inference.

    Key capabilities:
    - Infer beliefs from behavior and context
    - Infer desires from goals and actions
    - Predict actions based on mental state model
    - Handle false beliefs (Sally-Anne test)
    - Recursive mentalizing (I think they think I think...)
    """

    def __init__(self, self_id: str = "self"):
        self.self_id = self_id
        self.agent_models: Dict[str, MentalStateModel] = {}
        self.observations: List[Observation] = []
        self.world_state: Dict[str, Any] = {}

        self.recursion_depth = 2

    def observe(self, observation: Observation):
        """Record an observation of an agent"""
        self.observations.append(observation)

        if observation.agent_id not in self.agent_models:
            self.agent_models[observation.agent_id] = MentalStateModel(
                agent_id=observation.agent_id
            )

        self._update_model_from_observation(observation)

    def _update_model_from_observation(self, obs: Observation):
        """Update mental model based on observed behavior"""
        model = self.agent_models[obs.agent_id]

        if "target" in obs.context:
            desire = Desire(
                desire_id=f"desire_{obs.action}_{obs.context['target']}",
                content=f"wants {obs.context['target']}",
                intensity=0.7
            )
            model.add_desire(desire)

        if "looked_at" in obs.context:
            belief = Belief(
                belief_id=f"belief_location_{obs.context['looked_at']}",
                content=f"believes something is at {obs.context['looked_at']}",
                source="inferred_from_action"
            )
            model.add_belief(belief)

        if obs.action in ["search", "look", "check"]:
            if "location" in obs.context:
                model.knowledge.add(f"checked_{obs.context['location']}")

        model.last_updated = obs.timestamp
        model.confidence = min(1.0, model.confidence + 0.1)

    def infer_belief(self,
                     agent_id: str,
                     subject: str
                     ) -> Optional[Belief]:
        """Infer what an agent believes about a subject"""
        if agent_id not in self.agent_models:
            return None

        model = self.agent_models[agent_id]

        for belief in model.beliefs.values():
            if belief.subject == subject or subject in belief.content:
                return belief

        if subject in self.world_state:
            if f"informed_{subject}" in model.knowledge:
                return Belief(
                    belief_id=f"inferred_{subject}",
                    content=f"believes {subject} is {self.world_state[subject]}",
                    confidence=0.8,
                    is_true=True
                )
            else:
                return Belief(
                    belief_id=f"inferred_{subject}_outdated",
                    content=f"may have outdated belief about {subject}",
                    confidence=0.5,
                    is_true=None
                )

        return None

    def infer_desire(self,
                     agent_id: str,
                     from_actions: List[str] = None
                     ) -> List[Desire]:
        """Infer what an agent wants from their actions"""
        if agent_id not in self.agent_models:
            return []

        model = self.agent_models[agent_id]
        inferred_desires = list(model.desires.values())

        if from_actions:
            for action in from_actions:
                if "get" in action or "take" in action:
                    target = action.split()[-1] if len(action.split()) > 1 else "something"
                    desire = Desire(
                        desire_id=f"inferred_desire_{target}",
                        content=f"wants {target}",
                        intensity=0.6
                    )
                    inferred_desires.append(desire)

                if "avoid" in action or "escape" in action:
                    target = action.split()[-1] if len(action.split()) > 1 else "something"
                    desire = Desire(
                        desire_id=f"inferred_avoidance_{target}",
                        content=f"wants to avoid {target}",
                        intensity=0.7
                    )
                    inferred_desires.append(desire)

        return inferred_desires

    def infer_intention(self,
                        agent_id: str,
                        context: Dict[str, Any] = None
                        ) -> Optional[Intention]:
        """Infer what an agent intends to do"""
        if agent_id not in self.agent_models:
            return None

        model = self.agent_models[agent_id]

        if model.intentions:
            return list(model.intentions.values())[0]

        desires = list(model.desires.values())
        if desires:
            strongest = max(desires, key=lambda d: d.intensity)
            return Intention(
                intention_id=f"inferred_intention_{strongest.desire_id}",
                action=f"achieve_{strongest.content}",
                commitment=strongest.intensity
            )

        return None

    def predict_action(self,
                       agent_id: str,
                       context: Dict[str, Any] = None
                       ) -> Tuple[str, float]:
        """Predict what action an agent will take"""
        if agent_id not in self.agent_models:
            return "unknown", 0.0

        model = self.agent_models[agent_id]

        intention = self.infer_intention(agent_id, context)
        if intention:
            return intention.action, intention.commitment

        desires = list(model.desires.values())
        if desires:
            strongest = max(desires, key=lambda d: d.intensity)
            return f"pursue_{strongest.content}", strongest.intensity * 0.7

        recent_obs = [o for o in self.observations if o.agent_id == agent_id]
        if recent_obs:
            last_action = recent_obs[-1].action
            return last_action, 0.3

        return "idle", 0.1

    def update_world_state(self, key: str, value: Any, inform_agents: List[str] = None):
        """Update world state and optionally inform agents"""
        self.world_state[key] = value

        if inform_agents:
            for agent_id in inform_agents:
                if agent_id in self.agent_models:
                    self.agent_models[agent_id].knowledge.add(f"informed_{key}")

    def false_belief_test(self,
                          agent_id: str,
                          object_id: str,
                          true_location: str,
                          believed_location: str
                          ) -> Dict[str, Any]:
        """
        Sally-Anne test implementation.
        Agent believes object is at believed_location,
        but it's actually at true_location.
        """
        if agent_id not in self.agent_models:
            self.agent_models[agent_id] = MentalStateModel(agent_id=agent_id)

        model = self.agent_models[agent_id]

        false_belief = Belief(
            belief_id=f"belief_{object_id}_location",
            content=f"{object_id} is at {believed_location}",
            confidence=0.9,
            is_true=False
        )
        model.add_belief(false_belief)

        self.world_state[f"{object_id}_location"] = true_location

        predicted_search = believed_location

        return {
            "agent_id": agent_id,
            "object": object_id,
            "true_location": true_location,
            "agent_believes": believed_location,
            "predicted_search_location": predicted_search,
            "has_false_belief": true_location != believed_location
        }

    def recursive_belief(self,
                         about_agent: str,
                         belief_content: str,
                         depth: int = 1
                         ) -> str:
        """
        Generate recursive belief statement.
        e.g., "I think Alice thinks Bob thinks..."
        """
        if depth <= 0 or depth > self.recursion_depth:
            return belief_content

        return f"I think {about_agent} thinks " + self.recursive_belief(
            about_agent, belief_content, depth - 1
        )

    def what_does_agent_think_i_believe(self,
                                         agent_id: str,
                                         subject: str
                                         ) -> Optional[str]:
        """Second-order belief: What does agent think I believe?"""
        if agent_id not in self.agent_models:
            return None

        my_belief = self.infer_belief(self.self_id, subject) if self.self_id in self.agent_models else None

        if my_belief:
            return f"{agent_id} probably thinks I believe: {my_belief.content}"

        return f"{agent_id} may not know what I believe about {subject}"

    def empathize(self, agent_id: str) -> Dict[str, float]:
        """Infer emotional state of an agent"""
        if agent_id not in self.agent_models:
            return {}

        model = self.agent_models[agent_id]

        emotions = dict(model.emotions)

        achieved_desires = [d for d in model.desires.values() if d.achieved]
        if achieved_desires:
            emotions['happiness'] = emotions.get('happiness', 0) + 0.3

        unmet_desires = [d for d in model.desires.values()
                        if not d.achieved and d.intensity > 0.7]
        if unmet_desires:
            emotions['frustration'] = emotions.get('frustration', 0) + 0.2

        return emotions

    def simulate_perspective(self,
                             agent_id: str,
                             scenario: Dict[str, Any]
                             ) -> Dict[str, Any]:
        """Simulate how an agent would perceive a scenario"""
        if agent_id not in self.agent_models:
            return {"error": "unknown_agent"}

        model = self.agent_models[agent_id]

        perceived = {}

        for key, value in scenario.items():
            if f"informed_{key}" in model.knowledge:
                perceived[key] = value
            elif key in [b.subject for b in model.beliefs.values() if b.subject]:
                for belief in model.beliefs.values():
                    if belief.subject == key:
                        perceived[key] = belief.content
                        break
            else:
                perceived[key] = "unknown"

        return {
            "agent_id": agent_id,
            "scenario": scenario,
            "agent_perception": perceived,
            "knowledge_gaps": [k for k, v in perceived.items() if v == "unknown"]
        }
