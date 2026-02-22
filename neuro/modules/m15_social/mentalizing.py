"""Mentalizing - Inferring beliefs, intentions, desires

Core function: Infer the mental states underlying behavior
Neural basis: mPFC, pSTS (posterior superior temporal sulcus)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class MentalizingParams:
    """Parameters for mentalizing"""

    n_features: int = 50
    inference_strength: float = 0.7
    learning_rate: float = 0.1
    prior_strength: float = 0.3


class IntentionInference:
    """Infer intentions from observed actions

    Neural basis: pSTS - processes biological motion and intentional actions
    """

    def __init__(self, params: Optional[MentalizingParams] = None):
        self.params = params or MentalizingParams()

        # pSTS activation
        self.psts_activation = np.zeros(self.params.n_features)

        # Learned action-intention mappings
        self.action_intention_map: Dict[str, np.ndarray] = {}

        # Prior expectations about intentions
        self.intention_priors = np.ones(self.params.n_features) * 0.5

    def observe_action(self, action: np.ndarray, action_type: str = "unknown") -> Dict:
        """Observe an action and infer intention"""
        if len(action) != self.params.n_features:
            action = np.resize(action, self.params.n_features)

        # pSTS processes action
        self.psts_activation = np.tanh(self.psts_activation * 0.3 + action * 0.7)

        # Check for known action type
        if action_type in self.action_intention_map:
            learned_intention = self.action_intention_map[action_type]
            inferred = (
                learned_intention * self.params.inference_strength
                + self.intention_priors * self.params.prior_strength
            )
        else:
            # Infer from action pattern
            inferred = (
                action * self.params.inference_strength
                + self.intention_priors * self.params.prior_strength
            )

        inferred = np.tanh(inferred)

        return {
            "action_type": action_type,
            "inferred_intention": inferred,
            "psts_activity": np.mean(self.psts_activation),
            "confidence": self.params.inference_strength,
        }

    def learn_action_intention(self, action_type: str, intention: np.ndarray):
        """Learn mapping from action type to intention"""
        if len(intention) != self.params.n_features:
            intention = np.resize(intention, self.params.n_features)

        self.action_intention_map[action_type] = intention.copy()

    def update_priors(self, observed_intentions: List[np.ndarray]):
        """Update intention priors based on observations"""
        if not observed_intentions:
            return

        mean_intention = np.mean(observed_intentions, axis=0)
        self.intention_priors = (
            1 - self.params.learning_rate
        ) * self.intention_priors + self.params.learning_rate * mean_intention


class DesireModeling:
    """Model others' desires and goals"""

    def __init__(self, params: Optional[MentalizingParams] = None):
        self.params = params or MentalizingParams()

        # Modeled desires for different agents
        self.agent_desires: Dict[str, np.ndarray] = {}

        # Goal states inferred for agents
        self.agent_goals: Dict[str, List[np.ndarray]] = {}

    def infer_desire(
        self, agent: str, behavior: np.ndarray, outcome: Optional[np.ndarray] = None
    ) -> Dict:
        """Infer desire from behavior and outcome"""
        if len(behavior) != self.params.n_features:
            behavior = np.resize(behavior, self.params.n_features)

        # Desire is what drives behavior
        # If we see approach behavior, infer positive desire for target
        inferred_desire = np.tanh(behavior * self.params.inference_strength)

        if outcome is not None:
            if len(outcome) != self.params.n_features:
                outcome = np.resize(outcome, self.params.n_features)
            # Outcome achieved suggests that was the desired state
            inferred_desire = 0.5 * inferred_desire + 0.5 * outcome

        # Update agent's modeled desires
        if agent in self.agent_desires:
            self.agent_desires[agent] = 0.7 * self.agent_desires[agent] + 0.3 * inferred_desire
        else:
            self.agent_desires[agent] = inferred_desire

        return {
            "agent": agent,
            "inferred_desire": inferred_desire,
            "desire_strength": np.mean(np.abs(inferred_desire)),
        }

    def add_goal(self, agent: str, goal: np.ndarray):
        """Add inferred goal for agent"""
        if len(goal) != self.params.n_features:
            goal = np.resize(goal, self.params.n_features)

        if agent not in self.agent_goals:
            self.agent_goals[agent] = []

        self.agent_goals[agent].append(goal.copy())

    def get_desires(self, agent: str) -> Optional[np.ndarray]:
        """Get modeled desires for agent"""
        return self.agent_desires.get(agent)

    def predict_action(self, agent: str) -> Optional[np.ndarray]:
        """Predict action from modeled desires"""
        if agent not in self.agent_desires:
            return None

        # Action should move toward desired state
        desire = self.agent_desires[agent]
        predicted_action = np.tanh(desire)

        return predicted_action


class MentalizingNetwork:
    """Integrated mentalizing network

    Combines intention inference, desire modeling, and belief tracking
    """

    def __init__(self, params: Optional[MentalizingParams] = None):
        self.params = params or MentalizingParams()

        self.intention_inference = IntentionInference(params)
        self.desire_modeling = DesireModeling(params)

        # Belief representations (simpler than full ToM belief tracker)
        self.agent_beliefs: Dict[str, np.ndarray] = {}

        # mPFC activation for mentalizing
        self.mpfc_activation = np.zeros(self.params.n_features)

    def mentalize(
        self, agent: str, behavior: np.ndarray, context: Optional[np.ndarray] = None
    ) -> Dict:
        """Full mentalizing: infer beliefs, desires, intentions"""
        if len(behavior) != self.params.n_features:
            behavior = np.resize(behavior, self.params.n_features)

        # mPFC processing
        self.mpfc_activation = np.tanh(self.mpfc_activation * 0.3 + behavior * 0.7)

        # Infer intention
        intention_result = self.intention_inference.observe_action(behavior)

        # Infer desire
        desire_result = self.desire_modeling.infer_desire(agent, behavior)

        # Infer belief (what they think is true based on behavior)
        if context is not None:
            if len(context) != self.params.n_features:
                context = np.resize(context, self.params.n_features)
            # Belief is influenced by context
            inferred_belief = 0.5 * behavior + 0.5 * context
        else:
            inferred_belief = behavior * 0.8

        inferred_belief = np.tanh(inferred_belief)
        self.agent_beliefs[agent] = inferred_belief

        return {
            "agent": agent,
            "intention": intention_result["inferred_intention"],
            "desire": desire_result["inferred_desire"],
            "belief": inferred_belief,
            "mpfc_activity": np.mean(self.mpfc_activation),
            "psts_activity": intention_result["psts_activity"],
        }

    def predict_behavior(self, agent: str) -> Dict:
        """Predict agent's behavior from mental states"""
        intention = self.intention_inference.observe_action(
            self.agent_beliefs.get(agent, np.zeros(self.params.n_features))
        )["inferred_intention"]
        desire = self.desire_modeling.get_desires(agent)
        belief = self.agent_beliefs.get(agent)

        if desire is None:
            desire = np.zeros(self.params.n_features)
        if belief is None:
            belief = np.zeros(self.params.n_features)

        # Behavior = trying to satisfy desires given beliefs
        predicted = 0.4 * intention + 0.4 * desire + 0.2 * belief
        predicted = np.tanh(predicted)

        return {
            "agent": agent,
            "predicted_behavior": predicted,
            "based_on_intention": intention,
            "based_on_desire": desire,
            "based_on_belief": belief,
        }

    def update(self, dt: float = 1.0):
        """Update mentalizing network"""
        self.mpfc_activation *= 1 - 0.1 * dt
        self.intention_inference.psts_activation *= 1 - 0.1 * dt

    def get_state(self) -> Dict:
        """Get network state"""
        return {
            "mpfc_activity": np.mean(self.mpfc_activation),
            "psts_activity": np.mean(self.intention_inference.psts_activation),
            "tracked_agents": list(self.agent_beliefs.keys()),
            "known_action_types": list(self.intention_inference.action_intention_map.keys()),
        }
