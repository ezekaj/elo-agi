"""Theory of Mind - Mental state attribution

Core function: Attribute mental states (beliefs, desires, intentions) to self and others
Neural basis: mPFC (self-referential), TPJ (attribution to others)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class ToMParams:
    """Parameters for Theory of Mind"""

    n_features: int = 50
    belief_decay: float = 0.05
    inference_threshold: float = 0.5
    learning_rate: float = 0.1


class BeliefTracker:
    """Track beliefs about world state and others' beliefs

    Supports first-order (what X believes) and second-order (what X believes Y believes)
    """

    def __init__(self, params: Optional[ToMParams] = None):
        self.params = params or ToMParams()

        # Own beliefs about world
        self.own_beliefs: Dict[str, np.ndarray] = {}

        # Beliefs attributed to others (agent -> topic -> belief)
        self.others_beliefs: Dict[str, Dict[str, np.ndarray]] = {}

        # Second-order beliefs (what X thinks Y believes)
        self.second_order: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        # Belief confidence
        self.confidence: Dict[str, float] = {}

    def update_own_belief(self, topic: str, belief: np.ndarray, confidence: float = 1.0):
        """Update own belief about a topic"""
        if len(belief) != self.params.n_features:
            belief = np.resize(belief, self.params.n_features)

        self.own_beliefs[topic] = belief.copy()
        self.confidence[f"self_{topic}"] = confidence

    def attribute_belief(self, agent: str, topic: str, belief: np.ndarray, confidence: float = 0.8):
        """Attribute a belief to another agent"""
        if len(belief) != self.params.n_features:
            belief = np.resize(belief, self.params.n_features)

        if agent not in self.others_beliefs:
            self.others_beliefs[agent] = {}

        self.others_beliefs[agent][topic] = belief.copy()
        self.confidence[f"{agent}_{topic}"] = confidence

    def attribute_second_order(self, agent1: str, agent2: str, topic: str, belief: np.ndarray):
        """Attribute what agent1 thinks agent2 believes"""
        if len(belief) != self.params.n_features:
            belief = np.resize(belief, self.params.n_features)

        if agent1 not in self.second_order:
            self.second_order[agent1] = {}
        if agent2 not in self.second_order[agent1]:
            self.second_order[agent1][agent2] = {}

        self.second_order[agent1][agent2][topic] = belief.copy()

    def get_belief(self, agent: str, topic: str) -> Optional[np.ndarray]:
        """Get belief (own or attributed)"""
        if agent == "self":
            return self.own_beliefs.get(topic)
        return self.others_beliefs.get(agent, {}).get(topic)

    def check_false_belief(self, agent: str, topic: str) -> bool:
        """Check if agent has a false belief (different from reality/own belief)"""
        own = self.own_beliefs.get(topic)
        other = self.get_belief(agent, topic)

        if own is None or other is None:
            return False

        similarity = np.dot(own, other) / (np.linalg.norm(own) * np.linalg.norm(other) + 1e-8)

        return similarity < 0.5  # Different beliefs = false belief

    def decay_beliefs(self, dt: float = 1.0):
        """Decay confidence in beliefs over time"""
        for key in self.confidence:
            self.confidence[key] *= 1 - self.params.belief_decay * dt


class MentalStateAttribution:
    """Attribute mental states (beliefs, desires, intentions) to agents"""

    def __init__(self, params: Optional[ToMParams] = None):
        self.params = params or ToMParams()

        # Mental state templates
        self.belief_template = np.random.randn(self.params.n_features) * 0.1
        self.desire_template = np.random.randn(self.params.n_features) * 0.1
        self.intention_template = np.random.randn(self.params.n_features) * 0.1

        # Attributed states (agent -> state_type -> state)
        self.attributed_states: Dict[str, Dict[str, np.ndarray]] = {}

    def observe_behavior(self, agent: str, behavior: np.ndarray) -> Dict:
        """Infer mental states from observed behavior"""
        if len(behavior) != self.params.n_features:
            behavior = np.resize(behavior, self.params.n_features)

        if agent not in self.attributed_states:
            self.attributed_states[agent] = {}

        # Infer belief (what they think is true)
        belief_activation = np.dot(behavior, self.belief_template)
        inferred_belief = behavior * (1 + belief_activation * 0.5)

        # Infer desire (what they want)
        desire_activation = np.dot(behavior, self.desire_template)
        inferred_desire = np.tanh(behavior * desire_activation)

        # Infer intention (what they plan to do)
        intention_activation = np.dot(behavior, self.intention_template)
        inferred_intention = np.tanh(behavior + self.intention_template * intention_activation)

        self.attributed_states[agent]["belief"] = inferred_belief
        self.attributed_states[agent]["desire"] = inferred_desire
        self.attributed_states[agent]["intention"] = inferred_intention

        return {
            "agent": agent,
            "belief": inferred_belief,
            "desire": inferred_desire,
            "intention": inferred_intention,
            "belief_strength": abs(belief_activation),
            "desire_strength": abs(desire_activation),
            "intention_strength": abs(intention_activation),
        }

    def get_attributed_state(self, agent: str, state_type: str) -> Optional[np.ndarray]:
        """Get attributed mental state"""
        return self.attributed_states.get(agent, {}).get(state_type)

    def predict_behavior(self, agent: str) -> Optional[np.ndarray]:
        """Predict behavior from attributed mental states"""
        if agent not in self.attributed_states:
            return None

        states = self.attributed_states[agent]

        # Behavior is driven by desires constrained by beliefs
        belief = states.get("belief", np.zeros(self.params.n_features))
        desire = states.get("desire", np.zeros(self.params.n_features))
        intention = states.get("intention", np.zeros(self.params.n_features))

        predicted = intention * 0.5 + desire * 0.3 + belief * 0.2
        return np.tanh(predicted)


class TheoryOfMind:
    """Integrated Theory of Mind system"""

    def __init__(self, params: Optional[ToMParams] = None):
        self.params = params or ToMParams()

        self.belief_tracker = BeliefTracker(params)
        self.attribution = MentalStateAttribution(params)

        # mPFC activation (self-referential)
        self.mpfc_activation = np.zeros(self.params.n_features)
        # TPJ activation (other-referential)
        self.tpj_activation = np.zeros(self.params.n_features)

    def process_self(self, mental_state: np.ndarray, state_type: str = "belief"):
        """Process own mental state (mPFC)"""
        if len(mental_state) != self.params.n_features:
            mental_state = np.resize(mental_state, self.params.n_features)

        self.mpfc_activation = np.tanh(self.mpfc_activation * 0.5 + mental_state * 0.5)

        if state_type == "belief":
            self.belief_tracker.update_own_belief("current", mental_state)

        return {"mpfc_activity": np.mean(self.mpfc_activation)}

    def process_other(self, agent: str, observed_behavior: np.ndarray) -> Dict:
        """Process another's mental state (TPJ)"""
        if len(observed_behavior) != self.params.n_features:
            observed_behavior = np.resize(observed_behavior, self.params.n_features)

        self.tpj_activation = np.tanh(self.tpj_activation * 0.5 + observed_behavior * 0.5)

        attribution = self.attribution.observe_behavior(agent, observed_behavior)

        # Track inferred belief
        if attribution["belief"] is not None:
            self.belief_tracker.attribute_belief(
                agent, "current", attribution["belief"], confidence=attribution["belief_strength"]
            )

        attribution["tpj_activity"] = np.mean(self.tpj_activation)
        return attribution

    def run_false_belief_task(
        self, agent: str, reality: np.ndarray, agent_belief: np.ndarray
    ) -> Dict:
        """Run Sally-Anne style false belief task"""
        # Update own belief (reality)
        self.belief_tracker.update_own_belief("location", reality)

        # Attribute (false) belief to other
        self.belief_tracker.attribute_belief(agent, "location", agent_belief)

        # Check for false belief understanding
        has_false_belief = self.belief_tracker.check_false_belief(agent, "location")

        # Predict agent's behavior based on their belief
        predicted_behavior = self.attribution.predict_behavior(agent)

        return {
            "agent": agent,
            "reality": reality,
            "agent_belief": agent_belief,
            "false_belief_detected": has_false_belief,
            "predicted_behavior": predicted_behavior,
        }

    def update(self, dt: float = 1.0):
        """Update ToM system"""
        self.belief_tracker.decay_beliefs(dt)

        # Decay neural activations
        self.mpfc_activation *= 1 - 0.1 * dt
        self.tpj_activation *= 1 - 0.1 * dt

    def get_state(self) -> Dict:
        """Get ToM state"""
        return {
            "mpfc_activity": np.mean(self.mpfc_activation),
            "tpj_activity": np.mean(self.tpj_activation),
            "own_beliefs": list(self.belief_tracker.own_beliefs.keys()),
            "tracked_agents": list(self.belief_tracker.others_beliefs.keys()),
        }
