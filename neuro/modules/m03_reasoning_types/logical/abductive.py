"""
Abductive Reasoning - Effect → Cause

Inference to the best explanation. Probabilistic, prefers simpler explanations.
"The grass is wet" → "It probably rained"
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Effect:
    """An observed effect to be explained"""
    effect_id: str
    description: str
    features: Dict[str, Any] = field(default_factory=dict)
    certainty: float = 1.0
    timestamp: float = 0.0


@dataclass
class Cause:
    """A potential cause/hypothesis"""
    cause_id: str
    description: str
    typical_effects: List[str] = field(default_factory=list)
    prior_probability: float = 0.5
    complexity: float = 1.0
    mechanism: Optional[str] = None


@dataclass
class Explanation:
    """An explanation linking cause to effect"""
    explanation_id: str
    effect: Effect
    cause: Cause
    probability: float
    explanatory_power: float
    simplicity_score: float
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Combined score considering probability, power, and simplicity"""
        return (self.probability * 0.4 +
                self.explanatory_power * 0.4 +
                self.simplicity_score * 0.2)


class CausalModel:
    """A model of cause-effect relationships"""

    def __init__(self):
        self.causes: Dict[str, Cause] = {}
        self.effects: Dict[str, Effect] = {}
        self.cause_effect_links: Dict[str, Dict[str, float]] = {}
        self.effect_cause_links: Dict[str, Dict[str, float]] = {}

    def add_cause(self, cause: Cause):
        """Add a cause to the model"""
        self.causes[cause.cause_id] = cause
        if cause.cause_id not in self.cause_effect_links:
            self.cause_effect_links[cause.cause_id] = {}

    def add_effect(self, effect: Effect):
        """Add an effect to the model"""
        self.effects[effect.effect_id] = effect
        if effect.effect_id not in self.effect_cause_links:
            self.effect_cause_links[effect.effect_id] = {}

    def link_cause_effect(self,
                          cause_id: str,
                          effect_id: str,
                          probability: float):
        """Create a causal link with probability P(effect|cause)"""
        if cause_id not in self.cause_effect_links:
            self.cause_effect_links[cause_id] = {}
        if effect_id not in self.effect_cause_links:
            self.effect_cause_links[effect_id] = {}

        self.cause_effect_links[cause_id][effect_id] = probability
        self.effect_cause_links[effect_id][cause_id] = probability

    def get_possible_causes(self, effect_id: str) -> List[Tuple[str, float]]:
        """Get causes that could produce this effect"""
        if effect_id not in self.effect_cause_links:
            return []
        return list(self.effect_cause_links[effect_id].items())


class AbductiveReasoner:
    """
    Inference to the best explanation.
    Generates and ranks hypotheses that explain observations.

    "Grass is wet" → "It probably rained"
    Key: NOT certain - probabilistic
    Key: Prefers simpler explanations (Occam's razor)
    """

    def __init__(self,
                 simplicity_weight: float = 0.3,
                 prior_weight: float = 0.2):
        self.causal_model = CausalModel()
        self.observed_effects: Dict[str, Effect] = {}
        self.explanations: Dict[str, Explanation] = {}
        self.simplicity_weight = simplicity_weight
        self.prior_weight = prior_weight

        self._initialize_common_causes()

    def _initialize_common_causes(self):
        """Initialize with some common cause-effect relationships"""
        rain = Cause(
            cause_id="rain",
            description="It rained",
            typical_effects=["wet_grass", "wet_pavement", "puddles"],
            prior_probability=0.3,
            complexity=1.0
        )
        self.causal_model.add_cause(rain)

        sprinkler = Cause(
            cause_id="sprinkler",
            description="Sprinkler was on",
            typical_effects=["wet_grass"],
            prior_probability=0.2,
            complexity=1.0
        )
        self.causal_model.add_cause(sprinkler)

        dew = Cause(
            cause_id="dew",
            description="Morning dew formed",
            typical_effects=["wet_grass"],
            prior_probability=0.4,
            complexity=1.0
        )
        self.causal_model.add_cause(dew)

        for effect_id in rain.typical_effects:
            self.causal_model.link_cause_effect(rain.cause_id, effect_id, 0.9)

        for effect_id in sprinkler.typical_effects:
            self.causal_model.link_cause_effect(sprinkler.cause_id, effect_id, 0.95)

        for effect_id in dew.typical_effects:
            self.causal_model.link_cause_effect(dew.cause_id, effect_id, 0.7)

    def observe_effect(self, effect: Effect):
        """Observe an effect that needs explanation"""
        self.observed_effects[effect.effect_id] = effect
        self.causal_model.add_effect(effect)

    def generate_hypotheses(self, effect_id: str) -> List[Cause]:
        """Generate possible causes for an observed effect"""
        possible = self.causal_model.get_possible_causes(effect_id)

        if not possible:
            return list(self.causal_model.causes.values())

        causes = []
        for cause_id, prob in possible:
            if cause_id in self.causal_model.causes:
                causes.append(self.causal_model.causes[cause_id])

        return causes

    def rank_hypotheses(self,
                        effect: Effect,
                        hypotheses: List[Cause]
                        ) -> List[Explanation]:
        """Rank hypotheses by explanatory power"""
        explanations = []

        for cause in hypotheses:
            explanation = self._evaluate_explanation(effect, cause)
            explanations.append(explanation)

        explanations.sort(key=lambda e: e.overall_score, reverse=True)
        return explanations

    def _evaluate_explanation(self,
                              effect: Effect,
                              cause: Cause
                              ) -> Explanation:
        """Evaluate how well a cause explains an effect"""
        if effect.effect_id in self.causal_model.effect_cause_links:
            likelihood = self.causal_model.effect_cause_links[effect.effect_id].get(
                cause.cause_id, 0.1
            )
        else:
            likelihood = 0.1 if effect.effect_id in cause.typical_effects else 0.01

        prior = cause.prior_probability

        total_prior = sum(c.prior_probability for c in self.causal_model.causes.values())
        total_likelihood = 0.0
        for other_cause in self.causal_model.causes.values():
            if effect.effect_id in self.causal_model.effect_cause_links:
                other_likelihood = self.causal_model.effect_cause_links[effect.effect_id].get(
                    other_cause.cause_id, 0.1
                )
            else:
                other_likelihood = 0.1
            total_likelihood += other_likelihood * other_cause.prior_probability

        if total_likelihood > 0:
            posterior = (likelihood * prior) / total_likelihood
        else:
            posterior = prior

        explanatory_power = likelihood

        simplicity = 1.0 / cause.complexity

        supporting = []
        contradicting = []

        for eff_id, eff in self.observed_effects.items():
            if eff_id == effect.effect_id:
                continue
            if eff_id in cause.typical_effects:
                supporting.append(eff_id)
            elif eff_id in self.causal_model.effect_cause_links:
                if cause.cause_id not in self.causal_model.effect_cause_links[eff_id]:
                    pass

        return Explanation(
            explanation_id=f"expl_{effect.effect_id}_{cause.cause_id}",
            effect=effect,
            cause=cause,
            probability=posterior,
            explanatory_power=explanatory_power,
            simplicity_score=simplicity,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting
        )

    def select_best(self,
                    effect_id: str
                    ) -> Optional[Explanation]:
        """Select the best explanation for an effect"""
        if effect_id not in self.observed_effects:
            return None

        effect = self.observed_effects[effect_id]
        hypotheses = self.generate_hypotheses(effect_id)

        if not hypotheses:
            return None

        ranked = self.rank_hypotheses(effect, hypotheses)

        if ranked:
            best = ranked[0]
            self.explanations[best.explanation_id] = best
            return best

        return None

    def explain(self,
                effect_description: str,
                effect_features: Dict[str, Any] = None
                ) -> Explanation:
        """High-level interface to explain an observation"""
        effect = Effect(
            effect_id=f"eff_{len(self.observed_effects)}",
            description=effect_description,
            features=effect_features or {}
        )
        self.observe_effect(effect)

        return self.select_best(effect.effect_id)

    def add_causal_knowledge(self,
                             cause_description: str,
                             effects: List[str],
                             prior: float = 0.5,
                             complexity: float = 1.0,
                             cause_id: Optional[str] = None):
        """Add new causal knowledge to the model"""
        if cause_id is None:
            cause_id = f"cause_{len(self.causal_model.causes)}"
        cause = Cause(
            cause_id=cause_id,
            description=cause_description,
            typical_effects=effects,
            prior_probability=prior,
            complexity=complexity
        )
        self.causal_model.add_cause(cause)

        for effect_id in effects:
            self.causal_model.link_cause_effect(cause_id, effect_id, 0.8)

    def update_from_feedback(self,
                             explanation_id: str,
                             was_correct: bool):
        """Update model based on whether explanation was correct"""
        if explanation_id not in self.explanations:
            return

        explanation = self.explanations[explanation_id]
        cause = explanation.cause

        if was_correct:
            cause.prior_probability = min(0.95, cause.prior_probability * 1.1)

            if explanation.effect.effect_id not in cause.typical_effects:
                cause.typical_effects.append(explanation.effect.effect_id)
        else:
            cause.prior_probability = max(0.05, cause.prior_probability * 0.9)

    def differential_diagnosis(self,
                               effects: List[Effect]
                               ) -> List[Tuple[Cause, float]]:
        """Find causes that best explain multiple effects together"""
        if not effects:
            return []

        cause_scores: Dict[str, float] = {}

        for cause in self.causal_model.causes.values():
            explained_count = 0
            total_likelihood = 0.0

            for effect in effects:
                if effect.effect_id in cause.typical_effects:
                    explained_count += 1
                    if effect.effect_id in self.causal_model.effect_cause_links:
                        likelihood = self.causal_model.effect_cause_links[effect.effect_id].get(
                            cause.cause_id, 0.1
                        )
                        total_likelihood += likelihood

            coverage = explained_count / len(effects)
            avg_likelihood = total_likelihood / len(effects) if effects else 0

            score = (coverage * 0.5 +
                    avg_likelihood * 0.3 +
                    cause.prior_probability * 0.1 +
                    (1.0 / cause.complexity) * 0.1)

            cause_scores[cause.cause_id] = score

        ranked = sorted(cause_scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.causal_model.causes[cid], score) for cid, score in ranked
                if cid in self.causal_model.causes]
