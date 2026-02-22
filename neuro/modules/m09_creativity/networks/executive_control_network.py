"""
Executive Control Network - Idea Evaluation and Refinement

Active during critical assessment of ideas:
- Evaluating feasibility and quality
- Refining raw ideas into usable forms
- Filtering out non-viable options
- Goal-directed thinking

The ECN works in tension with the DMN - DMN generates,
ECN evaluates and selects.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class EvaluationCriterion(Enum):
    """Criteria for evaluating ideas"""

    NOVELTY = "novelty"
    USEFULNESS = "usefulness"
    FEASIBILITY = "feasibility"
    COHERENCE = "coherence"
    ELEGANCE = "elegance"
    IMPACT = "impact"


@dataclass
class Evaluation:
    """Evaluation result for an idea"""

    idea_id: str
    scores: Dict[EvaluationCriterion, float]
    overall_score: float
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str  # "accept", "refine", "reject"
    confidence: float


@dataclass
class Refinement:
    """A refinement applied to an idea"""

    original_id: str
    refined_id: str
    refinement_type: str  # "elaborate", "simplify", "combine", "constrain"
    changes: List[str]
    improvement_score: float


@dataclass
class Goal:
    """A goal to guide creative search"""

    id: str
    description: str
    criteria_weights: Dict[EvaluationCriterion, float]
    constraints: List[str]
    target_score: float = 0.7


class ExecutiveControlNetwork:
    """
    Executive Control Network - evaluation and refinement.

    Active during:
    - Critical assessment of ideas
    - Goal-directed thinking
    - Quality control
    - Strategic planning

    Key functions:
    - Evaluate ideas against criteria
    - Refine raw ideas
    - Filter non-viable options
    - Maintain goal focus
    """

    def __init__(self, default_weights: Optional[Dict[EvaluationCriterion, float]] = None):
        self.evaluations: Dict[str, Evaluation] = {}
        self.refinements: List[Refinement] = []
        self.current_goal: Optional[Goal] = None

        # Default weights for evaluation criteria
        self.default_weights = default_weights or {
            EvaluationCriterion.NOVELTY: 0.25,
            EvaluationCriterion.USEFULNESS: 0.25,
            EvaluationCriterion.FEASIBILITY: 0.20,
            EvaluationCriterion.COHERENCE: 0.15,
            EvaluationCriterion.ELEGANCE: 0.10,
            EvaluationCriterion.IMPACT: 0.05,
        }

        # Evaluation history for calibration
        self._evaluation_history: List[Evaluation] = []

    def set_goal(
        self,
        goal_id: str,
        description: str,
        criteria_weights: Optional[Dict[EvaluationCriterion, float]] = None,
        constraints: Optional[List[str]] = None,
        target_score: float = 0.7,
    ) -> Goal:
        """Set the current creative goal"""
        self.current_goal = Goal(
            id=goal_id,
            description=description,
            criteria_weights=criteria_weights or self.default_weights,
            constraints=constraints or [],
            target_score=target_score,
        )
        return self.current_goal

    def evaluate_idea(
        self,
        idea_id: str,
        idea_content: Any,
        idea_features: Dict[str, float],
        context: Optional[Dict[str, Any]] = None,
    ) -> Evaluation:
        """
        Evaluate an idea against criteria.

        This is the core ECN function - critical assessment
        of creative output from the DMN.
        """
        scores = {}

        # Evaluate each criterion
        scores[EvaluationCriterion.NOVELTY] = self._evaluate_novelty(
            idea_content, idea_features, context
        )
        scores[EvaluationCriterion.USEFULNESS] = self._evaluate_usefulness(
            idea_content, idea_features, context
        )
        scores[EvaluationCriterion.FEASIBILITY] = self._evaluate_feasibility(
            idea_content, idea_features, context
        )
        scores[EvaluationCriterion.COHERENCE] = self._evaluate_coherence(
            idea_content, idea_features, context
        )
        scores[EvaluationCriterion.ELEGANCE] = self._evaluate_elegance(
            idea_content, idea_features, context
        )
        scores[EvaluationCriterion.IMPACT] = self._evaluate_impact(
            idea_content, idea_features, context
        )

        # Compute weighted overall score
        weights = self.current_goal.criteria_weights if self.current_goal else self.default_weights
        overall = sum(scores[c] * weights.get(c, 0.1) for c in scores)

        # Identify strengths and weaknesses
        strengths = [c.value for c, s in scores.items() if s >= 0.7]
        weaknesses = [c.value for c, s in scores.items() if s < 0.4]

        # Make recommendation
        if overall >= 0.7:
            recommendation = "accept"
        elif overall >= 0.4:
            recommendation = "refine"
        else:
            recommendation = "reject"

        # Check constraints
        if self.current_goal:
            for constraint in self.current_goal.constraints:
                if not self._check_constraint(idea_content, constraint):
                    recommendation = "reject"
                    weaknesses.append(f"violates constraint: {constraint}")

        evaluation = Evaluation(
            idea_id=idea_id,
            scores=scores,
            overall_score=overall,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendation=recommendation,
            confidence=self._compute_confidence(scores),
        )

        self.evaluations[idea_id] = evaluation
        self._evaluation_history.append(evaluation)

        return evaluation

    def _evaluate_novelty(
        self, content: Any, features: Dict[str, float], context: Optional[Dict]
    ) -> float:
        """Evaluate how novel/original the idea is"""
        # Use provided novelty feature if available
        if "novelty" in features:
            return features["novelty"]

        # Otherwise estimate based on rarity of combination
        if "combination_rarity" in features:
            return features["combination_rarity"]

        return 0.5  # Default moderate novelty

    def _evaluate_usefulness(
        self, content: Any, features: Dict[str, float], context: Optional[Dict]
    ) -> float:
        """Evaluate practical usefulness"""
        if "usefulness" in features:
            return features["usefulness"]

        # Check if it addresses the goal
        if self.current_goal and context:
            goal_relevance = context.get("goal_relevance", 0.5)
            return goal_relevance

        return 0.5

    def _evaluate_feasibility(
        self, content: Any, features: Dict[str, float], context: Optional[Dict]
    ) -> float:
        """Evaluate implementation feasibility"""
        if "feasibility" in features:
            return features["feasibility"]

        # Simpler ideas are more feasible
        complexity = features.get("complexity", 0.5)
        return 1.0 - complexity * 0.5

    def _evaluate_coherence(
        self, content: Any, features: Dict[str, float], context: Optional[Dict]
    ) -> float:
        """Evaluate internal consistency"""
        if "coherence" in features:
            return features["coherence"]

        return 0.6  # Default reasonable coherence

    def _evaluate_elegance(
        self, content: Any, features: Dict[str, float], context: Optional[Dict]
    ) -> float:
        """Evaluate aesthetic quality/simplicity"""
        if "elegance" in features:
            return features["elegance"]

        # Elegance often inversely related to complexity
        complexity = features.get("complexity", 0.5)
        return 1.0 - complexity * 0.7

    def _evaluate_impact(
        self, content: Any, features: Dict[str, float], context: Optional[Dict]
    ) -> float:
        """Evaluate potential impact"""
        if "impact" in features:
            return features["impact"]

        # Combine novelty and usefulness for impact
        novelty = features.get("novelty", 0.5)
        usefulness = features.get("usefulness", 0.5)
        return (novelty + usefulness) / 2

    def _check_constraint(self, content: Any, constraint: str) -> bool:
        """Check if idea satisfies a constraint"""
        # Simple constraint checking - in practice would be more sophisticated
        return True  # Assume satisfied unless explicit violation

    def _compute_confidence(self, scores: Dict[EvaluationCriterion, float]) -> float:
        """Compute confidence in evaluation"""
        # Higher confidence when scores are not all similar
        values = list(scores.values())
        variance = np.var(values)

        # Higher variance = more confident discrimination
        return min(1.0, 0.5 + variance * 2)

    def refine_idea(
        self, idea_id: str, idea_content: Any, evaluation: Evaluation, refinement_type: str = "auto"
    ) -> Tuple[str, Any, Refinement]:
        """
        Refine an idea based on its evaluation.

        The ECN doesn't just evaluate - it helps improve ideas
        by identifying specific refinements.
        """
        changes = []
        refined_content = idea_content
        improvement = 0.0

        if refinement_type == "auto":
            # Choose refinement based on weaknesses
            if "feasibility" in evaluation.weaknesses:
                refinement_type = "simplify"
            elif "novelty" in evaluation.weaknesses:
                refinement_type = "elaborate"
            elif "coherence" in evaluation.weaknesses:
                refinement_type = "constrain"
            else:
                refinement_type = "elaborate"

        if refinement_type == "simplify":
            changes.append("Reduced complexity")
            changes.append("Focused on core elements")
            improvement = 0.15

        elif refinement_type == "elaborate":
            changes.append("Added detail and specificity")
            changes.append("Expanded implications")
            improvement = 0.10

        elif refinement_type == "combine":
            changes.append("Integrated complementary elements")
            changes.append("Created synthesis")
            improvement = 0.20

        elif refinement_type == "constrain":
            changes.append("Added structure and boundaries")
            changes.append("Improved coherence")
            improvement = 0.12

        refined_id = f"{idea_id}_refined_{len(self.refinements)}"

        refinement = Refinement(
            original_id=idea_id,
            refined_id=refined_id,
            refinement_type=refinement_type,
            changes=changes,
            improvement_score=improvement,
        )

        self.refinements.append(refinement)

        return refined_id, refined_content, refinement

    def filter_ideas(
        self, idea_evaluations: List[Evaluation], top_k: int = 5, min_score: float = 0.3
    ) -> List[Evaluation]:
        """
        Filter and rank ideas by evaluation scores.

        Applies executive control to select the best ideas
        from a set of candidates.
        """
        # Filter by minimum score
        viable = [e for e in idea_evaluations if e.overall_score >= min_score]

        # Sort by overall score
        ranked = sorted(viable, key=lambda e: e.overall_score, reverse=True)

        return ranked[:top_k]

    def should_continue_generating(
        self, evaluations: List[Evaluation], target_count: int = 3
    ) -> bool:
        """
        Decide whether to continue idea generation or stop.

        The ECN monitors progress toward goals and decides
        when enough good ideas have been generated.
        """
        if not self.current_goal:
            return len(evaluations) < target_count

        # Count ideas meeting target score
        good_ideas = sum(
            1 for e in evaluations if e.overall_score >= self.current_goal.target_score
        )

        return good_ideas < target_count

    def get_improvement_suggestions(self, evaluation: Evaluation) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []

        for weakness in evaluation.weaknesses:
            if weakness == "novelty":
                suggestions.append("Try combining with unexpected elements")
                suggestions.append("Look for analogies from distant domains")
            elif weakness == "usefulness":
                suggestions.append("Focus on specific user needs")
                suggestions.append("Identify concrete applications")
            elif weakness == "feasibility":
                suggestions.append("Break into smaller steps")
                suggestions.append("Identify and address key constraints")
            elif weakness == "coherence":
                suggestions.append("Ensure all parts work together")
                suggestions.append("Remove contradictory elements")
            elif weakness == "elegance":
                suggestions.append("Simplify where possible")
                suggestions.append("Find the essential core")

        return suggestions

    def compare_ideas(self, eval1: Evaluation, eval2: Evaluation) -> Dict[str, Any]:
        """Compare two ideas across all criteria"""
        comparison = {
            "overall_winner": eval1.idea_id
            if eval1.overall_score > eval2.overall_score
            else eval2.idea_id,
            "score_difference": abs(eval1.overall_score - eval2.overall_score),
            "criterion_winners": {},
            "tradeoffs": [],
        }

        for criterion in EvaluationCriterion:
            s1 = eval1.scores.get(criterion, 0)
            s2 = eval2.scores.get(criterion, 0)

            if s1 > s2:
                comparison["criterion_winners"][criterion.value] = eval1.idea_id
            elif s2 > s1:
                comparison["criterion_winners"][criterion.value] = eval2.idea_id
            else:
                comparison["criterion_winners"][criterion.value] = "tie"

            # Identify tradeoffs
            if abs(s1 - s2) > 0.3:
                winner = eval1.idea_id if s1 > s2 else eval2.idea_id
                comparison["tradeoffs"].append(
                    f"{winner} is significantly better on {criterion.value}"
                )

        return comparison
