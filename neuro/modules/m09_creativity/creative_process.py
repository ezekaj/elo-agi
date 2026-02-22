"""
Creative Process - Orchestrating Creativity

Integrates all components for complete creative cognition:
- DMN for idea generation
- ECN for evaluation and refinement
- Salience Network for dynamic switching
- Imagery System for mental simulation

Key insight: "Generating creative ideas led to significantly higher
network reconfiguration than generating non-creative ideas"
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time

from .networks import DefaultModeNetwork, ExecutiveControlNetwork, SalienceNetwork
from .networks.salience_network import NetworkState, SwitchTrigger
from .networks.executive_control_network import EvaluationCriterion, Evaluation
from .imagery import ImagerySystem, MultimodalImage


@dataclass
class Idea:
    """A creative idea"""

    id: str
    content: str
    source_concepts: List[str]
    novelty: float
    coherence: float
    imagery: Optional[MultimodalImage] = None
    evaluation: Optional[Evaluation] = None
    refinements: List[str] = field(default_factory=list)
    generation_time: float = field(default_factory=time.time)


@dataclass
class CreativeOutput:
    """Final output of creative process"""

    best_ideas: List[Idea]
    total_generated: int
    total_evaluated: int
    process_duration: float
    network_reconfigurations: int
    creativity_score: float


class CreativeProcess:
    """
    Complete creative cognition system.

    Orchestrates:
    - Idea generation (DMN)
    - Idea evaluation (ECN)
    - Network switching (Salience)
    - Mental imagery (all modalities)

    The creative process is a DYNAMIC DANCE between these systems,
    not a simple pipeline.
    """

    def __init__(self):
        self.dmn = DefaultModeNetwork()
        self.ecn = ExecutiveControlNetwork()
        self.salience = SalienceNetwork()
        self.imagery = ImagerySystem()

        self.ideas: Dict[str, Idea] = {}
        self._process_history: List[Dict[str, Any]] = []

    def setup_knowledge(self, concepts: List[Tuple[str, Any, Dict[str, float]]]):
        """
        Set up knowledge base for creative generation.

        Concepts are the raw material for creative combination.
        """
        for concept_id, content, features in concepts:
            self.dmn.add_concept(concept_id, content, features)

    def create_associations(self, associations: List[Tuple[str, str, float, str]]):
        """
        Create associations between concepts.

        Associations are the paths for creative exploration.
        """
        for source, target, strength, assoc_type in associations:
            self.dmn.create_association(source, target, strength, assoc_type)

    def set_creative_goal(
        self,
        goal_id: str,
        description: str,
        criteria_weights: Optional[Dict[EvaluationCriterion, float]] = None,
        constraints: Optional[List[str]] = None,
    ):
        """Set the goal for creative generation"""
        self.ecn.set_goal(goal_id, description, criteria_weights, constraints)

    def generate_ideas(
        self,
        num_ideas: int = 5,
        seed_concepts: Optional[List[str]] = None,
        use_imagery: bool = True,
    ) -> List[Idea]:
        """
        Generate creative ideas.

        Uses DMN for spontaneous thought generation,
        optionally enhanced with mental imagery.
        """
        generated = []

        # Ensure we're in DMN mode
        if self.salience.current_state != NetworkState.DMN:
            self.salience.force_switch_to(NetworkState.DMN)

        for i in range(num_ideas):
            # Generate spontaneous thought
            if seed_concepts:
                seed = np.random.choice(seed_concepts)
            else:
                seed = None

            thought = self.dmn.generate_spontaneous_thought(seed=seed)

            if not thought.concepts:
                continue

            # Create idea from thought
            idea_id = f"idea_{len(self.ideas)}"
            content = self._thought_to_content(thought)

            # Optionally create imagery
            imagery = None
            if use_imagery and len(thought.concepts) >= 2:
                imagery = self.imagery.create_multimodal_image(
                    f"imagery_{idea_id}",
                    content,
                    include_visual=True,
                    include_tactile=True,  # Tactile boosts creativity
                )

            idea = Idea(
                id=idea_id,
                content=content,
                source_concepts=thought.concepts,
                novelty=thought.novelty_score,
                coherence=thought.coherence_score,
                imagery=imagery,
            )

            self.ideas[idea_id] = idea
            generated.append(idea)

            # Record for salience network
            self.salience.record_idea_generated(thought.novelty_score)

        return generated

    def _thought_to_content(self, thought) -> str:
        """Convert spontaneous thought to idea content"""
        concepts = thought.concepts
        if len(concepts) == 1:
            return f"Idea based on: {concepts[0]}"
        elif len(concepts) == 2:
            return f"Combining {concepts[0]} with {concepts[1]}"
        else:
            return f"Synthesis of {', '.join(concepts[:3])} and more"

    def evaluate_ideas(self, idea_ids: Optional[List[str]] = None) -> List[Evaluation]:
        """
        Evaluate generated ideas.

        Uses ECN for critical assessment.
        """
        # Switch to ECN mode
        if self.salience.current_state != NetworkState.ECN:
            self.salience.execute_switch(SwitchTrigger.IDEA_GENERATED)

        if idea_ids is None:
            idea_ids = [i for i in self.ideas.keys() if self.ideas[i].evaluation is None]

        evaluations = []

        for idea_id in idea_ids:
            if idea_id not in self.ideas:
                continue

            idea = self.ideas[idea_id]

            # Build features for evaluation
            features = {
                "novelty": idea.novelty,
                "coherence": idea.coherence,
                "complexity": len(idea.source_concepts) / 10.0,
            }

            # Tactile imagery boosts creativity scores (2025 finding)
            if idea.imagery and idea.imagery.tactile:
                features["novelty"] *= 1.1  # Boost from tactile engagement
                features["usefulness"] = 0.6  # Tactile grounds ideas in reality

            context = {
                "goal_relevance": 0.6,  # Default moderate relevance
            }

            evaluation = self.ecn.evaluate_idea(idea_id, idea.content, features, context)

            idea.evaluation = evaluation
            evaluations.append(evaluation)

            self.salience.record_evaluation_complete(evaluation.overall_score)

        return evaluations

    def refine_ideas(
        self, idea_ids: Optional[List[str]] = None, target_score: float = 0.6
    ) -> List[Idea]:
        """
        Refine ideas that need improvement.

        ECN identifies weaknesses and suggests improvements.
        """
        if idea_ids is None:
            # Find ideas that need refinement
            idea_ids = [
                i
                for i, idea in self.ideas.items()
                if idea.evaluation
                and idea.evaluation.recommendation == "refine"
                and idea.evaluation.overall_score < target_score
            ]

        refined = []

        for idea_id in idea_ids:
            if idea_id not in self.ideas:
                continue

            idea = self.ideas[idea_id]
            if not idea.evaluation:
                continue

            # Get refinement from ECN
            refined_id, refined_content, refinement = self.ecn.refine_idea(
                idea_id, idea.content, idea.evaluation
            )

            # Create refined idea
            new_idea = Idea(
                id=refined_id,
                content=f"{idea.content} [refined: {', '.join(refinement.changes)}]",
                source_concepts=idea.source_concepts,
                novelty=idea.novelty * 0.9,  # Slight novelty loss from refinement
                coherence=min(1.0, idea.coherence + refinement.improvement_score),
                imagery=idea.imagery,
                refinements=[idea_id],
            )

            self.ideas[refined_id] = new_idea
            idea.refinements.append(refined_id)
            refined.append(new_idea)

        return refined

    def creative_session(
        self, goal: str, duration_seconds: float = 30.0, target_good_ideas: int = 3
    ) -> CreativeOutput:
        """
        Run a complete creative session.

        Dynamically switches between generation and evaluation
        until goals are met or time runs out.
        """
        start_time = time.time()
        self.set_creative_goal("session_goal", goal)

        total_generated = 0
        total_evaluated = 0

        while time.time() - start_time < duration_seconds:
            # Check current network state and act accordingly
            if self.salience.current_state == NetworkState.DMN:
                # Generate ideas
                new_ideas = self.generate_ideas(num_ideas=3)
                total_generated += len(new_ideas)

                # Check if should switch
                metrics = {
                    "ideas_generated": total_generated,
                    "target_ideas": 5,
                    "current_novelty": np.mean([i.novelty for i in new_ideas])
                    if new_ideas
                    else 0.5,
                }

                should_switch, trigger = self.salience.should_switch(metrics)
                if should_switch and trigger:
                    self.salience.execute_switch(trigger)

            elif self.salience.current_state == NetworkState.ECN:
                # Evaluate ideas
                evaluations = self.evaluate_ideas()
                total_evaluated += len(evaluations)

                # Check if we have enough good ideas
                good_ideas = [e for e in evaluations if e.overall_score >= 0.6]

                if len(good_ideas) >= target_good_ideas:
                    break

                # Maybe refine some ideas
                self.refine_ideas()

                # Check if should switch back
                metrics = {
                    "evaluation_complete": True,
                    "best_score": max((e.overall_score for e in evaluations), default=0),
                    "target_score": 0.7,
                    "ideas_evaluated": total_evaluated,
                }

                should_switch, trigger = self.salience.should_switch(metrics)
                if should_switch and trigger:
                    self.salience.execute_switch(trigger)

        # Compile results
        elapsed = time.time() - start_time

        # Get best ideas
        evaluated_ideas = [idea for idea in self.ideas.values() if idea.evaluation is not None]
        sorted_ideas = sorted(
            evaluated_ideas, key=lambda i: i.evaluation.overall_score, reverse=True
        )

        best_ideas = sorted_ideas[:target_good_ideas]

        # Compute creativity score
        if best_ideas:
            avg_novelty = np.mean([i.novelty for i in best_ideas])
            avg_quality = np.mean([i.evaluation.overall_score for i in best_ideas])
            creativity_score = (avg_novelty + avg_quality) / 2
        else:
            creativity_score = 0.0

        return CreativeOutput(
            best_ideas=best_ideas,
            total_generated=total_generated,
            total_evaluated=total_evaluated,
            process_duration=elapsed,
            network_reconfigurations=len(self.salience.switch_history),
            creativity_score=creativity_score,
        )

    def mind_wander_for_ideas(self, duration_steps: int = 10) -> List[Idea]:
        """
        Let the mind wander freely for creative inspiration.

        Unconstrained exploration can lead to unexpected insights.
        """
        self.salience.force_switch_to(NetworkState.DMN)

        thoughts = self.dmn.mind_wander(duration_steps)
        ideas = []

        for thought in thoughts:
            if thought.novelty_score > 0.5:  # Only keep novel thoughts
                idea_id = f"wander_idea_{len(self.ideas)}"
                idea = Idea(
                    id=idea_id,
                    content=self._thought_to_content(thought),
                    source_concepts=thought.concepts,
                    novelty=thought.novelty_score,
                    coherence=thought.coherence_score,
                )
                self.ideas[idea_id] = idea
                ideas.append(idea)

        return ideas

    def find_distant_connections(
        self, concept: str, max_results: int = 5
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Find distant but potentially creative connections.

        Creative insight often comes from unexpected connections.
        """
        return self.dmn.find_distant_associations(concept, min_distance=3, max_results=max_results)

    def imagine_idea(
        self, idea_id: str, modalities: Optional[List[str]] = None
    ) -> Optional[MultimodalImage]:
        """
        Create rich mental imagery for an idea.

        Imagery strengthens creative ideas and helps evaluation.
        """
        if idea_id not in self.ideas:
            return None

        idea = self.ideas[idea_id]

        if modalities is None:
            modalities = ["visual", "tactile"]  # Default for creativity

        imagery = self.imagery.simulate_experience(idea.content, modalities)
        idea.imagery = imagery

        return imagery

    def get_creative_statistics(self) -> Dict[str, Any]:
        """Get statistics about creative process"""
        network_stats = self.salience.get_switch_statistics()

        idea_scores = [
            idea.evaluation.overall_score for idea in self.ideas.values() if idea.evaluation
        ]

        return {
            "total_ideas": len(self.ideas),
            "evaluated_ideas": len([i for i in self.ideas.values() if i.evaluation]),
            "avg_novelty": np.mean([i.novelty for i in self.ideas.values()]) if self.ideas else 0,
            "avg_quality": np.mean(idea_scores) if idea_scores else 0,
            "network_switches": network_stats.get("total_switches", 0),
            "reconfiguration_level": network_stats.get("current_reconfiguration", 0),
        }
