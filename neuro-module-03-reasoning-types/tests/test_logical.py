"""Tests for logical reasoning components"""

import pytest
from neuro.modules.m03_reasoning_types.logical.inductive import InductiveReasoner, Observation
from neuro.modules.m03_reasoning_types.logical.deductive import (
    DeductiveReasoner,
    Proposition,
    PropositionType,
)
from neuro.modules.m03_reasoning_types.logical.abductive import AbductiveReasoner, Effect


class TestInductiveReasoning:
    def test_observation_and_hypothesis(self):
        """Test forming hypotheses from observations"""
        reasoner = InductiveReasoner(min_observations=2)

        obs1 = Observation("o1", {"color": "white", "type": "swan"})
        obs2 = Observation("o2", {"color": "white", "type": "swan"})
        obs3 = Observation("o3", {"color": "white", "type": "swan"})

        reasoner.observe(obs1)
        reasoner.observe(obs2)
        reasoner.observe(obs3)

        hypotheses = reasoner.hypothesize()

        assert len(hypotheses) > 0
        white_hyp = [h for h in hypotheses if "white" in h.description]
        assert len(white_hyp) > 0

    def test_counterexample_refutation(self):
        """Test that counterexamples reduce confidence"""
        reasoner = InductiveReasoner(min_observations=2)

        for i in range(5):
            reasoner.observe(Observation(f"w{i}", {"color": "white", "type": "swan"}))

        hypotheses = reasoner.hypothesize()
        initial_confidence = hypotheses[0].confidence if hypotheses else 0

        black_swan = Observation("bs", {"color": "black", "type": "swan"})
        reasoner.observe(black_swan)

        if hypotheses:
            assert hypotheses[0].confidence <= initial_confidence

    def test_generalization(self):
        """Test generalizing from examples"""
        reasoner = InductiveReasoner()

        for i in range(5):
            reasoner.observe(Observation(f"o{i}", {"size": "large", "habitat": "ocean"}))

        result = reasoner.generalize(list(reasoner.observations.keys()), "habitat")

        assert result is not None
        assert result[0] == "ocean"
        assert result[1] == 1.0


class TestDeductiveReasoning:
    def test_modus_ponens(self):
        """Test modus ponens: If P then Q, P ⊢ Q"""
        reasoner = DeductiveReasoner()

        p = Proposition("p", PropositionType.ATOMIC, "it is raining", "weather", "raining")
        q = Proposition("q", PropositionType.ATOMIC, "ground is wet", "ground", "wet")

        conditional = Proposition(
            "p_implies_q", PropositionType.CONDITIONAL, "if raining then wet", components=[p, q]
        )

        result = reasoner.modus_ponens(conditional, p)

        assert result is not None
        assert result.predicate == "wet"

    def test_syllogism(self):
        """Test categorical syllogism: All A are B, All B are C ⊢ All A are C"""
        reasoner = DeductiveReasoner()

        major = Proposition(
            "major",
            PropositionType.UNIVERSAL,
            "All men are mortal",
            subject="men",
            predicate="mortal",
        )

        minor = Proposition(
            "minor",
            PropositionType.ATOMIC,
            "Socrates is a man",
            subject="Socrates",
            predicate="men",
        )

        result = reasoner.syllogism(major, minor)

        assert result.is_valid
        assert result.conclusion is not None
        assert result.conclusion.subject == "Socrates"
        assert result.conclusion.predicate == "mortal"

    def test_chained_reasoning(self):
        """Test chained syllogisms"""
        reasoner = DeductiveReasoner()

        p1 = Proposition("p1", PropositionType.UNIVERSAL, "", subject="Greeks", predicate="humans")
        p2 = Proposition("p2", PropositionType.UNIVERSAL, "", subject="humans", predicate="mortal")

        reasoner.add_premise(p1)
        reasoner.add_premise(p2)

        inferences = reasoner.derive([p1, p2])

        greek_mortal = [
            i
            for i in inferences
            if i.conclusion.subject == "Greeks" and i.conclusion.predicate == "mortal"
        ]
        assert len(greek_mortal) > 0

    def test_validation(self):
        """Test conclusion validation"""
        reasoner = DeductiveReasoner()

        premises = [
            Proposition("p1", PropositionType.UNIVERSAL, "", subject="birds", predicate="animals"),
            Proposition("p2", PropositionType.UNIVERSAL, "", subject="animals", predicate="living"),
        ]

        valid_conclusion = Proposition(
            "c1", PropositionType.UNIVERSAL, "", subject="birds", predicate="living"
        )

        is_valid, explanation = reasoner.validate(valid_conclusion, premises)
        assert is_valid


class TestAbductiveReasoning:
    def test_hypothesis_generation(self):
        """Test generating explanatory hypotheses"""
        reasoner = AbductiveReasoner()

        wet_grass = Effect("wet_grass", "The grass is wet")
        reasoner.observe_effect(wet_grass)

        hypotheses = reasoner.generate_hypotheses("wet_grass")

        assert len(hypotheses) > 0

    def test_best_explanation(self):
        """Test selecting the best explanation"""
        reasoner = AbductiveReasoner()

        explanation = reasoner.explain("wet_grass")

        assert explanation is not None
        assert explanation.probability > 0
        assert explanation.cause is not None

    def test_multiple_effects(self):
        """Test diagnosis with multiple effects"""
        reasoner = AbductiveReasoner()

        effects = [
            Effect("wet_grass", "wet grass"),
            Effect("wet_pavement", "wet pavement"),
            Effect("puddles", "puddles on ground"),
        ]

        for e in effects:
            reasoner.observe_effect(e)

        diagnosis = reasoner.differential_diagnosis(effects)

        assert len(diagnosis) > 0
        best_cause, score = diagnosis[0]
        assert best_cause.cause_id == "rain"

    def test_feedback_learning(self):
        """Test updating model from feedback"""
        reasoner = AbductiveReasoner()

        explanation = reasoner.explain("wet_grass")
        initial_prior = explanation.cause.prior_probability

        reasoner.update_from_feedback(explanation.explanation_id, was_correct=True)

        assert explanation.cause.prior_probability >= initial_prior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
