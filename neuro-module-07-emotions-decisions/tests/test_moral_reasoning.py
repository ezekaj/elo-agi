"""Tests for moral reasoning."""

import pytest
from neuro.modules.m07_emotions_decisions.moral_reasoning import (
    DeontologicalSystem, UtilitarianSystem, MoralDilemmaProcessor,
    VMPFCLesionModel, MoralFramework, HarmType,
    create_trolley_switch, create_trolley_push, create_crying_baby
)

class TestDeontologicalSystem:
    """Test deontological (rule-based) moral reasoning."""

    def test_personal_harm_wrongness(self):
        deont = DeontologicalSystem()
        push = create_trolley_push()

        permissibility, _ = deont.evaluate(push)
        assert permissibility < 0  # Personal harm is wrong

    def test_impersonal_less_wrong(self):
        deont = DeontologicalSystem()
        switch = create_trolley_switch()
        push = create_trolley_push()

        switch_perm, _ = deont.evaluate(switch)
        push_perm, _ = deont.evaluate(push)

        # Personal (push) should be more wrong than impersonal (switch)
        assert push_perm < switch_perm

    def test_emotional_response(self):
        deont = DeontologicalSystem()
        push = create_trolley_push()

        emotional = deont.emotional_response(push)
        assert emotional < 0  # Negative emotional response to harm

class TestUtilitarianSystem:
    """Test utilitarian (outcome-based) moral reasoning."""

    def test_net_positive_utility(self):
        util = UtilitarianSystem()
        switch = create_trolley_switch()  # Save 5, lose 1

        permissibility, _ = util.evaluate(switch)
        assert permissibility > 0  # Net positive utility

    def test_same_utility_different_scenarios(self):
        util = UtilitarianSystem()
        switch = create_trolley_switch()  # 5 vs 1
        push = create_trolley_push()      # 5 vs 1

        switch_util, _ = util.evaluate(switch)
        push_util, _ = util.evaluate(push)

        # Utilitarian doesn't distinguish personal/impersonal
        assert abs(switch_util - push_util) < 0.1

class TestMoralDilemmaProcessor:
    """Test integrated moral dilemma processing."""

    def test_switch_more_action(self):
        processor = MoralDilemmaProcessor(vmpfc_intact=True)
        switch = create_trolley_switch()

        decision = processor.process_dilemma(switch)

        # Switch scenario typically gets utilitarian response
        assert decision.action_taken or decision.utilitarian_weight > 0.5

    def test_push_less_action(self):
        processor = MoralDilemmaProcessor(vmpfc_intact=True)
        push = create_trolley_push()

        decision = processor.process_dilemma(push)

        # Push scenario typically gets deontological (no action) response
        assert decision.deontological_weight > 0.5

    def test_framework_detection(self):
        processor = MoralDilemmaProcessor()

        switch = create_trolley_switch()
        push = create_trolley_push()

        switch_decision = processor.process_dilemma(switch)
        push_decision = processor.process_dilemma(push)

        # Impersonal should lean utilitarian
        assert switch_decision.utilitarian_weight > switch_decision.deontological_weight

        # Personal should lean deontological
        assert push_decision.deontological_weight > push_decision.utilitarian_weight

class TestVMPFCLesionModel:
    """Test VMPFC lesion effects on moral reasoning."""

    def test_more_utilitarian_with_lesion(self):
        healthy = MoralDilemmaProcessor(vmpfc_intact=True)
        lesion = VMPFCLesionModel()

        push = create_trolley_push()

        healthy_decision = healthy.process_dilemma(push)
        lesion_decision = lesion.process_moral_dilemma(push)

        # Lesioned should be more willing to act (more utilitarian)
        assert lesion_decision.utilitarian_weight >= healthy_decision.utilitarian_weight

    def test_emotional_blunting(self):
        lesion = VMPFCLesionModel()
        push = create_trolley_push()

        decision = lesion.process_moral_dilemma(push)

        # Emotional response should be blunted
        assert abs(decision.emotional_response) < 0.5

    def test_compare_healthy_vs_lesioned(self):
        lesion = VMPFCLesionModel()
        push = create_trolley_push()

        comparison = lesion.compare_with_healthy(push)

        assert 'healthy' in comparison
        assert 'lesioned' in comparison

        # Lesioned should have reduced emotional response
        assert abs(comparison['lesioned'].emotional_response) <= abs(comparison['healthy'].emotional_response)

class TestTrolleyProblemPredictions:
    """Test that model matches known trolley problem findings."""

    def test_switch_more_endorsed(self):
        processor = MoralDilemmaProcessor()
        switch = create_trolley_switch()
        push = create_trolley_push()

        switch_dec = processor.process_dilemma(switch)
        push_dec = processor.process_dilemma(push)

        # Research shows: people more willing to flip switch than push
        # Our model should reflect this
        switch_willingness = switch_dec.confidence if switch_dec.action_taken else -switch_dec.confidence
        push_willingness = push_dec.confidence if push_dec.action_taken else -push_dec.confidence

        assert switch_willingness >= push_willingness

    def test_crying_baby_extreme(self):
        processor = MoralDilemmaProcessor()
        baby = create_crying_baby()

        decision = processor.process_dilemma(baby)

        # Extremely personal - should have very high deontological weight
        assert decision.deontological_weight > 0.6
        assert decision.emotional_response < -0.3  # Strong negative emotion
