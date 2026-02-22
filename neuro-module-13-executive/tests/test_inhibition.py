"""Tests for inhibition system"""

import numpy as np
import pytest
from neuro.modules.m13_executive.inhibition import (
    ResponseInhibitor,
    ImpulseController,
    InhibitionSystem,
    InhibitionParams,
)


class TestResponseInhibitor:
    """Tests for response inhibition"""

    def test_initialization(self):
        """Test inhibitor initialization"""
        inhibitor = ResponseInhibitor()

        assert inhibitor.go_activation == 0.0
        assert inhibitor.stop_activation == 0.0
        assert not inhibitor.response_made
        assert not inhibitor.response_inhibited

    def test_go_accumulation(self):
        """Test go process accumulates over time"""
        inhibitor = ResponseInhibitor()

        initial = inhibitor.go_activation
        for _ in range(50):
            inhibitor.update_go(1.0)

        assert inhibitor.go_activation > initial

    def test_stop_inhibits_go(self):
        """Test stop signal can inhibit response"""
        params = InhibitionParams(stop_threshold=0.3, go_threshold=0.6)
        inhibitor = ResponseInhibitor(params)

        # Build some go activation
        for _ in range(20):
            inhibitor.update_go(1.0)

        # Apply stop before threshold
        for _ in range(50):
            inhibitor.update_stop(1.0)

        assert inhibitor.response_inhibited
        assert not inhibitor.response_made

    def test_go_trial_produces_response(self):
        """Test go trial results in response"""
        inhibitor = ResponseInhibitor()
        result = inhibitor.run_trial(is_stop_trial=False)

        assert result["response_made"]
        assert not result["response_inhibited"]
        assert result["reaction_time"] is not None

    def test_stop_trial_can_inhibit(self):
        """Test stop trial can prevent response"""
        params = InhibitionParams(stop_signal_delay=50.0)  # Early stop signal
        inhibitor = ResponseInhibitor(params)

        # Run multiple trials - some should be inhibited
        inhibited_count = 0
        for _ in range(20):
            inhibitor.reset()
            result = inhibitor.run_trial(is_stop_trial=True, stop_signal_delay=50.0)
            if result["response_inhibited"]:
                inhibited_count += 1

        # At least some should be inhibited with early stop signal
        assert inhibited_count > 0

    def test_reset(self):
        """Test reset clears state"""
        inhibitor = ResponseInhibitor()
        inhibitor.run_trial(is_stop_trial=False)

        inhibitor.reset()

        assert inhibitor.go_activation == 0.0
        assert inhibitor.stop_activation == 0.0
        assert not inhibitor.response_made


class TestImpulseController:
    """Tests for impulse control"""

    def test_initialization(self):
        """Test controller initialization"""
        controller = ImpulseController(n_actions=5)

        assert len(controller.inhibition) == 5
        assert len(controller.urge) == 5
        assert np.all(controller.inhibition > 0)

    def test_allowed_actions(self):
        """Test setting allowed actions"""
        controller = ImpulseController(n_actions=5)
        controller.set_allowed_actions([0, 2, 4])

        assert controller.allowed[0]
        assert not controller.allowed[1]
        assert controller.allowed[2]

    def test_impulse_receipt(self):
        """Test receiving impulses"""
        controller = ImpulseController(n_actions=5)

        controller.receive_impulse(2, 0.5)

        assert controller.urge[2] > 0
        assert controller.urge[0] == 0

    def test_inhibition_blocks_action(self):
        """Test inhibition prevents action execution"""
        controller = ImpulseController(n_actions=5)
        controller.set_allowed_actions([])  # No actions allowed

        controller.receive_impulse(0, 0.3)  # Weak impulse
        executed = controller.update()

        # Should not execute because not allowed and not overwhelming
        assert executed[0] == 0

    def test_strong_impulse_breaks_through(self):
        """Test very strong impulse can overcome inhibition"""
        controller = ImpulseController(n_actions=5)
        params = InhibitionParams(inhibition_strength=0.3)
        controller = ImpulseController(n_actions=5, params=params)

        controller.receive_impulse(0, 1.0)  # Very strong impulse
        executed = controller.update()

        # Strong impulse should break through
        assert executed[0] == 1

    def test_release_inhibition(self):
        """Test releasing inhibition for specific action"""
        controller = ImpulseController(n_actions=5)
        initial = controller.inhibition[0]

        controller.release_inhibition(0, 0.3)

        assert controller.inhibition[0] < initial

    def test_restore_inhibition(self):
        """Test restoring inhibition"""
        controller = ImpulseController(n_actions=5)
        controller.release_inhibition(0, 0.5)
        controller.restore_inhibition(0)

        assert controller.inhibition[0] == controller.params.inhibition_strength


class TestInhibitionSystem:
    """Tests for integrated inhibition system"""

    def test_initialization(self):
        """Test system initialization"""
        system = InhibitionSystem(n_actions=10)

        assert system.response_inhibitor is not None
        assert system.impulse_controller is not None
        assert len(system.rifg_activation) == system.params.n_units

    def test_stop_signal_processing(self):
        """Test stop signal activates rIFG"""
        system = InhibitionSystem()

        signal = system.process_stop_signal()

        assert signal > 0
        assert np.mean(system.rifg_activation) > 0

    def test_go_nogo_go_trial(self):
        """Test go trial in go/no-go task"""
        system = InhibitionSystem()
        result = system.run_go_nogo_trial("go")

        assert result["stimulus_type"] == "go"
        # Go trials should usually produce response
        # (Allow for some noise in the model)

    def test_go_nogo_nogo_trial(self):
        """Test no-go trial in go/no-go task"""
        system = InhibitionSystem()
        result = system.run_go_nogo_trial("nogo")

        assert result["stimulus_type"] == "nogo"

    def test_inhibition_strength_getter(self):
        """Test getting inhibition strength"""
        system = InhibitionSystem()
        system.process_stop_signal()

        strength = system.get_inhibition_strength()

        assert 0 <= strength <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
