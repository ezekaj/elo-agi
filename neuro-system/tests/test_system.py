"""
Tests for the Neuro System: Unified Cognitive Architecture.

Covers:
- Unit tests for individual components
- Integration tests for full system
- Stress tests for stability
"""

import pytest
import numpy as np
import time
from pathlib import Path

# Add src to path
from neuro.modules.system.config import SystemConfig, ModuleConfig, DEFAULT_CONFIG
from neuro.modules.system.module_loader import ModuleLoader, ModuleStatus, ModuleStub
from neuro.modules.system.sensory_interface import SensoryInterface, SensoryInput, InputType
from neuro.modules.system.motor_interface import MotorInterface, MotorOutput, OutputType, ActionCategory
from neuro.modules.system.active_inference import ActiveInferenceController, Policy, EFEResult
from neuro.modules.system.cognitive_core import CognitiveCore, CognitiveState, CycleResult

# =============================================================================
# Unit Tests: Config
# =============================================================================

class TestConfig:
    """Tests for SystemConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = SystemConfig()
        assert config.input_dim == 64
        assert config.output_dim == 32
        assert len(config.module_configs) == 20

    def test_module_configs(self):
        """Test module configuration initialization."""
        config = SystemConfig()
        assert "00" in config.module_configs
        assert "17" in config.module_configs
        assert config.module_configs["00"].priority == 10.0

    def test_get_enabled_modules(self):
        """Test getting enabled modules sorted by priority."""
        config = SystemConfig()
        enabled = config.get_enabled_modules()
        assert len(enabled) == 20
        assert enabled[0] == "00"  # Highest priority

    def test_enable_disable_module(self):
        """Test enabling/disabling modules."""
        config = SystemConfig()
        config.disable_module("05")
        assert not config.module_configs["05"].enabled

        config.enable_module("05")
        assert config.module_configs["05"].enabled

# =============================================================================
# Unit Tests: Module Loader
# =============================================================================

class TestModuleLoader:
    """Tests for ModuleLoader."""

    def test_loader_creation(self):
        """Test loader initialization."""
        loader = ModuleLoader()
        assert loader.config is not None

    def test_load_module_missing(self):
        """Test loading a non-existent module."""
        config = SystemConfig()
        config.modules_base_path = Path("/nonexistent/path")
        loader = ModuleLoader(config)

        result = loader.load_module("00")
        assert result.status == ModuleStatus.FAILED

    def test_get_statistics(self):
        """Test getting loader statistics."""
        loader = ModuleLoader()
        stats = loader.get_statistics()
        assert 'total_modules' in stats
        assert 'loaded' in stats

    def test_module_stub(self):
        """Test module stub functionality."""
        stub = ModuleStub("test")
        assert stub.propose() == []
        stub.receive_broadcast()  # Should not raise
        stub.process()  # Should not raise

# =============================================================================
# Unit Tests: Sensory Interface
# =============================================================================

class TestSensoryInterface:
    """Tests for SensoryInterface."""

    def test_interface_creation(self):
        """Test interface initialization."""
        interface = SensoryInterface()
        assert interface.config is not None

    def test_process_vector(self):
        """Test processing vector input."""
        interface = SensoryInterface()
        data = np.random.randn(32)

        result = interface.process(data, InputType.VECTOR)

        assert result.input_type == InputType.VECTOR
        assert result.processed is not None
        assert len(result.processed) == interface.config.input_dim

    def test_process_text(self):
        """Test processing text input."""
        interface = SensoryInterface()
        data = "Hello, world!"

        result = interface.process(data, InputType.TEXT)

        assert result.input_type == InputType.TEXT
        assert result.processed is not None
        assert len(result.processed) == interface.config.input_dim

    def test_auto_detect_type(self):
        """Test automatic type detection."""
        interface = SensoryInterface()

        # Vector
        result1 = interface.process(np.array([1, 2, 3]))
        assert result1.input_type == InputType.VECTOR

        # Text
        result2 = interface.process("test string")
        assert result2.input_type == InputType.TEXT

    def test_buffer(self):
        """Test sensory buffer."""
        interface = SensoryInterface()

        for i in range(5):
            interface.process(np.array([i]))

        recent = interface.buffer.get_recent(3)
        assert len(recent) == 3

    def test_combined_input(self):
        """Test getting combined input."""
        interface = SensoryInterface()

        for i in range(3):
            interface.process(np.ones(32) * i)

        combined = interface.get_combined_input(3)
        assert len(combined) == interface.config.input_dim

# =============================================================================
# Unit Tests: Motor Interface
# =============================================================================

class TestMotorInterface:
    """Tests for MotorInterface."""

    def test_interface_creation(self):
        """Test interface initialization."""
        interface = MotorInterface()
        assert interface.config is not None

    def test_generate_vector(self):
        """Test generating vector output."""
        interface = MotorInterface()
        state = np.random.randn(64)

        output = interface.generate(state, OutputType.VECTOR)

        assert output.output_type == OutputType.VECTOR
        assert isinstance(output.value, np.ndarray)

    def test_generate_discrete(self):
        """Test generating discrete output."""
        interface = MotorInterface()
        state = np.random.randn(64)

        output = interface.generate(state, OutputType.DISCRETE)

        assert output.output_type == OutputType.DISCRETE
        assert isinstance(output.value, ActionCategory)

    def test_generate_text(self):
        """Test generating text output."""
        interface = MotorInterface()
        state = np.random.randn(64)

        output = interface.generate(state, OutputType.TEXT)

        assert output.output_type == OutputType.TEXT
        assert isinstance(output.value, str)

    def test_propose_action(self):
        """Test proposing actions."""
        interface = MotorInterface()

        interface.propose_action(np.array([1, 2, 3]), confidence=0.8)
        interface.propose_action(np.array([4, 5, 6]), confidence=0.6)

        selected = interface.select_action()
        assert selected is not None

# =============================================================================
# Unit Tests: Active Inference Controller
# =============================================================================

class TestActiveInference:
    """Tests for ActiveInferenceController."""

    def test_controller_creation(self):
        """Test controller initialization."""
        controller = ActiveInferenceController()
        assert controller.belief is not None

    def test_update_belief(self):
        """Test belief update."""
        controller = ActiveInferenceController()
        obs = np.random.randn(64)

        controller.update_belief(obs)

        assert not np.allclose(controller.belief.state, 0)

    def test_set_goals(self):
        """Test setting goals."""
        controller = ActiveInferenceController()
        goals = np.ones(64)

        controller.set_goals(goals)

        assert np.allclose(controller.belief.goals, 1)

    def test_generate_policies(self):
        """Test policy generation."""
        controller = ActiveInferenceController()

        policies = controller.generate_policies(n_policies=5)

        assert len(policies) == 5
        for p in policies:
            assert len(p.actions) == controller.config.efe_horizon

    def test_compute_efe(self):
        """Test EFE computation."""
        controller = ActiveInferenceController()
        controller.set_goals(np.ones(64))

        policies = controller.generate_policies(n_policies=1)
        result = controller.compute_efe(policies[0])

        assert isinstance(result, EFEResult)
        assert result.total_efe is not None

    def test_select_action(self):
        """Test action selection."""
        controller = ActiveInferenceController()
        controller.set_goals(np.ones(64))
        state = np.random.randn(64)

        action = controller.select_action(state)

        assert isinstance(action, np.ndarray)
        assert len(action) == controller.config.output_dim

# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full system."""

    def test_full_cycle(self):
        """Test complete perceive -> think -> act cycle."""
        core = CognitiveCore()
        core.initialize()

        # Perceive
        input_data = np.random.randn(32)
        sensory_input = core.perceive(input_data)
        assert sensory_input.processed is not None

        # Think
        proposals = core.think()
        assert proposals >= 0

        # Act
        output = core.act()
        assert output is not None

    def test_memory_persistence(self):
        """Test state persists across cycles."""
        core = CognitiveCore()
        core.initialize()

        # Run multiple cycles
        for i in range(5):
            core.perceive(np.ones(32) * i)
            core.think()

        # State should reflect history
        assert core.state.cycle_count == 5

    def test_learning(self):
        """Test system adapts over time."""
        core = CognitiveCore()
        core.initialize()

        # Set goals
        goals = np.ones(64)
        core.set_goals(goals)

        # Run cycles and track actions
        actions = []
        for i in range(10):
            core.perceive(np.random.randn(32))
            core.think()
            output = core.act()
            if isinstance(output.value, np.ndarray):
                actions.append(output.value)

        # Should have generated actions
        assert len(actions) > 0

    def test_multi_step(self):
        """Test long-running stability."""
        core = CognitiveCore()
        core.initialize()

        def input_gen(step):
            return np.sin(np.linspace(0, 2 * np.pi, 32) + step * 0.1)

        results = core.run(steps=20, input_generator=input_gen)

        assert len(results) == 20
        for r in results:
            assert isinstance(r, CycleResult)

# =============================================================================
# Stress Tests
# =============================================================================

class TestStress:
    """Stress tests for system stability."""

    def test_1000_cycles(self):
        """Test 1000 cognitive cycles."""
        core = CognitiveCore()
        core.initialize()

        start_time = time.time()

        for i in range(1000):
            core.perceive(np.random.randn(32))
            core.think(dt=0.01)
            core.act()

        elapsed = time.time() - start_time

        assert core.state.cycle_count == 1000
        # Should complete in reasonable time
        assert elapsed < 60  # Less than 1 minute

    def test_concurrent_inputs(self):
        """Test handling multiple rapid inputs."""
        core = CognitiveCore()
        core.initialize()

        # Rapid input sequence
        for i in range(100):
            core.perceive(np.random.randn(64))
            core.perceive("text input")
            core.perceive([1, 2, 3, 4, 5])

        # Should handle without errors
        stats = core.get_statistics()
        assert stats['error_count'] == 0

    def test_reset_stability(self):
        """Test system stability after reset."""
        core = CognitiveCore()
        core.initialize()

        # Run some cycles
        for _ in range(10):
            core.perceive(np.random.randn(32))
            core.think()
            core.act()

        # Reset
        core.reset()

        # Run more cycles
        for _ in range(10):
            core.perceive(np.random.randn(32))
            core.think()
            core.act()

        assert core.state.cycle_count == 10

# =============================================================================
# Additional Tests
# =============================================================================

class TestCognitiveCore:
    """Additional tests for CognitiveCore."""

    def test_core_creation(self):
        """Test core initialization."""
        core = CognitiveCore()
        assert core.config is not None
        assert not core._initialized

    def test_initialize(self):
        """Test module loading."""
        core = CognitiveCore()
        stats = core.initialize()

        assert core._initialized
        assert 'total_modules' in stats

    def test_get_state(self):
        """Test getting cognitive state."""
        core = CognitiveCore()
        core.initialize()

        state = core.get_state()

        assert isinstance(state, CognitiveState)
        assert state.cycle_count == 0

    def test_get_statistics(self):
        """Test getting comprehensive statistics."""
        core = CognitiveCore()
        core.initialize()

        stats = core.get_statistics()

        assert 'initialized' in stats
        assert 'cycle_count' in stats
        assert 'loader' in stats
        assert 'sensory' in stats
        assert 'motor' in stats
        assert 'controller' in stats

    def test_shutdown(self):
        """Test system shutdown."""
        core = CognitiveCore()
        core.initialize()

        core.shutdown()

        assert not core._initialized
        assert len(core._modules) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
