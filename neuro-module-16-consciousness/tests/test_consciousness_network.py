"""Tests for consciousness network"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.consciousness_network import GlobalWorkspace, ConsciousnessNetwork


class TestGlobalWorkspace:
    """Tests for global workspace"""

    def test_initialization(self):
        """Test workspace initialization"""
        workspace = GlobalWorkspace()
        assert len(workspace.workspace) == 0

    def test_submit_for_broadcast(self):
        """Test submitting content"""
        workspace = GlobalWorkspace(n_features=50, capacity=5)
        content = np.random.rand(50)

        success = workspace.submit_for_broadcast("test", content, activation=0.8)

        assert success
        assert "test" in workspace.workspace

    def test_capacity_limit(self):
        """Test capacity is enforced"""
        workspace = GlobalWorkspace(n_features=50, capacity=3)

        for i in range(5):
            workspace.submit_for_broadcast(f"item{i}", np.random.rand(50), 0.5 + i * 0.1)

        assert len(workspace.workspace) <= 3

    def test_competition(self):
        """Test high activation wins competition"""
        workspace = GlobalWorkspace(n_features=50, capacity=2)

        workspace.submit_for_broadcast("weak", np.random.rand(50), 0.3)
        workspace.submit_for_broadcast("strong", np.random.rand(50), 0.9)
        workspace.submit_for_broadcast("stronger", np.random.rand(50), 0.95)

        # Strongest should be in workspace
        assert "stronger" in workspace.workspace

    def test_broadcast(self):
        """Test broadcasting"""
        workspace = GlobalWorkspace()
        workspace.submit_for_broadcast("content", np.random.rand(50), 0.8)

        broadcast = workspace.broadcast()

        assert "content" in broadcast
        assert workspace.is_broadcasting

    def test_decay(self):
        """Test activation decay"""
        workspace = GlobalWorkspace()
        workspace.submit_for_broadcast("content", np.random.rand(50), 0.5)

        initial = workspace.activations["content"]
        workspace.decay(rate=0.5)

        assert workspace.activations.get("content", 0) < initial


class TestConsciousnessNetwork:
    """Tests for full consciousness network"""

    def test_initialization(self):
        """Test network initialization"""
        network = ConsciousnessNetwork()

        assert network.minimal_self is not None
        assert network.narrative_self is not None
        assert network.metacognition is not None
        assert network.introspection is not None
        assert network.workspace is not None

    def test_process_experience(self):
        """Test processing an experience"""
        network = ConsciousnessNetwork()

        sensory = np.random.rand(50)
        action = np.random.rand(50)

        result = network.process_experience(sensory, action=action)

        assert "minimal_self" in result
        assert "narrative_self" in result
        assert "metacognition" in result
        assert "workspace_contents" in result

    def test_consciousness_level_updates(self):
        """Test consciousness level changes"""
        network = ConsciousnessNetwork()

        # Process several experiences
        for _ in range(5):
            network.process_experience(np.random.rand(50), np.random.rand(50))

        # Consciousness level should be valid
        assert 0 <= network.consciousness_level <= 1

    def test_introspect(self):
        """Test introspection"""
        network = ConsciousnessNetwork()
        network.process_experience(np.random.rand(50))

        result = network.introspect_current_state()

        assert "conscious_contents" in result
        assert "consciousness_level" in result

    def test_make_decision(self):
        """Test decision making"""
        network = ConsciousnessNetwork()

        options = [np.random.rand(50) for _ in range(3)]
        evidence = np.random.rand(50)

        result = network.make_decision(options, evidence)

        assert "selected_option" in result
        assert "confidence" in result
        assert 0 <= result["selected_option"] < 3

    def test_recall_autobiographical(self):
        """Test autobiographical recall"""
        network = ConsciousnessNetwork()

        # Create some memories
        for _ in range(5):
            network.process_experience(np.random.rand(50))

        cue = np.random.rand(50)
        result = network.recall_autobiographical(cue)

        assert "memories" in result

    def test_update(self):
        """Test network update"""
        network = ConsciousnessNetwork()
        network.process_experience(np.random.rand(50))

        network.update(dt=1.0)
        # Should not error

    def test_get_state(self):
        """Test comprehensive state"""
        network = ConsciousnessNetwork()
        network.process_experience(np.random.rand(50))

        state = network.get_consciousness_state()

        assert "consciousness_level" in state
        assert "minimal_self" in state
        assert "narrative_self" in state
        assert "metacognition" in state
        assert "introspection" in state
        assert "workspace_contents" in state


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
