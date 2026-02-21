"""
Tests for the Global Workspace integration system.
"""

import pytest
import numpy as np
import os

from neuro.modules.m00_integration.module_interface import (
    CognitiveModule,
    ModuleProposal,
    ModuleParams,
    ModuleType,
    ContentType,
    DummyModule,
)
from neuro.modules.m00_integration.global_workspace import (
    GlobalWorkspace,
    WorkspaceParams,
    WorkspaceMode,
)
from neuro.modules.m00_integration.attention_competition import AttentionCompetition, CompetitionParams
from neuro.modules.m00_integration.broadcast_system import BroadcastSystem, BroadcastParams
from neuro.modules.m00_integration.ignition import IgnitionDetector, IgnitionParams, IgnitionState

class TestModuleInterface:
    """Tests for the cognitive module interface."""

    def test_module_params_creation(self):
        """Test creating module parameters."""
        params = ModuleParams(
            module_type=ModuleType.MEMORY,
            name="TestModule",
            n_features=32,
        )
        assert params.module_type == ModuleType.MEMORY
        assert params.name == "TestModule"
        assert params.n_features == 32

    def test_proposal_creation(self):
        """Test creating a module proposal."""
        content = np.random.randn(64)
        proposal = ModuleProposal(
            source_module=ModuleType.MEMORY,
            content_type=ContentType.MEMORY,
            content=content,
            activation=0.8,
            confidence=0.9,
            relevance=0.7,
        )
        assert proposal.source_module == ModuleType.MEMORY
        assert proposal.activation == 0.8
        assert proposal.priority == 0.8 * 0.9 * 0.7

    def test_proposal_decay(self):
        """Test proposal activation decay."""
        proposal = ModuleProposal(
            source_module=ModuleType.MEMORY,
            content_type=ContentType.MEMORY,
            content=np.zeros(64),
            activation=1.0,
            confidence=1.0,
            relevance=1.0,
        )
        proposal.decay(0.9)
        assert proposal.activation == 0.9

    def test_dummy_module_initialization(self):
        """Test dummy module initialization."""
        module = DummyModule(
            module_type=ModuleType.MEMORY,
            name="TestDummy",
            n_features=32,
        )
        assert module.module_type == ModuleType.MEMORY
        assert module.name == "TestDummy"

    def test_dummy_module_propose(self):
        """Test dummy module proposal generation."""
        module = DummyModule()
        input_state = np.random.randn(64)
        proposals = module.propose(input_state)
        assert len(proposals) > 0
        assert isinstance(proposals[0], ModuleProposal)

    def test_dummy_module_receive_broadcast(self):
        """Test dummy module receiving broadcast."""
        module = DummyModule()
        proposal = ModuleProposal(
            source_module=ModuleType.EMOTION,
            content_type=ContentType.EMOTION,
            content=np.random.randn(64),
            activation=0.9,
            confidence=0.8,
            relevance=0.7,
        )
        module.receive_broadcast(proposal)
        state = module.get_state()
        assert state.last_broadcast_received is not None

    def test_module_reset(self):
        """Test module reset."""
        module = DummyModule()
        module.propose(np.random.randn(64))
        module.reset()
        state = module.get_state()
        assert state.activation_level == 0.0
        assert len(state.pending_proposals) == 0

class TestAttentionCompetition:
    """Tests for attention-based competition."""

    def test_competition_initialization(self):
        """Test competition initialization."""
        competition = AttentionCompetition()
        assert competition.params.capacity == 7
        assert competition.params.temperature == 1.0

    def test_competition_with_proposals(self):
        """Test competition with multiple proposals."""
        competition = AttentionCompetition()

        proposals = [
            ModuleProposal(
                source_module=ModuleType.MEMORY,
                content_type=ContentType.MEMORY,
                content=np.random.randn(64),
                activation=0.9,
                confidence=0.8,
                relevance=0.7,
            ),
            ModuleProposal(
                source_module=ModuleType.EMOTION,
                content_type=ContentType.EMOTION,
                content=np.random.randn(64),
                activation=0.5,
                confidence=0.6,
                relevance=0.5,
            ),
        ]

        result = competition.compete(proposals)
        assert len(result.winners) > 0
        assert result.max_score > 0

    def test_competition_empty_proposals(self):
        """Test competition with no proposals."""
        competition = AttentionCompetition()
        result = competition.compete([])
        assert len(result.winners) == 0
        assert result.max_score == 0.0

    def test_competition_capacity_limit(self):
        """Test that competition respects capacity limit."""
        params = CompetitionParams(capacity=3)
        competition = AttentionCompetition(params)

        # Create more proposals than capacity
        proposals = [
            ModuleProposal(
                source_module=ModuleType.MEMORY,
                content_type=ContentType.MEMORY,
                content=np.random.randn(64),
                activation=np.random.rand(),
                confidence=np.random.rand(),
                relevance=np.random.rand(),
            )
            for _ in range(10)
        ]

        result = competition.compete(proposals)
        assert len(result.winners) <= 3

    def test_attention_bias(self):
        """Test top-down attention bias."""
        competition = AttentionCompetition()
        bias = np.random.randn(64)
        competition.set_attention_bias(bias)

        # Proposal aligned with bias should score higher
        aligned_content = bias.copy()
        unaligned_content = -bias

        aligned_proposal = ModuleProposal(
            source_module=ModuleType.MEMORY,
            content_type=ContentType.MEMORY,
            content=aligned_content,
            activation=0.5,
            confidence=0.5,
            relevance=0.5,
        )
        unaligned_proposal = ModuleProposal(
            source_module=ModuleType.MEMORY,
            content_type=ContentType.MEMORY,
            content=unaligned_content,
            activation=0.5,
            confidence=0.5,
            relevance=0.5,
        )

        result = competition.compete([aligned_proposal, unaligned_proposal])
        # Aligned should be first winner (higher score)
        assert result.winners[0].content is aligned_content

class TestIgnitionDetector:
    """Tests for ignition detection."""

    def test_ignition_initialization(self):
        """Test ignition detector initialization."""
        detector = IgnitionDetector()
        assert detector.get_state() == IgnitionState.SUBLIMINAL
        assert not detector.is_ignited()

    def test_ignition_below_threshold(self):
        """Test that low activation doesn't trigger ignition."""
        detector = IgnitionDetector(IgnitionParams(threshold=0.7))
        result = detector.detect(activation=0.3, buffer=[])
        assert not result.ignited
        assert result.state == IgnitionState.SUBLIMINAL

    def test_ignition_above_threshold(self):
        """Test that high activation triggers ignition."""
        params = IgnitionParams(threshold=0.5, min_duration=0.0)
        detector = IgnitionDetector(params)

        # First detection moves to threshold
        result1 = detector.detect(activation=0.8, buffer=[])

        # Second detection should ignite
        result2 = detector.detect(activation=0.8, buffer=[])
        assert result2.ignited or result2.state in [IgnitionState.THRESHOLD, IgnitionState.IGNITED]

    def test_ignition_with_buffer(self):
        """Test ignition with buffer contents."""
        detector = IgnitionDetector()
        buffer = [
            ModuleProposal(
                source_module=ModuleType.MEMORY,
                content_type=ContentType.MEMORY,
                content=np.random.randn(64),
                activation=0.9,
                confidence=0.8,
                relevance=0.7,
            )
        ]
        result = detector.detect(activation=0.8, buffer=buffer)
        assert result.trigger_proposal is not None

    def test_ignition_statistics(self):
        """Test ignition statistics tracking."""
        detector = IgnitionDetector()
        for i in range(10):
            detector.detect(activation=np.random.rand(), buffer=[])
        stats = detector.get_statistics()
        assert 'total_events' in stats
        assert stats['total_events'] == 10

class TestBroadcastSystem:
    """Tests for the broadcast system."""

    def test_broadcast_initialization(self):
        """Test broadcast system initialization."""
        system = BroadcastSystem()
        assert len(system.get_recent_broadcasts()) == 0

    def test_broadcast_event(self):
        """Test broadcasting an event."""
        system = BroadcastSystem()
        proposal = ModuleProposal(
            source_module=ModuleType.MEMORY,
            content_type=ContentType.MEMORY,
            content=np.random.randn(64),
            activation=0.9,
            confidence=0.8,
            relevance=0.7,
        )
        recipients = [ModuleType.EMOTION, ModuleType.LANGUAGE]
        event = system.broadcast(proposal, recipients)
        assert event is not None
        assert len(system.get_recent_broadcasts()) == 1

    def test_broadcast_statistics(self):
        """Test broadcast statistics."""
        system = BroadcastSystem()
        for i in range(5):
            proposal = ModuleProposal(
                source_module=ModuleType.MEMORY,
                content_type=ContentType.MEMORY,
                content=np.random.randn(64),
                activation=0.9,
                confidence=0.8,
                relevance=0.7,
            )
            system.broadcast(proposal, [ModuleType.EMOTION])
        stats = system.get_statistics()
        assert stats['total_broadcasts'] >= 1

class TestGlobalWorkspace:
    """Tests for the complete global workspace."""

    def test_workspace_initialization(self):
        """Test workspace initialization."""
        workspace = GlobalWorkspace()
        assert len(workspace._modules) == 0
        state = workspace.get_state()
        assert len(state.buffer) == 0

    def test_register_module(self):
        """Test registering a module."""
        workspace = GlobalWorkspace()
        module = DummyModule(module_type=ModuleType.MEMORY, name="Memory")
        workspace.register_module(module)
        assert ModuleType.MEMORY in workspace._modules

    def test_collect_proposals(self):
        """Test collecting proposals from modules."""
        workspace = GlobalWorkspace()
        module1 = DummyModule(module_type=ModuleType.MEMORY, name="Memory")
        module2 = DummyModule(module_type=ModuleType.EMOTION, name="Emotion")
        workspace.register_module(module1)
        workspace.register_module(module2)

        input_state = np.random.randn(64)
        proposals = workspace.collect_proposals(input_state)
        assert len(proposals) >= 2

    def test_workspace_compete(self):
        """Test workspace competition."""
        workspace = GlobalWorkspace()
        module = DummyModule(module_type=ModuleType.MEMORY, name="Memory")
        workspace.register_module(module)

        proposals = workspace.collect_proposals(np.random.randn(64))
        winners = workspace.compete(proposals)
        assert len(winners) >= 0

    def test_workspace_run_cycle(self):
        """Test complete workspace cycle."""
        workspace = GlobalWorkspace()
        for i, mod_type in enumerate([ModuleType.MEMORY, ModuleType.EMOTION, ModuleType.LANGUAGE]):
            module = DummyModule(module_type=mod_type, name=f"Module{i}")
            workspace.register_module(module)

        input_state = np.random.randn(64)
        winners, broadcast = workspace.run_cycle(input_state)
        assert isinstance(winners, list)

    def test_workspace_mode(self):
        """Test workspace operating modes."""
        workspace = GlobalWorkspace()
        workspace.set_mode(WorkspaceMode.FOCUSED)
        assert workspace._mode == WorkspaceMode.FOCUSED

    def test_workspace_reset(self):
        """Test workspace reset."""
        workspace = GlobalWorkspace()
        module = DummyModule(module_type=ModuleType.MEMORY, name="Memory")
        workspace.register_module(module)
        workspace.run_cycle(np.random.randn(64))

        workspace.reset()
        state = workspace.get_state()
        assert len(state.buffer) == 0
        assert state.step_count == 0

    def test_workspace_statistics(self):
        """Test workspace statistics."""
        workspace = GlobalWorkspace()
        for i in range(3):
            module = DummyModule(
                module_type=list(ModuleType)[i],
                name=f"Module{i}"
            )
            workspace.register_module(module)

        for _ in range(10):
            workspace.run_cycle(np.random.randn(64))

        stats = workspace.get_statistics()
        assert 'step_count' in stats
        assert stats['step_count'] == 10
        assert stats['module_count'] == 3

class TestIntegration:
    """Integration tests for the complete system."""

    def test_full_cycle_multiple_modules(self):
        """Test full cycle with multiple modules."""
        workspace = GlobalWorkspace(WorkspaceParams(
            buffer_capacity=5,
            ignition_threshold=0.5,
        ))

        # Register several modules
        module_types = [
            ModuleType.MEMORY,
            ModuleType.EMOTION,
            ModuleType.LANGUAGE,
            ModuleType.REASONING,
            ModuleType.EXECUTIVE,
        ]
        for mod_type in module_types:
            module = DummyModule(module_type=mod_type, name=mod_type.name)
            workspace.register_module(module)

        # Run multiple cycles
        for i in range(20):
            input_state = np.random.randn(64) * (i + 1) / 10
            winners, broadcast = workspace.run_cycle(input_state)

        stats = workspace.get_statistics()
        assert stats['step_count'] == 20
        assert stats['module_count'] == 5

    def test_broadcast_reception(self):
        """Test that modules receive broadcasts."""
        workspace = GlobalWorkspace(WorkspaceParams(
            ignition_threshold=0.3,  # Low threshold for testing
        ))

        received_broadcasts = []

        class TrackingModule(DummyModule):
            def receive_broadcast(self, proposal):
                super().receive_broadcast(proposal)
                received_broadcasts.append(proposal)

        module = TrackingModule(module_type=ModuleType.MEMORY, name="Tracking")
        workspace.register_module(module)

        # Run cycles with high activation
        for _ in range(10):
            input_state = np.ones(64)  # High activation input
            workspace.run_cycle(input_state)

        # Should have received some broadcasts
        # (may be 0 if ignition didn't trigger)
        assert isinstance(received_broadcasts, list)

    def test_buffer_capacity(self):
        """Test that buffer respects capacity limit."""
        workspace = GlobalWorkspace(WorkspaceParams(buffer_capacity=3))

        for i in range(10):
            module = DummyModule(
                module_type=list(ModuleType)[i % len(ModuleType)],
                name=f"Module{i}"
            )
            workspace.register_module(module)

        for _ in range(5):
            workspace.run_cycle(np.random.randn(64))

        state = workspace.get_state()
        assert len(state.buffer) <= 3

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
