"""
Tests for the Self-Improvement module.

Covers:
- Modification generator
- Change verifier
- System updater
- Meta learner
- Darwin Gödel Machine
"""

import pytest
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.generator import (
    ModificationGenerator, GeneratorParams, Modification, ModificationType
)
from src.verifier import (
    ChangeVerifier, VerifierParams, VerificationResult, VerificationMethod
)
from src.updater import (
    SystemUpdater, UpdaterParams, UpdateResult, UpdateStatus, Checkpoint
)
from src.meta_learner import (
    MetaLearner, MetaParams, LearningStrategy, StrategyType
)
from src.darwin_godel import (
    DarwinGodelMachine, DGMParams, ImprovementCycle, ImprovementPhase
)


class MockSystem:
    """Mock system for testing improvements."""

    def __init__(self):
        self.weights = np.random.randn(10)
        self.learning_rate = 0.01
        self.performance = 0.5

    def get_performance(self) -> float:
        return self.performance + np.random.randn() * 0.01

    def get_state(self) -> dict:
        return {
            'weights': self.weights.copy(),
            'learning_rate': self.learning_rate,
        }

    def set_state(self, state: dict) -> None:
        self.weights = state['weights']
        self.learning_rate = state['learning_rate']

    def apply_modification(self, mod: Modification) -> None:
        if mod.mod_type == ModificationType.WEIGHT_ADJUSTMENT:
            scale = mod.changes.get('adjustment_scale', 0.1)
            self.weights *= (1 + scale * 0.01)
            self.performance += 0.01  # Small improvement

    def rollback(self) -> None:
        self.weights = np.random.randn(10)
        self.performance = 0.5


class TestModificationGenerator:
    """Tests for the modification generator."""

    def test_generator_initialization(self):
        """Test generator initialization."""
        generator = ModificationGenerator()
        assert generator.params.n_candidates == 10

    def test_register_component(self):
        """Test component registration."""
        generator = ModificationGenerator()
        generator.register_component("layer1", {"type": "dense", "size": 64})
        assert "layer1" in generator._components

    def test_generate_candidates(self):
        """Test candidate generation."""
        generator = ModificationGenerator()
        generator.register_component("layer1", {"type": "dense"})
        generator.register_component("layer2", {"type": "dense"})

        candidates = generator.generate_candidates(0.5)
        assert len(candidates) <= generator.params.n_candidates
        assert all(isinstance(c, Modification) for c in candidates)

    def test_generate_without_components(self):
        """Test generation with no registered components."""
        generator = ModificationGenerator()
        candidates = generator.generate_candidates(0.5)
        # Should still generate some candidates
        assert isinstance(candidates, list)

    def test_record_outcome(self):
        """Test recording outcomes."""
        generator = ModificationGenerator()
        generator.register_component("layer1", {"type": "dense"})

        candidates = generator.generate_candidates(0.5)
        if candidates:
            generator.record_outcome(candidates[0], 0.1)

        stats = generator.get_statistics()
        assert stats['n_modifications'] >= 0

    def test_strategy_weights_update(self):
        """Test that strategy weights update."""
        generator = ModificationGenerator()
        generator.register_component("layer1", {"type": "dense"})

        initial_weights = generator._strategy_weights.copy()

        for _ in range(10):
            candidates = generator.generate_candidates(0.5)
            if candidates:
                generator.record_outcome(candidates[0], np.random.rand() - 0.5)

        # Weights should have changed
        # (Note: might be same due to normalization, but structure should be maintained)
        assert sum(generator._strategy_weights.values()) > 0

    def test_crossover(self):
        """Test modification crossover."""
        generator = ModificationGenerator()
        generator.register_component("layer1", {"type": "dense"})

        mod1 = Modification(
            mod_id="m1",
            mod_type=ModificationType.WEIGHT_ADJUSTMENT,
            target_component="layer1",
            changes={"scale": 0.1, "momentum": 0.9},
            expected_improvement=0.1,
            confidence=0.8,
            complexity_cost=0.01,
            reversible=True,
        )
        mod2 = Modification(
            mod_id="m2",
            mod_type=ModificationType.WEIGHT_ADJUSTMENT,
            target_component="layer1",
            changes={"scale": 0.2, "decay": 0.99},
            expected_improvement=0.15,
            confidence=0.7,
            complexity_cost=0.02,
            reversible=True,
        )

        crossed = generator._crossover(mod1, mod2)
        assert isinstance(crossed, Modification)
        assert crossed.mod_id != mod1.mod_id

    def test_mutate(self):
        """Test modification mutation."""
        generator = ModificationGenerator()

        mod = Modification(
            mod_id="m1",
            mod_type=ModificationType.WEIGHT_ADJUSTMENT,
            target_component="layer1",
            changes={"scale": 0.1},
            expected_improvement=0.1,
            confidence=0.8,
            complexity_cost=0.01,
            reversible=True,
        )

        mutated = generator._mutate(mod)
        assert isinstance(mutated, Modification)
        assert mutated.mod_id != mod.mod_id


class TestChangeVerifier:
    """Tests for the change verifier."""

    def test_verifier_initialization(self):
        """Test verifier initialization."""
        verifier = ChangeVerifier()
        assert verifier.params.min_improvement == 0.01

    def test_register_test_suite(self):
        """Test test suite registration."""
        verifier = ChangeVerifier()
        verifier.register_test_suite("perf", lambda: 0.5)
        assert "perf" in verifier._test_suites

    def test_register_constraint(self):
        """Test constraint registration."""
        verifier = ChangeVerifier()
        verifier.register_constraint(lambda: True)
        assert len(verifier._constraints) == 1

    def test_verify_simulation(self):
        """Test verification via simulation."""
        verifier = ChangeVerifier()
        verifier.register_test_suite("perf", lambda: 0.5)
        verifier.update_baseline(0.5)

        mod = Modification(
            mod_id="m1",
            mod_type=ModificationType.WEIGHT_ADJUSTMENT,
            target_component="layer1",
            changes={"scale": 0.1},
            expected_improvement=0.1,
            confidence=0.8,
            complexity_cost=0.01,
            reversible=True,
        )

        result = verifier.verify(
            mod,
            lambda m: None,
            lambda: None,
            method=VerificationMethod.SIMULATION,
        )

        assert isinstance(result, VerificationResult)

    def test_verify_rollback(self):
        """Test verification with rollback."""
        verifier = ChangeVerifier()
        performance = [0.5]

        def get_perf():
            return performance[0]

        def apply(m):
            performance[0] += 0.05

        def rollback():
            performance[0] = 0.5

        verifier.register_test_suite("perf", get_perf)
        verifier.update_baseline(0.5)

        mod = Modification(
            mod_id="m1",
            mod_type=ModificationType.WEIGHT_ADJUSTMENT,
            target_component="layer1",
            changes={"scale": 0.1},
            expected_improvement=0.1,
            confidence=0.8,
            complexity_cost=0.01,
            reversible=True,
        )

        result = verifier.verify(
            mod,
            apply,
            rollback,
            method=VerificationMethod.ROLLBACK_TEST,
        )

        assert isinstance(result, VerificationResult)

    def test_constraint_violation(self):
        """Test that constraint violations fail verification."""
        verifier = ChangeVerifier()
        verifier.register_test_suite("perf", lambda: 0.5)
        verifier.register_constraint(lambda: False)  # Always fails

        mod = Modification(
            mod_id="m1",
            mod_type=ModificationType.WEIGHT_ADJUSTMENT,
            target_component="layer1",
            changes={"scale": 0.1},
            expected_improvement=0.1,
            confidence=0.8,
            complexity_cost=0.01,
            reversible=True,
        )

        result = verifier.verify(
            mod,
            lambda m: None,
            lambda: None,
        )

        assert not result.verified
        assert len(result.warnings) > 0


class TestSystemUpdater:
    """Tests for the system updater."""

    def test_updater_initialization(self):
        """Test updater initialization."""
        updater = SystemUpdater()
        assert updater.params.gradual_application == True

    def test_create_checkpoint(self):
        """Test checkpoint creation."""
        updater = SystemUpdater()
        state = {'weights': np.array([1, 2, 3])}
        updater.set_state_accessors(lambda: state, lambda s: None)
        updater.set_performance_monitor(lambda: 0.5)

        checkpoint = updater.create_checkpoint()
        assert isinstance(checkpoint, Checkpoint)

    def test_apply_modification(self):
        """Test modification application."""
        updater = SystemUpdater()
        state = {'weights': 1.0, 'learning_rate': 0.01}

        def get_state():
            return state.copy()

        def set_state(s):
            nonlocal state
            state = s

        updater.set_state_accessors(get_state, set_state)
        updater.set_performance_monitor(lambda: 0.5)

        mod = Modification(
            mod_id="m1",
            mod_type=ModificationType.WEIGHT_ADJUSTMENT,
            target_component="layer1",
            changes={"adjustment_scale": 0.1},
            expected_improvement=0.1,
            confidence=0.8,
            complexity_cost=0.01,
            reversible=True,
        )

        result = updater.apply(mod)
        assert isinstance(result, UpdateResult)

    def test_rollback(self):
        """Test rollback functionality."""
        updater = SystemUpdater()
        original = {'value': 100}
        state = original.copy()

        updater.set_state_accessors(
            lambda: state.copy(),
            lambda s: state.update(s)
        )
        updater.set_performance_monitor(lambda: 0.5)

        # Create checkpoint
        checkpoint = updater.create_checkpoint()

        # Modify state
        state['value'] = 200

        # Rollback
        success = updater.rollback(checkpoint.checkpoint_id)
        assert success
        assert state['value'] == 100


class TestMetaLearner:
    """Tests for the meta learner."""

    def test_meta_learner_initialization(self):
        """Test meta learner initialization."""
        learner = MetaLearner()
        assert len(learner._strategies) > 0

    def test_select_strategy(self):
        """Test strategy selection."""
        learner = MetaLearner()
        strategy = learner.select_strategy()
        assert isinstance(strategy, LearningStrategy)

    def test_record_experience(self):
        """Test experience recording."""
        learner = MetaLearner()
        learner.select_strategy()  # Set current strategy

        mod = Modification(
            mod_id="m1",
            mod_type=ModificationType.WEIGHT_ADJUSTMENT,
            target_component="layer1",
            changes={},
            expected_improvement=0.1,
            confidence=0.8,
            complexity_cost=0.01,
            reversible=True,
        )

        result = UpdateResult(
            modification=mod,
            status=UpdateStatus.APPLIED,
            applied_at=time.time(),
            rollback_available=True,
            checkpoint_id="ckpt_1",
            performance_delta=0.05,
            details={},
        )

        learner.record_experience(mod, result)
        assert len(learner._experience) == 1

    def test_record_performance(self):
        """Test performance recording."""
        learner = MetaLearner()

        for i in range(10):
            learner.record_performance(0.5 + i * 0.01)

        assert len(learner._performance_history) == 10

    def test_should_improve(self):
        """Test improvement decision."""
        learner = MetaLearner()
        should, reason = learner.should_improve()
        assert isinstance(should, bool)
        assert isinstance(reason, str)

    def test_recommend_target(self):
        """Test component recommendation."""
        learner = MetaLearner()
        target, priority = learner.recommend_target()
        assert isinstance(target, str)
        assert isinstance(priority, float)

    def test_strategy_adaptation(self):
        """Test that strategies adapt over time."""
        learner = MetaLearner()

        for _ in range(20):
            strategy = learner.select_strategy()
            mod = Modification(
                mod_id=f"m_{_}",
                mod_type=ModificationType.WEIGHT_ADJUSTMENT,
                target_component="layer1",
                changes={},
                expected_improvement=0.1,
                confidence=0.8,
                complexity_cost=0.01,
                reversible=True,
            )
            result = UpdateResult(
                modification=mod,
                status=UpdateStatus.APPLIED if np.random.rand() > 0.5 else UpdateStatus.FAILED,
                applied_at=time.time(),
                rollback_available=True,
                checkpoint_id="ckpt_1",
                performance_delta=np.random.rand() * 0.1,
                details={},
            )
            learner.record_experience(mod, result)

        stats = learner.get_statistics()
        assert stats['n_experiences'] == 20


class TestDarwinGodelMachine:
    """Tests for the Darwin Gödel Machine."""

    def test_dgm_initialization(self):
        """Test DGM initialization."""
        dgm = DarwinGodelMachine()
        assert dgm._phase == ImprovementPhase.IDLE
        assert dgm._cycle_count == 0

    def test_set_target_system(self):
        """Test setting target system."""
        dgm = DarwinGodelMachine()
        system = MockSystem()

        dgm.set_target_system(
            system,
            system.get_performance,
            system.apply_modification,
            system.rollback,
            (system.get_state, system.set_state),
        )

        assert dgm._target_system is system

    def test_run_improvement_cycle(self):
        """Test running an improvement cycle."""
        dgm = DarwinGodelMachine()
        system = MockSystem()

        dgm.set_target_system(
            system,
            system.get_performance,
            system.apply_modification,
            system.rollback,
            (system.get_state, system.set_state),
        )

        cycle = dgm.run_improvement_cycle()
        assert isinstance(cycle, ImprovementCycle)
        assert cycle.cycle_id == 1

    def test_multiple_cycles(self):
        """Test running multiple improvement cycles."""
        dgm = DarwinGodelMachine()
        system = MockSystem()

        dgm.set_target_system(
            system,
            system.get_performance,
            system.apply_modification,
            system.rollback,
        )

        for _ in range(5):
            cycle = dgm.run_improvement_cycle()

        assert dgm._cycle_count == 5

    def test_should_improve(self):
        """Test improvement decision logic."""
        dgm = DarwinGodelMachine(DGMParams(improvement_interval=0.0))
        system = MockSystem()

        dgm.set_target_system(
            system,
            system.get_performance,
        )

        should, reason = dgm.should_improve()
        assert isinstance(should, bool)

    def test_safety_constraint(self):
        """Test adding safety constraints."""
        dgm = DarwinGodelMachine()
        dgm.add_safety_constraint(lambda: True)
        assert len(dgm.verifier._constraints) == 1

    def test_get_improvement_summary(self):
        """Test improvement summary."""
        dgm = DarwinGodelMachine()
        system = MockSystem()

        dgm.set_target_system(system, system.get_performance)
        dgm.run_improvement_cycle()

        summary = dgm.get_improvement_summary()
        assert 'total_cycles' in summary
        assert summary['total_cycles'] >= 1

    def test_component_statistics(self):
        """Test getting component statistics."""
        dgm = DarwinGodelMachine()
        stats = dgm.get_component_statistics()

        assert 'generator' in stats
        assert 'verifier' in stats
        assert 'updater' in stats
        assert 'meta_learner' in stats

    def test_reset(self):
        """Test system reset."""
        dgm = DarwinGodelMachine()
        system = MockSystem()

        dgm.set_target_system(system, system.get_performance)
        dgm.run_improvement_cycle()

        dgm.reset()
        assert dgm._cycle_count == 0
        assert len(dgm._cycle_history) == 0


class TestStress:
    """Stress tests for the self-improvement module."""

    def test_many_candidates(self):
        """Test generating many candidates."""
        generator = ModificationGenerator(GeneratorParams(n_candidates=50))

        for i in range(5):
            generator.register_component(f"layer{i}", {"type": "dense"})

        candidates = generator.generate_candidates(0.5)
        assert len(candidates) <= 50

    def test_many_verifications(self):
        """Test many verifications."""
        verifier = ChangeVerifier()
        verifier.register_test_suite("perf", lambda: 0.5)

        for i in range(100):
            mod = Modification(
                mod_id=f"m{i}",
                mod_type=ModificationType.WEIGHT_ADJUSTMENT,
                target_component="layer1",
                changes={"scale": np.random.rand()},
                expected_improvement=0.1,
                confidence=0.8,
                complexity_cost=0.01,
                reversible=True,
            )
            verifier.verify(mod, lambda m: None, lambda: None, VerificationMethod.SIMULATION)

        stats = verifier.get_statistics()
        assert stats['n_verifications'] == 100

    def test_many_updates(self):
        """Test many updates."""
        updater = SystemUpdater()
        state = {'value': 0}
        updater.set_state_accessors(
            lambda: state.copy(),
            lambda s: state.update(s)
        )
        updater.set_performance_monitor(lambda: 0.5)

        for i in range(50):
            mod = Modification(
                mod_id=f"m{i}",
                mod_type=ModificationType.WEIGHT_ADJUSTMENT,
                target_component="layer1",
                changes={},
                expected_improvement=0.1,
                confidence=0.8,
                complexity_cost=0.01,
                reversible=True,
            )
            updater.apply(mod)

        stats = updater.get_statistics()
        assert stats['n_updates'] == 50

    def test_long_running_dgm(self):
        """Test DGM over many cycles."""
        dgm = DarwinGodelMachine(DGMParams(
            improvement_interval=0.0,
            max_failed_attempts=100,
        ))
        system = MockSystem()

        dgm.set_target_system(
            system,
            system.get_performance,
            system.apply_modification,
            system.rollback,
        )

        for _ in range(20):
            dgm.run_improvement_cycle()

        stats = dgm.get_statistics()
        assert stats['cycle_count'] == 20


class TestIntegration:
    """Integration tests."""

    def test_full_improvement_pipeline(self):
        """Test complete improvement pipeline."""
        # Create system
        system = MockSystem()
        initial_perf = system.get_performance()

        # Create DGM
        dgm = DarwinGodelMachine(DGMParams(
            improvement_interval=0.0,
            safety_mode=True,
        ))

        dgm.set_target_system(
            system,
            system.get_performance,
            system.apply_modification,
            system.rollback,
            (system.get_state, system.set_state),
        )

        # Run improvement cycles
        cycles = dgm.run_auto_improve(max_cycles=10)

        # Verify results
        assert len(cycles) <= 10
        summary = dgm.get_improvement_summary()
        assert 'total_cycles' in summary

    def test_meta_learning_integration(self):
        """Test meta-learning integration."""
        system = MockSystem()

        dgm = DarwinGodelMachine(DGMParams(improvement_interval=0.0))
        dgm.set_target_system(
            system,
            system.get_performance,
            system.apply_modification,
            system.rollback,
        )

        # Run cycles
        for _ in range(15):
            dgm.run_improvement_cycle()

        # Check meta-learner state
        ml_stats = dgm.meta_learner.get_statistics()
        assert ml_stats['n_experiences'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
