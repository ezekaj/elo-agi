"""Tests for catastrophic forgetting prevention."""

import pytest
import numpy as np

from neuro.modules.continual.forgetting_prevention import (
    CatastrophicForgettingPrevention,
    ForgettingPreventionConfig,
    ForgettingPreventionMethod,
    TaskMemory,
)

class TestForgettingPreventionConfig:
    """Tests for ForgettingPreventionConfig class."""

    def test_default_config(self):
        config = ForgettingPreventionConfig()
        assert config.method == ForgettingPreventionMethod.EWC
        assert config.ewc_lambda == 1000.0
        assert config.si_c == 0.1

    def test_custom_config(self):
        config = ForgettingPreventionConfig(
            method=ForgettingPreventionMethod.SYNAPTIC_INTELLIGENCE,
            ewc_lambda=500.0,
        )
        assert config.method == ForgettingPreventionMethod.SYNAPTIC_INTELLIGENCE
        assert config.ewc_lambda == 500.0

class TestCatastrophicForgettingPrevention:
    """Tests for CatastrophicForgettingPrevention class."""

    def test_creation(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)
        assert cfp is not None

    def test_register_task(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        params = {"layer1": np.random.randn(10, 10)}
        cfp.register_task("task1", params)

        assert "task1" in cfp.get_registered_tasks()

    def test_compute_fisher_information(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        params = {"layer1": np.random.randn(5, 5)}
        samples = [(np.random.randn(5), np.random.randn(5)) for _ in range(10)]

        fisher = cfp.compute_fisher_information(params, samples)

        assert "layer1" in fisher
        assert fisher["layer1"].shape == params["layer1"].shape

    def test_compute_fisher_empty_samples(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        params = {"layer1": np.random.randn(5, 5)}
        fisher = cfp.compute_fisher_information(params, [])

        assert np.allclose(fisher["layer1"], 0)

    def test_compute_ewc_loss_no_tasks(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        params = {"layer1": np.random.randn(5, 5)}
        loss = cfp.compute_ewc_loss(params)

        assert loss == 0.0

    def test_compute_ewc_loss_with_task(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        old_params = {"layer1": np.zeros((5, 5))}
        cfp.register_task("task1", old_params)

        new_params = {"layer1": np.ones((5, 5))}
        loss = cfp.compute_ewc_loss(new_params)

        assert loss > 0

    def test_ewc_loss_unchanged_params(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        params = {"layer1": np.random.randn(5, 5)}
        cfp.register_task("task1", params)

        loss = cfp.compute_ewc_loss(params)

        assert loss == pytest.approx(0.0, abs=1e-6)

    def test_allocate_capacity(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        shapes = {"layer1": (10,), "layer2": (5, 5)}
        masks = cfp.allocate_capacity("task1", shapes, required_fraction=0.5)

        assert "layer1" in masks
        assert "layer2" in masks
        assert masks["layer1"].shape == shapes["layer1"]
        assert masks["layer2"].shape == shapes["layer2"]

    def test_allocate_capacity_multiple_tasks(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        shapes = {"layer1": (10,)}

        masks1 = cfp.allocate_capacity("task1", shapes, required_fraction=0.5)
        masks2 = cfp.allocate_capacity("task2", shapes, required_fraction=0.5)

        overlap = np.sum(masks1["layer1"] & masks2["layer1"])
        assert overlap == 0

    def test_get_task_mask(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        params = {"layer1": np.random.randn(10)}
        shapes = {"layer1": (10,)}

        masks = cfp.allocate_capacity("task1", shapes)
        cfp.register_task("task1", params, mask=masks)

        retrieved = cfp.get_task_mask("task1")
        assert retrieved is not None
        assert np.array_equal(retrieved["layer1"], masks["layer1"])

    def test_compute_synaptic_importance(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        params = {"layer1": np.random.randn(5, 5)}
        grads = {"layer1": np.random.randn(5, 5)}

        importance = cfp.compute_synaptic_importance(grads, params)

        assert "layer1" in importance
        assert importance["layer1"].shape == params["layer1"].shape

    def test_compute_si_loss_no_tasks(self):
        config = ForgettingPreventionConfig(method=ForgettingPreventionMethod.SYNAPTIC_INTELLIGENCE)
        cfp = CatastrophicForgettingPrevention(config=config, random_seed=42)

        params = {"layer1": np.random.randn(5, 5)}
        loss = cfp.compute_si_loss(params)

        assert loss == 0.0

    def test_compute_combined_loss(self):
        config = ForgettingPreventionConfig(method=ForgettingPreventionMethod.COMBINED)
        cfp = CatastrophicForgettingPrevention(config=config, random_seed=42)

        old_params = {"layer1": np.zeros((5, 5))}
        cfp.register_task("task1", old_params)

        new_params = {"layer1": np.ones((5, 5))}
        total, losses = cfp.compute_combined_loss(new_params)

        assert "ewc" in losses
        assert "si" in losses
        assert total >= 0

    def test_apply_mask(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        params = {"layer1": np.ones(10)}
        shapes = {"layer1": (10,)}
        masks = cfp.allocate_capacity("task1", shapes)
        cfp.register_task("task1", params, mask=masks)

        grads = {"layer1": np.ones(10) * 2}
        masked_grads = cfp.apply_mask(grads, "task1")

        assert masked_grads["layer1"].sum() <= grads["layer1"].sum()

    def test_get_task_memory(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        params = {"layer1": np.random.randn(5, 5)}
        cfp.register_task("task1", params)

        memory = cfp.get_task_memory("task1")

        assert isinstance(memory, TaskMemory)
        assert memory.task_id == "task1"

    def test_reset(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        params = {"layer1": np.random.randn(5, 5)}
        cfp.register_task("task1", params)

        cfp.reset()

        assert len(cfp.get_registered_tasks()) == 0

    def test_statistics(self):
        cfp = CatastrophicForgettingPrevention(random_seed=42)

        params = {"layer1": np.random.randn(5, 5)}
        cfp.register_task("task1", params)

        stats = cfp.statistics()

        assert "num_tasks" in stats
        assert "method" in stats
        assert stats["num_tasks"] == 1

class TestForgettingPreventionMethod:
    """Tests for ForgettingPreventionMethod enum."""

    def test_methods(self):
        assert ForgettingPreventionMethod.EWC.value == "ewc"
        assert ForgettingPreventionMethod.PACKNET.value == "packnet"
        assert ForgettingPreventionMethod.SYNAPTIC_INTELLIGENCE.value == "si"
        assert ForgettingPreventionMethod.COMBINED.value == "combined"

    def test_all_methods_work(self):
        for method in ForgettingPreventionMethod:
            config = ForgettingPreventionConfig(method=method)
            cfp = CatastrophicForgettingPrevention(config=config, random_seed=42)

            params = {"layer1": np.random.randn(5, 5)}
            cfp.register_task("task1", params)

            total, losses = cfp.compute_combined_loss(params)
            assert total >= 0
