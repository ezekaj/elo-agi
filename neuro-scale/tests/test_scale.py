"""
Comprehensive tests for neuro-scale module.

Tests distributed processing, GPU kernels, sparse optimizations,
and efficiency optimization components.
"""

import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distributed.coordinator import (
    TaskStatus,
    SchedulingPolicy,
    CoordinatorConfig,
    DistributedTask,
    WorkerInfo,
    TaskQueue,
    LoadBalancer,
    TaskCoordinator,
)
from distributed.worker import (
    WorkerStatus,
    WorkResult,
    WorkerConfig,
    Worker,
    WorkerPool,
)
from distributed.aggregator import (
    AggregationStrategy,
    AggregatedResult,
    WeightedAverage,
    Voting,
    ConcatAggregator,
    ResultAggregator,
)

from gpu.kernels import (
    KernelConfig,
    KernelStats,
    GPUKernel,
    MatMulKernel,
    ConvolutionKernel,
    SoftmaxKernel,
    ReductionKernel,
    KernelManager,
)
from gpu.batch_inference import (
    BatchConfig,
    InferenceBatch,
    BatchResult,
    InferenceCache,
    BatchBuilder,
    BatchInference,
)

from sparse.pruning import (
    PruningStrategy,
    PruningResult,
    PruningConfig,
    PruningMask,
    MagnitudePruning,
    RandomPruning,
    StructuredPruning,
    NetworkPruner,
)
from sparse.quantization import (
    QuantizationLevel,
    QuantizationConfig,
    QuantizedTensor,
    QuantizedModel,
    Quantizer,
)

from efficiency import (
    LatencyProfile,
    ThroughputMetrics,
    MemoryProfile,
    OptimizationConfig,
    Profiler,
    OperationFuser,
    MemoryOptimizer,
    EfficiencyOptimizer,
)


# =============================================================================
# DISTRIBUTED TESTS
# =============================================================================

class TestTaskQueue:
    """Tests for TaskQueue."""

    def test_task_queue_creation(self):
        """Test TaskQueue creation."""
        queue = TaskQueue(max_size=100)
        assert queue.size() == 0

    def test_task_queue_put_get(self):
        """Test putting and getting tasks."""
        queue = TaskQueue()
        task = DistributedTask(
            id="task_1",
            name="test_task",
            function=lambda: None,
        )
        assert queue.put(task) is True
        assert queue.size() == 1
        retrieved = queue.get()
        assert retrieved.id == "task_1"
        assert queue.size() == 0

    def test_task_queue_priority(self):
        """Test priority ordering."""
        queue = TaskQueue()
        low_priority = DistributedTask(id="low", name="low", function=lambda: None, priority=0)
        high_priority = DistributedTask(id="high", name="high", function=lambda: None, priority=10)

        queue.put(low_priority)
        queue.put(high_priority)

        # High priority should come first
        first = queue.get()
        assert first.id == "high"

    def test_task_queue_max_size(self):
        """Test max size enforcement."""
        queue = TaskQueue(max_size=2)
        task1 = DistributedTask(id="1", name="t1", function=lambda: None)
        task2 = DistributedTask(id="2", name="t2", function=lambda: None)
        task3 = DistributedTask(id="3", name="t3", function=lambda: None)

        assert queue.put(task1) is True
        assert queue.put(task2) is True
        assert queue.put(task3) is False  # Should fail

    def test_task_queue_peek(self):
        """Test peeking without removal."""
        queue = TaskQueue()
        task = DistributedTask(id="task_1", name="test", function=lambda: None)
        queue.put(task)

        peeked = queue.peek()
        assert peeked.id == "task_1"
        assert queue.size() == 1  # Still there


class TestLoadBalancer:
    """Tests for LoadBalancer."""

    def test_load_balancer_creation(self):
        """Test LoadBalancer creation."""
        balancer = LoadBalancer()
        assert balancer.get_worker_loads() == {}

    def test_load_balancer_update(self):
        """Test updating worker loads."""
        balancer = LoadBalancer()
        balancer.update_load("worker_1", 0.5)
        balancer.update_load("worker_2", 0.3)

        loads = balancer.get_worker_loads()
        assert loads["worker_1"] == 0.5
        assert loads["worker_2"] == 0.3

    def test_load_balancer_select_worker(self):
        """Test selecting best worker."""
        balancer = LoadBalancer()
        balancer.update_load("worker_1", 0.8)
        balancer.update_load("worker_2", 0.2)

        selected = balancer.select_worker(["worker_1", "worker_2"])
        assert selected == "worker_2"  # Lower load

    def test_load_balancer_remove_worker(self):
        """Test removing a worker."""
        balancer = LoadBalancer()
        balancer.update_load("worker_1", 0.5)
        balancer.remove_worker("worker_1")
        assert "worker_1" not in balancer.get_worker_loads()


class TestTaskCoordinator:
    """Tests for TaskCoordinator."""

    def test_coordinator_creation(self):
        """Test TaskCoordinator creation."""
        coordinator = TaskCoordinator()
        assert coordinator.config.max_workers == 4

    def test_coordinator_custom_config(self):
        """Test with custom config."""
        config = CoordinatorConfig(max_workers=8, retry_limit=5)
        coordinator = TaskCoordinator(config)
        assert coordinator.config.max_workers == 8

    def test_coordinator_register_worker(self):
        """Test registering workers."""
        coordinator = TaskCoordinator()
        info = coordinator.register_worker("worker_1")
        assert info.id == "worker_1"
        assert info.status == "idle"

    def test_coordinator_submit_task(self):
        """Test submitting tasks."""
        coordinator = TaskCoordinator()
        task = coordinator.submit_task(
            name="test_task",
            function=lambda x: x * 2,
            args=(5,),
        )
        assert task.name == "test_task"
        assert task.status == TaskStatus.PENDING

    def test_coordinator_schedule_next(self):
        """Test scheduling tasks to workers."""
        coordinator = TaskCoordinator()
        coordinator.register_worker("worker_1")
        task = coordinator.submit_task("task", lambda: 42)

        scheduled = coordinator.schedule_next()
        assert scheduled is not None
        assert scheduled.status == TaskStatus.SCHEDULED

    def test_coordinator_complete_task(self):
        """Test completing tasks."""
        coordinator = TaskCoordinator()
        coordinator.register_worker("worker_1")
        task = coordinator.submit_task("task", lambda: 42)
        coordinator.schedule_next()

        coordinator.start_task(task.id)
        coordinator.complete_task(task.id, result=42)

        result = coordinator.get_result(task.id)
        assert result == 42

    def test_coordinator_task_failure_retry(self):
        """Test task failure and retry."""
        coordinator = TaskCoordinator()
        coordinator.register_worker("worker_1")
        task = coordinator.submit_task("task", lambda: None)
        coordinator.schedule_next()

        coordinator.complete_task(task.id, result=None, error="Test error")

        # Task should be requeued for retry
        updated_task = coordinator.get_task(task.id)
        assert updated_task.retry_count == 1

    def test_coordinator_cancel_task(self):
        """Test cancelling tasks."""
        coordinator = TaskCoordinator()
        task = coordinator.submit_task("task", lambda: 42)
        assert coordinator.cancel_task(task.id) is True
        assert coordinator.get_task(task.id).status == TaskStatus.CANCELLED

    def test_coordinator_worker_stats(self):
        """Test getting worker stats."""
        coordinator = TaskCoordinator()
        coordinator.register_worker("worker_1")
        coordinator.register_worker("worker_2")

        stats = coordinator.get_worker_stats()
        assert stats["n_workers"] == 2

    def test_coordinator_task_stats(self):
        """Test getting task stats."""
        coordinator = TaskCoordinator()
        coordinator.register_worker("worker_1")
        coordinator.submit_task("task1", lambda: 1)
        coordinator.submit_task("task2", lambda: 2)

        stats = coordinator.get_task_stats()
        assert stats["total_tasks"] == 2


class TestWorker:
    """Tests for Worker."""

    def test_worker_creation(self):
        """Test Worker creation."""
        worker = Worker(worker_id="test_worker")
        assert worker.id == "test_worker"
        assert worker.status == WorkerStatus.IDLE

    def test_worker_execute(self):
        """Test executing a task."""
        worker = Worker()
        result = worker.execute("task_1", lambda x: x * 2, (5,))

        assert result.success is True
        assert result.result == 10

    def test_worker_execute_with_kwargs(self):
        """Test executing with keyword arguments."""
        worker = Worker()
        result = worker.execute(
            "task_1",
            lambda x, y=1: x + y,
            (5,),
            {"y": 10}
        )
        assert result.result == 15

    def test_worker_execute_failure(self):
        """Test handling execution failures."""
        worker = Worker()

        def failing_func():
            raise ValueError("Test error")

        result = worker.execute("task_1", failing_func)
        assert result.success is False
        assert "ValueError" in result.error

    def test_worker_pause_resume(self):
        """Test pausing and resuming worker."""
        worker = Worker()
        worker.pause()
        assert worker.status == WorkerStatus.PAUSED

        worker.resume()
        assert worker.status == WorkerStatus.IDLE

    def test_worker_stop(self):
        """Test stopping worker."""
        worker = Worker()
        worker.stop()
        assert worker.status == WorkerStatus.STOPPED

    def test_worker_statistics(self):
        """Test worker statistics."""
        worker = Worker(worker_id="test")
        worker.execute("t1", lambda: 1)
        worker.execute("t2", lambda: 2)

        stats = worker.statistics()
        assert stats["tasks_completed"] == 2


class TestWorkerPool:
    """Tests for WorkerPool."""

    def test_worker_pool_creation(self):
        """Test WorkerPool creation."""
        pool = WorkerPool(n_workers=4)
        assert pool.n_workers == 4

    def test_worker_pool_submit_execute(self):
        """Test submitting and executing tasks."""
        pool = WorkerPool(n_workers=2)
        pool.submit("task_1", lambda x: x * 2, (5,))
        pool.submit("task_2", lambda x: x + 1, (10,))

        results = pool.execute_all()
        assert len(results) == 2

    def test_worker_pool_map(self):
        """Test mapping over items."""
        pool = WorkerPool(n_workers=2)
        results = pool.map(lambda x: x * 2, [1, 2, 3, 4])

        assert len(results) == 4
        values = [r.result for r in results]
        assert sorted(values) == [2, 4, 6, 8]

    def test_worker_pool_get_idle_workers(self):
        """Test getting idle workers."""
        pool = WorkerPool(n_workers=3)
        idle = pool.get_idle_workers()
        assert len(idle) == 3

    def test_worker_pool_shutdown(self):
        """Test pool shutdown."""
        pool = WorkerPool(n_workers=2)
        pool.shutdown()
        for worker in pool._workers.values():
            assert worker.status == WorkerStatus.STOPPED

    def test_worker_pool_statistics(self):
        """Test pool statistics."""
        pool = WorkerPool(n_workers=2)
        pool.submit("t1", lambda: 1)
        pool.execute_all()

        stats = pool.statistics()
        assert stats["n_workers"] == 2
        assert stats["completed_tasks"] == 1


class TestWeightedAverage:
    """Tests for WeightedAverage aggregator."""

    def test_weighted_average_creation(self):
        """Test WeightedAverage creation."""
        agg = WeightedAverage()
        assert agg.normalize is True

    def test_weighted_average_aggregate(self):
        """Test weighted averaging."""
        agg = WeightedAverage()
        results = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        aggregated = agg.aggregate(results)
        assert aggregated.n_inputs == 2
        np.testing.assert_array_almost_equal(aggregated.value, [2.0, 3.0])

    def test_weighted_average_with_weights(self):
        """Test with explicit weights."""
        agg = WeightedAverage()
        results = [np.array([0.0]), np.array([10.0])]
        weights = [1.0, 9.0]

        aggregated = agg.aggregate(results, weights)
        np.testing.assert_array_almost_equal(aggregated.value, [9.0])

    def test_weighted_average_empty(self):
        """Test with empty input."""
        agg = WeightedAverage()
        aggregated = agg.aggregate([])
        assert aggregated.n_inputs == 0


class TestVoting:
    """Tests for Voting aggregator."""

    def test_voting_creation(self):
        """Test Voting creation."""
        voter = Voting()
        assert voter.min_agreement == 0.5

    def test_voting_aggregate_unanimous(self):
        """Test unanimous voting."""
        voter = Voting()
        results = ["yes", "yes", "yes"]

        aggregated = voter.aggregate(results)
        assert aggregated.value == "yes"
        assert aggregated.confidence == 1.0

    def test_voting_aggregate_majority(self):
        """Test majority voting."""
        voter = Voting()
        results = ["A", "B", "A", "A"]

        aggregated = voter.aggregate(results)
        assert aggregated.value == "A"
        assert aggregated.confidence == 0.75

    def test_voting_with_weights(self):
        """Test weighted voting."""
        voter = Voting()
        results = ["A", "B"]
        weights = [1.0, 2.0]

        aggregated = voter.aggregate(results, weights)
        assert aggregated.value == "B"


class TestConcatAggregator:
    """Tests for ConcatAggregator."""

    def test_concat_arrays(self):
        """Test concatenating arrays."""
        agg = ConcatAggregator(axis=0)
        results = [np.array([1, 2]), np.array([3, 4])]

        aggregated = agg.aggregate(results)
        np.testing.assert_array_equal(aggregated.value, [1, 2, 3, 4])

    def test_concat_lists(self):
        """Test concatenating lists."""
        agg = ConcatAggregator()
        results = [[1, 2], [3, 4]]

        aggregated = agg.aggregate(results)
        # ConcatAggregator tries numpy first, then falls back to list concat
        np.testing.assert_array_equal(aggregated.value, [1, 2, 3, 4])


class TestResultAggregator:
    """Tests for ResultAggregator."""

    def test_result_aggregator_creation(self):
        """Test ResultAggregator creation."""
        agg = ResultAggregator()
        assert agg.default_strategy == AggregationStrategy.MEAN

    def test_result_aggregator_mean(self):
        """Test mean aggregation."""
        agg = ResultAggregator()
        results = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        aggregated = agg.aggregate(results, AggregationStrategy.MEAN)
        np.testing.assert_array_almost_equal(aggregated.value, [2.0, 3.0])

    def test_result_aggregator_first(self):
        """Test first aggregation."""
        agg = ResultAggregator()
        results = [1, 2, 3]

        aggregated = agg.aggregate(results, AggregationStrategy.FIRST)
        assert aggregated.value == 1

    def test_result_aggregator_last(self):
        """Test last aggregation."""
        agg = ResultAggregator()
        results = [1, 2, 3]

        aggregated = agg.aggregate(results, AggregationStrategy.LAST)
        assert aggregated.value == 3

    def test_result_aggregator_custom(self):
        """Test custom aggregation."""
        agg = ResultAggregator()
        agg.register_custom("product", lambda x: np.prod(x))

        aggregated = agg.aggregate(
            [2, 3, 4],
            AggregationStrategy.CUSTOM,
            custom_name="product"
        )
        assert aggregated.value == 24

    def test_result_aggregator_dict(self):
        """Test dictionary aggregation."""
        agg = ResultAggregator()
        results = [
            {"accuracy": 0.9, "loss": 0.1},
            {"accuracy": 0.8, "loss": 0.2},
        ]

        aggregated = agg.aggregate_dict(results)
        assert "accuracy" in aggregated
        assert "loss" in aggregated


# =============================================================================
# GPU KERNEL TESTS
# =============================================================================

class TestMatMulKernel:
    """Tests for MatMulKernel."""

    def test_matmul_kernel_creation(self):
        """Test MatMulKernel creation."""
        kernel = MatMulKernel(tile_size=32)
        assert kernel.tile_size == 32

    def test_matmul_execute(self):
        """Test matrix multiplication."""
        kernel = MatMulKernel()
        A = np.random.randn(64, 32).astype(np.float32)
        B = np.random.randn(32, 16).astype(np.float32)

        result = kernel.execute(A, B)

        expected = A @ B
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_matmul_stats_recorded(self):
        """Test that stats are recorded."""
        kernel = MatMulKernel()
        A = np.random.randn(16, 16).astype(np.float32)
        B = np.random.randn(16, 16).astype(np.float32)

        kernel.execute(A, B)
        stats = kernel.get_stats()
        assert len(stats) == 1
        assert stats[0].kernel_name == "matmul"

    def test_matmul_shape_validation(self):
        """Test shape validation."""
        kernel = MatMulKernel()
        A = np.random.randn(10, 5).astype(np.float32)
        B = np.random.randn(8, 4).astype(np.float32)  # Incompatible

        with pytest.raises(ValueError):
            kernel.execute(A, B)


class TestConvolutionKernel:
    """Tests for ConvolutionKernel."""

    def test_conv_kernel_creation(self):
        """Test ConvolutionKernel creation."""
        kernel = ConvolutionKernel(padding="same")
        assert kernel.padding == "same"

    def test_conv_2d(self):
        """Test 2D convolution."""
        kernel = ConvolutionKernel()
        input = np.ones((8, 8))
        conv_filter = np.ones((3, 3)) / 9.0

        result = kernel.execute(input, conv_filter)
        assert result.shape == (8, 8)

    def test_conv_3d_input(self):
        """Test convolution with channel dimension."""
        kernel = ConvolutionKernel()
        input = np.ones((3, 8, 8))  # 3 channels
        conv_filter = np.ones((3, 3)) / 9.0

        result = kernel.execute(input, conv_filter)
        assert result.shape == (3, 8, 8)


class TestSoftmaxKernel:
    """Tests for SoftmaxKernel."""

    def test_softmax_kernel_creation(self):
        """Test SoftmaxKernel creation."""
        kernel = SoftmaxKernel(axis=-1)
        assert kernel.axis == -1

    def test_softmax_probabilities(self):
        """Test softmax produces valid probabilities."""
        kernel = SoftmaxKernel()
        x = np.random.randn(5, 10)

        result = kernel.execute(x)

        # Should sum to 1 along last axis
        np.testing.assert_array_almost_equal(
            result.sum(axis=-1), np.ones(5)
        )

    def test_softmax_numerical_stability(self):
        """Test numerical stability with large values."""
        kernel = SoftmaxKernel()
        x = np.array([1000.0, 1001.0, 1002.0])

        result = kernel.execute(x)

        # Should not produce NaN or Inf
        assert np.all(np.isfinite(result))
        np.testing.assert_almost_equal(result.sum(), 1.0)


class TestReductionKernel:
    """Tests for ReductionKernel."""

    def test_reduction_sum(self):
        """Test sum reduction."""
        kernel = ReductionKernel("sum")
        x = np.array([1, 2, 3, 4, 5])

        result = kernel.execute(x)
        assert result == 15

    def test_reduction_mean(self):
        """Test mean reduction."""
        kernel = ReductionKernel("mean")
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result = kernel.execute(x)
        assert result == 3.0

    def test_reduction_max(self):
        """Test max reduction."""
        kernel = ReductionKernel("max")
        x = np.array([1, 5, 3, 2, 4])

        result = kernel.execute(x)
        assert result == 5

    def test_reduction_with_axis(self):
        """Test reduction along axis."""
        kernel = ReductionKernel("sum")
        x = np.array([[1, 2], [3, 4]])

        result = kernel.execute(x, axis=1)
        np.testing.assert_array_equal(result, [3, 7])


class TestKernelManager:
    """Tests for KernelManager."""

    def test_kernel_manager_creation(self):
        """Test KernelManager creation."""
        manager = KernelManager()
        assert "matmul" in manager._kernels
        assert "softmax" in manager._kernels

    def test_kernel_manager_execute(self):
        """Test executing kernels by name."""
        manager = KernelManager()
        A = np.eye(4)
        B = np.ones((4, 4))

        result = manager.execute("matmul", A, B)
        np.testing.assert_array_equal(result, B)

    def test_kernel_manager_register(self):
        """Test registering custom kernel."""
        manager = KernelManager()
        kernel = ReductionKernel("min")
        manager.register("reduce_min", kernel)

        assert manager.get("reduce_min") is not None

    def test_kernel_manager_statistics(self):
        """Test statistics."""
        manager = KernelManager()
        stats = manager.statistics()
        assert stats["n_kernels"] == 6


# =============================================================================
# BATCH INFERENCE TESTS
# =============================================================================

class TestInferenceCache:
    """Tests for InferenceCache."""

    def test_cache_creation(self):
        """Test cache creation."""
        cache = InferenceCache(max_size=100)
        assert cache.max_size == 100
        assert cache.size == 0

    def test_cache_put_get(self):
        """Test putting and getting from cache."""
        cache = InferenceCache()
        x = np.array([1, 2, 3])
        result = np.array([2, 4, 6])

        cache.put(x, result)
        retrieved = cache.get(x)

        np.testing.assert_array_equal(retrieved, result)

    def test_cache_miss(self):
        """Test cache miss."""
        cache = InferenceCache()
        x = np.array([1, 2, 3])

        retrieved = cache.get(x)
        assert retrieved is None

    def test_cache_eviction(self):
        """Test LRU eviction."""
        cache = InferenceCache(max_size=2)
        cache.put(np.array([1]), np.array([1]))
        cache.put(np.array([2]), np.array([2]))
        cache.put(np.array([3]), np.array([3]))

        # First entry should be evicted
        assert cache.size == 2


class TestBatchBuilder:
    """Tests for BatchBuilder."""

    def test_batch_builder_creation(self):
        """Test BatchBuilder creation."""
        builder = BatchBuilder(batch_size=16)
        assert builder.batch_size == 16

    def test_batch_builder_add(self):
        """Test adding requests."""
        builder = BatchBuilder(batch_size=3)
        builder.add("req_1", np.array([1]))
        builder.add("req_2", np.array([2]))

        assert builder.pending_count == 2

    def test_batch_builder_auto_flush(self):
        """Test automatic flushing when full."""
        builder = BatchBuilder(batch_size=2)
        builder.add("req_1", np.array([1]))
        batch = builder.add("req_2", np.array([2]))

        assert batch is not None
        assert batch.size == 2

    def test_batch_builder_manual_flush(self):
        """Test manual flushing."""
        builder = BatchBuilder(batch_size=10)
        builder.add("req_1", np.array([1]))

        batch = builder.flush()
        assert batch is not None
        assert batch.size == 1


class TestBatchInference:
    """Tests for BatchInference."""

    def test_batch_inference_creation(self):
        """Test BatchInference creation."""
        def model_fn(x):
            return x * 2

        batch_inf = BatchInference(model_fn)
        assert batch_inf.config.batch_size == 32

    def test_batch_inference_single(self):
        """Test single inference."""
        def model_fn(x):
            return x * 2

        batch_inf = BatchInference(model_fn)
        result = batch_inf.infer_single(np.array([1, 2, 3]))

        np.testing.assert_array_equal(result, [2, 4, 6])

    def test_batch_inference_batch(self):
        """Test batch inference."""
        def model_fn(x):
            return x * 2

        batch_inf = BatchInference(model_fn)
        inputs = [np.array([1]), np.array([2]), np.array([3])]
        results = batch_inf.infer_batch(inputs)

        assert len(results) == 3

    def test_batch_inference_caching(self):
        """Test inference caching."""
        call_count = [0]

        def model_fn(x):
            call_count[0] += 1
            return x * 2

        batch_inf = BatchInference(model_fn, BatchConfig(cache_enabled=True))

        x = np.array([1, 2, 3])
        batch_inf.infer_single(x)
        batch_inf.infer_single(x)  # Should use cache

        # Model should only be called once due to caching
        assert call_count[0] == 1

    def test_batch_inference_statistics(self):
        """Test statistics collection."""
        def model_fn(x):
            return x * 2

        batch_inf = BatchInference(model_fn)
        batch_inf.infer_single(np.array([1]))

        stats = batch_inf.statistics()
        assert stats["total_requests"] == 1


# =============================================================================
# SPARSE PRUNING TESTS
# =============================================================================

class TestPruningMask:
    """Tests for PruningMask."""

    def test_mask_creation(self):
        """Test mask creation."""
        mask = PruningMask((10, 10))
        assert mask.sparsity == 0.0  # All ones

    def test_mask_apply(self):
        """Test applying mask."""
        mask = PruningMask((4,))
        mask.update(np.array([True, True, False, False]))

        weights = np.array([1.0, 2.0, 3.0, 4.0])
        pruned = mask.apply(weights)

        np.testing.assert_array_equal(pruned, [1.0, 2.0, 0.0, 0.0])

    def test_mask_sparsity(self):
        """Test sparsity calculation."""
        mask = PruningMask((4,))
        mask.update(np.array([True, True, False, False]))

        assert mask.sparsity == 0.5


class TestMagnitudePruning:
    """Tests for MagnitudePruning."""

    def test_magnitude_pruning_creation(self):
        """Test MagnitudePruning creation."""
        pruner = MagnitudePruning()
        assert pruner is not None

    def test_magnitude_pruning_mask(self):
        """Test magnitude-based mask computation."""
        pruner = MagnitudePruning()
        weights = np.array([0.1, 0.5, 0.2, 0.8, 0.3])

        mask = pruner.compute_mask(weights, sparsity=0.4)

        # 40% should be pruned (smallest magnitudes)
        assert mask.sum() >= 3  # At least 60% kept

    def test_magnitude_pruning_zero_sparsity(self):
        """Test with zero sparsity."""
        pruner = MagnitudePruning()
        weights = np.random.randn(10)

        mask = pruner.compute_mask(weights, sparsity=0.0)
        assert mask.all()  # All ones

    def test_magnitude_pruning_full_sparsity(self):
        """Test with full sparsity."""
        pruner = MagnitudePruning()
        weights = np.random.randn(10)

        mask = pruner.compute_mask(weights, sparsity=1.0)
        assert not mask.any()  # All zeros


class TestRandomPruning:
    """Tests for RandomPruning."""

    def test_random_pruning_creation(self):
        """Test RandomPruning creation."""
        pruner = RandomPruning(seed=42)
        assert pruner is not None

    def test_random_pruning_mask(self):
        """Test random mask computation."""
        pruner = RandomPruning(seed=42)
        weights = np.ones((100,))

        mask = pruner.compute_mask(weights, sparsity=0.5)

        # Should be approximately 50% pruned
        actual_sparsity = 1.0 - mask.mean()
        assert 0.3 < actual_sparsity < 0.7


class TestStructuredPruning:
    """Tests for StructuredPruning."""

    def test_structured_pruning_creation(self):
        """Test StructuredPruning creation."""
        pruner = StructuredPruning(axis=0)
        assert pruner.axis == 0

    def test_structured_pruning_mask(self):
        """Test structured mask computation."""
        pruner = StructuredPruning(axis=0)
        weights = np.array([
            [0.1, 0.1, 0.1],
            [0.5, 0.5, 0.5],
            [0.2, 0.2, 0.2],
            [0.8, 0.8, 0.8],
        ])

        mask = pruner.compute_mask(weights, sparsity=0.5)

        # Entire channels should be pruned/kept
        for i in range(4):
            assert mask[i].all() or not mask[i].any()


class TestNetworkPruner:
    """Tests for NetworkPruner."""

    def test_network_pruner_creation(self):
        """Test NetworkPruner creation."""
        pruner = NetworkPruner()
        assert pruner.config.target_sparsity == 0.5

    def test_network_pruner_prune(self):
        """Test pruning a network."""
        pruner = NetworkPruner(PruningConfig(
            strategy=PruningStrategy.MAGNITUDE
        ))

        weights = {
            "layer1": np.random.randn(100),
            "layer2": np.random.randn(200),
        }

        pruned_weights, result = pruner.prune(weights, sparsity=0.3)

        assert len(pruned_weights) == 2
        assert result.sparsity > 0

    def test_network_pruner_iterative(self):
        """Test iterative pruning."""
        pruner = NetworkPruner()
        weights = {"layer1": np.random.randn(100)}

        final_weights, results = pruner.prune_iterative(
            weights,
            target_sparsity=0.5,
            n_iterations=5,
        )

        assert len(results) == 5
        # Sparsity should increase
        assert results[-1].sparsity > results[0].sparsity

    def test_network_pruner_statistics(self):
        """Test statistics."""
        pruner = NetworkPruner()
        weights = {"layer1": np.random.randn(100)}
        pruner.prune(weights)

        stats = pruner.statistics()
        assert "strategy" in stats


# =============================================================================
# QUANTIZATION TESTS
# =============================================================================

class TestQuantizer:
    """Tests for Quantizer."""

    def test_quantizer_creation(self):
        """Test Quantizer creation."""
        quantizer = Quantizer()
        assert quantizer.config.level == QuantizationLevel.INT8

    def test_quantizer_int8(self):
        """Test INT8 quantization."""
        quantizer = Quantizer()
        tensor = np.random.randn(100).astype(np.float32)

        qtensor = quantizer.quantize(tensor, QuantizationLevel.INT8)

        assert qtensor.level == QuantizationLevel.INT8
        assert qtensor.data.dtype == np.int8

    def test_quantizer_fp16(self):
        """Test FP16 quantization."""
        quantizer = Quantizer()
        tensor = np.random.randn(100).astype(np.float32)

        qtensor = quantizer.quantize(tensor, QuantizationLevel.FP16)

        assert qtensor.level == QuantizationLevel.FP16
        assert qtensor.data.dtype == np.float16

    def test_quantizer_dequantize(self):
        """Test dequantization."""
        quantizer = Quantizer()
        tensor = np.random.randn(100).astype(np.float32)

        qtensor = quantizer.quantize(tensor)
        recovered = quantizer.dequantize(qtensor)

        # Should be close to original
        assert recovered.shape == tensor.shape

    def test_quantizer_model(self):
        """Test model quantization."""
        quantizer = Quantizer()
        weights = {
            "layer1": np.random.randn(100, 50).astype(np.float32),
            "layer2": np.random.randn(50, 10).astype(np.float32),
        }

        qmodel = quantizer.quantize_model(weights)

        assert len(qmodel.weights) == 2
        assert qmodel.memory_reduction > 0

    def test_quantizer_error_metrics(self):
        """Test quantization error computation."""
        quantizer = Quantizer()
        tensor = np.random.randn(100).astype(np.float32)
        qtensor = quantizer.quantize(tensor)

        error = quantizer.compute_quantization_error(tensor, qtensor)

        assert "mse" in error
        assert "mae" in error
        assert "snr_db" in error


class TestQuantizedTensor:
    """Tests for QuantizedTensor."""

    def test_quantized_tensor_dequantize(self):
        """Test QuantizedTensor dequantization."""
        qtensor = QuantizedTensor(
            data=np.array([0, 64, 127], dtype=np.int8),
            scale=np.array([0.01]),
            zero_point=np.array([0]),
            original_dtype=np.float32,
            level=QuantizationLevel.INT8,
        )

        result = qtensor.dequantize()
        expected = np.array([0.0, 0.64, 1.27])
        np.testing.assert_array_almost_equal(result, expected, decimal=2)


# =============================================================================
# EFFICIENCY TESTS
# =============================================================================

class TestProfiler:
    """Tests for Profiler."""

    def test_profiler_creation(self):
        """Test Profiler creation."""
        profiler = Profiler()
        assert profiler is not None

    def test_profiler_profile(self):
        """Test profiling an operation."""
        profiler = Profiler()

        def slow_func():
            return sum(range(1000))

        profile = profiler.profile("sum", slow_func, n_runs=5, warmup_runs=1)

        assert profile.operation == "sum"
        assert profile.n_samples == 5
        assert profile.mean_latency_ms > 0

    def test_profiler_get_profile(self):
        """Test getting stored profile."""
        profiler = Profiler()

        profiler.profile("test", lambda: 42, n_runs=3)
        profile = profiler.get_profile("test")

        assert profile is not None
        assert profile.n_samples == 3

    def test_profiler_clear(self):
        """Test clearing profiles."""
        profiler = Profiler()
        profiler.profile("test", lambda: 42)
        profiler.clear()

        assert profiler.get_profile("test") is None

    def test_profiler_summary(self):
        """Test summary."""
        profiler = Profiler()
        profiler.profile("op1", lambda: 1)
        profiler.profile("op2", lambda: 2)

        summary = profiler.summary()
        assert "op1" in summary
        assert "op2" in summary


class TestOperationFuser:
    """Tests for OperationFuser."""

    def test_fuser_creation(self):
        """Test OperationFuser creation."""
        fuser = OperationFuser()
        assert len(fuser._fusion_rules) > 0

    def test_fuser_can_fuse(self):
        """Test checking fusability."""
        fuser = OperationFuser()
        assert fuser.can_fuse("matmul", "add") is True
        assert fuser.can_fuse("add", "matmul") is False

    def test_fuser_fuse(self):
        """Test getting fused operation."""
        fuser = OperationFuser()
        fused = fuser.fuse("matmul", "relu")
        assert fused == "matmul_relu"

    def test_fuser_optimize_graph(self):
        """Test graph optimization."""
        fuser = OperationFuser()
        operations = ["matmul", "add", "relu", "conv2d", "relu"]

        optimized = fuser.optimize_graph(operations)

        # Should have fewer operations due to fusion
        assert len(optimized) < len(operations)


class TestMemoryOptimizer:
    """Tests for MemoryOptimizer."""

    def test_memory_optimizer_creation(self):
        """Test MemoryOptimizer creation."""
        optimizer = MemoryOptimizer()
        assert optimizer is not None

    def test_memory_optimizer_estimate(self):
        """Test memory estimation."""
        optimizer = MemoryOptimizer()
        shapes = [(100, 100), (200, 200)]

        # Pass an actual dtype instance
        memory_mb = optimizer.estimate_memory(shapes, dtype=np.dtype(np.float32))
        assert memory_mb > 0

    def test_memory_optimizer_peak_memory(self):
        """Test peak memory computation."""
        optimizer = MemoryOptimizer()
        lifetimes = {
            "tensor1": (0, 2),
            "tensor2": (1, 3),
            "tensor3": (2, 4),
        }
        sizes = {
            "tensor1": 1000,
            "tensor2": 2000,
            "tensor3": 1500,
        }

        peak = optimizer.compute_peak_memory(lifetimes, sizes, n_ops=5)
        assert peak > 0


class TestEfficiencyOptimizer:
    """Tests for EfficiencyOptimizer."""

    def test_efficiency_optimizer_creation(self):
        """Test EfficiencyOptimizer creation."""
        optimizer = EfficiencyOptimizer()
        assert optimizer.config is not None

    def test_efficiency_optimizer_profile_model(self):
        """Test model profiling."""
        optimizer = EfficiencyOptimizer()

        def forward_fn(x):
            return x * 2

        profile = optimizer.profile_model(forward_fn, np.array([1, 2, 3]))
        assert profile.operation == "forward"

    def test_efficiency_optimizer_optimize(self):
        """Test optimization."""
        optimizer = EfficiencyOptimizer()
        operations = ["matmul", "add", "matmul", "relu"]

        optimized, report = optimizer.optimize(operations)

        assert "original_ops" in report
        assert "optimized_ops" in report

    def test_efficiency_optimizer_check_targets(self):
        """Test target checking."""
        config = OptimizationConfig(
            target_latency_ms=10.0,
            target_throughput=1000.0,
        )
        optimizer = EfficiencyOptimizer(config)

        results = optimizer.check_targets(latency_ms=5.0, throughput=2000.0)

        assert results.get("latency_target") is True
        assert results.get("throughput_target") is True

    def test_efficiency_optimizer_recommendations(self):
        """Test optimization recommendations."""
        config = OptimizationConfig(
            target_latency_ms=1.0,
            max_memory_mb=100.0,
        )
        optimizer = EfficiencyOptimizer(config)

        profile = LatencyProfile(
            operation="test",
            mean_latency_ms=10.0,
            std_latency_ms=5.0,
            min_latency_ms=5.0,
            max_latency_ms=15.0,
            p50_latency_ms=10.0,
            p99_latency_ms=14.0,
            n_samples=10,
        )

        recommendations = optimizer.recommend_optimizations(profile, memory_mb=200.0)
        assert len(recommendations) > 0

    def test_efficiency_optimizer_statistics(self):
        """Test statistics."""
        optimizer = EfficiencyOptimizer()
        stats = optimizer.statistics()
        assert "n_optimizations_applied" in stats


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for neuro-scale."""

    def test_distributed_workflow(self):
        """Test complete distributed workflow."""
        # Create coordinator
        coordinator = TaskCoordinator()

        # Register workers
        for i in range(3):
            coordinator.register_worker(f"worker_{i}")

        # Submit tasks
        def compute(x):
            return x * 2

        tasks = []
        for i in range(5):
            task = coordinator.submit_task(
                f"compute_{i}",
                compute,
                args=(i,),
            )
            tasks.append(task)

        # Schedule and complete
        for _ in range(5):
            scheduled = coordinator.schedule_next()
            if scheduled:
                coordinator.start_task(scheduled.id)
                result = scheduled.function(*scheduled.args)
                coordinator.complete_task(scheduled.id, result)

        # Aggregate results
        aggregator = ResultAggregator()
        results = [coordinator.get_result(t.id) for t in tasks]
        aggregated = aggregator.aggregate(results, AggregationStrategy.MEAN)

        assert aggregated.n_inputs == 5

    def test_gpu_inference_pipeline(self):
        """Test GPU inference pipeline."""
        # Create kernel manager
        manager = KernelManager()

        # Create batch inference
        def model_fn(x):
            # Softmax classifier
            kernel = manager.get("softmax")
            return kernel.execute(x)

        batch_inf = BatchInference(model_fn)

        # Run batched inference
        inputs = [np.random.randn(10) for _ in range(5)]
        results = batch_inf.infer_batch(inputs)

        assert len(results) == 5
        for r in results:
            np.testing.assert_almost_equal(r.sum(), 1.0)

    def test_model_compression_pipeline(self):
        """Test model compression pipeline."""
        # Create model weights
        weights = {
            "conv1": np.random.randn(64, 3, 3, 3).astype(np.float32),
            "fc1": np.random.randn(1000, 512).astype(np.float32),
        }

        # Prune
        pruner = NetworkPruner(PruningConfig(target_sparsity=0.3))
        pruned_weights, prune_result = pruner.prune(weights)

        # Quantize
        quantizer = Quantizer(QuantizationConfig(level=QuantizationLevel.INT8))
        qmodel = quantizer.quantize_model(pruned_weights)

        # Check compression
        assert prune_result.sparsity > 0
        assert qmodel.memory_reduction > 0

    def test_efficiency_analysis(self):
        """Test efficiency analysis workflow."""
        optimizer = EfficiencyOptimizer(OptimizationConfig(
            target_latency_ms=5.0
        ))

        # Profile operation - use a function that accepts input_data
        def matrix_op(x):
            A = np.random.randn(100, 100)
            B = np.random.randn(100, 100)
            return A @ B

        profile = optimizer.profile_model(matrix_op, np.array([1]), n_runs=5)

        # Check targets
        targets = optimizer.check_targets(latency_ms=profile.mean_latency_ms)

        # Get recommendations
        recommendations = optimizer.recommend_optimizations(profile, memory_mb=10.0)

        # Get statistics
        stats = optimizer.statistics()

        assert profile.n_samples == 5
        assert "n_optimizations_applied" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
