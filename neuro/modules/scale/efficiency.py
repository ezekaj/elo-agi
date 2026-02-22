"""
Efficiency: Latency and throughput optimization.

Implements profiling, optimization, and efficiency analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
import time
import numpy as np


@dataclass
class LatencyProfile:
    """Profile of operation latencies."""

    operation: str
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    n_samples: int


@dataclass
class ThroughputMetrics:
    """Throughput metrics."""

    operation: str
    samples_per_second: float
    tokens_per_second: Optional[float] = None
    batches_per_second: float = 0.0
    gpu_utilization: float = 0.0


@dataclass
class MemoryProfile:
    """Memory usage profile."""

    peak_memory_mb: float
    average_memory_mb: float
    memory_by_layer: Dict[str, float] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """Configuration for optimization."""

    target_latency_ms: Optional[float] = None
    target_throughput: Optional[float] = None
    max_memory_mb: Optional[float] = None
    optimization_passes: List[str] = field(
        default_factory=lambda: [
            "fuse_operations",
            "optimize_memory",
            "parallelize",
        ]
    )


class Profiler:
    """
    Profile operation performance.
    """

    def __init__(self):
        self._latencies: Dict[str, List[float]] = {}
        self._memory_samples: List[float] = []

    def profile(
        self,
        operation: str,
        func: Callable,
        *args,
        n_runs: int = 10,
        warmup_runs: int = 3,
        **kwargs,
    ) -> LatencyProfile:
        """
        Profile an operation.

        Args:
            operation: Operation name
            func: Function to profile
            n_runs: Number of profiling runs
            warmup_runs: Number of warmup runs

        Returns:
            LatencyProfile
        """
        # Warmup
        for _ in range(warmup_runs):
            func(*args, **kwargs)

        # Profile
        latencies = []
        for _ in range(n_runs):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

        latencies = np.array(latencies)

        if operation not in self._latencies:
            self._latencies[operation] = []
        self._latencies[operation].extend(latencies)

        return LatencyProfile(
            operation=operation,
            mean_latency_ms=float(np.mean(latencies)),
            std_latency_ms=float(np.std(latencies)),
            min_latency_ms=float(np.min(latencies)),
            max_latency_ms=float(np.max(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            n_samples=n_runs,
        )

    def measure_throughput(
        self,
        operation: str,
        func: Callable,
        batch_size: int,
        duration_seconds: float = 5.0,
    ) -> ThroughputMetrics:
        """
        Measure throughput.

        Args:
            operation: Operation name
            func: Function to measure
            batch_size: Batch size used
            duration_seconds: Measurement duration

        Returns:
            ThroughputMetrics
        """
        n_batches = 0
        n_samples = 0
        start = time.perf_counter()

        while time.perf_counter() - start < duration_seconds:
            func()
            n_batches += 1
            n_samples += batch_size

        elapsed = time.perf_counter() - start

        return ThroughputMetrics(
            operation=operation,
            samples_per_second=n_samples / elapsed,
            batches_per_second=n_batches / elapsed,
        )

    def get_profile(self, operation: str) -> Optional[LatencyProfile]:
        """Get profile for an operation."""
        latencies = self._latencies.get(operation)
        if not latencies:
            return None

        latencies = np.array(latencies)
        return LatencyProfile(
            operation=operation,
            mean_latency_ms=float(np.mean(latencies)),
            std_latency_ms=float(np.std(latencies)),
            min_latency_ms=float(np.min(latencies)),
            max_latency_ms=float(np.max(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            n_samples=len(latencies),
        )

    def clear(self) -> None:
        """Clear profiling data."""
        self._latencies.clear()
        self._memory_samples.clear()

    def summary(self) -> Dict[str, LatencyProfile]:
        """Get summary of all profiles."""
        return {op: self.get_profile(op) for op in self._latencies}


class OperationFuser:
    """
    Fuse operations for efficiency.
    """

    def __init__(self):
        self._fusion_rules: Dict[Tuple[str, str], str] = {
            ("matmul", "add"): "matmul_bias",
            ("matmul", "relu"): "matmul_relu",
            ("conv2d", "batchnorm"): "conv2d_bn",
            ("conv2d", "relu"): "conv2d_relu",
        }

    def can_fuse(self, op1: str, op2: str) -> bool:
        """Check if two operations can be fused."""
        return (op1, op2) in self._fusion_rules

    def fuse(self, op1: str, op2: str) -> Optional[str]:
        """Get fused operation name."""
        return self._fusion_rules.get((op1, op2))

    def optimize_graph(
        self,
        operations: List[str],
    ) -> List[str]:
        """
        Optimize a sequence of operations by fusion.

        Args:
            operations: List of operation names

        Returns:
            Optimized list of operations
        """
        if len(operations) < 2:
            return operations

        optimized = []
        i = 0

        while i < len(operations):
            if i < len(operations) - 1:
                fused = self.fuse(operations[i], operations[i + 1])
                if fused:
                    optimized.append(fused)
                    i += 2
                    continue

            optimized.append(operations[i])
            i += 1

        return optimized


class MemoryOptimizer:
    """
    Optimize memory usage.
    """

    def __init__(self):
        self._memory_estimates: Dict[str, float] = {}

    def estimate_memory(
        self,
        tensor_shapes: List[Tuple[int, ...]],
        dtype: np.dtype = None,
    ) -> float:
        """Estimate memory for tensors."""
        if dtype is None:
            dtype = np.dtype(np.float32)
        total_bytes = 0
        for shape in tensor_shapes:
            total_bytes += np.prod(shape) * dtype.itemsize
        return total_bytes / (1024 * 1024)  # MB

    def optimize_allocation(
        self,
        tensor_lifetimes: Dict[str, Tuple[int, int]],
        tensor_sizes: Dict[str, int],
    ) -> Dict[str, int]:
        """
        Optimize memory allocation by reusing buffers.

        Args:
            tensor_lifetimes: tensor -> (start_op, end_op)
            tensor_sizes: tensor -> size in bytes

        Returns:
            tensor -> buffer offset
        """
        # Sort by lifetime start
        sorted_tensors = sorted(tensor_lifetimes.keys(), key=lambda t: tensor_lifetimes[t][0])

        allocations = {}
        free_regions: List[Tuple[int, int]] = []  # (offset, size)
        current_offset = 0

        for tensor in sorted_tensors:
            size = tensor_sizes.get(tensor, 0)
            start, end = tensor_lifetimes[tensor]

            # Try to reuse freed memory
            allocated = False
            for i, (offset, region_size) in enumerate(free_regions):
                if region_size >= size:
                    allocations[tensor] = offset
                    if region_size > size:
                        free_regions[i] = (offset + size, region_size - size)
                    else:
                        free_regions.pop(i)
                    allocated = True
                    break

            if not allocated:
                allocations[tensor] = current_offset
                current_offset += size

        return allocations

    def compute_peak_memory(
        self,
        tensor_lifetimes: Dict[str, Tuple[int, int]],
        tensor_sizes: Dict[str, int],
        n_ops: int,
    ) -> float:
        """Compute peak memory usage."""
        memory_at_op = [0] * (n_ops + 1)

        for tensor, (start, end) in tensor_lifetimes.items():
            size = tensor_sizes.get(tensor, 0)
            for op in range(start, min(end + 1, n_ops + 1)):
                memory_at_op[op] += size

        return max(memory_at_op) / (1024 * 1024)  # MB


class EfficiencyOptimizer:
    """
    Complete efficiency optimization system.

    Implements:
    - Latency profiling
    - Throughput optimization
    - Memory optimization
    - Operation fusion
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()

        self.profiler = Profiler()
        self.fuser = OperationFuser()
        self.memory_optimizer = MemoryOptimizer()

        # Optimization history
        self._optimizations_applied: List[str] = []

    def profile_model(
        self,
        forward_fn: Callable,
        input_data: Any,
        n_runs: int = 10,
    ) -> LatencyProfile:
        """Profile model forward pass."""
        return self.profiler.profile(
            "forward",
            forward_fn,
            input_data,
            n_runs=n_runs,
        )

    def optimize(
        self,
        operations: List[str],
        tensor_info: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Apply optimization passes.

        Args:
            operations: List of operations
            tensor_info: Optional tensor information

        Returns:
            Tuple of (optimized_operations, optimization_report)
        """
        optimized = operations
        report = {
            "original_ops": len(operations),
            "passes_applied": [],
        }

        for pass_name in self.config.optimization_passes:
            if pass_name == "fuse_operations":
                optimized = self.fuser.optimize_graph(optimized)
                report["passes_applied"].append("fuse_operations")
                self._optimizations_applied.append("fuse_operations")

            elif pass_name == "optimize_memory" and tensor_info:
                # Memory optimization would modify tensor allocations
                report["passes_applied"].append("optimize_memory")
                self._optimizations_applied.append("optimize_memory")

            elif pass_name == "parallelize":
                # Would identify parallel opportunities
                report["passes_applied"].append("parallelize")
                self._optimizations_applied.append("parallelize")

        report["optimized_ops"] = len(optimized)
        report["reduction"] = 1.0 - len(optimized) / max(len(operations), 1)

        return optimized, report

    def check_targets(
        self,
        latency_ms: Optional[float] = None,
        throughput: Optional[float] = None,
        memory_mb: Optional[float] = None,
    ) -> Dict[str, bool]:
        """Check if optimization targets are met."""
        results = {}

        if self.config.target_latency_ms and latency_ms:
            results["latency_target"] = latency_ms <= self.config.target_latency_ms

        if self.config.target_throughput and throughput:
            results["throughput_target"] = throughput >= self.config.target_throughput

        if self.config.max_memory_mb and memory_mb:
            results["memory_target"] = memory_mb <= self.config.max_memory_mb

        return results

    def recommend_optimizations(
        self,
        profile: LatencyProfile,
        memory_mb: float,
    ) -> List[str]:
        """Recommend optimizations based on profiling."""
        recommendations = []

        # Latency recommendations
        if (
            self.config.target_latency_ms
            and profile.mean_latency_ms > self.config.target_latency_ms
        ):
            recommendations.append("Consider operation fusion to reduce latency")
            recommendations.append("Enable batching to improve throughput")

        # Memory recommendations
        if self.config.max_memory_mb and memory_mb > self.config.max_memory_mb:
            recommendations.append("Apply pruning to reduce model size")
            recommendations.append("Use quantization to reduce memory")
            recommendations.append("Optimize tensor allocation")

        # Variance recommendations
        if profile.std_latency_ms > profile.mean_latency_ms * 0.2:
            recommendations.append(
                "High latency variance - consider warming up or using deterministic execution"
            )

        return recommendations

    def get_optimization_history(self) -> List[str]:
        """Get history of applied optimizations."""
        return self._optimizations_applied.copy()

    def statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        profiles = self.profiler.summary()

        return {
            "n_optimizations_applied": len(self._optimizations_applied),
            "optimizations": self._optimizations_applied,
            "n_profiles": len(profiles),
            "target_latency_ms": self.config.target_latency_ms,
            "target_throughput": self.config.target_throughput,
            "max_memory_mb": self.config.max_memory_mb,
        }
