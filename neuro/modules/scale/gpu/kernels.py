"""
Kernels: GPU-accelerated compute kernels.

Provides GPU-like operations using numpy (can be extended to CUDA/Metal).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class KernelConfig:
    """Configuration for GPU kernels."""

    block_size: int = 256
    grid_size: int = 1024
    shared_memory: int = 48 * 1024  # 48KB
    device: str = "cpu"  # cpu, cuda, metal


@dataclass
class KernelStats:
    """Statistics from kernel execution."""

    kernel_name: str
    execution_time: float
    memory_used: int
    flops: int


class GPUKernel(ABC):
    """
    Base class for GPU kernels.

    Simulates GPU-like parallelism using numpy.
    Can be extended for actual CUDA/Metal implementation.
    """

    def __init__(self, config: Optional[KernelConfig] = None):
        self.config = config or KernelConfig()
        self._stats: List[KernelStats] = []

    @abstractmethod
    def execute(self, *args, **kwargs) -> np.ndarray:
        """Execute the kernel."""
        pass

    def get_stats(self) -> List[KernelStats]:
        """Get execution statistics."""
        return self._stats

    def _record_stats(
        self,
        kernel_name: str,
        execution_time: float,
        memory_used: int,
        flops: int,
    ) -> None:
        """Record kernel execution stats."""
        self._stats.append(
            KernelStats(
                kernel_name=kernel_name,
                execution_time=execution_time,
                memory_used=memory_used,
                flops=flops,
            )
        )


class MatMulKernel(GPUKernel):
    """
    Matrix multiplication kernel.

    Simulates tiled matrix multiplication for GPU efficiency.
    """

    def __init__(
        self,
        tile_size: int = 32,
        config: Optional[KernelConfig] = None,
    ):
        super().__init__(config)
        self.tile_size = tile_size

    def execute(
        self,
        A: np.ndarray,
        B: np.ndarray,
    ) -> np.ndarray:
        """
        Perform matrix multiplication A @ B.

        Args:
            A: Matrix of shape (M, K)
            B: Matrix of shape (K, N)

        Returns:
            Result matrix of shape (M, N)
        """
        import time

        start = time.time()

        # Validate shapes
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError("Inputs must be 2D matrices")

        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Incompatible shapes: {A.shape} @ {B.shape}")

        M, K = A.shape
        K2, N = B.shape

        # Tiled matrix multiplication (simulated GPU)
        result = np.zeros((M, N), dtype=A.dtype)

        for i in range(0, M, self.tile_size):
            i_end = min(i + self.tile_size, M)
            for j in range(0, N, self.tile_size):
                j_end = min(j + self.tile_size, N)
                for k in range(0, K, self.tile_size):
                    k_end = min(k + self.tile_size, K)

                    # Tile computation
                    result[i:i_end, j:j_end] += A[i:i_end, k:k_end] @ B[k:k_end, j:j_end]

        execution_time = time.time() - start
        flops = 2 * M * K * N  # multiply-add

        self._record_stats(
            "matmul",
            execution_time,
            A.nbytes + B.nbytes + result.nbytes,
            flops,
        )

        return result


class ConvolutionKernel(GPUKernel):
    """
    2D convolution kernel.
    """

    def __init__(
        self,
        padding: str = "same",
        config: Optional[KernelConfig] = None,
    ):
        super().__init__(config)
        self.padding = padding

    def execute(
        self,
        input: np.ndarray,
        kernel: np.ndarray,
    ) -> np.ndarray:
        """
        Perform 2D convolution.

        Args:
            input: Input array of shape (H, W) or (C, H, W)
            kernel: Kernel of shape (KH, KW)

        Returns:
            Convolved output
        """
        import time
        from scipy.ndimage import convolve

        start = time.time()

        # Handle different input shapes
        if input.ndim == 2:
            result = convolve(input, kernel, mode="constant")
        elif input.ndim == 3:
            # Apply to each channel
            result = np.stack(
                [convolve(input[c], kernel, mode="constant") for c in range(input.shape[0])]
            )
        else:
            raise ValueError(f"Unsupported input shape: {input.shape}")

        execution_time = time.time() - start
        flops = np.prod(input.shape) * np.prod(kernel.shape)

        self._record_stats(
            "conv2d",
            execution_time,
            input.nbytes + kernel.nbytes + result.nbytes,
            flops,
        )

        return result


class SoftmaxKernel(GPUKernel):
    """
    Numerically stable softmax kernel.
    """

    def __init__(
        self,
        axis: int = -1,
        config: Optional[KernelConfig] = None,
    ):
        super().__init__(config)
        self.axis = axis

    def execute(self, x: np.ndarray) -> np.ndarray:
        """
        Compute softmax along specified axis.

        Args:
            x: Input array

        Returns:
            Softmax probabilities
        """
        import time

        start = time.time()

        # Numerically stable softmax
        x_max = np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        result = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)

        execution_time = time.time() - start

        self._record_stats(
            "softmax",
            execution_time,
            x.nbytes + result.nbytes,
            3 * x.size,  # exp, sum, div
        )

        return result


class ReductionKernel(GPUKernel):
    """
    Parallel reduction kernel.
    """

    def __init__(
        self,
        operation: str = "sum",
        config: Optional[KernelConfig] = None,
    ):
        super().__init__(config)
        self.operation = operation

    def execute(
        self,
        x: np.ndarray,
        axis: Optional[int] = None,
    ) -> np.ndarray:
        """
        Perform parallel reduction.

        Args:
            x: Input array
            axis: Axis to reduce over (None for all)

        Returns:
            Reduced result
        """
        import time

        start = time.time()

        if self.operation == "sum":
            result = np.sum(x, axis=axis)
        elif self.operation == "mean":
            result = np.mean(x, axis=axis)
        elif self.operation == "max":
            result = np.max(x, axis=axis)
        elif self.operation == "min":
            result = np.min(x, axis=axis)
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        execution_time = time.time() - start

        self._record_stats(
            f"reduce_{self.operation}",
            execution_time,
            x.nbytes,
            x.size,
        )

        return result


class KernelManager:
    """
    Manage GPU kernels and execution.
    """

    def __init__(self, config: Optional[KernelConfig] = None):
        self.config = config or KernelConfig()

        # Kernel registry
        self._kernels: Dict[str, GPUKernel] = {
            "matmul": MatMulKernel(config=config),
            "conv2d": ConvolutionKernel(config=config),
            "softmax": SoftmaxKernel(config=config),
            "reduce_sum": ReductionKernel("sum", config=config),
            "reduce_mean": ReductionKernel("mean", config=config),
            "reduce_max": ReductionKernel("max", config=config),
        }

    def register(self, name: str, kernel: GPUKernel) -> None:
        """Register a kernel."""
        self._kernels[name] = kernel

    def get(self, name: str) -> Optional[GPUKernel]:
        """Get a kernel by name."""
        return self._kernels.get(name)

    def execute(
        self,
        kernel_name: str,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Execute a kernel by name."""
        kernel = self._kernels.get(kernel_name)
        if kernel is None:
            raise ValueError(f"Unknown kernel: {kernel_name}")
        return kernel.execute(*args, **kwargs)

    def get_all_stats(self) -> Dict[str, List[KernelStats]]:
        """Get stats for all kernels."""
        return {name: kernel.get_stats() for name, kernel in self._kernels.items()}

    def statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        all_stats = self.get_all_stats()
        total_executions = sum(len(stats) for stats in all_stats.values())
        total_time = sum(s.execution_time for stats in all_stats.values() for s in stats)

        return {
            "n_kernels": len(self._kernels),
            "total_executions": total_executions,
            "total_time": total_time,
            "device": self.config.device,
        }
