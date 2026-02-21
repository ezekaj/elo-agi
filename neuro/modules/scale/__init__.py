"""
neuro-scale: Distributed and parallel processing.

Provides scalability through distributed computation, GPU acceleration,
and efficiency optimizations.
"""

from .distributed.coordinator import (
    TaskCoordinator,
    DistributedTask,
    TaskStatus,
    CoordinatorConfig,
)
from .distributed.worker import (
    Worker,
    WorkerPool,
    WorkerStatus,
    WorkResult,
)
from .distributed.aggregator import (
    ResultAggregator,
    AggregationStrategy,
    WeightedAverage,
    Voting,
)

from .gpu.kernels import (
    GPUKernel,
    MatMulKernel,
    ConvolutionKernel,
    SoftmaxKernel,
    KernelConfig,
)
from .gpu.batch_inference import (
    BatchInference,
    BatchConfig,
    InferenceBatch,
    BatchResult,
)

from .sparse.pruning import (
    NetworkPruner,
    PruningStrategy,
    MagnitudePruning,
    StructuredPruning,
    PruningResult,
)
from .sparse.quantization import (
    Quantizer,
    QuantizationConfig,
    QuantizedModel,
    QuantizationLevel,
)

from .efficiency import (
    EfficiencyOptimizer,
    LatencyProfile,
    ThroughputMetrics,
    MemoryProfile,
    OptimizationConfig,
)

__version__ = "0.1.0"

__all__ = [
    # Distributed
    "TaskCoordinator",
    "DistributedTask",
    "TaskStatus",
    "CoordinatorConfig",
    "Worker",
    "WorkerPool",
    "WorkerStatus",
    "WorkResult",
    "ResultAggregator",
    "AggregationStrategy",
    "WeightedAverage",
    "Voting",
    # GPU
    "GPUKernel",
    "MatMulKernel",
    "ConvolutionKernel",
    "SoftmaxKernel",
    "KernelConfig",
    "BatchInference",
    "BatchConfig",
    "InferenceBatch",
    "BatchResult",
    # Sparse
    "NetworkPruner",
    "PruningStrategy",
    "MagnitudePruning",
    "StructuredPruning",
    "PruningResult",
    "Quantizer",
    "QuantizationConfig",
    "QuantizedModel",
    "QuantizationLevel",
    # Efficiency
    "EfficiencyOptimizer",
    "LatencyProfile",
    "ThroughputMetrics",
    "MemoryProfile",
    "OptimizationConfig",
]
