"""
Batch Inference: Batched processing for efficiency.

Implements batched inference with dynamic batching and caching.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
import time


@dataclass
class BatchConfig:
    """Configuration for batch inference."""

    batch_size: int = 32
    max_wait_ms: float = 10.0
    dynamic_batching: bool = True
    cache_enabled: bool = True
    cache_size: int = 1000


@dataclass
class InferenceBatch:
    """A batch of inference requests."""

    id: str
    inputs: List[np.ndarray]
    request_ids: List[str]
    created_at: float = field(default_factory=time.time)

    @property
    def size(self) -> int:
        return len(self.inputs)


@dataclass
class BatchResult:
    """Result of batch inference."""

    batch_id: str
    outputs: List[np.ndarray]
    request_ids: List[str]
    inference_time: float
    batch_size: int


class InferenceCache:
    """
    Cache for inference results.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []

    def _hash_input(self, x: np.ndarray) -> str:
        """Hash an input for caching."""
        return str(hash(x.tobytes()))

    def get(self, x: np.ndarray) -> Optional[np.ndarray]:
        """Get cached result."""
        key = self._hash_input(x)
        if key in self._cache:
            # Move to end (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, x: np.ndarray, result: np.ndarray) -> None:
        """Cache a result."""
        key = self._hash_input(x)

        # Evict if full
        while len(self._cache) >= self.max_size:
            old_key = self._access_order.pop(0)
            self._cache.pop(old_key, None)

        self._cache[key] = result.copy()
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._access_order.clear()

    @property
    def size(self) -> int:
        return len(self._cache)


class BatchBuilder:
    """
    Build batches from individual requests.
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_wait_ms: float = 10.0,
    ):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms

        self._pending: List[Tuple[str, np.ndarray]] = []
        self._batch_counter = 0
        self._last_batch_time = time.time()

    def add(self, request_id: str, input: np.ndarray) -> Optional[InferenceBatch]:
        """
        Add a request to the pending batch.

        Returns:
            InferenceBatch if batch is ready, None otherwise
        """
        self._pending.append((request_id, input))

        # Check if batch is ready
        if self._should_flush():
            return self._flush()
        return None

    def _should_flush(self) -> bool:
        """Check if batch should be flushed."""
        if len(self._pending) >= self.batch_size:
            return True

        elapsed_ms = (time.time() - self._last_batch_time) * 1000
        if self._pending and elapsed_ms >= self.max_wait_ms:
            return True

        return False

    def _flush(self) -> Optional[InferenceBatch]:
        """Flush pending requests into a batch."""
        if not self._pending:
            return None

        self._batch_counter += 1
        request_ids = [r[0] for r in self._pending]
        inputs = [r[1] for r in self._pending]

        batch = InferenceBatch(
            id=f"batch_{self._batch_counter}",
            inputs=inputs,
            request_ids=request_ids,
        )

        self._pending.clear()
        self._last_batch_time = time.time()

        return batch

    def flush(self) -> Optional[InferenceBatch]:
        """Force flush pending requests."""
        return self._flush()

    @property
    def pending_count(self) -> int:
        return len(self._pending)


class BatchInference:
    """
    Batched inference system.

    Implements:
    - Dynamic batching
    - Result caching
    - Parallel execution
    """

    def __init__(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        config: Optional[BatchConfig] = None,
    ):
        self.model_fn = model_fn
        self.config = config or BatchConfig()

        self._batch_builder = BatchBuilder(
            batch_size=self.config.batch_size,
            max_wait_ms=self.config.max_wait_ms,
        )

        self._cache = InferenceCache(self.config.cache_size) if self.config.cache_enabled else None

        # Statistics
        self._total_requests = 0
        self._cache_hits = 0
        self._total_batches = 0
        self._total_inference_time = 0.0

    def infer_single(
        self,
        input: np.ndarray,
        request_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Infer on a single input.

        Args:
            input: Input array
            request_id: Optional request identifier

        Returns:
            Model output
        """
        self._total_requests += 1

        # Check cache
        if self._cache:
            cached = self._cache.get(input)
            if cached is not None:
                self._cache_hits += 1
                return cached

        # Direct inference
        start = time.time()
        output = self.model_fn(input[np.newaxis, ...])[0]
        inference_time = time.time() - start

        self._total_inference_time += inference_time

        # Cache result
        if self._cache:
            self._cache.put(input, output)

        return output

    def infer_batch(
        self,
        inputs: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Infer on a batch of inputs.

        Args:
            inputs: List of input arrays

        Returns:
            List of outputs
        """
        self._total_requests += len(inputs)
        self._total_batches += 1

        outputs = []
        uncached_inputs = []
        uncached_indices = []

        # Check cache for each input
        for i, input in enumerate(inputs):
            if self._cache:
                cached = self._cache.get(input)
                if cached is not None:
                    outputs.append(cached)
                    self._cache_hits += 1
                    continue

            uncached_inputs.append(input)
            uncached_indices.append(i)
            outputs.append(None)

        # Batch inference on uncached
        if uncached_inputs:
            batch = np.stack(uncached_inputs)
            start = time.time()
            batch_outputs = self.model_fn(batch)
            inference_time = time.time() - start
            self._total_inference_time += inference_time

            # Fill in outputs and cache
            for j, idx in enumerate(uncached_indices):
                output = batch_outputs[j]
                outputs[idx] = output
                if self._cache:
                    self._cache.put(uncached_inputs[j], output)

        return outputs

    def submit(
        self,
        input: np.ndarray,
        request_id: Optional[str] = None,
    ) -> Optional[BatchResult]:
        """
        Submit a request for batched inference.

        Returns:
            BatchResult if a batch was processed, None otherwise
        """
        request_id = request_id or f"req_{self._total_requests}"
        self._total_requests += 1

        # Check cache first
        if self._cache:
            cached = self._cache.get(input)
            if cached is not None:
                self._cache_hits += 1
                return BatchResult(
                    batch_id="cached",
                    outputs=[cached],
                    request_ids=[request_id],
                    inference_time=0.0,
                    batch_size=1,
                )

        # Add to batch
        batch = self._batch_builder.add(request_id, input)

        if batch:
            return self._process_batch(batch)
        return None

    def flush(self) -> Optional[BatchResult]:
        """Flush pending requests."""
        batch = self._batch_builder.flush()
        if batch:
            return self._process_batch(batch)
        return None

    def _process_batch(self, batch: InferenceBatch) -> BatchResult:
        """Process a batch."""
        self._total_batches += 1

        # Stack inputs
        stacked = np.stack(batch.inputs)

        # Inference
        start = time.time()
        outputs = self.model_fn(stacked)
        inference_time = time.time() - start
        self._total_inference_time += inference_time

        # Cache results
        if self._cache:
            for i, (input, output) in enumerate(zip(batch.inputs, outputs)):
                self._cache.put(input, output)

        return BatchResult(
            batch_id=batch.id,
            outputs=list(outputs),
            request_ids=batch.request_ids,
            inference_time=inference_time,
            batch_size=batch.size,
        )

    def get_pending_count(self) -> int:
        """Get number of pending requests."""
        return self._batch_builder.pending_count

    def clear_cache(self) -> None:
        """Clear the inference cache."""
        if self._cache:
            self._cache.clear()

    def statistics(self) -> Dict[str, Any]:
        """Get inference statistics."""
        cache_hit_rate = self._cache_hits / max(self._total_requests, 1)
        avg_batch_size = self._total_requests / max(self._total_batches, 1)
        avg_inference_time = self._total_inference_time / max(self._total_batches, 1)

        return {
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": cache_hit_rate,
            "avg_batch_size": avg_batch_size,
            "avg_inference_time": avg_inference_time,
            "cache_size": self._cache.size if self._cache else 0,
            "pending": self.get_pending_count(),
        }
