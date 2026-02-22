"""
Quantization: Reduce precision for efficiency.

Implements various quantization methods for model compression.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np


class QuantizationLevel(Enum):
    """Quantization precision levels."""

    FP32 = 32  # Full precision
    FP16 = 16  # Half precision
    INT8 = 8  # 8-bit integer
    INT4 = 4  # 4-bit integer
    BINARY = 1  # Binary


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""

    level: QuantizationLevel = QuantizationLevel.INT8
    symmetric: bool = True
    per_channel: bool = False
    calibration_samples: int = 100


@dataclass
class QuantizedTensor:
    """A quantized tensor with scale and zero-point."""

    data: np.ndarray  # Quantized values
    scale: np.ndarray  # Scale factor(s)
    zero_point: np.ndarray  # Zero point(s)
    original_dtype: np.dtype
    level: QuantizationLevel

    def dequantize(self) -> np.ndarray:
        """Convert back to original precision."""
        return (self.data.astype(np.float32) - self.zero_point) * self.scale


@dataclass
class QuantizedModel:
    """A quantized model."""

    weights: Dict[str, QuantizedTensor]
    level: QuantizationLevel
    memory_reduction: float
    accuracy_change: Optional[float] = None


class Quantizer:
    """
    Quantizer for neural network weights and activations.

    Implements:
    - Symmetric and asymmetric quantization
    - Per-tensor and per-channel quantization
    - Various bit widths
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig()

        # Bit ranges
        self._bit_ranges = {
            QuantizationLevel.INT8: (-128, 127),
            QuantizationLevel.INT4: (-8, 7),
            QuantizationLevel.BINARY: (-1, 1),
            QuantizationLevel.FP16: None,  # No range limit
            QuantizationLevel.FP32: None,
        }

    def quantize(
        self,
        tensor: np.ndarray,
        level: Optional[QuantizationLevel] = None,
        symmetric: Optional[bool] = None,
    ) -> QuantizedTensor:
        """
        Quantize a tensor.

        Args:
            tensor: Input tensor (float32)
            level: Target quantization level
            symmetric: Use symmetric quantization

        Returns:
            QuantizedTensor
        """
        level = level or self.config.level
        symmetric = symmetric if symmetric is not None else self.config.symmetric

        # FP16 - just convert
        if level == QuantizationLevel.FP16:
            return QuantizedTensor(
                data=tensor.astype(np.float16),
                scale=np.array([1.0]),
                zero_point=np.array([0.0]),
                original_dtype=tensor.dtype,
                level=level,
            )

        # FP32 - no quantization
        if level == QuantizationLevel.FP32:
            return QuantizedTensor(
                data=tensor.copy(),
                scale=np.array([1.0]),
                zero_point=np.array([0.0]),
                original_dtype=tensor.dtype,
                level=level,
            )

        # Integer quantization
        qmin, qmax = self._bit_ranges[level]

        if self.config.per_channel and tensor.ndim > 1:
            return self._quantize_per_channel(tensor, qmin, qmax, symmetric, level)
        else:
            return self._quantize_per_tensor(tensor, qmin, qmax, symmetric, level)

    def _quantize_per_tensor(
        self,
        tensor: np.ndarray,
        qmin: int,
        qmax: int,
        symmetric: bool,
        level: QuantizationLevel,
    ) -> QuantizedTensor:
        """Quantize entire tensor with single scale/zero-point."""
        if symmetric:
            max_abs = np.max(np.abs(tensor))
            scale = max_abs / max(abs(qmin), abs(qmax))
            zero_point = 0
        else:
            min_val, max_val = tensor.min(), tensor.max()
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - np.round(min_val / scale)

        scale = max(scale, 1e-8)  # Prevent divide by zero

        # Quantize
        quantized = np.round(tensor / scale + zero_point)
        quantized = np.clip(quantized, qmin, qmax)

        return QuantizedTensor(
            data=quantized.astype(np.int8 if level == QuantizationLevel.INT8 else np.int8),
            scale=np.array([scale]),
            zero_point=np.array([zero_point]),
            original_dtype=tensor.dtype,
            level=level,
        )

    def _quantize_per_channel(
        self,
        tensor: np.ndarray,
        qmin: int,
        qmax: int,
        symmetric: bool,
        level: QuantizationLevel,
    ) -> QuantizedTensor:
        """Quantize with per-channel scale/zero-point."""
        n_channels = tensor.shape[0]
        scales = np.zeros(n_channels)
        zero_points = np.zeros(n_channels)
        quantized = np.zeros_like(tensor, dtype=np.int8)

        for c in range(n_channels):
            channel_data = tensor[c]

            if symmetric:
                max_abs = np.max(np.abs(channel_data))
                scale = max_abs / max(abs(qmin), abs(qmax))
                zp = 0
            else:
                min_val, max_val = channel_data.min(), channel_data.max()
                scale = (max_val - min_val) / (qmax - qmin)
                zp = qmin - np.round(min_val / scale)

            scale = max(scale, 1e-8)
            scales[c] = scale
            zero_points[c] = zp

            # Quantize channel
            q = np.round(channel_data / scale + zp)
            quantized[c] = np.clip(q, qmin, qmax)

        return QuantizedTensor(
            data=quantized,
            scale=scales,
            zero_point=zero_points,
            original_dtype=tensor.dtype,
            level=level,
        )

    def dequantize(self, qtensor: QuantizedTensor) -> np.ndarray:
        """Dequantize a tensor."""
        return qtensor.dequantize()

    def quantize_model(
        self,
        weights: Dict[str, np.ndarray],
        level: Optional[QuantizationLevel] = None,
    ) -> QuantizedModel:
        """
        Quantize all weights in a model.

        Args:
            weights: Dictionary of layer_name -> weight tensor
            level: Target quantization level

        Returns:
            QuantizedModel
        """
        level = level or self.config.level

        quantized_weights = {}
        original_size = 0
        quantized_size = 0

        for name, weight in weights.items():
            original_size += weight.nbytes

            qtensor = self.quantize(weight, level)
            quantized_weights[name] = qtensor

            quantized_size += qtensor.data.nbytes + qtensor.scale.nbytes + qtensor.zero_point.nbytes

        memory_reduction = 1.0 - quantized_size / original_size if original_size > 0 else 0.0

        return QuantizedModel(
            weights=quantized_weights,
            level=level,
            memory_reduction=memory_reduction,
        )

    def dequantize_model(
        self,
        qmodel: QuantizedModel,
    ) -> Dict[str, np.ndarray]:
        """Dequantize all weights in a model."""
        return {name: qtensor.dequantize() for name, qtensor in qmodel.weights.items()}

    def compute_quantization_error(
        self,
        original: np.ndarray,
        quantized: QuantizedTensor,
    ) -> Dict[str, float]:
        """Compute quantization error metrics."""
        dequantized = quantized.dequantize()

        mse = np.mean((original - dequantized) ** 2)
        mae = np.mean(np.abs(original - dequantized))
        max_error = np.max(np.abs(original - dequantized))

        # Signal-to-noise ratio
        signal_power = np.mean(original**2)
        noise_power = mse
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

        return {
            "mse": float(mse),
            "mae": float(mae),
            "max_error": float(max_error),
            "snr_db": float(snr_db),
        }

    def calibrate(
        self,
        tensor: np.ndarray,
        calibration_data: List[np.ndarray],
    ) -> Tuple[float, float]:
        """
        Calibrate scale and zero-point using data.

        Args:
            tensor: Weight tensor
            calibration_data: Activation samples for calibration

        Returns:
            Tuple of (scale, zero_point)
        """
        # Collect activation statistics
        all_vals = []
        for data in calibration_data[: self.config.calibration_samples]:
            all_vals.extend(data.flatten())

        all_vals = np.array(all_vals)

        # Use percentile to handle outliers
        min_val = np.percentile(all_vals, 0.1)
        max_val = np.percentile(all_vals, 99.9)

        qmin, qmax = self._bit_ranges[self.config.level]

        if self.config.symmetric:
            max_abs = max(abs(min_val), abs(max_val))
            scale = max_abs / max(abs(qmin), abs(qmax))
            zero_point = 0.0
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = qmin - min_val / scale

        return float(scale), float(zero_point)

    def statistics(self) -> Dict[str, Any]:
        """Get quantizer statistics."""
        return {
            "level": self.config.level.value,
            "bits": self.config.level.value,
            "symmetric": self.config.symmetric,
            "per_channel": self.config.per_channel,
        }
