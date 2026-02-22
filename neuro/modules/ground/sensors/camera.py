"""
Camera: Vision input processing.

Implements camera interface for visual perception.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import time


class CameraType(Enum):
    """Types of cameras."""

    RGB = "rgb"
    DEPTH = "depth"
    RGBD = "rgbd"
    STEREO = "stereo"
    INFRARED = "infrared"


class ColorSpace(Enum):
    """Color spaces."""

    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    HSV = "hsv"
    LAB = "lab"


@dataclass
class CameraConfig:
    """Configuration for camera."""

    camera_type: CameraType = CameraType.RGB
    width: int = 640
    height: int = 480
    fps: float = 30.0
    color_space: ColorSpace = ColorSpace.RGB
    device_id: int = 0
    auto_exposure: bool = True
    auto_focus: bool = True


@dataclass
class CameraFrame:
    """A single frame from a camera."""

    data: np.ndarray
    timestamp: float
    frame_id: int
    camera_id: str
    width: int
    height: int
    channels: int
    color_space: ColorSpace
    depth_data: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape


@dataclass
class CameraCalibration:
    """Camera calibration parameters."""

    intrinsic_matrix: np.ndarray  # 3x3
    distortion_coeffs: np.ndarray
    extrinsic_matrix: Optional[np.ndarray] = None  # 4x4


class Camera:
    """
    Camera interface for visual input.

    Provides:
    - Frame capture
    - Color space conversion
    - Basic preprocessing
    - Depth processing (for RGBD)
    """

    def __init__(
        self,
        camera_id: str = "camera_0",
        config: Optional[CameraConfig] = None,
    ):
        self.camera_id = camera_id
        self.config = config or CameraConfig()

        self._is_open = False
        self._frame_count = 0
        self._calibration: Optional[CameraCalibration] = None

        # Callbacks
        self._on_frame: Optional[Callable[[CameraFrame], None]] = None

        # Simulated frame buffer for testing
        self._simulated = True

    def open(self) -> bool:
        """Open the camera."""
        self._is_open = True
        return True

    def close(self) -> None:
        """Close the camera."""
        self._is_open = False

    def is_open(self) -> bool:
        """Check if camera is open."""
        return self._is_open

    def capture(self) -> Optional[CameraFrame]:
        """
        Capture a single frame.

        Returns:
            CameraFrame or None if capture failed
        """
        if not self._is_open:
            return None

        self._frame_count += 1

        # Generate simulated frame
        if self._simulated:
            data = self._generate_test_frame()
        else:
            data = self._capture_real_frame()

        channels = 3 if self.config.color_space != ColorSpace.GRAY else 1

        frame = CameraFrame(
            data=data,
            timestamp=time.time(),
            frame_id=self._frame_count,
            camera_id=self.camera_id,
            width=self.config.width,
            height=self.config.height,
            channels=channels,
            color_space=self.config.color_space,
        )

        # Add depth data for RGBD cameras
        if self.config.camera_type in [CameraType.DEPTH, CameraType.RGBD]:
            frame.depth_data = self._generate_depth_frame()

        # Callback
        if self._on_frame:
            self._on_frame(frame)

        return frame

    def _generate_test_frame(self) -> np.ndarray:
        """Generate a test frame for simulation."""
        h, w = self.config.height, self.config.width

        if self.config.color_space == ColorSpace.GRAY:
            # Gradient pattern
            x = np.linspace(0, 255, w)
            y = np.linspace(0, 255, h)
            xx, yy = np.meshgrid(x, y)
            frame = ((xx + yy) / 2).astype(np.uint8)
        else:
            # RGB pattern
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            # Red gradient
            frame[:, :, 0] = np.linspace(0, 255, w).astype(np.uint8)
            # Green gradient
            frame[:, :, 1] = np.linspace(0, 255, h).reshape(-1, 1).astype(np.uint8)
            # Blue noise
            frame[:, :, 2] = np.random.randint(0, 128, (h, w), dtype=np.uint8)

        return frame

    def _generate_depth_frame(self) -> np.ndarray:
        """Generate simulated depth data."""
        h, w = self.config.height, self.config.width
        # Simulate depth with distance gradient
        depth = np.linspace(0.5, 5.0, w).reshape(1, -1)
        depth = np.tile(depth, (h, 1))
        # Add some noise
        depth += np.random.randn(h, w) * 0.05
        return depth.astype(np.float32)

    def _capture_real_frame(self) -> np.ndarray:
        """Capture from real camera (stub for actual implementation)."""
        return self._generate_test_frame()

    def set_calibration(self, calibration: CameraCalibration) -> None:
        """Set camera calibration."""
        self._calibration = calibration

    def get_calibration(self) -> Optional[CameraCalibration]:
        """Get camera calibration."""
        return self._calibration

    def set_callback(
        self,
        on_frame: Optional[Callable[[CameraFrame], None]] = None,
    ) -> None:
        """Set frame callback."""
        self._on_frame = on_frame

    def convert_color(
        self,
        frame: CameraFrame,
        target_space: ColorSpace,
    ) -> CameraFrame:
        """
        Convert frame to different color space.

        Args:
            frame: Input frame
            target_space: Target color space

        Returns:
            Converted frame
        """
        if frame.color_space == target_space:
            return frame

        data = frame.data.copy()

        # Simple conversions (full implementation would use OpenCV)
        if target_space == ColorSpace.GRAY and frame.color_space == ColorSpace.RGB:
            # RGB to grayscale
            data = (0.299 * data[:, :, 0] + 0.587 * data[:, :, 1] + 0.114 * data[:, :, 2]).astype(
                np.uint8
            )
        elif target_space == ColorSpace.RGB and frame.color_space == ColorSpace.BGR:
            # BGR to RGB
            data = data[:, :, ::-1]

        channels = 1 if target_space == ColorSpace.GRAY else 3

        return CameraFrame(
            data=data,
            timestamp=frame.timestamp,
            frame_id=frame.frame_id,
            camera_id=frame.camera_id,
            width=frame.width,
            height=frame.height,
            channels=channels,
            color_space=target_space,
            depth_data=frame.depth_data,
        )

    def undistort(self, frame: CameraFrame) -> CameraFrame:
        """
        Remove lens distortion using calibration.

        Args:
            frame: Input frame

        Returns:
            Undistorted frame
        """
        if self._calibration is None:
            return frame

        # Simplified undistortion (real implementation would use cv2.undistort)
        return frame

    def statistics(self) -> Dict[str, Any]:
        """Get camera statistics."""
        return {
            "camera_id": self.camera_id,
            "is_open": self._is_open,
            "frame_count": self._frame_count,
            "config": {
                "width": self.config.width,
                "height": self.config.height,
                "fps": self.config.fps,
                "type": self.config.camera_type.value,
            },
            "calibrated": self._calibration is not None,
        }


class VisionProcessor:
    """
    Process camera frames for perception.

    Implements:
    - Frame preprocessing
    - Feature extraction
    - Object detection (stub)
    - Optical flow
    """

    def __init__(self):
        self._prev_frame: Optional[np.ndarray] = None
        self._frame_count = 0

    def preprocess(
        self,
        frame: CameraFrame,
        resize: Optional[Tuple[int, int]] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Preprocess frame for neural network input.

        Args:
            frame: Input frame
            resize: Target size (width, height)
            normalize: Normalize to [0, 1]

        Returns:
            Preprocessed array
        """
        data = frame.data.astype(np.float32)

        if normalize:
            data = data / 255.0

        if resize:
            data = self._resize(data, resize)

        return data

    def _resize(
        self,
        data: np.ndarray,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """Simple nearest-neighbor resize."""
        h, w = data.shape[:2]
        new_w, new_h = size

        # Compute indices
        x_indices = np.linspace(0, w - 1, new_w).astype(int)
        y_indices = np.linspace(0, h - 1, new_h).astype(int)

        if data.ndim == 2:
            return data[np.ix_(y_indices, x_indices)]
        else:
            return data[np.ix_(y_indices, x_indices, np.arange(data.shape[2]))]

    def compute_optical_flow(
        self,
        frame: CameraFrame,
    ) -> Optional[np.ndarray]:
        """
        Compute optical flow between frames.

        Args:
            frame: Current frame

        Returns:
            Flow field (H, W, 2) or None if first frame
        """
        # Convert to grayscale if needed
        if frame.channels > 1:
            gray = (
                0.299 * frame.data[:, :, 0]
                + 0.587 * frame.data[:, :, 1]
                + 0.114 * frame.data[:, :, 2]
            )
        else:
            gray = frame.data

        if self._prev_frame is None:
            self._prev_frame = gray
            return None

        # Simple block matching flow (simplified)
        h, w = gray.shape
        flow = np.zeros((h, w, 2))

        # This is a stub - real implementation would use optical flow algorithms
        diff = gray.astype(float) - self._prev_frame.astype(float)
        flow[:, :, 0] = np.gradient(diff, axis=1)  # dx
        flow[:, :, 1] = np.gradient(diff, axis=0)  # dy

        self._prev_frame = gray
        self._frame_count += 1

        return flow

    def extract_edges(
        self,
        frame: CameraFrame,
        threshold: float = 50.0,
    ) -> np.ndarray:
        """
        Extract edges using Sobel operator.

        Args:
            frame: Input frame
            threshold: Edge threshold

        Returns:
            Edge map
        """
        # Convert to grayscale
        if frame.channels > 1:
            gray = (
                0.299 * frame.data[:, :, 0]
                + 0.587 * frame.data[:, :, 1]
                + 0.114 * frame.data[:, :, 2]
            ).astype(float)
        else:
            gray = frame.data.astype(float)

        # Sobel operators
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Simple convolution
        from scipy.ndimage import convolve

        gx = convolve(gray, sobel_x)
        gy = convolve(gray, sobel_y)

        magnitude = np.sqrt(gx**2 + gy**2)

        # Threshold
        edges = (magnitude > threshold).astype(np.uint8) * 255

        return edges

    def compute_histogram(
        self,
        frame: CameraFrame,
        bins: int = 256,
    ) -> Dict[str, np.ndarray]:
        """
        Compute color histogram.

        Args:
            frame: Input frame
            bins: Number of histogram bins

        Returns:
            Dictionary of histograms per channel
        """
        histograms = {}

        if frame.channels == 1:
            histograms["gray"] = np.histogram(frame.data, bins=bins, range=(0, 256))[0]
        else:
            channels = ["red", "green", "blue"]
            for i, name in enumerate(channels):
                histograms[name] = np.histogram(frame.data[:, :, i], bins=bins, range=(0, 256))[0]

        return histograms

    def detect_motion(
        self,
        frame: CameraFrame,
        motion_threshold: float = 10.0,
    ) -> Tuple[bool, float]:
        """
        Detect motion in frame.

        Args:
            frame: Current frame
            motion_threshold: Threshold for motion detection

        Returns:
            Tuple of (motion_detected, motion_magnitude)
        """
        flow = self.compute_optical_flow(frame)

        if flow is None:
            return False, 0.0

        magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)
        mean_magnitude = np.mean(magnitude)

        return mean_magnitude > motion_threshold, float(mean_magnitude)

    def statistics(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "frames_processed": self._frame_count,
        }
