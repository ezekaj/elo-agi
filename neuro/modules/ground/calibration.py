"""
Calibration: Sensor and actuator calibration.

Implements calibration procedures for real-world sensors and actuators.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np
import time


class CalibrationType(Enum):
    """Types of calibration."""

    CAMERA_INTRINSIC = "camera_intrinsic"
    CAMERA_EXTRINSIC = "camera_extrinsic"
    IMU = "imu"
    JOINT_ENCODER = "joint_encoder"
    FORCE_SENSOR = "force_sensor"
    MICROPHONE = "microphone"


class CalibrationStatus(Enum):
    """Calibration status."""

    NOT_CALIBRATED = "not_calibrated"
    IN_PROGRESS = "in_progress"
    CALIBRATED = "calibrated"
    NEEDS_RECALIBRATION = "needs_recalibration"


@dataclass
class CalibrationConfig:
    """Configuration for calibration."""

    calibration_type: CalibrationType
    n_samples: int = 100
    warmup_samples: int = 10
    outlier_threshold: float = 3.0  # Standard deviations
    auto_recalibrate: bool = False
    recalibration_interval: float = 3600.0  # seconds


@dataclass
class CalibrationResult:
    """Result of a calibration procedure."""

    sensor_id: str
    calibration_type: CalibrationType
    status: CalibrationStatus
    parameters: Dict[str, Any]
    error_metrics: Dict[str, float]
    timestamp: float
    n_samples_used: int
    valid: bool = True


@dataclass
class CameraCalibrationParams:
    """Camera calibration parameters."""

    intrinsic_matrix: np.ndarray  # 3x3
    distortion_coeffs: np.ndarray  # 5 or 8 coeffs
    reprojection_error: float


@dataclass
class IMUCalibrationParams:
    """IMU calibration parameters."""

    accelerometer_bias: np.ndarray  # 3D
    accelerometer_scale: np.ndarray  # 3x3
    gyroscope_bias: np.ndarray  # 3D
    gyroscope_scale: np.ndarray  # 3x3


class SensorCalibrator:
    """
    Calibrate sensors for accurate measurements.

    Implements:
    - Camera calibration
    - IMU calibration
    - Joint encoder calibration
    - Force sensor calibration
    """

    def __init__(self, calibrator_id: str = "calibrator_0"):
        self.calibrator_id = calibrator_id

        # Calibration storage
        self._calibrations: Dict[str, CalibrationResult] = {}

        # Calibration data buffers
        self._data_buffers: Dict[str, List[np.ndarray]] = {}

        # Calibration status
        self._status: Dict[str, CalibrationStatus] = {}

    def start_calibration(
        self,
        sensor_id: str,
        config: CalibrationConfig,
    ) -> bool:
        """
        Start a calibration procedure.

        Args:
            sensor_id: Sensor to calibrate
            config: Calibration configuration

        Returns:
            True if calibration started
        """
        self._status[sensor_id] = CalibrationStatus.IN_PROGRESS
        self._data_buffers[sensor_id] = []
        return True

    def add_calibration_sample(
        self,
        sensor_id: str,
        sample: np.ndarray,
    ) -> None:
        """
        Add a calibration sample.

        Args:
            sensor_id: Sensor identifier
            sample: Calibration sample
        """
        if sensor_id not in self._data_buffers:
            self._data_buffers[sensor_id] = []

        self._data_buffers[sensor_id].append(sample)

    def calibrate_camera_intrinsic(
        self,
        sensor_id: str,
        image_points: List[np.ndarray],
        object_points: List[np.ndarray],
        image_size: Tuple[int, int],
    ) -> CalibrationResult:
        """
        Calibrate camera intrinsic parameters.

        Args:
            sensor_id: Camera identifier
            image_points: 2D points in images
            object_points: 3D points in world
            image_size: Image dimensions (width, height)

        Returns:
            CalibrationResult
        """
        # Simplified calibration (real implementation would use cv2.calibrateCamera)

        # Estimate focal length from image size
        focal = max(image_size) * 1.2

        # Build intrinsic matrix
        cx, cy = image_size[0] / 2, image_size[1] / 2
        intrinsic = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])

        # No distortion for simplified version
        distortion = np.zeros(5)

        # Compute reprojection error (simplified)
        errors = []
        for img_pts, obj_pts in zip(image_points, object_points):
            # Project points
            projected = obj_pts[:, :2] * (focal / obj_pts[:, 2:3]) + np.array([cx, cy])
            error = np.mean(np.linalg.norm(projected - img_pts, axis=1))
            errors.append(error)

        reprojection_error = np.mean(errors) if errors else 0.0

        params = CameraCalibrationParams(
            intrinsic_matrix=intrinsic,
            distortion_coeffs=distortion,
            reprojection_error=reprojection_error,
        )

        result = CalibrationResult(
            sensor_id=sensor_id,
            calibration_type=CalibrationType.CAMERA_INTRINSIC,
            status=CalibrationStatus.CALIBRATED,
            parameters={
                "intrinsic_matrix": intrinsic.tolist(),
                "distortion_coeffs": distortion.tolist(),
            },
            error_metrics={
                "reprojection_error": reprojection_error,
            },
            timestamp=time.time(),
            n_samples_used=len(image_points),
            valid=reprojection_error < 1.0,
        )

        self._calibrations[sensor_id] = result
        self._status[sensor_id] = CalibrationStatus.CALIBRATED

        return result

    def calibrate_imu(
        self,
        sensor_id: str,
        static_samples: List[np.ndarray],
    ) -> CalibrationResult:
        """
        Calibrate IMU sensors.

        Args:
            sensor_id: IMU identifier
            static_samples: Samples taken while IMU is static

        Returns:
            CalibrationResult
        """
        samples = np.array(static_samples)

        # Assume samples are [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
        if samples.shape[1] < 6:
            # Pad if needed
            samples = np.pad(samples, ((0, 0), (0, 6 - samples.shape[1])))

        acc_samples = samples[:, :3]
        gyro_samples = samples[:, 3:6]

        # Accelerometer calibration
        # When static, should read [0, 0, g]
        acc_mean = np.mean(acc_samples, axis=0)
        gravity = np.array([0, 0, 9.81])

        # Bias is difference from expected
        acc_bias = acc_mean - gravity

        # Scale (simplified - assume no scale error)
        acc_scale = np.eye(3)

        # Gyroscope calibration
        # When static, should read [0, 0, 0]
        gyro_bias = np.mean(gyro_samples, axis=0)
        gyro_scale = np.eye(3)

        params = IMUCalibrationParams(
            accelerometer_bias=acc_bias,
            accelerometer_scale=acc_scale,
            gyroscope_bias=gyro_bias,
            gyroscope_scale=gyro_scale,
        )

        # Compute error
        corrected_acc = (acc_samples - acc_bias) @ acc_scale
        acc_error = np.mean(np.linalg.norm(corrected_acc - gravity, axis=1))

        corrected_gyro = (gyro_samples - gyro_bias) @ gyro_scale
        gyro_error = np.mean(np.linalg.norm(corrected_gyro, axis=1))

        result = CalibrationResult(
            sensor_id=sensor_id,
            calibration_type=CalibrationType.IMU,
            status=CalibrationStatus.CALIBRATED,
            parameters={
                "accelerometer_bias": acc_bias.tolist(),
                "accelerometer_scale": acc_scale.tolist(),
                "gyroscope_bias": gyro_bias.tolist(),
                "gyroscope_scale": gyro_scale.tolist(),
            },
            error_metrics={
                "accelerometer_error": acc_error,
                "gyroscope_error": gyro_error,
            },
            timestamp=time.time(),
            n_samples_used=len(static_samples),
            valid=acc_error < 0.5 and gyro_error < 0.01,
        )

        self._calibrations[sensor_id] = result
        self._status[sensor_id] = CalibrationStatus.CALIBRATED

        return result

    def calibrate_joint_encoder(
        self,
        sensor_id: str,
        encoder_readings: List[float],
        reference_positions: List[float],
    ) -> CalibrationResult:
        """
        Calibrate joint encoder.

        Args:
            sensor_id: Encoder identifier
            encoder_readings: Raw encoder values
            reference_positions: Known reference positions

        Returns:
            CalibrationResult
        """
        readings = np.array(encoder_readings)
        references = np.array(reference_positions)

        # Linear fit: position = scale * reading + offset
        A = np.vstack([readings, np.ones(len(readings))]).T
        scale, offset = np.linalg.lstsq(A, references, rcond=None)[0]

        # Compute error
        predicted = scale * readings + offset
        error = np.sqrt(np.mean((predicted - references) ** 2))

        result = CalibrationResult(
            sensor_id=sensor_id,
            calibration_type=CalibrationType.JOINT_ENCODER,
            status=CalibrationStatus.CALIBRATED,
            parameters={
                "scale": float(scale),
                "offset": float(offset),
            },
            error_metrics={
                "rmse": float(error),
                "max_error": float(np.max(np.abs(predicted - references))),
            },
            timestamp=time.time(),
            n_samples_used=len(encoder_readings),
            valid=error < 0.01,  # 0.01 rad
        )

        self._calibrations[sensor_id] = result
        self._status[sensor_id] = CalibrationStatus.CALIBRATED

        return result

    def calibrate_force_sensor(
        self,
        sensor_id: str,
        sensor_readings: List[np.ndarray],
        known_forces: List[np.ndarray],
    ) -> CalibrationResult:
        """
        Calibrate force/torque sensor.

        Args:
            sensor_id: Sensor identifier
            sensor_readings: Raw sensor readings
            known_forces: Known applied forces

        Returns:
            CalibrationResult
        """
        readings = np.array(sensor_readings)
        forces = np.array(known_forces)

        # Linear calibration: force = calibration_matrix @ reading + bias
        # Simplified: per-axis calibration

        n_axes = readings.shape[1] if readings.ndim > 1 else 1

        if n_axes == 1:
            # Single axis
            A = np.vstack([readings.flatten(), np.ones(len(readings))]).T
            scale, offset = np.linalg.lstsq(A, forces.flatten(), rcond=None)[0]
            calibration_matrix = np.array([[scale]])
            bias = np.array([offset])
        else:
            # Multiple axes
            calibration_matrix = np.zeros((n_axes, n_axes))
            bias = np.zeros(n_axes)

            for i in range(n_axes):
                A = np.vstack([readings[:, i], np.ones(len(readings))]).T
                calibration_matrix[i, i], bias[i] = np.linalg.lstsq(A, forces[:, i], rcond=None)[0]

        # Compute error
        predicted = readings @ calibration_matrix.T + bias
        error = np.sqrt(np.mean((predicted - forces) ** 2))

        result = CalibrationResult(
            sensor_id=sensor_id,
            calibration_type=CalibrationType.FORCE_SENSOR,
            status=CalibrationStatus.CALIBRATED,
            parameters={
                "calibration_matrix": calibration_matrix.tolist(),
                "bias": bias.tolist(),
            },
            error_metrics={
                "rmse": float(error),
            },
            timestamp=time.time(),
            n_samples_used=len(sensor_readings),
            valid=error < 1.0,  # 1N
        )

        self._calibrations[sensor_id] = result
        self._status[sensor_id] = CalibrationStatus.CALIBRATED

        return result

    def get_calibration(
        self,
        sensor_id: str,
    ) -> Optional[CalibrationResult]:
        """Get calibration for a sensor."""
        return self._calibrations.get(sensor_id)

    def get_status(self, sensor_id: str) -> CalibrationStatus:
        """Get calibration status for a sensor."""
        return self._status.get(sensor_id, CalibrationStatus.NOT_CALIBRATED)

    def apply_calibration(
        self,
        sensor_id: str,
        raw_data: np.ndarray,
    ) -> np.ndarray:
        """
        Apply calibration to raw sensor data.

        Args:
            sensor_id: Sensor identifier
            raw_data: Raw sensor reading

        Returns:
            Calibrated data
        """
        calibration = self._calibrations.get(sensor_id)
        if calibration is None:
            return raw_data

        params = calibration.parameters

        if calibration.calibration_type == CalibrationType.IMU:
            # Apply IMU calibration
            acc_bias = np.array(params.get("accelerometer_bias", [0, 0, 0]))
            gyro_bias = np.array(params.get("gyroscope_bias", [0, 0, 0]))

            if raw_data.shape[-1] >= 6:
                calibrated = raw_data.copy()
                calibrated[..., :3] -= acc_bias
                calibrated[..., 3:6] -= gyro_bias
                return calibrated

        elif calibration.calibration_type == CalibrationType.JOINT_ENCODER:
            scale = params.get("scale", 1.0)
            offset = params.get("offset", 0.0)
            return raw_data * scale + offset

        elif calibration.calibration_type == CalibrationType.FORCE_SENSOR:
            matrix = np.array(params.get("calibration_matrix", [[1.0]]))
            bias = np.array(params.get("bias", [0.0]))
            return raw_data @ matrix.T + bias

        return raw_data

    def check_needs_recalibration(
        self,
        sensor_id: str,
        max_age: float = 3600.0,
    ) -> bool:
        """
        Check if sensor needs recalibration.

        Args:
            sensor_id: Sensor identifier
            max_age: Maximum calibration age in seconds

        Returns:
            True if recalibration needed
        """
        calibration = self._calibrations.get(sensor_id)
        if calibration is None:
            return True

        age = time.time() - calibration.timestamp
        return age > max_age or not calibration.valid

    def statistics(self) -> Dict[str, Any]:
        """Get calibrator statistics."""
        return {
            "calibrator_id": self.calibrator_id,
            "n_calibrations": len(self._calibrations),
            "calibrations": {
                sensor_id: {
                    "type": cal.calibration_type.value,
                    "status": cal.status.value,
                    "valid": cal.valid,
                    "age_seconds": time.time() - cal.timestamp,
                }
                for sensor_id, cal in self._calibrations.items()
            },
        }
