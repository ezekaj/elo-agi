"""
Proprioception: Body state sensing.

Implements proprioceptive sensing for embodied cognition.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
import time


class JointType(Enum):
    """Types of joints."""
    REVOLUTE = "revolute"    # Rotational
    PRISMATIC = "prismatic"  # Linear
    SPHERICAL = "spherical"  # Ball joint
    FIXED = "fixed"          # No movement


@dataclass
class JointState:
    """State of a single joint."""
    joint_id: str
    joint_type: JointType
    position: float          # Radians for revolute, meters for prismatic
    velocity: float          # rad/s or m/s
    acceleration: float      # rad/s^2 or m/s^2
    torque: float           # Nm or N
    min_position: float
    max_position: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class EndEffectorState:
    """State of an end effector (hand, foot, etc.)."""
    effector_id: str
    position: np.ndarray     # 3D position
    orientation: np.ndarray  # Quaternion (w, x, y, z)
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    in_contact: bool = False
    contact_force: Optional[np.ndarray] = None


@dataclass
class BodyState:
    """Complete body state."""
    timestamp: float
    joint_states: Dict[str, JointState]
    end_effectors: Dict[str, EndEffectorState]
    center_of_mass: np.ndarray
    total_mass: float
    base_position: np.ndarray
    base_orientation: np.ndarray
    is_balanced: bool = True


@dataclass
class IMUReading:
    """Inertial measurement unit reading."""
    timestamp: float
    acceleration: np.ndarray  # m/s^2, body frame
    angular_velocity: np.ndarray  # rad/s, body frame
    orientation: np.ndarray  # Quaternion (w, x, y, z)


class ProprioceptionSensor:
    """
    Proprioception sensor for body state awareness.

    Provides:
    - Joint state sensing
    - End effector tracking
    - Balance detection
    - IMU integration
    """

    def __init__(
        self,
        sensor_id: str = "proprio_0",
        n_joints: int = 7,
    ):
        self.sensor_id = sensor_id
        self.n_joints = n_joints

        # Joint configuration
        self._joints: Dict[str, JointState] = {}
        self._end_effectors: Dict[str, EndEffectorState] = {}

        # History
        self._state_history: List[BodyState] = []
        self._max_history = 100

        # IMU
        self._imu_readings: List[IMUReading] = []
        self._max_imu_history = 1000

        # Initialize default joints
        self._initialize_default_joints()

        # Simulated mode
        self._simulated = True
        self._sim_time = 0.0

    def _initialize_default_joints(self) -> None:
        """Initialize default joint configuration."""
        joint_names = [f"joint_{i}" for i in range(self.n_joints)]

        for i, name in enumerate(joint_names):
            self._joints[name] = JointState(
                joint_id=name,
                joint_type=JointType.REVOLUTE,
                position=0.0,
                velocity=0.0,
                acceleration=0.0,
                torque=0.0,
                min_position=-np.pi,
                max_position=np.pi,
            )

        # Default end effector
        self._end_effectors["hand"] = EndEffectorState(
            effector_id="hand",
            position=np.array([0.5, 0.0, 0.5]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )

    def read_state(self) -> BodyState:
        """
        Read current body state.

        Returns:
            Complete body state
        """
        if self._simulated:
            self._update_simulation()

        # Compute center of mass (simplified)
        positions = [ee.position for ee in self._end_effectors.values()]
        if positions:
            com = np.mean(positions, axis=0)
        else:
            com = np.zeros(3)

        state = BodyState(
            timestamp=time.time(),
            joint_states=dict(self._joints),
            end_effectors=dict(self._end_effectors),
            center_of_mass=com,
            total_mass=10.0,  # kg
            base_position=np.zeros(3),
            base_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            is_balanced=self._check_balance(),
        )

        # Store history
        self._state_history.append(state)
        if len(self._state_history) > self._max_history:
            self._state_history.pop(0)

        return state

    def _update_simulation(self) -> None:
        """Update simulated joint positions."""
        self._sim_time += 0.01

        for name, joint in self._joints.items():
            # Sinusoidal motion
            idx = int(name.split("_")[1])
            joint.position = 0.5 * np.sin(self._sim_time + idx * 0.3)
            joint.velocity = 0.5 * np.cos(self._sim_time + idx * 0.3)
            joint.acceleration = -0.5 * np.sin(self._sim_time + idx * 0.3)

        # Update end effector
        for name, ee in self._end_effectors.items():
            ee.position = np.array([
                0.5 + 0.1 * np.sin(self._sim_time),
                0.1 * np.sin(self._sim_time * 0.5),
                0.5 + 0.05 * np.cos(self._sim_time),
            ])
            ee.linear_velocity = np.array([
                0.1 * np.cos(self._sim_time),
                0.05 * np.cos(self._sim_time * 0.5),
                -0.05 * np.sin(self._sim_time),
            ])

    def read_joint(self, joint_id: str) -> Optional[JointState]:
        """Read state of a single joint."""
        return self._joints.get(joint_id)

    def read_end_effector(self, effector_id: str) -> Optional[EndEffectorState]:
        """Read state of an end effector."""
        return self._end_effectors.get(effector_id)

    def read_imu(self) -> IMUReading:
        """
        Read IMU data.

        Returns:
            IMU reading
        """
        if self._simulated:
            reading = IMUReading(
                timestamp=time.time(),
                acceleration=np.array([0, 0, 9.81]) + np.random.randn(3) * 0.1,
                angular_velocity=np.random.randn(3) * 0.01,
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        else:
            reading = self._read_real_imu()

        self._imu_readings.append(reading)
        if len(self._imu_readings) > self._max_imu_history:
            self._imu_readings.pop(0)

        return reading

    def _read_real_imu(self) -> IMUReading:
        """Read from real IMU (stub)."""
        return IMUReading(
            timestamp=time.time(),
            acceleration=np.array([0, 0, 9.81]),
            angular_velocity=np.zeros(3),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )

    def _check_balance(self) -> bool:
        """Check if body is balanced."""
        if not self._imu_readings:
            return True

        recent = self._imu_readings[-10:]
        accelerations = np.array([r.acceleration for r in recent])

        # Check for large deviations from gravity
        mean_acc = np.mean(accelerations, axis=0)
        gravity = np.array([0, 0, 9.81])

        deviation = np.linalg.norm(mean_acc - gravity)
        return deviation < 2.0  # m/s^2

    def get_joint_positions(self) -> np.ndarray:
        """Get all joint positions as array."""
        return np.array([j.position for j in self._joints.values()])

    def get_joint_velocities(self) -> np.ndarray:
        """Get all joint velocities as array."""
        return np.array([j.velocity for j in self._joints.values()])

    def get_state_history(self, n_states: int = 10) -> List[BodyState]:
        """Get recent state history."""
        return self._state_history[-n_states:]

    def forward_kinematics(
        self,
        joint_positions: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Compute end effector positions from joint positions.

        Args:
            joint_positions: Array of joint positions

        Returns:
            Dictionary of effector_id -> position
        """
        # Simplified FK for a serial chain
        # In reality, this would use DH parameters or similar

        base_position = np.zeros(3)
        link_length = 0.1  # 10cm links

        positions = {}
        current_pos = base_position.copy()
        current_angle = 0.0

        for i, theta in enumerate(joint_positions):
            current_angle += theta
            dx = link_length * np.cos(current_angle)
            dy = link_length * np.sin(current_angle)
            current_pos = current_pos + np.array([dx, dy, 0])

        positions["hand"] = current_pos
        return positions

    def statistics(self) -> Dict[str, Any]:
        """Get sensor statistics."""
        return {
            "sensor_id": self.sensor_id,
            "n_joints": len(self._joints),
            "n_end_effectors": len(self._end_effectors),
            "history_length": len(self._state_history),
            "imu_readings": len(self._imu_readings),
        }


class ProprioceptionProcessor:
    """
    Process proprioceptive data for motor control.

    Implements:
    - State estimation
    - Velocity/acceleration computation
    - Balance analysis
    - Movement classification
    """

    def __init__(self):
        self._state_buffer: List[BodyState] = []
        self._max_buffer = 100

    def estimate_velocity(
        self,
        states: List[BodyState],
    ) -> Dict[str, float]:
        """
        Estimate joint velocities from position history.

        Args:
            states: List of body states

        Returns:
            Dictionary of joint_id -> estimated velocity
        """
        if len(states) < 2:
            return {}

        velocities = {}
        dt = states[-1].timestamp - states[-2].timestamp

        if dt <= 0:
            return {}

        for joint_id in states[-1].joint_states.keys():
            pos_now = states[-1].joint_states[joint_id].position
            pos_prev = states[-2].joint_states[joint_id].position
            velocities[joint_id] = (pos_now - pos_prev) / dt

        return velocities

    def compute_jacobian(
        self,
        joint_positions: np.ndarray,
        link_lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute Jacobian matrix for velocity kinematics.

        Args:
            joint_positions: Current joint positions
            link_lengths: Length of each link

        Returns:
            Jacobian matrix
        """
        n_joints = len(joint_positions)
        if link_lengths is None:
            link_lengths = np.ones(n_joints) * 0.1

        # For a planar serial chain
        jacobian = np.zeros((2, n_joints))  # 2D position

        for i in range(n_joints):
            angle_sum = np.sum(joint_positions[:i + 1])

            for j in range(i, n_joints):
                jacobian[0, i] -= link_lengths[j] * np.sin(np.sum(joint_positions[:j + 1]))
                jacobian[1, i] += link_lengths[j] * np.cos(np.sum(joint_positions[:j + 1]))

        return jacobian

    def analyze_movement(
        self,
        states: List[BodyState],
    ) -> Dict[str, Any]:
        """
        Analyze movement patterns.

        Args:
            states: List of body states

        Returns:
            Movement analysis
        """
        if len(states) < 2:
            return {"movement_type": "static", "magnitude": 0.0}

        # Compute total movement
        total_movement = 0.0
        max_velocity = 0.0

        for state in states:
            for joint in state.joint_states.values():
                total_movement += abs(joint.velocity)
                max_velocity = max(max_velocity, abs(joint.velocity))

        avg_movement = total_movement / (len(states) * len(states[0].joint_states))

        # Classify movement
        if avg_movement < 0.01:
            movement_type = "static"
        elif avg_movement < 0.1:
            movement_type = "slow"
        elif avg_movement < 1.0:
            movement_type = "moderate"
        else:
            movement_type = "fast"

        return {
            "movement_type": movement_type,
            "average_velocity": avg_movement,
            "max_velocity": max_velocity,
            "total_movement": total_movement,
        }

    def detect_contact(
        self,
        state: BodyState,
        force_threshold: float = 1.0,
    ) -> List[str]:
        """
        Detect which end effectors are in contact.

        Args:
            state: Current body state
            force_threshold: Force threshold for contact

        Returns:
            List of effector IDs in contact
        """
        contacts = []

        for effector_id, ee in state.end_effectors.items():
            if ee.in_contact:
                contacts.append(effector_id)
            elif ee.contact_force is not None:
                if np.linalg.norm(ee.contact_force) > force_threshold:
                    contacts.append(effector_id)

        return contacts

    def compute_energy(
        self,
        state: BodyState,
    ) -> Dict[str, float]:
        """
        Compute kinetic and potential energy.

        Args:
            state: Body state

        Returns:
            Energy breakdown
        """
        # Simplified energy computation
        kinetic = 0.0
        for joint in state.joint_states.values():
            # Assume unit inertia
            kinetic += 0.5 * joint.velocity ** 2

        # Potential energy from height
        potential = state.total_mass * 9.81 * state.center_of_mass[2]

        return {
            "kinetic_energy": kinetic,
            "potential_energy": potential,
            "total_energy": kinetic + potential,
        }

    def statistics(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            "buffer_size": len(self._state_buffer),
        }
