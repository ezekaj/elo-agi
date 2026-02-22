"""
Motor Controller: Movement command execution.

Implements motor control for actuator commands.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np
import time


class ControlMode(Enum):
    """Motor control modes."""

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    IMPEDANCE = "impedance"


class TrajectoryType(Enum):
    """Trajectory interpolation types."""

    LINEAR = "linear"
    CUBIC = "cubic"
    QUINTIC = "quintic"
    MINIMUM_JERK = "minimum_jerk"


@dataclass
class MotorConfig:
    """Configuration for motor control."""

    control_mode: ControlMode = ControlMode.POSITION
    max_velocity: float = 1.0  # rad/s
    max_acceleration: float = 5.0  # rad/s^2
    max_torque: float = 10.0  # Nm
    position_gain: float = 100.0
    velocity_gain: float = 10.0
    dt: float = 0.001  # Control loop timestep


@dataclass
class MotorCommand:
    """A command to a motor."""

    motor_id: str
    target_position: Optional[float] = None
    target_velocity: Optional[float] = None
    target_torque: Optional[float] = None
    stiffness: Optional[float] = None
    damping: Optional[float] = None
    duration: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class MotorState:
    """Current state of a motor."""

    motor_id: str
    position: float
    velocity: float
    torque: float
    temperature: float = 25.0
    error: float = 0.0
    is_moving: bool = False


@dataclass
class Trajectory:
    """A motion trajectory."""

    motor_id: str
    waypoints: np.ndarray  # Nx1 or Nx3 (pos, vel, acc)
    timestamps: np.ndarray
    trajectory_type: TrajectoryType
    duration: float


class MotorController:
    """
    Motor controller for actuator commands.

    Implements:
    - Position/velocity/torque control
    - Trajectory execution
    - Safety limits
    - Multi-motor coordination
    """

    def __init__(
        self,
        controller_id: str = "motor_ctrl_0",
        config: Optional[MotorConfig] = None,
    ):
        self.controller_id = controller_id
        self.config = config or MotorConfig()

        # Motor states
        self._motors: Dict[str, MotorState] = {}
        self._active_trajectories: Dict[str, Trajectory] = {}

        # Command history
        self._command_history: List[MotorCommand] = []
        self._max_history = 1000

        # Safety
        self._emergency_stop = False
        self._limits: Dict[str, Tuple[float, float]] = {}

        # Callbacks
        self._on_error: Optional[Callable[[str, str], None]] = None

        # Simulated mode
        self._simulated = True

    def register_motor(
        self,
        motor_id: str,
        initial_position: float = 0.0,
        limits: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Register a motor for control.

        Args:
            motor_id: Motor identifier
            initial_position: Initial position
            limits: (min, max) position limits
        """
        self._motors[motor_id] = MotorState(
            motor_id=motor_id,
            position=initial_position,
            velocity=0.0,
            torque=0.0,
        )

        if limits:
            self._limits[motor_id] = limits

    def send_command(self, command: MotorCommand) -> bool:
        """
        Send a command to a motor.

        Args:
            command: Motor command

        Returns:
            True if command accepted
        """
        if self._emergency_stop:
            return False

        if command.motor_id not in self._motors:
            return False

        # Check limits
        if command.target_position is not None:
            if command.motor_id in self._limits:
                min_pos, max_pos = self._limits[command.motor_id]
                command.target_position = np.clip(command.target_position, min_pos, max_pos)

        # Store command
        self._command_history.append(command)
        if len(self._command_history) > self._max_history:
            self._command_history.pop(0)

        # Execute command
        if self._simulated:
            self._simulate_command(command)
        else:
            self._execute_real_command(command)

        return True

    def _simulate_command(self, command: MotorCommand) -> None:
        """Simulate command execution."""
        state = self._motors[command.motor_id]

        if command.target_position is not None:
            # Simple position control
            error = command.target_position - state.position
            velocity = np.clip(
                self.config.position_gain * error,
                -self.config.max_velocity,
                self.config.max_velocity,
            )
            state.velocity = velocity
            state.position += velocity * self.config.dt
            state.error = abs(error)
            state.is_moving = abs(error) > 0.001

        elif command.target_velocity is not None:
            state.velocity = np.clip(
                command.target_velocity, -self.config.max_velocity, self.config.max_velocity
            )
            state.position += state.velocity * self.config.dt
            state.is_moving = abs(state.velocity) > 0.001

        elif command.target_torque is not None:
            state.torque = np.clip(
                command.target_torque, -self.config.max_torque, self.config.max_torque
            )

    def _execute_real_command(self, command: MotorCommand) -> None:
        """Execute real command (stub)."""
        self._simulate_command(command)

    def execute_trajectory(
        self,
        trajectory: Trajectory,
    ) -> bool:
        """
        Execute a trajectory.

        Args:
            trajectory: Motion trajectory

        Returns:
            True if trajectory started
        """
        if trajectory.motor_id not in self._motors:
            return False

        self._active_trajectories[trajectory.motor_id] = trajectory
        return True

    def update(self, dt: float = None) -> None:
        """
        Update controller state.

        Args:
            dt: Time step (default: config.dt)
        """
        dt = dt or self.config.dt

        # Update active trajectories
        current_time = time.time()

        for motor_id, traj in list(self._active_trajectories.items()):
            # Find current position on trajectory
            elapsed = current_time - traj.timestamps[0]

            if elapsed >= traj.duration:
                # Trajectory complete
                del self._active_trajectories[motor_id]
                continue

            # Interpolate
            target = self._interpolate_trajectory(traj, elapsed)

            # Send position command
            self.send_command(
                MotorCommand(
                    motor_id=motor_id,
                    target_position=target,
                )
            )

    def _interpolate_trajectory(
        self,
        trajectory: Trajectory,
        t: float,
    ) -> float:
        """Interpolate trajectory at time t."""
        # Find surrounding waypoints
        idx = np.searchsorted(trajectory.timestamps - trajectory.timestamps[0], t)
        idx = np.clip(idx, 1, len(trajectory.waypoints) - 1)

        t0 = trajectory.timestamps[idx - 1] - trajectory.timestamps[0]
        t1 = trajectory.timestamps[idx] - trajectory.timestamps[0]
        p0 = trajectory.waypoints[idx - 1]
        p1 = trajectory.waypoints[idx]

        if trajectory.trajectory_type == TrajectoryType.LINEAR:
            alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0
            return p0 + alpha * (p1 - p0)

        elif trajectory.trajectory_type == TrajectoryType.CUBIC:
            # Cubic interpolation
            tau = (t - t0) / (t1 - t0) if t1 != t0 else 0
            h00 = 2 * tau**3 - 3 * tau**2 + 1
            h10 = tau**3 - 2 * tau**2 + tau
            h01 = -2 * tau**3 + 3 * tau**2
            h11 = tau**3 - tau**2

            return h00 * p0 + h01 * p1

        return float(trajectory.waypoints[idx])

    def get_state(self, motor_id: str) -> Optional[MotorState]:
        """Get state of a motor."""
        return self._motors.get(motor_id)

    def get_all_states(self) -> Dict[str, MotorState]:
        """Get states of all motors."""
        return dict(self._motors)

    def emergency_stop(self) -> None:
        """Trigger emergency stop."""
        self._emergency_stop = True

        # Stop all motors
        for motor_id in self._motors:
            self.send_command(
                MotorCommand(
                    motor_id=motor_id,
                    target_velocity=0.0,
                )
            )

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop."""
        self._emergency_stop = False

    def is_emergency_stopped(self) -> bool:
        """Check if emergency stopped."""
        return self._emergency_stop

    def set_error_callback(
        self,
        callback: Callable[[str, str], None],
    ) -> None:
        """Set error callback."""
        self._on_error = callback

    def statistics(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            "controller_id": self.controller_id,
            "n_motors": len(self._motors),
            "active_trajectories": len(self._active_trajectories),
            "commands_sent": len(self._command_history),
            "emergency_stopped": self._emergency_stop,
            "control_mode": self.config.control_mode.value,
        }


class TrajectoryPlanner:
    """
    Plan motion trajectories.

    Implements:
    - Point-to-point motion
    - Via-point trajectories
    - Minimum jerk planning
    - Multi-joint coordination
    """

    def __init__(self):
        self._planned_trajectories: List[Trajectory] = []

    def plan_point_to_point(
        self,
        motor_id: str,
        start_pos: float,
        end_pos: float,
        duration: float,
        trajectory_type: TrajectoryType = TrajectoryType.MINIMUM_JERK,
    ) -> Trajectory:
        """
        Plan point-to-point trajectory.

        Args:
            motor_id: Motor identifier
            start_pos: Starting position
            end_pos: Ending position
            duration: Trajectory duration
            trajectory_type: Type of interpolation

        Returns:
            Planned trajectory
        """
        n_points = int(duration * 100)  # 100 Hz
        timestamps = np.linspace(0, duration, n_points) + time.time()

        if trajectory_type == TrajectoryType.MINIMUM_JERK:
            waypoints = self._minimum_jerk(start_pos, end_pos, n_points)
        elif trajectory_type == TrajectoryType.CUBIC:
            waypoints = self._cubic_interpolation(start_pos, end_pos, n_points)
        else:
            waypoints = np.linspace(start_pos, end_pos, n_points)

        trajectory = Trajectory(
            motor_id=motor_id,
            waypoints=waypoints,
            timestamps=timestamps,
            trajectory_type=trajectory_type,
            duration=duration,
        )

        self._planned_trajectories.append(trajectory)
        return trajectory

    def _minimum_jerk(
        self,
        start: float,
        end: float,
        n_points: int,
    ) -> np.ndarray:
        """Generate minimum jerk trajectory."""
        tau = np.linspace(0, 1, n_points)
        # Minimum jerk polynomial
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        return start + (end - start) * s

    def _cubic_interpolation(
        self,
        start: float,
        end: float,
        n_points: int,
    ) -> np.ndarray:
        """Generate cubic interpolation."""
        tau = np.linspace(0, 1, n_points)
        # Cubic with zero velocity at endpoints
        s = 3 * tau**2 - 2 * tau**3
        return start + (end - start) * s

    def plan_via_points(
        self,
        motor_id: str,
        via_points: List[float],
        durations: List[float],
        trajectory_type: TrajectoryType = TrajectoryType.MINIMUM_JERK,
    ) -> Trajectory:
        """
        Plan trajectory through via points.

        Args:
            motor_id: Motor identifier
            via_points: List of positions to pass through
            durations: Duration for each segment
            trajectory_type: Type of interpolation

        Returns:
            Planned trajectory
        """
        all_waypoints = []
        all_timestamps = []
        current_time = time.time()

        for i in range(len(via_points) - 1):
            segment = self.plan_point_to_point(
                motor_id,
                via_points[i],
                via_points[i + 1],
                durations[i],
                trajectory_type,
            )

            # Adjust timestamps
            segment_timestamps = segment.timestamps - segment.timestamps[0] + current_time

            all_waypoints.extend(segment.waypoints)
            all_timestamps.extend(segment_timestamps)

            current_time = segment_timestamps[-1]

        return Trajectory(
            motor_id=motor_id,
            waypoints=np.array(all_waypoints),
            timestamps=np.array(all_timestamps),
            trajectory_type=trajectory_type,
            duration=sum(durations),
        )

    def plan_coordinated(
        self,
        motor_configs: Dict[str, Tuple[float, float]],
        duration: float,
        trajectory_type: TrajectoryType = TrajectoryType.MINIMUM_JERK,
    ) -> Dict[str, Trajectory]:
        """
        Plan coordinated multi-motor trajectory.

        Args:
            motor_configs: motor_id -> (start_pos, end_pos)
            duration: Total duration
            trajectory_type: Type of interpolation

        Returns:
            Dictionary of motor trajectories
        """
        trajectories = {}

        for motor_id, (start, end) in motor_configs.items():
            trajectories[motor_id] = self.plan_point_to_point(
                motor_id, start, end, duration, trajectory_type
            )

        return trajectories

    def compute_velocity_profile(
        self,
        trajectory: Trajectory,
    ) -> np.ndarray:
        """
        Compute velocity profile of trajectory.

        Args:
            trajectory: Input trajectory

        Returns:
            Velocity at each waypoint
        """
        dt = np.diff(trajectory.timestamps)
        dp = np.diff(trajectory.waypoints)

        # Avoid division by zero
        dt = np.where(dt == 0, 1e-6, dt)

        velocity = dp / dt

        # Pad to match waypoints length
        velocity = np.append(velocity, velocity[-1])

        return velocity

    def check_limits(
        self,
        trajectory: Trajectory,
        max_velocity: float,
        max_acceleration: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if trajectory respects limits.

        Args:
            trajectory: Trajectory to check
            max_velocity: Maximum allowed velocity
            max_acceleration: Maximum allowed acceleration

        Returns:
            (is_valid, error_message)
        """
        velocity = self.compute_velocity_profile(trajectory)

        if np.any(np.abs(velocity) > max_velocity):
            return False, f"Velocity exceeds limit: {np.max(np.abs(velocity)):.2f} > {max_velocity}"

        # Compute acceleration
        dt = np.diff(trajectory.timestamps)
        dt = np.where(dt == 0, 1e-6, dt)
        acceleration = np.diff(velocity) / dt[:-1]

        if np.any(np.abs(acceleration) > max_acceleration):
            return (
                False,
                f"Acceleration exceeds limit: {np.max(np.abs(acceleration)):.2f} > {max_acceleration}",
            )

        return True, None

    def statistics(self) -> Dict[str, Any]:
        """Get planner statistics."""
        return {
            "planned_trajectories": len(self._planned_trajectories),
        }
