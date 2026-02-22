"""
Comprehensive tests for neuro-ground module.

Tests sensors, actuators, sim2real, and calibration components.
"""

import pytest
import numpy as np
import time
import os

from neuro.modules.ground.sensors.camera import (
    CameraType,
    ColorSpace,
    CameraConfig,
    CameraFrame,
    CameraCalibration,
    Camera,
    VisionProcessor,
)
from neuro.modules.ground.sensors.microphone import (
    AudioFormat,
    MicrophoneConfig,
    AudioBuffer,
    AudioEvent,
    Microphone,
    AudioProcessor,
)
from neuro.modules.ground.sensors.proprioception import (
    JointType,
    JointState,
    EndEffectorState,
    BodyState,
    IMUReading,
    ProprioceptionSensor,
    ProprioceptionProcessor,
)

from neuro.modules.ground.actuators.motor_controller import (
    ControlMode,
    TrajectoryType,
    MotorConfig,
    MotorCommand,
    MotorState,
    Trajectory,
    MotorController,
    TrajectoryPlanner,
)
from neuro.modules.ground.actuators.speech_synth import (
    Voice,
    EmotionType,
    SpeechConfig,
    Phoneme,
    Utterance,
    ProsodyParams,
    SpeechSynthesizer,
    ProsodyController,
)

from neuro.modules.ground.sim2real import (
    DomainType,
    RandomizationType,
    RandomizationConfig,
    RealityGapMetrics,
    DomainRandomization,
    RealityGap,
    SimToRealTransfer,
)
from neuro.modules.ground.calibration import (
    CalibrationType,
    CalibrationStatus,
    CalibrationConfig,
    CalibrationResult,
    SensorCalibrator,
)

# =============================================================================
# CAMERA TESTS
# =============================================================================


class TestCamera:
    """Tests for Camera."""

    def test_camera_creation(self):
        """Test Camera creation."""
        camera = Camera(camera_id="test_cam")
        assert camera.camera_id == "test_cam"
        assert not camera.is_open()

    def test_camera_open_close(self):
        """Test opening and closing camera."""
        camera = Camera()
        assert camera.open()
        assert camera.is_open()
        camera.close()
        assert not camera.is_open()

    def test_camera_capture(self):
        """Test frame capture."""
        camera = Camera()
        camera.open()

        frame = camera.capture()
        assert frame is not None
        assert frame.width == camera.config.width
        assert frame.height == camera.config.height

    def test_camera_capture_closed(self):
        """Test capture when closed."""
        camera = Camera()
        frame = camera.capture()
        assert frame is None

    def test_camera_custom_config(self):
        """Test with custom config."""
        config = CameraConfig(width=1280, height=720, fps=60.0)
        camera = Camera(config=config)
        camera.open()

        frame = camera.capture()
        assert frame.width == 1280
        assert frame.height == 720

    def test_camera_depth(self):
        """Test depth camera."""
        config = CameraConfig(camera_type=CameraType.RGBD)
        camera = Camera(config=config)
        camera.open()

        frame = camera.capture()
        assert frame.depth_data is not None

    def test_camera_color_convert(self):
        """Test color space conversion."""
        camera = Camera()
        camera.open()

        frame = camera.capture()
        gray_frame = camera.convert_color(frame, ColorSpace.GRAY)

        assert gray_frame.color_space == ColorSpace.GRAY
        assert gray_frame.channels == 1

    def test_camera_statistics(self):
        """Test camera statistics."""
        camera = Camera()
        camera.open()
        camera.capture()

        stats = camera.statistics()
        assert stats["frame_count"] == 1


class TestVisionProcessor:
    """Tests for VisionProcessor."""

    def test_processor_creation(self):
        """Test VisionProcessor creation."""
        processor = VisionProcessor()
        assert processor is not None

    def test_processor_preprocess(self):
        """Test frame preprocessing."""
        processor = VisionProcessor()
        camera = Camera()
        camera.open()
        frame = camera.capture()

        processed = processor.preprocess(frame, normalize=True)
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0

    def test_processor_resize(self):
        """Test resize operation."""
        processor = VisionProcessor()
        camera = Camera()
        camera.open()
        frame = camera.capture()

        processed = processor.preprocess(frame, resize=(320, 240))
        assert processed.shape[1] == 320
        assert processed.shape[0] == 240

    def test_processor_optical_flow(self):
        """Test optical flow computation."""
        processor = VisionProcessor()
        camera = Camera()
        camera.open()

        frame1 = camera.capture()
        flow = processor.compute_optical_flow(frame1)
        assert flow is None  # First frame

        frame2 = camera.capture()
        flow = processor.compute_optical_flow(frame2)
        assert flow is not None
        assert flow.shape[2] == 2

    def test_processor_edge_detection(self):
        """Test edge detection."""
        processor = VisionProcessor()
        camera = Camera()
        camera.open()
        frame = camera.capture()

        edges = processor.extract_edges(frame)
        assert edges.shape == (frame.height, frame.width)

    def test_processor_histogram(self):
        """Test histogram computation."""
        processor = VisionProcessor()
        camera = Camera()
        camera.open()
        frame = camera.capture()

        histograms = processor.compute_histogram(frame)
        assert "red" in histograms or "gray" in histograms


# =============================================================================
# MICROPHONE TESTS
# =============================================================================


class TestMicrophone:
    """Tests for Microphone."""

    def test_microphone_creation(self):
        """Test Microphone creation."""
        mic = Microphone(mic_id="test_mic")
        assert mic.mic_id == "test_mic"

    def test_microphone_open_close(self):
        """Test opening and closing microphone."""
        mic = Microphone()
        assert mic.open()
        assert mic.is_open()
        mic.close()
        assert not mic.is_open()

    def test_microphone_read(self):
        """Test audio read."""
        mic = Microphone()
        mic.open()

        buffer = mic.read()
        assert buffer is not None
        assert buffer.sample_rate == mic.config.sample_rate

    def test_microphone_custom_config(self):
        """Test with custom config."""
        config = MicrophoneConfig(sample_rate=44100, buffer_size=2048)
        mic = Microphone(config=config)
        mic.open()

        buffer = mic.read()
        assert buffer.sample_rate == 44100

    def test_microphone_recent_audio(self):
        """Test getting recent audio."""
        mic = Microphone()
        mic.open()

        # Read several buffers
        for _ in range(5):
            mic.read()

        recent = mic.get_recent_audio(0.5)
        assert len(recent) > 0

    def test_microphone_statistics(self):
        """Test microphone statistics."""
        mic = Microphone()
        mic.open()
        mic.read()
        mic.read()

        stats = mic.statistics()
        assert stats["buffer_count"] == 2


class TestAudioProcessor:
    """Tests for AudioProcessor."""

    def test_processor_creation(self):
        """Test AudioProcessor creation."""
        processor = AudioProcessor()
        assert processor.sample_rate == 16000

    def test_processor_spectrum(self):
        """Test spectrum computation."""
        processor = AudioProcessor()
        mic = Microphone()
        mic.open()
        buffer = mic.read(n_samples=1024)

        spectrum = processor.compute_spectrum(buffer)
        assert spectrum.shape[1] == 257  # window_size/2 + 1

    def test_processor_mel_spectrum(self):
        """Test mel spectrum computation."""
        processor = AudioProcessor()
        mic = Microphone()
        mic.open()
        buffer = mic.read(n_samples=1024)

        mel = processor.compute_mel_spectrum(buffer, n_mels=40)
        assert mel.shape[1] == 40

    def test_processor_mfcc(self):
        """Test MFCC extraction."""
        processor = AudioProcessor()
        mic = Microphone()
        mic.open()
        buffer = mic.read(n_samples=1024)

        mfcc = processor.extract_mfcc(buffer, n_mfcc=13)
        assert mfcc.shape[1] == 13

    def test_processor_vad(self):
        """Test voice activity detection."""
        processor = AudioProcessor()
        mic = Microphone()
        mic.open()
        buffer = mic.read()

        detected, confidence = processor.detect_voice_activity(buffer)
        assert isinstance(detected, (bool, np.bool_))
        assert 0 <= confidence <= 1

    def test_processor_pitch(self):
        """Test pitch estimation."""
        processor = AudioProcessor()
        mic = Microphone()
        mic.open()
        buffer = mic.read(n_samples=2048)

        pitch = processor.compute_pitch(buffer)
        # May or may not detect pitch depending on test signal


# =============================================================================
# PROPRIOCEPTION TESTS
# =============================================================================


class TestProprioceptionSensor:
    """Tests for ProprioceptionSensor."""

    def test_sensor_creation(self):
        """Test ProprioceptionSensor creation."""
        sensor = ProprioceptionSensor(n_joints=7)
        assert sensor.n_joints == 7

    def test_sensor_read_state(self):
        """Test reading body state."""
        sensor = ProprioceptionSensor()
        state = sensor.read_state()

        assert state is not None
        assert len(state.joint_states) == sensor.n_joints

    def test_sensor_read_joint(self):
        """Test reading single joint."""
        sensor = ProprioceptionSensor()
        joint = sensor.read_joint("joint_0")

        assert joint is not None
        assert joint.joint_type == JointType.REVOLUTE

    def test_sensor_read_imu(self):
        """Test IMU reading."""
        sensor = ProprioceptionSensor()
        imu = sensor.read_imu()

        assert imu.acceleration.shape == (3,)
        assert imu.angular_velocity.shape == (3,)

    def test_sensor_joint_positions(self):
        """Test getting joint positions."""
        sensor = ProprioceptionSensor()
        sensor.read_state()  # Update

        positions = sensor.get_joint_positions()
        assert len(positions) == sensor.n_joints

    def test_sensor_state_history(self):
        """Test state history."""
        sensor = ProprioceptionSensor()

        for _ in range(5):
            sensor.read_state()

        history = sensor.get_state_history(n_states=3)
        assert len(history) == 3

    def test_sensor_forward_kinematics(self):
        """Test forward kinematics."""
        sensor = ProprioceptionSensor(n_joints=3)
        positions = np.array([0.0, 0.5, -0.5])

        ee_positions = sensor.forward_kinematics(positions)
        assert "hand" in ee_positions


class TestProprioceptionProcessor:
    """Tests for ProprioceptionProcessor."""

    def test_processor_creation(self):
        """Test ProprioceptionProcessor creation."""
        processor = ProprioceptionProcessor()
        assert processor is not None

    def test_processor_velocity_estimation(self):
        """Test velocity estimation."""
        processor = ProprioceptionProcessor()
        sensor = ProprioceptionSensor()

        states = [sensor.read_state() for _ in range(3)]
        velocities = processor.estimate_velocity(states)

        assert len(velocities) > 0

    def test_processor_jacobian(self):
        """Test Jacobian computation."""
        processor = ProprioceptionProcessor()
        positions = np.array([0.1, 0.2, 0.3])

        jacobian = processor.compute_jacobian(positions)
        assert jacobian.shape == (2, 3)

    def test_processor_movement_analysis(self):
        """Test movement analysis."""
        processor = ProprioceptionProcessor()
        sensor = ProprioceptionSensor()

        states = [sensor.read_state() for _ in range(10)]
        analysis = processor.analyze_movement(states)

        assert "movement_type" in analysis
        assert "average_velocity" in analysis

    def test_processor_energy(self):
        """Test energy computation."""
        processor = ProprioceptionProcessor()
        sensor = ProprioceptionSensor()
        state = sensor.read_state()

        energy = processor.compute_energy(state)
        assert "kinetic_energy" in energy
        assert "potential_energy" in energy


# =============================================================================
# MOTOR CONTROLLER TESTS
# =============================================================================


class TestMotorController:
    """Tests for MotorController."""

    def test_controller_creation(self):
        """Test MotorController creation."""
        controller = MotorController()
        assert controller is not None

    def test_controller_register_motor(self):
        """Test registering motors."""
        controller = MotorController()
        controller.register_motor("motor_0", initial_position=0.5)

        state = controller.get_state("motor_0")
        assert state is not None
        assert state.position == 0.5

    def test_controller_send_command(self):
        """Test sending commands."""
        controller = MotorController()
        controller.register_motor("motor_0")

        command = MotorCommand(
            motor_id="motor_0",
            target_position=1.0,
        )

        success = controller.send_command(command)
        assert success

    def test_controller_position_control(self):
        """Test position control."""
        controller = MotorController()
        controller.register_motor("motor_0", initial_position=0.0)

        # Send position command
        controller.send_command(
            MotorCommand(
                motor_id="motor_0",
                target_position=0.5,
            )
        )

        state = controller.get_state("motor_0")
        assert state.is_moving or abs(state.error) > 0

    def test_controller_emergency_stop(self):
        """Test emergency stop."""
        controller = MotorController()
        controller.register_motor("motor_0")

        controller.emergency_stop()
        assert controller.is_emergency_stopped()

        # Commands should fail
        success = controller.send_command(
            MotorCommand(
                motor_id="motor_0",
                target_position=1.0,
            )
        )
        assert not success

        controller.reset_emergency_stop()
        assert not controller.is_emergency_stopped()

    def test_controller_statistics(self):
        """Test controller statistics."""
        controller = MotorController()
        controller.register_motor("motor_0")

        stats = controller.statistics()
        assert stats["n_motors"] == 1


class TestTrajectoryPlanner:
    """Tests for TrajectoryPlanner."""

    def test_planner_creation(self):
        """Test TrajectoryPlanner creation."""
        planner = TrajectoryPlanner()
        assert planner is not None

    def test_planner_point_to_point(self):
        """Test point-to-point planning."""
        planner = TrajectoryPlanner()

        trajectory = planner.plan_point_to_point(
            motor_id="motor_0",
            start_pos=0.0,
            end_pos=1.0,
            duration=2.0,
        )

        assert trajectory.motor_id == "motor_0"
        assert trajectory.duration == 2.0
        assert trajectory.waypoints[0] == pytest.approx(0.0, abs=0.1)
        assert trajectory.waypoints[-1] == pytest.approx(1.0, abs=0.1)

    def test_planner_minimum_jerk(self):
        """Test minimum jerk trajectory."""
        planner = TrajectoryPlanner()

        trajectory = planner.plan_point_to_point(
            motor_id="motor_0",
            start_pos=0.0,
            end_pos=1.0,
            duration=1.0,
            trajectory_type=TrajectoryType.MINIMUM_JERK,
        )

        # Check smoothness (velocity should be zero at endpoints)
        velocity = planner.compute_velocity_profile(trajectory)
        assert abs(velocity[0]) < 0.1
        assert abs(velocity[-1]) < 0.1

    def test_planner_via_points(self):
        """Test via-point trajectory."""
        planner = TrajectoryPlanner()

        trajectory = planner.plan_via_points(
            motor_id="motor_0",
            via_points=[0.0, 0.5, 1.0],
            durations=[1.0, 1.0],
        )

        assert trajectory.duration == 2.0

    def test_planner_coordinated(self):
        """Test coordinated multi-motor planning."""
        planner = TrajectoryPlanner()

        trajectories = planner.plan_coordinated(
            motor_configs={
                "motor_0": (0.0, 1.0),
                "motor_1": (0.5, -0.5),
            },
            duration=2.0,
        )

        assert len(trajectories) == 2
        assert "motor_0" in trajectories
        assert "motor_1" in trajectories

    def test_planner_limit_check(self):
        """Test limit checking."""
        planner = TrajectoryPlanner()

        # Fast trajectory
        trajectory = planner.plan_point_to_point(
            motor_id="motor_0",
            start_pos=0.0,
            end_pos=10.0,
            duration=0.1,  # Very short duration
        )

        valid, error = planner.check_limits(
            trajectory,
            max_velocity=5.0,
            max_acceleration=10.0,
        )

        # Should exceed limits
        assert not valid or error is not None


# =============================================================================
# SPEECH SYNTHESIZER TESTS
# =============================================================================


class TestSpeechSynthesizer:
    """Tests for SpeechSynthesizer."""

    def test_synth_creation(self):
        """Test SpeechSynthesizer creation."""
        synth = SpeechSynthesizer()
        assert synth is not None

    def test_synth_synthesize(self):
        """Test speech synthesis."""
        synth = SpeechSynthesizer()
        utterance = synth.synthesize("hello world")

        assert utterance.text == "hello world"
        assert len(utterance.audio) > 0
        assert utterance.duration_seconds > 0

    def test_synth_with_prosody(self):
        """Test synthesis with prosody."""
        synth = SpeechSynthesizer()
        prosody = ProsodyParams(emotion=EmotionType.HAPPY)

        utterance = synth.synthesize("I am happy", prosody)
        assert utterance.metadata["prosody"]["emotion"] == "happy"

    def test_synth_different_voices(self):
        """Test different voices."""
        config = SpeechConfig(voice=Voice.FEMALE_1)
        synth = SpeechSynthesizer(config=config)

        utterance = synth.synthesize("test")
        assert utterance.metadata["voice"] == "female_1"

    def test_synth_statistics(self):
        """Test synthesizer statistics."""
        synth = SpeechSynthesizer()
        synth.synthesize("hello")
        synth.synthesize("world")

        stats = synth.statistics()
        assert stats["utterances_generated"] == 2


class TestProsodyController:
    """Tests for ProsodyController."""

    def test_controller_creation(self):
        """Test ProsodyController creation."""
        controller = ProsodyController()
        assert controller is not None

    def test_controller_emotion_prosody(self):
        """Test emotion prosody."""
        controller = ProsodyController()

        happy = controller.get_emotion_prosody(EmotionType.HAPPY)
        sad = controller.get_emotion_prosody(EmotionType.SAD)

        assert happy.pitch_shift > sad.pitch_shift
        assert happy.energy_scale > sad.energy_scale

    def test_controller_blend_emotions(self):
        """Test emotion blending."""
        controller = ProsodyController()

        blended = controller.blend_emotions(
            EmotionType.HAPPY,
            EmotionType.SAD,
            weight=0.5,
        )

        happy = controller.get_emotion_prosody(EmotionType.HAPPY)
        sad = controller.get_emotion_prosody(EmotionType.SAD)

        expected_pitch = (happy.pitch_shift + sad.pitch_shift) / 2
        assert abs(blended.pitch_shift - expected_pitch) < 0.1

    def test_controller_emphasis(self):
        """Test applying emphasis."""
        controller = ProsodyController()
        base = controller.get_emotion_prosody(EmotionType.NEUTRAL)

        emphasized = controller.apply_emphasis(base, emphasis_level=1.5)
        assert emphasized.energy_scale > base.energy_scale

    def test_controller_sentiment_analysis(self):
        """Test sentiment analysis."""
        controller = ProsodyController()

        emotion = controller.analyze_text_sentiment("I am so happy!")
        assert emotion == EmotionType.HAPPY

        emotion = controller.analyze_text_sentiment("I am sad")
        assert emotion == EmotionType.SAD


# =============================================================================
# SIM2REAL TESTS
# =============================================================================


class TestDomainRandomization:
    """Tests for DomainRandomization."""

    def test_randomization_creation(self):
        """Test DomainRandomization creation."""
        rand = DomainRandomization()
        assert rand is not None

    def test_randomize_visual(self):
        """Test visual randomization."""
        rand = DomainRandomization(seed=42)
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128

        randomized = rand.randomize_visual(image)

        # Should be different
        assert not np.array_equal(image, randomized)

    def test_randomize_dynamics(self):
        """Test dynamics randomization."""
        rand = DomainRandomization(seed=42)

        new_mass, new_friction, new_damping = rand.randomize_dynamics(
            mass=1.0, friction=0.5, damping=0.1
        )

        # Should be within range
        assert 0.5 <= new_mass / 1.0 <= 1.5
        assert 0.25 <= new_friction / 0.5 <= 1.5

    def test_add_sensor_noise(self):
        """Test sensor noise."""
        rand = DomainRandomization(seed=42)
        observation = np.zeros(10)

        noisy = rand.add_sensor_noise(observation)

        # Should have noise
        assert not np.array_equal(observation, noisy)

    def test_randomization_history(self):
        """Test randomization history."""
        rand = DomainRandomization()
        rand.randomize_visual(np.ones((10, 10)))
        rand.randomize_dynamics(1.0, 0.5, 0.1)

        history = rand.get_history()
        assert len(history) == 2


class TestRealityGap:
    """Tests for RealityGap."""

    def test_gap_creation(self):
        """Test RealityGap creation."""
        gap = RealityGap()
        assert gap is not None

    def test_gap_add_samples(self):
        """Test adding samples."""
        gap = RealityGap()

        gap.add_sim_sample(np.array([1, 2, 3]), reward=1.0)
        gap.add_real_sample(np.array([1.1, 2.1, 3.1]), reward=0.9)

        stats = gap.statistics()
        assert stats["sim_samples"] == 1
        assert stats["real_samples"] == 1

    def test_gap_distribution_distance(self):
        """Test distribution distance."""
        gap = RealityGap()

        # Add similar samples
        for _ in range(10):
            gap.add_sim_sample(np.random.randn(5))
            gap.add_real_sample(np.random.randn(5) + 1.0)  # Shifted

        distance = gap.compute_distribution_distance()
        assert distance > 0

    def test_gap_reward_gap(self):
        """Test reward gap."""
        gap = RealityGap()

        for _ in range(10):
            gap.add_sim_sample(np.zeros(3), reward=1.0)
            gap.add_real_sample(np.zeros(3), reward=0.7)

        reward_gap = gap.compute_reward_gap()
        assert abs(reward_gap - 0.3) < 0.1


class TestSimToRealTransfer:
    """Tests for SimToRealTransfer."""

    def test_transfer_creation(self):
        """Test SimToRealTransfer creation."""
        transfer = SimToRealTransfer()
        assert transfer is not None

    def test_transfer_observation(self):
        """Test observation transfer."""
        transfer = SimToRealTransfer()
        sim_obs = np.array([1.0, 2.0, 3.0])

        real_obs = transfer.transfer_observation(sim_obs)
        assert real_obs.shape == sim_obs.shape

    def test_transfer_action(self):
        """Test action transfer."""
        transfer = SimToRealTransfer()
        sim_action = np.array([0.5, -0.5])

        real_action = transfer.transfer_action(sim_action, action_scale=0.9)
        np.testing.assert_array_almost_equal(real_action, sim_action * 0.9)

    def test_transfer_evaluation(self):
        """Test transfer evaluation."""
        transfer = SimToRealTransfer()

        # Add some samples
        for _ in range(5):
            transfer.reality_gap.add_sim_sample(np.random.randn(3), reward=1.0)
            transfer.reality_gap.add_real_sample(np.random.randn(3), reward=0.8)

        eval_result = transfer.evaluate_transfer_quality()
        assert "distribution_distance" in eval_result
        assert "reward_gap" in eval_result


# =============================================================================
# CALIBRATION TESTS
# =============================================================================


class TestSensorCalibrator:
    """Tests for SensorCalibrator."""

    def test_calibrator_creation(self):
        """Test SensorCalibrator creation."""
        calibrator = SensorCalibrator()
        assert calibrator is not None

    def test_calibrator_imu(self):
        """Test IMU calibration."""
        calibrator = SensorCalibrator()

        # Static samples (should measure gravity)
        samples = [np.array([0.1, 0.1, 9.9, 0.01, 0.01, -0.01]) for _ in range(50)]

        result = calibrator.calibrate_imu("imu_0", samples)

        assert result.status == CalibrationStatus.CALIBRATED
        assert "accelerometer_bias" in result.parameters

    def test_calibrator_joint_encoder(self):
        """Test joint encoder calibration."""
        calibrator = SensorCalibrator()

        # Encoder readings and known positions
        readings = [0, 100, 200, 300, 400]
        positions = [0.0, 0.5, 1.0, 1.5, 2.0]

        result = calibrator.calibrate_joint_encoder("encoder_0", readings, positions)

        assert result.status == CalibrationStatus.CALIBRATED
        assert "scale" in result.parameters

    def test_calibrator_force_sensor(self):
        """Test force sensor calibration."""
        calibrator = SensorCalibrator()

        # Sensor readings and known forces
        readings = [np.array([i * 10]) for i in range(10)]
        forces = [np.array([i * 1.0]) for i in range(10)]

        result = calibrator.calibrate_force_sensor("force_0", readings, forces)

        assert result.status == CalibrationStatus.CALIBRATED

    def test_calibrator_apply(self):
        """Test applying calibration."""
        calibrator = SensorCalibrator()

        # Calibrate encoder
        readings = [0, 1000]
        positions = [0.0, 1.0]
        calibrator.calibrate_joint_encoder("encoder_0", readings, positions)

        # Apply calibration
        raw = np.array([500])
        calibrated = calibrator.apply_calibration("encoder_0", raw)

        assert calibrated == pytest.approx(0.5, abs=0.01)

    def test_calibrator_status(self):
        """Test calibration status."""
        calibrator = SensorCalibrator()

        assert calibrator.get_status("unknown") == CalibrationStatus.NOT_CALIBRATED

        calibrator.calibrate_joint_encoder("enc_0", [0, 100], [0.0, 1.0])
        assert calibrator.get_status("enc_0") == CalibrationStatus.CALIBRATED


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for neuro-ground."""

    def test_camera_to_processor(self):
        """Test camera to processor pipeline."""
        camera = Camera()
        processor = VisionProcessor()

        camera.open()
        frame = camera.capture()

        processed = processor.preprocess(frame)
        edges = processor.extract_edges(frame)

        assert processed.shape[:2] == (frame.height, frame.width)
        assert edges.shape == (frame.height, frame.width)

    def test_microphone_to_processor(self):
        """Test microphone to processor pipeline."""
        mic = Microphone()
        processor = AudioProcessor()

        mic.open()
        buffer = mic.read(n_samples=2048)

        mfcc = processor.extract_mfcc(buffer)
        vad, _ = processor.detect_voice_activity(buffer)

        assert mfcc.shape[1] == 13
        assert isinstance(vad, (bool, np.bool_))

    def test_proprioception_to_motor(self):
        """Test proprioception feedback to motor control."""
        sensor = ProprioceptionSensor(n_joints=3)
        controller = MotorController()
        planner = TrajectoryPlanner()

        # Register motors
        for i in range(3):
            controller.register_motor(f"joint_{i}")

        # Read current state
        state = sensor.read_state()
        current_positions = sensor.get_joint_positions()

        # Plan trajectory
        trajectory = planner.plan_point_to_point(
            motor_id="joint_0",
            start_pos=current_positions[0],
            end_pos=current_positions[0] + 0.5,
            duration=1.0,
        )

        assert trajectory.duration == 1.0

    def test_full_sim2real_pipeline(self):
        """Test complete sim2real pipeline."""
        transfer = SimToRealTransfer()
        camera = Camera()

        camera.open()
        frame = camera.capture()

        # Randomize visual input
        randomized = transfer.randomization.randomize_visual(frame.data)

        # Add sensor noise to observation
        obs = np.array([1.0, 2.0, 3.0])
        noisy_obs = transfer.transfer_observation(obs)

        # Track reality gap
        transfer.reality_gap.add_sim_sample(obs, reward=1.0)
        transfer.reality_gap.add_real_sample(noisy_obs, reward=0.9)

        metrics = transfer.reality_gap.compute_metrics()
        assert metrics.reward_difference >= 0

    def test_calibration_pipeline(self):
        """Test full calibration pipeline."""
        calibrator = SensorCalibrator()

        # Calibrate multiple sensors
        calibrator.calibrate_joint_encoder("enc_0", [0, 500, 1000], [0.0, 0.5, 1.0])
        calibrator.calibrate_force_sensor(
            "force_0",
            [np.array([i * 100]) for i in range(5)],
            [np.array([i * 1.0]) for i in range(5)],
        )

        # Apply calibrations
        enc_raw = np.array([250])
        enc_calibrated = calibrator.apply_calibration("enc_0", enc_raw)

        force_raw = np.array([300])
        force_calibrated = calibrator.apply_calibration("force_0", force_raw)

        assert enc_calibrated == pytest.approx(0.25, abs=0.05)

        stats = calibrator.statistics()
        assert stats["n_calibrations"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
