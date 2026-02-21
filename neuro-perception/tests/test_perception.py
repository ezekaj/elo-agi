"""
Tests for neuro-perception module.

Tests visual, auditory, and multimodal processing.
"""

import pytest
import numpy as np
import os

from neuro.modules.perception.visual.retina import Retina, RetinaOutput, PhotoreceptorType, GanglionCellType
from neuro.modules.perception.visual.v1_v2 import V1Processor, V2Processor, V1Output, V2Output
from neuro.modules.perception.visual.v4_it import V4Processor, ITProcessor, V4Output, ITOutput, ShapeDescriptor
from neuro.modules.perception.visual.dorsal_ventral import DorsalStream, VentralStream, VisualPathways

from neuro.modules.perception.auditory.cochlea import Cochlea, CochleaOutput, GammatoneFilterbank
from neuro.modules.perception.auditory.a1 import A1Processor, A1Output
from neuro.modules.perception.auditory.speech import SpeechProcessor, SpeechOutput, PhonemeRecognizer, FormantTracker, VoicingDetector

from neuro.modules.perception.multimodal.binding import (
    CrossModalBinder, BindingOutput, ModalityInput, Modality,
    TemporalBinder, SpatialBinder, FeatureBinder
)
from neuro.modules.perception.multimodal.attention import (
    SelectiveAttention, AttentionOutput, SaliencyComputer,
    TopDownController, AttentionGate
)

from neuro.modules.perception.interface import (
    PerceptionSystem, VisualPipeline, AuditoryPipeline,
    VisualPercept, AuditoryPercept, MultimodalPercept,
    create_perception_system
)

# ============== Visual Tests ==============

class TestRetina:
    """Tests for retinal processing."""

    def test_retina_creation(self):
        """Test retina creation."""
        retina = Retina()
        assert retina.center_size == 1.0
        assert retina.surround_size == 3.0

    def test_retina_process_grayscale(self):
        """Test processing grayscale image."""
        retina = Retina()
        image = np.random.rand(64, 64)
        output = retina.process(image)

        assert isinstance(output, RetinaOutput)
        assert output.size == (64, 64)
        assert output.on_center.shape == (64, 64)
        assert output.off_center.shape == (64, 64)

    def test_retina_process_rgb(self):
        """Test processing RGB image."""
        retina = Retina()
        image = np.random.rand(64, 64, 3)
        output = retina.process(image)

        assert output.size == (64, 64)
        assert output.red_green.shape == (64, 64)
        assert output.blue_yellow.shape == (64, 64)

    def test_retina_pathways(self):
        """Test magno and parvo pathways."""
        retina = Retina()
        image = np.random.rand(32, 32, 3)
        output = retina.process(image)

        assert output.magno.shape == (32, 32)
        assert output.parvo.shape == (32, 32)

    def test_retina_adaptation(self):
        """Test light adaptation."""
        retina = Retina()
        bright = np.ones((32, 32)) * 0.9
        dark = np.ones((32, 32)) * 0.1

        retina.process(bright)
        level1 = retina._adaptation_level

        retina.process(dark)
        level2 = retina._adaptation_level

        assert level1 != level2

    def test_receptive_field_map(self):
        """Test receptive field visualization."""
        retina = Retina()
        rf_map = retina.get_receptive_field_map(size=(32, 32))

        assert rf_map.shape == (32, 32)
        # Center should be different from surround
        center_val = rf_map[16, 16]
        edge_val = rf_map[0, 0]
        assert center_val != edge_val

class TestV1V2:
    """Tests for V1 and V2 processing."""

    def test_v1_creation(self):
        """Test V1 processor creation."""
        v1 = V1Processor(n_orientations=8, n_frequencies=4)
        assert v1.n_orientations == 8
        assert v1.n_frequencies == 4
        assert len(v1._filters) == 8 * 4 * 2  # orientations * frequencies * phases

    def test_v1_gabor_bank(self):
        """Test Gabor filter bank creation."""
        v1 = V1Processor()
        filters = v1._filters

        # Check filter properties
        for f in filters:
            assert 0 <= f.orientation < np.pi
            assert f.frequency > 0
            assert f.sigma > 0

    def test_v1_process(self):
        """Test V1 processing."""
        retina = Retina()
        v1 = V1Processor(n_orientations=4, n_frequencies=2)

        image = np.random.rand(32, 32)
        retina_out = retina.process(image)
        v1_out = v1.process(retina_out)

        assert isinstance(v1_out, V1Output)
        assert v1_out.size == (32, 32)
        assert v1_out.orientation_map.shape == (32, 32, 4)
        assert v1_out.frequency_map.shape == (32, 32, 2)

    def test_v1_edge_map(self):
        """Test edge detection."""
        retina = Retina()
        v1 = V1Processor()

        # Create image with edge
        image = np.zeros((64, 64))
        image[:, 32:] = 1.0

        retina_out = retina.process(image)
        v1_out = v1.process(retina_out)

        # Edge should be detected
        assert v1_out.edge_map.max() > 0

    def test_v2_creation(self):
        """Test V2 processor creation."""
        v2 = V2Processor()
        assert v2.contour_length == 5
        assert v2.curvature_radius == 3

    def test_v2_process(self):
        """Test V2 processing."""
        retina = Retina()
        v1 = V1Processor(n_orientations=4, n_frequencies=2)
        v2 = V2Processor()

        image = np.random.rand(32, 32)
        retina_out = retina.process(image)
        v1_out = v1.process(retina_out)
        v2_out = v2.process(v1_out)

        assert isinstance(v2_out, V2Output)
        assert v2_out.size == (32, 32)
        assert v2_out.contour_map.shape == (32, 32)
        assert v2_out.curvature_map.shape == (32, 32)

    def test_v2_corner_detection(self):
        """Test corner detection."""
        retina = Retina()
        v1 = V1Processor()
        v2 = V2Processor()

        # Create L-shaped image
        image = np.zeros((64, 64))
        image[20:40, 20:30] = 1.0
        image[30:40, 20:40] = 1.0

        retina_out = retina.process(image)
        v1_out = v1.process(retina_out)
        v2_out = v2.process(v1_out)

        # Corners should be detected
        assert v2_out.corner_map.max() > 0

class TestV4IT:
    """Tests for V4 and IT processing."""

    def test_v4_creation(self):
        """Test V4 processor creation."""
        v4 = V4Processor(n_curvature_cells=8, n_shape_cells=16)
        assert v4.n_curvature_cells == 8
        assert len(v4._shape_templates) > 0

    def test_v4_shape_templates(self):
        """Test shape template creation."""
        v4 = V4Processor()
        templates = v4._shape_templates

        assert len(templates) >= 5  # At least basic shapes
        for t in templates:
            assert t.shape == (16, 16)

    def test_v4_process(self):
        """Test V4 processing."""
        retina = Retina()
        v1 = V1Processor(n_orientations=4, n_frequencies=2)
        v2 = V2Processor()
        v4 = V4Processor()

        image = np.random.rand(32, 32)
        retina_out = retina.process(image)
        v1_out = v1.process(retina_out)
        v2_out = v2.process(v1_out)
        v4_out = v4.process(v2_out)

        assert isinstance(v4_out, V4Output)
        assert v4_out.size == (32, 32)
        assert v4_out.shape_map.shape == (32, 32)

    def test_v4_intermediate_forms(self):
        """Test shape extraction."""
        retina = Retina()
        v1 = V1Processor()
        v2 = V2Processor()
        v4 = V4Processor()

        # Create simple shape
        image = np.zeros((64, 64))
        image[20:40, 20:40] = 1.0  # Square

        retina_out = retina.process(image)
        v1_out = v1.process(retina_out)
        v2_out = v2.process(v1_out)
        v4_out = v4.process(v2_out)

        # Should extract shape descriptors
        assert isinstance(v4_out.intermediate_forms, list)

    def test_it_creation(self):
        """Test IT processor creation."""
        it = ITProcessor(n_categories=20)
        assert len(it._prototypes) > 0
        assert it.embedding_dim == 128

    def test_it_categories(self):
        """Test IT category prototypes."""
        it = ITProcessor()
        assert "face" in it._prototypes
        assert "car" in it._prototypes

        for proto in it._prototypes.values():
            # Prototypes should be unit vectors
            assert abs(np.linalg.norm(proto) - 1.0) < 1e-6

    def test_it_process(self):
        """Test IT processing."""
        retina = Retina()
        v1 = V1Processor(n_orientations=4, n_frequencies=2)
        v2 = V2Processor()
        v4 = V4Processor()
        it = ITProcessor()

        image = np.random.rand(32, 32)
        retina_out = retina.process(image)
        v1_out = v1.process(retina_out)
        v2_out = v2.process(v1_out)
        v4_out = v4.process(v2_out)
        it_out = it.process(v4_out)

        assert isinstance(it_out, ITOutput)
        assert len(it_out.object_responses) == len(it._prototypes)

    def test_it_learn_category(self):
        """Test learning new category."""
        it = ITProcessor()
        n_before = len(it._prototypes)

        examples = [np.random.randn(128) for _ in range(5)]
        it.learn_category("new_object", examples)

        assert len(it._prototypes) == n_before + 1
        assert "new_object" in it._prototypes

class TestDorsalVentral:
    """Tests for dorsal/ventral streams."""

    def test_dorsal_creation(self):
        """Test dorsal stream creation."""
        dorsal = DorsalStream()
        assert dorsal.motion_window == 3
        assert len(dorsal._affordances) > 0

    def test_ventral_creation(self):
        """Test ventral stream creation."""
        ventral = VentralStream()
        assert ventral.embedding_dim == 128

    def test_visual_pathways(self):
        """Test combined pathways."""
        retina = Retina()
        v1 = V1Processor(n_orientations=4, n_frequencies=2)
        v2 = V2Processor()
        v4 = V4Processor()
        it = ITProcessor()
        pathways = VisualPathways()

        image = np.random.rand(32, 32)
        retina_out = retina.process(image)
        v1_out = v1.process(retina_out)
        v2_out = v2.process(v1_out)
        v4_out = v4.process(v2_out)
        it_out = it.process(v4_out)

        dorsal_out, ventral_out = pathways.process(v1_out, v2_out, v4_out, it_out)

        assert dorsal_out.depth_map.shape == (32, 32)
        assert dorsal_out.motion_map.shape == (32, 32, 2)
        assert len(dorsal_out.action_affordances) > 0

    def test_dorsal_motion_estimation(self):
        """Test motion estimation."""
        dorsal = DorsalStream()

        # Simulate two frames
        frame1 = np.random.rand(32, 32)
        frame2 = np.roll(frame1, 2, axis=1)  # Shifted right

        motion1 = dorsal._estimate_motion(frame1)
        motion2 = dorsal._estimate_motion(frame2)

        # First frame should be zero motion
        assert np.allclose(motion1, 0)
        # Second frame should detect motion
        assert motion2.max() > 0

    def test_affordances(self):
        """Test action affordance computation."""
        dorsal = DorsalStream()

        motion = np.zeros((32, 32, 2))
        depth = np.random.rand(32, 32) * 0.5  # Near objects
        edges = np.random.rand(32, 32)

        affordances = dorsal._compute_affordances(motion, depth, edges)

        assert "grasp" in affordances
        assert "reach" in affordances
        assert "avoid" in affordances
        assert "track" in affordances

# ============== Auditory Tests ==============

class TestCochlea:
    """Tests for cochlear processing."""

    def test_gammatone_creation(self):
        """Test gammatone filterbank creation."""
        fb = GammatoneFilterbank(n_channels=32)
        assert len(fb.center_frequencies) == 32
        assert fb.center_frequencies[0] < fb.center_frequencies[-1]

    def test_gammatone_erb_spacing(self):
        """Test ERB frequency spacing."""
        fb = GammatoneFilterbank(n_channels=64, low_freq=80, high_freq=8000)

        # Frequencies should be on ERB scale (denser at low frequencies)
        diffs = np.diff(fb.center_frequencies)
        assert diffs[0] < diffs[-1]

    def test_gammatone_process(self):
        """Test filterbank processing."""
        fb = GammatoneFilterbank(sample_rate=16000, n_channels=32)

        signal = np.random.randn(8000)  # 0.5s of audio
        output = fb.process(signal)

        assert output.shape == (32, 8000)

    def test_cochlea_creation(self):
        """Test cochlea creation."""
        cochlea = Cochlea(sample_rate=16000, n_channels=64)
        assert cochlea.sample_rate == 16000
        assert cochlea.n_channels == 64

    def test_cochlea_process(self):
        """Test cochlea processing."""
        cochlea = Cochlea(sample_rate=16000, n_channels=32)

        signal = np.random.randn(8000)
        output = cochlea.process(signal)

        assert isinstance(output, CochleaOutput)
        assert output.basilar_membrane.shape[0] == 32
        assert output.auditory_nerve.shape[0] == 32

    def test_cochlea_compression(self):
        """Test cochlear compression."""
        cochlea = Cochlea(compression_power=0.3)

        # Louder signal should have compressed response
        quiet = np.random.randn(4000) * 0.1
        loud = np.random.randn(4000) * 1.0

        out_quiet = cochlea.process(quiet)
        cochlea._adaptation = np.ones(cochlea.n_channels)  # Reset
        out_loud = cochlea.process(loud)

        # Ratio should be less than 10 due to compression
        ratio = out_loud.auditory_nerve.max() / (out_quiet.auditory_nerve.max() + 1e-8)
        assert ratio < 10

    def test_cochlea_spectrogram(self):
        """Test spectrogram generation."""
        cochlea = Cochlea(n_channels=32)
        signal = np.random.randn(8000)
        output = cochlea.process(signal)

        spec = cochlea.get_spectrogram(output, frame_size=256, hop_size=128)
        assert spec.shape[0] == 32
        assert spec.shape[1] > 0

class TestA1:
    """Tests for A1 processing."""

    def test_a1_creation(self):
        """Test A1 processor creation."""
        a1 = A1Processor(n_rates=8, n_scales=8)
        assert a1.n_rates == 8
        assert a1.n_scales == 8

    def test_a1_strf_bank(self):
        """Test STRF bank creation."""
        a1 = A1Processor()
        assert len(a1._strfs) > 0

        for strf in a1._strfs:
            assert strf.kernel.shape == (11, 21)
            assert strf.best_rate > 0

    def test_a1_process(self):
        """Test A1 processing."""
        cochlea = Cochlea(n_channels=32)
        a1 = A1Processor(n_rates=4, n_scales=4)

        signal = np.random.randn(8000)
        cochlea_out = cochlea.process(signal)
        a1_out = a1.process(cochlea_out)

        assert isinstance(a1_out, A1Output)
        assert a1_out.tonotopic_map.shape[0] == 32
        assert a1_out.onset_map.shape[0] == 32

    def test_a1_modulation_spectrum(self):
        """Test modulation spectrum extraction."""
        cochlea = Cochlea(n_channels=32)
        a1 = A1Processor(n_rates=4, n_scales=4)

        signal = np.random.randn(8000)
        cochlea_out = cochlea.process(signal)
        a1_out = a1.process(cochlea_out)

        mod_spec = a1.get_modulation_spectrum(a1_out)

        assert "rate_spectrum" in mod_spec
        assert "scale_spectrum" in mod_spec
        assert len(mod_spec["rate_spectrum"]) == 4

class TestSpeech:
    """Tests for speech processing."""

    def test_formant_tracker_creation(self):
        """Test formant tracker creation."""
        tracker = FormantTracker(n_formants=3)
        assert tracker.n_formants == 3
        assert len(tracker.formant_ranges) == 3

    def test_formant_tracking(self):
        """Test formant extraction."""
        tracker = FormantTracker()

        # Create spectrum with peaks
        frequencies = np.linspace(0, 8000, 256)
        spectrum = np.zeros(256)
        spectrum[20] = 1.0  # ~600 Hz (F1)
        spectrum[60] = 0.8  # ~1800 Hz (F2)
        spectrum[100] = 0.5  # ~3000 Hz (F3)

        formants = tracker.track(spectrum, frequencies)
        assert len(formants) == 3

    def test_voicing_detector(self):
        """Test voicing detection."""
        detector = VoicingDetector(sample_rate=16000)

        # Create periodic signal (voiced)
        t = np.arange(8000) / 16000
        voiced = np.sin(2 * np.pi * 150 * t)  # 150 Hz pitch

        voicing, pitch = detector.detect(voiced)
        assert voicing.max() > 0.3  # Should detect voicing

    def test_phoneme_recognizer(self):
        """Test phoneme recognizer."""
        recognizer = PhonemeRecognizer()
        assert len(recognizer._phoneme_templates) > 0
        assert "AA" in recognizer._phoneme_templates
        assert "S" in recognizer._phoneme_templates

    def test_speech_processor(self):
        """Test speech processor."""
        cochlea = Cochlea(n_channels=32)
        a1 = A1Processor(n_rates=4, n_scales=4)
        speech = SpeechProcessor()

        signal = np.random.randn(8000)
        cochlea_out = cochlea.process(signal)
        a1_out = a1.process(cochlea_out)
        speech_out = speech.process(a1_out)

        assert isinstance(speech_out, SpeechOutput)
        assert speech_out.formant_tracks.shape[0] == 3

# ============== Multimodal Tests ==============

class TestBinding:
    """Tests for multimodal binding."""

    def test_temporal_binder(self):
        """Test temporal binding."""
        binder = TemporalBinder(binding_window=0.1)

        inputs = [
            ModalityInput(Modality.VISUAL, np.zeros(10), timestamp=0.0),
            ModalityInput(Modality.AUDITORY, np.zeros(10), timestamp=0.05),
            ModalityInput(Modality.VISUAL, np.zeros(10), timestamp=0.5),
        ]

        binding = binder.compute_temporal_binding(inputs)

        # First two should be bound (within window)
        assert binding[0, 1] > 0.5
        # Third should be separate
        assert binding[0, 2] < 0.1

    def test_spatial_binder(self):
        """Test spatial binding."""
        binder = SpatialBinder(spatial_sigma=0.1)

        inputs = [
            ModalityInput(Modality.VISUAL, np.zeros(10), spatial_location=(0.0, 0.0, 1.0)),
            ModalityInput(Modality.AUDITORY, np.zeros(10), spatial_location=(0.05, 0.0, 1.0)),
            ModalityInput(Modality.VISUAL, np.zeros(10), spatial_location=(1.0, 1.0, 1.0)),
        ]

        binding = binder.compute_spatial_binding(inputs)

        # First two should be bound (nearby)
        assert binding[0, 1] > 0.5
        # Third should be separate
        assert binding[0, 2] < 0.1

    def test_feature_binder(self):
        """Test feature binding."""
        binder = FeatureBinder(unified_dim=64)

        features = np.random.randn(32)
        unified = binder.project_to_unified(features, Modality.VISUAL)

        assert len(unified) == 64
        assert abs(np.linalg.norm(unified) - 1.0) < 1e-6

    def test_cross_modal_binder(self):
        """Test complete binding."""
        binder = CrossModalBinder()

        inputs = [
            ModalityInput(
                Modality.VISUAL,
                np.random.randn(64),
                spatial_location=(0.0, 0.0, 1.0),
                timestamp=0.0,
            ),
            ModalityInput(
                Modality.AUDITORY,
                np.random.randn(64),
                spatial_location=(0.05, 0.0, 1.0),
                timestamp=0.02,
            ),
        ]

        output = binder.bind(inputs)

        assert isinstance(output, BindingOutput)
        assert len(output.percepts) > 0
        assert output.coherence > 0

    def test_bound_percept_creation(self):
        """Test bound percept properties."""
        binder = CrossModalBinder()

        inputs = [
            ModalityInput(
                Modality.VISUAL,
                np.random.randn(64),
                spatial_location=(0.0, 0.0, 1.0),
                timestamp=0.0,
            ),
            ModalityInput(
                Modality.AUDITORY,
                np.random.randn(64),
                timestamp=0.01,
            ),
        ]

        output = binder.bind(inputs)
        percept = output.percepts[0]

        assert Modality.VISUAL in percept.modalities
        assert Modality.AUDITORY in percept.modalities
        assert len(percept.unified_features) == 256

class TestAttention:
    """Tests for attention mechanisms."""

    def test_saliency_computer(self):
        """Test saliency computation."""
        computer = SaliencyComputer()

        intensity = np.random.rand(64, 64)
        saliency = computer.compute_visual_saliency(intensity)

        assert saliency.shape == (64, 64)
        assert 0 <= saliency.max() <= 1

    def test_auditory_saliency(self):
        """Test auditory saliency."""
        computer = SaliencyComputer()

        spectrogram = np.random.rand(32, 100)
        saliency = computer.compute_auditory_saliency(spectrogram)

        assert saliency.shape == (32, 100)

    def test_top_down_controller(self):
        """Test top-down attention control."""
        controller = TopDownController()

        template = np.random.randn(256)
        controller.set_goal(template=template, modality=Modality.VISUAL)

        inputs = [
            ModalityInput(Modality.VISUAL, template + np.random.randn(256) * 0.1),
            ModalityInput(Modality.AUDITORY, np.random.randn(256)),
        ]

        relevance = controller.compute_goal_relevance(inputs)

        # Visual input matching template should be more relevant
        assert relevance[0] > relevance[1]

    def test_attention_gate(self):
        """Test attention gating."""
        gate = AttentionGate(capacity=2)

        from neuro.modules.perception.multimodal.binding import BoundPercept
        percepts = [
            BoundPercept(
                id="p1",
                modalities={Modality.VISUAL},
                unified_features=np.zeros(64),
                spatial_location=None,
                temporal_window=(0, 0.1),
                binding_strength=0.8,
            ),
            BoundPercept(
                id="p2",
                modalities={Modality.AUDITORY},
                unified_features=np.zeros(64),
                spatial_location=None,
                temporal_window=(0, 0.1),
                binding_strength=0.6,
            ),
            BoundPercept(
                id="p3",
                modalities={Modality.VISUAL},
                unified_features=np.zeros(64),
                spatial_location=None,
                temporal_window=(0, 0.1),
                binding_strength=0.3,
            ),
        ]

        scores = np.array([0.9, 0.7, 0.2])
        attended, suppressed = gate.gate(percepts, scores)

        assert len(attended) == 2
        assert len(suppressed) == 1

    def test_selective_attention(self):
        """Test complete selective attention."""
        attention = SelectiveAttention()

        from neuro.modules.perception.multimodal.binding import BoundPercept
        inputs = [
            ModalityInput(Modality.VISUAL, np.random.randn(64)),
        ]
        percepts = [
            BoundPercept(
                id="p1",
                modalities={Modality.VISUAL},
                unified_features=np.random.randn(256),
                spatial_location=(0.0, 0.0, 1.0),
                temporal_window=(0, 0.1),
                binding_strength=0.8,
            ),
        ]

        output = attention.attend(inputs, percepts)

        assert isinstance(output, AttentionOutput)
        assert len(output.attended_percepts) > 0

# ============== Interface Tests ==============

class TestVisualPipeline:
    """Tests for visual pipeline."""

    def test_pipeline_creation(self):
        """Test pipeline creation."""
        pipeline = VisualPipeline()
        assert pipeline.retina is not None
        assert pipeline.v1 is not None
        assert pipeline.v2 is not None

    def test_pipeline_process(self):
        """Test complete visual pipeline."""
        pipeline = VisualPipeline(
            n_orientations=4,
            n_frequencies=2,
            n_categories=10,
        )

        image = np.random.rand(32, 32)
        percept = pipeline.process(image)

        assert isinstance(percept, VisualPercept)
        assert percept.retina is not None
        assert percept.v1 is not None
        assert percept.v4 is not None
        assert percept.dorsal is not None

    def test_pipeline_features(self):
        """Test feature extraction."""
        pipeline = VisualPipeline(
            n_orientations=4,
            n_frequencies=2,
        )

        image = np.random.rand(32, 32)
        percept = pipeline.process(image)
        features = pipeline.get_features(percept)

        assert len(features) > 0
        assert abs(np.linalg.norm(features) - 1.0) < 1e-6

class TestAuditoryPipeline:
    """Tests for auditory pipeline."""

    def test_pipeline_creation(self):
        """Test pipeline creation."""
        pipeline = AuditoryPipeline()
        assert pipeline.cochlea is not None
        assert pipeline.a1 is not None

    def test_pipeline_process(self):
        """Test complete auditory pipeline."""
        pipeline = AuditoryPipeline(
            sample_rate=16000,
            n_channels=32,
        )

        audio = np.random.randn(8000)
        percept = pipeline.process(audio)

        assert isinstance(percept, AuditoryPercept)
        assert percept.cochlea is not None
        assert percept.a1 is not None

    def test_pipeline_features(self):
        """Test feature extraction."""
        pipeline = AuditoryPipeline(n_channels=32)

        audio = np.random.randn(8000)
        percept = pipeline.process(audio)
        features = pipeline.get_features(percept)

        assert len(features) > 0

class TestPerceptionSystem:
    """Tests for complete perception system."""

    def test_system_creation(self):
        """Test system creation."""
        system = PerceptionSystem()
        assert system.visual is not None
        assert system.auditory is not None
        assert system.binder is not None
        assert system.attention is not None

    def test_system_visual_only(self):
        """Test visual-only processing."""
        system = PerceptionSystem()

        image = np.random.rand(32, 32)
        percept = system.process(visual_input=image)

        assert isinstance(percept, MultimodalPercept)
        assert percept.visual is not None
        assert percept.auditory is None

    def test_system_auditory_only(self):
        """Test auditory-only processing."""
        system = PerceptionSystem()

        audio = np.random.randn(8000)
        percept = system.process(auditory_input=audio)

        assert percept.visual is None
        assert percept.auditory is not None

    def test_system_multimodal(self):
        """Test multimodal processing."""
        system = PerceptionSystem()

        image = np.random.rand(32, 32)
        audio = np.random.randn(8000)
        percept = system.process(visual_input=image, auditory_input=audio)

        assert percept.visual is not None
        assert percept.auditory is not None
        assert percept.binding is not None

    def test_system_attention_goal(self):
        """Test setting attention goal."""
        system = PerceptionSystem()

        template = np.random.randn(256)
        system.set_attention_goal(template=template, modality=Modality.VISUAL)

        # Process should use the goal
        image = np.random.rand(32, 32)
        percept = system.process(visual_input=image)

        assert percept.attention is not None

    def test_system_reset(self):
        """Test system reset."""
        system = PerceptionSystem()

        # Process some input
        image = np.random.rand(32, 32)
        system.process(visual_input=image)

        # Reset
        system.reset()
        assert system._current_time == 0.0

    def test_create_perception_system(self):
        """Test convenience function."""
        system = create_perception_system()
        assert isinstance(system, PerceptionSystem)

    def test_system_statistics(self):
        """Test statistics collection."""
        system = PerceptionSystem()
        stats = system.statistics()

        assert "visual" in stats
        assert "auditory" in stats
        assert "binding" in stats
        assert "attention" in stats

# ============== Integration Tests ==============

class TestIntegration:
    """Integration tests for perception system."""

    def test_full_visual_pipeline(self):
        """Test complete visual processing pipeline."""
        system = create_perception_system()

        # Create test image with features
        image = np.zeros((64, 64))
        image[20:40, 20:40] = 1.0  # White square

        percept = system.process(visual_input=image)

        # Should detect objects
        assert percept.visual is not None
        assert percept.visual.v4 is not None
        assert percept.visual.dorsal is not None

    def test_full_auditory_pipeline(self):
        """Test complete auditory processing pipeline."""
        system = create_perception_system()

        # Create test audio (sine wave)
        t = np.arange(16000) / 16000
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

        percept = system.process(auditory_input=audio)

        assert percept.auditory is not None
        assert percept.auditory.a1 is not None

    def test_multimodal_binding(self):
        """Test binding of visual and auditory input."""
        system = create_perception_system()

        image = np.random.rand(32, 32)
        audio = np.random.randn(8000)

        percept = system.process(visual_input=image, auditory_input=audio)

        assert percept.binding is not None
        assert len(percept.binding.percepts) > 0

    def test_sequential_frames(self):
        """Test processing sequential frames."""
        system = create_perception_system()

        # Process sequence of frames
        for i in range(5):
            image = np.random.rand(32, 32)
            percept = system.process(visual_input=image)
            assert percept.visual is not None

        # Motion should be computed after first frame
        assert system.visual.pathways.dorsal._previous_frame is not None

    def test_attention_modulates_perception(self):
        """Test that attention affects perception."""
        system = create_perception_system()

        # Set attention to visual modality
        system.set_attention_goal(modality=Modality.VISUAL)

        image = np.random.rand(32, 32)
        audio = np.random.randn(8000)
        percept = system.process(visual_input=image, auditory_input=audio)

        # Visual should be attended
        if percept.attention and percept.attention.attended_percepts:
            attended_modalities = set()
            for p in percept.attention.attended_percepts:
                attended_modalities.update(p.modalities)
            assert Modality.VISUAL in attended_modalities

    def test_perception_statistics(self):
        """Test statistics collection."""
        system = create_perception_system()

        # Process some inputs
        for _ in range(3):
            image = np.random.rand(32, 32)
            system.process(visual_input=image)

        stats = system.statistics()

        assert "visual" in stats
        assert "retina" in stats["visual"]
        assert "v1" in stats["visual"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
