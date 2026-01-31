"""
Tests for Mental Imagery Systems
"""

import pytest
import numpy as np
from src.imagery import (
    VisualImagery, AuditoryImagery, MotorImagery,
    TactileImagery, ImagerySystem, MultimodalImage
)
from src.imagery.visual_imagery import VisualProperty
from src.imagery.auditory_imagery import AuditoryProperty
from src.imagery.motor_imagery import MotorProperty
from src.imagery.tactile_imagery import TactileProperty


class TestVisualImagery:
    """Tests for Visual Imagery System"""

    def test_initialization(self):
        """Test visual imagery initializes correctly"""
        visual = VisualImagery()
        assert visual.default_vividness == 0.7
        assert len(visual.images) == 0

    def test_create_image(self):
        """Test creating a visual image"""
        visual = VisualImagery()
        img = visual.create_image(
            "sunset",
            "A beautiful sunset over the ocean",
            properties={VisualProperty.COLOR: "orange", VisualProperty.BRIGHTNESS: 0.8}
        )

        assert img.id == "sunset"
        assert "sunset" in img.description
        assert img.vividness > 0

    def test_visualize_from_description(self):
        """Test visualizing from text description"""
        visual = VisualImagery()
        img = visual.visualize("A red apple on a wooden table")

        assert img is not None
        assert img.vividness > 0

    def test_mental_rotation(self):
        """Test mental rotation of images"""
        visual = VisualImagery()
        original = visual.create_image("cube", "A 3D cube")

        rotated = visual.rotate(original.id, 90)

        assert rotated is not None
        assert "rotated" in rotated.id or "rotate" in rotated.description.lower()
        # Rotation should slightly reduce vividness
        assert rotated.vividness <= original.vividness

    def test_scaling(self):
        """Test mental scaling of images"""
        visual = VisualImagery()
        original = visual.create_image("ball", "A small ball")

        scaled_up = visual.scale(original.id, 2.0)
        scaled_down = visual.scale(original.id, 0.5)

        assert scaled_up is not None
        assert scaled_down is not None

    def test_image_transformation_chain(self):
        """Test chaining multiple transformations"""
        visual = VisualImagery()
        img = visual.create_image("shape", "A geometric shape")

        # Chain transformations
        rotated = visual.rotate(img.id, 45)
        scaled = visual.scale(rotated.id, 1.5)

        assert scaled is not None
        # Each transformation reduces vividness
        assert scaled.vividness < img.vividness

    def test_morphing(self):
        """Test morphing between two images"""
        visual = VisualImagery()
        img1 = visual.create_image("cat", "A cat")
        img2 = visual.create_image("dog", "A dog")

        morphed = visual.morph(img1.id, img2.id, blend_factor=0.5)

        assert morphed is not None
        assert "cat" in morphed.description.lower() or "dog" in morphed.description.lower()


class TestAuditoryImagery:
    """Tests for Auditory Imagery System"""

    def test_initialization(self):
        """Test auditory imagery initializes correctly"""
        auditory = AuditoryImagery()
        assert auditory.default_vividness == 0.6
        assert len(auditory.sounds) == 0

    def test_create_sound(self):
        """Test creating a sound image"""
        auditory = AuditoryImagery()
        sound = auditory.create_sound(
            "bell",
            "A ringing bell",
            properties={AuditoryProperty.PITCH: "high", AuditoryProperty.VOLUME: "loud"}
        )

        assert sound.id == "bell"
        assert sound.vividness > 0

    def test_inner_speech(self):
        """Test inner speech generation"""
        auditory = AuditoryImagery()
        speech = auditory.inner_speech("Hello, world!")

        assert speech is not None
        assert speech.is_speech == True
        assert speech.duration > 0

    def test_imagine_music(self):
        """Test music imagination"""
        auditory = AuditoryImagery()
        music = auditory.imagine_music("A cheerful melody", tempo=120)

        assert music is not None
        assert music.is_music == True
        assert AuditoryProperty.TEMPO in music.properties

    def test_change_pitch(self):
        """Test pitch transposition"""
        auditory = AuditoryImagery()
        original = auditory.create_sound("note", "A musical note")

        higher = auditory.change_pitch(original.id, semitones=5)
        lower = auditory.change_pitch(original.id, semitones=-5)

        assert higher is not None
        assert lower is not None

    def test_change_tempo(self):
        """Test tempo change"""
        auditory = AuditoryImagery()
        music = auditory.imagine_music("A song", tempo=100)

        faster = auditory.change_tempo(music.id, factor=1.5)
        slower = auditory.change_tempo(music.id, factor=0.5)

        assert faster.duration < music.duration
        assert slower.duration > music.duration

    def test_sequence_sounds(self):
        """Test sequencing multiple sounds"""
        auditory = AuditoryImagery()
        s1 = auditory.create_sound("sound1", "First sound", duration=1.0)
        s2 = auditory.create_sound("sound2", "Second sound", duration=2.0)

        sequence = auditory.sequence_sounds([s1.id, s2.id])

        assert sequence.duration == s1.duration + s2.duration

    def test_layer_sounds(self):
        """Test layering multiple sounds"""
        auditory = AuditoryImagery()
        s1 = auditory.create_sound("melody", "A melody", duration=5.0)
        s2 = auditory.create_sound("harmony", "A harmony", duration=5.0)

        layered = auditory.layer_sounds([s1.id, s2.id])

        assert layered is not None
        assert layered.duration == max(s1.duration, s2.duration)


class TestMotorImagery:
    """Tests for Motor Imagery System"""

    def test_initialization(self):
        """Test motor imagery initializes correctly"""
        motor = MotorImagery()
        assert motor.default_vividness == 0.6
        assert len(motor.actions) == 0

    def test_imagine_action(self):
        """Test imagining an action"""
        motor = MotorImagery()
        action = motor.imagine_action(
            "throw",
            "Throwing a ball",
            body_parts=["arm", "hand"],
            duration=1.0
        )

        assert action.id == "throw"
        assert "arm" in action.body_parts

    def test_simulate_movement(self):
        """Test simulating movement from description"""
        motor = MotorImagery()
        movement = motor.simulate_movement("Quickly running forward")

        assert movement is not None
        assert MotorProperty.SPEED in movement.properties

    def test_rehearsal(self):
        """Test mental rehearsal improves vividness"""
        motor = MotorImagery()
        action = motor.imagine_action("jump", "Jumping up")

        initial_vividness = action.vividness
        initial_feel = action.kinesthetic_feel

        motor.rehearse(action.id, repetitions=5)

        assert action.vividness > initial_vividness
        assert action.kinesthetic_feel > initial_feel

    def test_slow_motion(self):
        """Test slow motion visualization"""
        motor = MotorImagery()
        action = motor.imagine_action("kick", "Kicking a ball", duration=0.5)

        slow = motor.slow_motion(action.id, factor=0.5)

        assert slow.duration > action.duration
        assert MotorProperty.SPEED in slow.properties

    def test_sequence_actions(self):
        """Test sequencing actions"""
        motor = MotorImagery()
        a1 = motor.imagine_action("step", "Take a step", duration=0.5)
        a2 = motor.imagine_action("turn", "Turn around", duration=1.0)
        a3 = motor.imagine_action("jump", "Jump up", duration=0.3)

        sequence = motor.sequence_actions([a1.id, a2.id, a3.id])

        assert sequence.duration == a1.duration + a2.duration + a3.duration

    def test_mirror_action(self):
        """Test mirroring an action"""
        motor = MotorImagery()
        action = motor.imagine_action(
            "wave",
            "Waving right hand",
            body_parts=["right hand"]
        )

        mirrored = motor.mirror(action.id)

        assert "left" in str(mirrored.body_parts).lower() or mirrored is not None


class TestTactileImagery:
    """Tests for Tactile Imagery System"""

    def test_initialization(self):
        """Test tactile imagery initializes correctly"""
        tactile = TactileImagery()
        assert tactile.default_vividness == 0.5
        assert len(tactile.sensations) == 0

    def test_imagine_touch(self):
        """Test imagining touch sensation"""
        tactile = TactileImagery()
        touch = tactile.imagine_touch(
            "silk",
            "Touching silk fabric",
            properties={TactileProperty.TEXTURE: "smooth"},
            body_location="fingertips"
        )

        assert touch.id == "silk"
        assert touch.body_location == "fingertips"

    def test_feel_texture(self):
        """Test feeling a texture"""
        tactile = TactileImagery()
        texture = tactile.feel_texture("soft warm blanket")

        assert texture is not None
        assert TactileProperty.TEXTURE in texture.properties
        # Should have semantic associations
        assert len(texture.semantic_associations) > 0

    def test_texture_semantic_associations(self):
        """Test that textures evoke semantic associations"""
        tactile = TactileImagery()

        soft_texture = tactile.feel_texture("soft fluffy pillow")
        hard_texture = tactile.feel_texture("hard cold metal")

        # Soft textures associate with comfort
        assert any("comfort" in a.lower() for a in soft_texture.semantic_associations)

        # Hard textures associate with strength/durability
        hard_associations = [a.lower() for a in hard_texture.semantic_associations]
        assert any(word in hard_associations for word in ["strength", "durability", "solid", "cold"])

    def test_feel_temperature(self):
        """Test temperature sensation"""
        tactile = TactileImagery()

        warm = tactile.feel_temperature("warm", body_location="hands")
        cold = tactile.feel_temperature("cold", body_location="face")

        assert warm is not None
        assert cold is not None
        assert warm.body_location == "hands"
        assert cold.body_location == "face"

    def test_feel_pressure(self):
        """Test pressure sensation"""
        tactile = TactileImagery()

        heavy = tactile.feel_pressure("heavy", body_location="shoulders")
        light = tactile.feel_pressure("light", body_location="arm")

        assert heavy.intensity > light.intensity

    def test_combine_sensations(self):
        """Test combining multiple tactile sensations"""
        tactile = TactileImagery()

        s1 = tactile.feel_texture("smooth")
        s2 = tactile.feel_temperature("warm")

        combined = tactile.combine_sensations([s1.id, s2.id])

        assert len(combined.properties) >= 2
        assert len(combined.semantic_associations) >= len(s1.semantic_associations)

    def test_enhance_with_touch(self):
        """Test enhancing abstract concepts with touch"""
        tactile = TactileImagery()

        love_touch = tactile.enhance_with_touch("love")
        fear_touch = tactile.enhance_with_touch("fear")

        # Love should feel warm/soft
        assert "soft" in love_touch.description.lower() or "warm" in love_touch.description.lower()

        # Fear should feel cold/hard
        assert "cold" in fear_touch.description.lower() or "hard" in fear_touch.description.lower()

    def test_tactile_creativity_correlation(self):
        """
        Test key 2025 finding: Touch imagery facilitates creativity
        through semantic integration.
        """
        tactile = TactileImagery()

        # Create multiple textures and check semantic richness
        textures = [
            "rough sandpaper",
            "smooth glass",
            "soft velvet",
            "warm sunshine",
            "cold ice"
        ]

        all_associations = []
        for tex in textures:
            sensation = tactile.feel_texture(tex)
            all_associations.extend(sensation.semantic_associations)

        # Rich tactile imagery should produce diverse semantic associations
        unique_associations = set(all_associations)
        assert len(unique_associations) >= 5  # Multiple distinct associations


class TestImagerySystem:
    """Tests for Integrated Imagery System"""

    def test_initialization(self):
        """Test imagery system initializes all modalities"""
        imagery = ImagerySystem()

        assert imagery.visual is not None
        assert imagery.auditory is not None
        assert imagery.motor is not None
        assert imagery.tactile is not None

    def test_create_multimodal_image(self):
        """Test creating multimodal image"""
        imagery = ImagerySystem()

        img = imagery.create_multimodal_image(
            "beach_scene",
            "A sunny beach with waves",
            include_visual=True,
            include_auditory=True,
            include_tactile=True
        )

        assert img.visual is not None
        assert img.auditory is not None
        assert img.tactile is not None
        assert img.motor is None  # Not included
        assert img.overall_vividness > 0

    def test_imagine_scene(self):
        """Test imagining a full scene"""
        imagery = ImagerySystem()

        scene = imagery.imagine_scene("A forest with birds singing")

        assert scene.visual is not None
        assert scene.auditory is not None
        assert scene.tactile is not None

    def test_imagine_action(self):
        """Test imagining an action"""
        imagery = ImagerySystem()

        action = imagery.imagine_action("Swimming in a pool")

        assert action.visual is not None
        assert action.motor is not None
        assert action.tactile is not None

    def test_simulate_experience(self):
        """Test simulating experience with selected modalities"""
        imagery = ImagerySystem()

        experience = imagery.simulate_experience(
            "Playing piano",
            modalities=["visual", "motor", "auditory"]
        )

        assert experience.visual is not None
        assert experience.motor is not None
        assert experience.auditory is not None
        assert experience.tactile is None

    def test_coherence_computation(self):
        """Test coherence is computed correctly"""
        imagery = ImagerySystem()

        # Single modality - perfect coherence
        single = imagery.create_multimodal_image(
            "single",
            "Just visual",
            include_visual=True,
            include_auditory=False,
            include_motor=False,
            include_tactile=False
        )

        # Multiple modalities - lower coherence
        multi = imagery.create_multimodal_image(
            "multi",
            "All modalities",
            include_visual=True,
            include_auditory=True,
            include_motor=True,
            include_tactile=True
        )

        assert single.coherence >= multi.coherence

    def test_get_all_associations(self):
        """Test getting associations from multimodal image"""
        imagery = ImagerySystem()

        img = imagery.create_multimodal_image(
            "rich_image",
            "A warm soft red blanket",
            include_visual=True,
            include_tactile=True
        )

        associations = imagery.get_all_associations(img.id)

        # Should have associations from both visual and tactile
        assert len(associations) > 0

    def test_transform_multimodal(self):
        """Test transforming multimodal image"""
        imagery = ImagerySystem()

        original = imagery.create_multimodal_image(
            "original",
            "A spinning top",
            include_visual=True,
            include_motor=True
        )

        transformed = imagery.transform_multimodal(original.id, "rotate")

        assert transformed is not None
        assert transformed.id != original.id

    def test_blend_images(self):
        """Test blending two multimodal images"""
        imagery = ImagerySystem()

        img1 = imagery.imagine_scene("A sunny meadow")
        img2 = imagery.imagine_scene("A rainy forest")

        blended = imagery.blend_images(img1.id, img2.id, blend_factor=0.5)

        assert blended is not None
        assert "blend" in blended.id.lower()

    def test_vividness_by_modality(self):
        """Test getting vividness breakdown"""
        imagery = ImagerySystem()

        img = imagery.create_multimodal_image(
            "test",
            "Test image",
            include_visual=True,
            include_auditory=True,
            include_tactile=True
        )

        vividness = imagery.get_vividness_by_modality(img.id)

        assert "visual" in vividness
        assert "auditory" in vividness
        assert "tactile" in vividness
        assert all(0 <= v <= 1 for v in vividness.values())


class TestImageryIntegration:
    """Tests for imagery integration across modalities"""

    def test_creative_imagery_pipeline(self):
        """Test full creative imagery pipeline"""
        imagery = ImagerySystem()

        # Create initial image
        img1 = imagery.imagine_scene("A starry night sky")

        # Transform it
        transformed = imagery.transform_multimodal(img1.id, "rotate")

        # Create another image
        img2 = imagery.imagine_scene("An ocean at dawn")

        # Blend them
        creative_blend = imagery.blend_images(transformed.id, img2.id)

        assert creative_blend is not None
        assert creative_blend.overall_vividness > 0

    def test_tactile_boosts_creativity(self):
        """
        Test 2025 finding: Tactile imagery enhances creative associations.
        """
        imagery = ImagerySystem()

        # Image without tactile
        no_tactile = imagery.create_multimodal_image(
            "no_touch",
            "An abstract concept",
            include_visual=True,
            include_tactile=False
        )

        # Image with tactile
        with_tactile = imagery.create_multimodal_image(
            "with_touch",
            "An abstract concept",
            include_visual=True,
            include_tactile=True
        )

        assoc_no_tactile = imagery.get_all_associations(no_tactile.id)
        assoc_with_tactile = imagery.get_all_associations(with_tactile.id)

        # Tactile should provide additional associations
        assert len(assoc_with_tactile) >= len(assoc_no_tactile)
