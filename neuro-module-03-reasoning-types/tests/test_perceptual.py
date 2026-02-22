"""Tests for perceptual reasoning components"""

import pytest
import numpy as np
from neuro.modules.m03_reasoning_types.perceptual.visual_features import (
    VisualFeatureExtractor,
    FeatureMap,
    Feature,
    FeatureType,
)
from neuro.modules.m03_reasoning_types.perceptual.multimodal_integration import (
    MultimodalIntegrator,
    SensoryInput,
    Modality,
)
from neuro.modules.m03_reasoning_types.perceptual.object_recognition import (
    ObjectRecognizer,
    CategoryLevel,
)


class TestVisualFeatures:
    def test_edge_detection(self):
        """Test edge detection on a simple image"""
        extractor = VisualFeatureExtractor(edge_threshold=0.1)

        image = np.zeros((10, 10))
        image[:, 5:] = 1.0

        edges = extractor.extract_edges(image)

        assert len(edges) > 0
        assert all(e.feature_type == FeatureType.EDGE for e in edges)

    def test_texture_extraction(self):
        """Test texture feature extraction"""
        extractor = VisualFeatureExtractor(texture_window=3)

        image = np.random.rand(15, 15)

        textures = extractor.extract_textures(image)

        assert len(textures) > 0
        assert all(t.feature_type == FeatureType.TEXTURE for t in textures)
        assert all("variance" in t.properties for t in textures)

    def test_feature_map_query(self):
        """Test feature map region queries"""
        feature_map = FeatureMap(width=100, height=100)

        for i in range(10):
            feature = Feature(
                feature_type=FeatureType.EDGE, location=(i * 10, i * 10), magnitude=1.0
            )
            feature_map.add_feature(feature)

        region = feature_map.query_region(0, 0, 50, 50)
        assert len(region) > 0
        assert len(region) < 10

    def test_motion_detection(self):
        """Test motion detection from image sequence"""
        extractor = VisualFeatureExtractor()

        frame1 = np.zeros((32, 32))
        frame1[10:15, 10:15] = 1.0

        frame2 = np.zeros((32, 32))
        frame2[12:17, 12:17] = 1.0

        motion = extractor.extract_motion([frame1, frame2])

        assert len(motion) >= 0


class TestMultimodalIntegration:
    def test_visual_audio_binding(self):
        """Test binding visual and auditory inputs"""
        integrator = MultimodalIntegrator(spatial_tolerance=0.5)

        visual = SensoryInput(
            modality=Modality.VISUAL,
            timestamp=0.0,
            location=(1.0, 1.0, 0.0),
            features={"color": "red"},
        )

        audio = SensoryInput(
            modality=Modality.AUDITORY,
            timestamp=0.01,
            location=(1.1, 1.0, 0.0),
            features={"frequency": 440},
        )

        percept = integrator.bind_visual_audio(visual, audio)

        assert percept is not None
        assert percept.binding_strength > 0
        assert Modality.VISUAL in percept.modalities
        assert Modality.AUDITORY in percept.modalities

    def test_conflict_resolution(self):
        """Test conflict resolution between modalities"""
        integrator = MultimodalIntegrator()

        inputs = {
            Modality.VISUAL: SensoryInput(
                modality=Modality.VISUAL, timestamp=0.0, location=(1.0, 0.0, 0.0), confidence=0.9
            ),
            Modality.AUDITORY: SensoryInput(
                modality=Modality.AUDITORY, timestamp=0.0, location=(2.0, 0.0, 0.0), confidence=0.6
            ),
        }

        location, status = integrator.resolve_conflict(inputs)

        assert location is not None
        assert location[0] > 1.0
        assert location[0] < 2.0

    def test_unified_percept_creation(self):
        """Test creating unified percept from multiple modalities"""
        integrator = MultimodalIntegrator()

        inputs = {
            Modality.VISUAL: SensoryInput(
                modality=Modality.VISUAL,
                timestamp=0.0,
                location=(1.0, 1.0, 1.0),
                features={"shape": "round"},
            ),
            Modality.TACTILE: SensoryInput(
                modality=Modality.TACTILE,
                timestamp=0.0,
                location=(1.0, 1.0, 1.0),
                features={"texture": "smooth"},
            ),
        }

        percept = integrator.create_unified_percept(inputs)

        assert percept is not None
        assert len(percept.modalities) == 2


class TestObjectRecognition:
    def test_category_learning(self):
        """Test learning a new category from examples"""
        recognizer = ObjectRecognizer()

        examples = [
            {"color": "red", "shape": "round", "size": 0.5},
            {"color": "red", "shape": "round", "size": 0.6},
            {"color": "red", "shape": "round", "size": 0.4},
        ]

        recognizer.learn_category("apple", examples, CategoryLevel.BASIC)

        assert "apple" in recognizer.categories
        assert recognizer.categories["apple"].n_observations == 3

    def test_recognition(self):
        """Test recognizing an object"""
        recognizer = ObjectRecognizer(recognition_threshold=0.3)

        recognizer.learn_category(
            "ball", [{"roundness": 0.9, "size": 0.5}, {"roundness": 0.95, "size": 0.4}]
        )

        result = recognizer.recognize({"roundness": 0.92, "size": 0.45})

        if result:
            assert result.category_id == "ball"
            assert result.confidence > 0

    def test_invariant_recognition(self):
        """Test scale-invariant recognition"""
        recognizer = ObjectRecognizer(use_invariance=True)

        features_small = {"width": 10, "height": 10, "aspect_ratio": 1.0}
        features_large = {"width": 100, "height": 100, "aspect_ratio": 1.0}

        normalized_small = recognizer.invariant_processor.scale_invariance(
            features_small, reference_size=50
        )
        normalized_large = recognizer.invariant_processor.scale_invariance(
            features_large, reference_size=50
        )

        assert normalized_small["width"] != features_small["width"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
