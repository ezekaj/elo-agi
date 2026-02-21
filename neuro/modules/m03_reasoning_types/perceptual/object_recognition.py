"""
Object Recognition - Temporal Lobe Simulation

Category and object identification with transformation invariance.
Handles viewpoint, scale, and lighting invariance.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum

from .visual_features import Feature, FeatureMap, FeatureType


class CategoryLevel(Enum):
    SUPERORDINATE = "superordinate"
    BASIC = "basic"
    SUBORDINATE = "subordinate"


@dataclass
class CategoryPrototype:
    """Prototype representation of a category"""
    category_id: str
    level: CategoryLevel
    parent_category: Optional[str] = None
    child_categories: List[str] = field(default_factory=list)

    feature_weights: Dict[str, float] = field(default_factory=dict)
    typical_features: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    exemplars: List[Dict[str, float]] = field(default_factory=list)
    n_observations: int = 0

    def match_score(self, features: Dict[str, Any]) -> float:
        """How well do features match this category?"""
        if not self.typical_features:
            return 0.0

        total_score = 0.0
        total_weight = 0.0

        for feat_name, (mean, std) in self.typical_features.items():
            weight = self.feature_weights.get(feat_name, 1.0)

            if feat_name in features:
                value = features[feat_name]
                is_numeric = isinstance(value, (int, float)) and isinstance(mean, (int, float))

                if is_numeric and std > 0:
                    z_score = abs(value - mean) / std
                    score = np.exp(-0.5 * z_score**2)
                else:
                    score = 1.0 if value == mean else 0.0
                total_score += weight * score

            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def update_from_exemplar(self, features: Dict[str, Any]):
        """Update prototype from a new exemplar"""
        self.exemplars.append(features)
        self.n_observations += 1

        for feat_name, value in features.items():
            is_numeric = isinstance(value, (int, float))

            if feat_name not in self.typical_features:
                if is_numeric:
                    self.typical_features[feat_name] = (value, 0.1)
                else:
                    self.typical_features[feat_name] = (value, 0.0)
                self.feature_weights[feat_name] = 1.0
            else:
                old_mean, old_std = self.typical_features[feat_name]

                if is_numeric and isinstance(old_mean, (int, float)):
                    n = self.n_observations
                    new_mean = old_mean + (value - old_mean) / n

                    if n > 1:
                        values = [e.get(feat_name, old_mean) for e in self.exemplars
                                 if isinstance(e.get(feat_name), (int, float))]
                        new_std = np.std(values) if len(values) > 1 else 0.1
                    else:
                        new_std = old_std

                    self.typical_features[feat_name] = (new_mean, max(new_std, 0.01))
                else:
                    self.typical_features[feat_name] = (value, 0.0)


@dataclass
class RecognitionResult:
    """Result of object recognition"""
    category_id: str
    confidence: float
    level: CategoryLevel
    alternative_categories: List[Tuple[str, float]] = field(default_factory=list)
    features_used: List[str] = field(default_factory=list)


class InvariantRecognition:
    """Handles recognition invariant to transformations"""

    def __init__(self):
        self.canonical_views: Dict[str, np.ndarray] = {}

    def viewpoint_invariance(self,
                             features: Dict[str, float],
                             estimated_viewpoint: Optional[Tuple[float, float, float]] = None
                             ) -> Dict[str, float]:
        """Normalize features to canonical viewpoint"""
        normalized = features.copy()

        if estimated_viewpoint is not None:
            azimuth, elevation, roll = estimated_viewpoint

            if 'aspect_ratio' in normalized:
                correction = np.cos(elevation) * np.cos(azimuth)
                if abs(correction) > 0.1:
                    normalized['aspect_ratio'] /= correction

        return normalized

    def scale_invariance(self,
                         features: Dict[str, float],
                         reference_size: Optional[float] = None
                         ) -> Dict[str, float]:
        """Normalize features to canonical scale"""
        normalized = features.copy()

        size_features = ['width', 'height', 'area', 'perimeter']
        current_size = None

        for sf in size_features:
            if sf in features:
                current_size = features[sf]
                break

        if current_size is not None and current_size > 0:
            scale_factor = (reference_size or 1.0) / current_size

            for key, value in normalized.items():
                if key in size_features:
                    normalized[key] = value * scale_factor

        return normalized

    def lighting_invariance(self,
                            features: Dict[str, float]
                            ) -> Dict[str, float]:
        """Normalize features to canonical lighting"""
        normalized = features.copy()

        if 'brightness' in normalized:
            brightness = normalized['brightness']

            if brightness > 0:
                for key in ['color_r', 'color_g', 'color_b']:
                    if key in normalized:
                        normalized[key] /= brightness

                normalized['brightness'] = 1.0

        if 'contrast' in normalized and normalized['contrast'] > 0:
            pass

        return normalized

    def apply_all_invariances(self,
                               features: Dict[str, float],
                               viewpoint: Optional[Tuple[float, float, float]] = None,
                               reference_size: Optional[float] = None
                               ) -> Dict[str, float]:
        """Apply all invariance transformations"""
        result = features.copy()
        result = self.viewpoint_invariance(result, viewpoint)
        result = self.scale_invariance(result, reference_size)
        result = self.lighting_invariance(result)
        return result


class ObjectRecognizer:
    """
    Object and category recognition system.
    Simulates temporal lobe object recognition.
    """

    def __init__(self,
                 recognition_threshold: float = 0.5,
                 use_invariance: bool = True):
        self.categories: Dict[str, CategoryPrototype] = {}
        self.recognition_threshold = recognition_threshold
        self.use_invariance = use_invariance
        self.invariant_processor = InvariantRecognition()

        self._initialize_basic_categories()

    def _initialize_basic_categories(self):
        """Initialize with some basic category structure"""
        self.categories['object'] = CategoryPrototype(
            category_id='object',
            level=CategoryLevel.SUPERORDINATE,
            child_categories=['living', 'artifact']
        )

        self.categories['living'] = CategoryPrototype(
            category_id='living',
            level=CategoryLevel.SUPERORDINATE,
            parent_category='object',
            child_categories=['animal', 'plant']
        )

        self.categories['artifact'] = CategoryPrototype(
            category_id='artifact',
            level=CategoryLevel.SUPERORDINATE,
            parent_category='object',
            child_categories=['tool', 'vehicle', 'furniture']
        )

    def recognize(self,
                  features: Dict[str, float],
                  level: CategoryLevel = CategoryLevel.BASIC
                  ) -> Optional[RecognitionResult]:
        """Recognize object category from features"""
        if self.use_invariance:
            features = self.invariant_processor.apply_all_invariances(features)

        candidates = []

        for cat_id, prototype in self.categories.items():
            if prototype.level == level or level == CategoryLevel.SUPERORDINATE:
                score = prototype.match_score(features)
                if score > 0:
                    candidates.append((cat_id, score))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)

        best_cat, best_score = candidates[0]

        if best_score < self.recognition_threshold:
            return None

        return RecognitionResult(
            category_id=best_cat,
            confidence=best_score,
            level=self.categories[best_cat].level,
            alternative_categories=candidates[1:4],
            features_used=list(features.keys())
        )

    def categorize(self,
                   features: Dict[str, float],
                   target_level: CategoryLevel
                   ) -> Optional[RecognitionResult]:
        """Categorize at a specific level of abstraction"""
        basic_result = self.recognize(features, CategoryLevel.BASIC)

        if basic_result is None:
            return self.recognize(features, target_level)

        if target_level == CategoryLevel.BASIC:
            return basic_result

        if target_level == CategoryLevel.SUPERORDINATE:
            current = basic_result.category_id
            while current in self.categories:
                parent = self.categories[current].parent_category
                if parent is None:
                    break
                if self.categories.get(parent, CategoryPrototype(
                    category_id='', level=CategoryLevel.BASIC
                )).level == CategoryLevel.SUPERORDINATE:
                    return RecognitionResult(
                        category_id=parent,
                        confidence=basic_result.confidence * 0.9,
                        level=CategoryLevel.SUPERORDINATE,
                        features_used=basic_result.features_used
                    )
                current = parent

        if target_level == CategoryLevel.SUBORDINATE:
            current_cat = self.categories.get(basic_result.category_id)
            if current_cat and current_cat.child_categories:
                best_child = None
                best_score = 0

                for child_id in current_cat.child_categories:
                    if child_id in self.categories:
                        score = self.categories[child_id].match_score(features)
                        if score > best_score:
                            best_score = score
                            best_child = child_id

                if best_child and best_score > self.recognition_threshold:
                    return RecognitionResult(
                        category_id=best_child,
                        confidence=best_score,
                        level=CategoryLevel.SUBORDINATE,
                        features_used=basic_result.features_used
                    )

        return basic_result

    def learn_category(self,
                       category_id: str,
                       examples: List[Dict[str, float]],
                       level: CategoryLevel = CategoryLevel.BASIC,
                       parent_category: Optional[str] = None):
        """Learn a new category from examples"""
        if category_id not in self.categories:
            self.categories[category_id] = CategoryPrototype(
                category_id=category_id,
                level=level,
                parent_category=parent_category
            )

            if parent_category and parent_category in self.categories:
                self.categories[parent_category].child_categories.append(category_id)

        prototype = self.categories[category_id]

        for example in examples:
            prototype.update_from_exemplar(example)

        self._compute_feature_weights(category_id)

    def _compute_feature_weights(self, category_id: str):
        """Compute diagnostic feature weights for a category"""
        prototype = self.categories[category_id]

        if not prototype.typical_features:
            return

        other_categories = [
            c for c_id, c in self.categories.items()
            if c_id != category_id and c.level == prototype.level
        ]

        for feat_name, (mean, std) in prototype.typical_features.items():
            discriminability = 1.0

            for other in other_categories:
                if feat_name in other.typical_features:
                    other_mean, other_std = other.typical_features[feat_name]
                    pooled_std = np.sqrt((std**2 + other_std**2) / 2)
                    if pooled_std > 0:
                        d_prime = abs(mean - other_mean) / pooled_std
                        discriminability = max(discriminability, d_prime)

            prototype.feature_weights[feat_name] = min(discriminability, 3.0)

    def recognize_from_feature_map(self,
                                    feature_map: FeatureMap
                                    ) -> List[RecognitionResult]:
        """Recognize objects from a visual feature map"""
        results = []

        shapes = feature_map.query_type(FeatureType.SHAPE)

        for shape in shapes:
            features = {
                'x': float(shape.location[0]),
                'y': float(shape.location[1]),
                'magnitude': shape.magnitude,
                'orientation': shape.orientation
            }
            features.update(shape.properties)

            result = self.recognize(features)
            if result:
                results.append(result)

        return results
