"""
Abstraction: Extract domain-general principles.

Implements hierarchical abstraction, structural analogy,
and principle extraction for cross-domain transfer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from enum import Enum
import numpy as np


class AbstractionLevel(Enum):
    """Levels of abstraction."""
    CONCRETE = "concrete"        # Specific instances
    CATEGORICAL = "categorical"  # Categories/types
    STRUCTURAL = "structural"    # Relational patterns
    PRINCIPLED = "principled"    # Abstract rules
    UNIVERSAL = "universal"      # Domain-invariant


@dataclass
class AbstractConcept:
    """An abstract concept extracted from examples."""
    id: str
    name: str
    level: AbstractionLevel
    features: Dict[str, Any]
    instances: List[str] = field(default_factory=list)
    relations: Dict[str, List[str]] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class StructuralAnalogy:
    """A structural analogy between domains."""
    source_domain: str
    target_domain: str
    mappings: Dict[str, str]  # source_element -> target_element
    structural_similarity: float
    preserved_relations: List[Tuple[str, str, str]]  # (relation, arg1, arg2)


@dataclass
class DomainPrinciple:
    """A domain-general principle."""
    id: str
    name: str
    description: str
    conditions: List[Callable[[Dict], bool]]
    actions: List[Callable[[Dict], Any]]
    source_domains: Set[str]
    generality_score: float


class RelationExtractor:
    """
    Extract relational structure from examples.
    """

    def __init__(self):
        self._relation_types = [
            "is_a", "has_a", "part_of", "causes", "precedes",
            "similar_to", "opposite_of", "requires", "produces"
        ]

    def extract(
        self,
        examples: List[Dict[str, Any]],
        domain: str,
    ) -> List[Tuple[str, str, str, float]]:
        """
        Extract relations from examples.

        Returns:
            List of (relation, subject, object, confidence)
        """
        relations = []

        for example in examples:
            # Extract implicit relations from structure
            if "type" in example and "properties" in example:
                # IS-A relation
                relations.append((
                    "is_a",
                    example.get("id", "unknown"),
                    example["type"],
                    1.0
                ))

                # HAS-A relations from properties
                for prop in example.get("properties", {}):
                    relations.append((
                        "has_a",
                        example.get("id", "unknown"),
                        prop,
                        0.8
                    ))

            # Causal relations
            if "causes" in example:
                for effect in example["causes"]:
                    relations.append((
                        "causes",
                        example.get("id", "unknown"),
                        effect,
                        0.9
                    ))

            # Temporal relations
            if "precedes" in example:
                for successor in example["precedes"]:
                    relations.append((
                        "precedes",
                        example.get("id", "unknown"),
                        successor,
                        0.9
                    ))

        return relations


class StructureMapper:
    """
    Map structural patterns between domains (SMT-based).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
    ):
        self.similarity_threshold = similarity_threshold

    def find_mapping(
        self,
        source_structure: Dict[str, List[Tuple]],
        target_structure: Dict[str, List[Tuple]],
    ) -> StructuralAnalogy:
        """
        Find structural analogy between source and target.

        Args:
            source_structure: {entity: [(relation, other_entity), ...]}
            target_structure: Same format

        Returns:
            StructuralAnalogy with element mappings
        """
        source_entities = set(source_structure.keys())
        target_entities = set(target_structure.keys())

        # Compute structural features for each entity
        source_features = self._compute_structural_features(source_structure)
        target_features = self._compute_structural_features(target_structure)

        # Find best mapping using similarity
        mappings = {}
        used_targets = set()

        for s_entity in sorted(source_entities, key=lambda x: -len(source_structure.get(x, []))):
            best_match = None
            best_similarity = -1

            for t_entity in target_entities - used_targets:
                sim = self._entity_similarity(
                    source_features.get(s_entity, {}),
                    target_features.get(t_entity, {}),
                )
                if sim > best_similarity and sim >= self.similarity_threshold:
                    best_similarity = sim
                    best_match = t_entity

            if best_match:
                mappings[s_entity] = best_match
                used_targets.add(best_match)

        # Find preserved relations
        preserved = self._find_preserved_relations(
            source_structure, target_structure, mappings
        )

        # Compute overall similarity
        if mappings:
            similarity = len(preserved) / max(
                sum(len(rels) for rels in source_structure.values()),
                1
            )
        else:
            similarity = 0.0

        return StructuralAnalogy(
            source_domain="source",
            target_domain="target",
            mappings=mappings,
            structural_similarity=similarity,
            preserved_relations=preserved,
        )

    def _compute_structural_features(
        self,
        structure: Dict[str, List[Tuple]],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute structural features for each entity."""
        features = {}

        for entity, relations in structure.items():
            relation_counts = {}
            for rel, other in relations:
                relation_counts[rel] = relation_counts.get(rel, 0) + 1

            features[entity] = {
                "degree": len(relations),
                "relation_types": set(rel for rel, _ in relations),
                "relation_counts": relation_counts,
            }

        return features

    def _entity_similarity(
        self,
        features1: Dict,
        features2: Dict,
    ) -> float:
        """Compute similarity between entity features."""
        if not features1 or not features2:
            return 0.0

        # Degree similarity
        degree_sim = 1.0 / (1.0 + abs(features1.get("degree", 0) - features2.get("degree", 0)))

        # Relation type overlap
        types1 = features1.get("relation_types", set())
        types2 = features2.get("relation_types", set())
        if types1 or types2:
            type_sim = len(types1 & types2) / len(types1 | types2)
        else:
            type_sim = 1.0

        return (degree_sim + type_sim) / 2

    def _find_preserved_relations(
        self,
        source: Dict,
        target: Dict,
        mappings: Dict[str, str],
    ) -> List[Tuple[str, str, str]]:
        """Find relations preserved under the mapping."""
        preserved = []

        for s_entity, relations in source.items():
            if s_entity not in mappings:
                continue

            t_entity = mappings[s_entity]
            target_rels = target.get(t_entity, [])
            target_rel_set = {(rel, mappings.get(other, other)) for rel, other in target_rels}

            for rel, other in relations:
                mapped_other = mappings.get(other, other)
                if (rel, mapped_other) in target_rel_set:
                    preserved.append((rel, s_entity, other))

        return preserved


class PrincipleExtractor:
    """
    Extract abstract principles from examples.
    """

    def __init__(
        self,
        min_support: int = 2,
    ):
        self.min_support = min_support
        self._principle_counter = 0

    def extract(
        self,
        examples: List[Dict[str, Any]],
        domain: str,
    ) -> List[DomainPrinciple]:
        """
        Extract principles from examples.

        Args:
            examples: List of example situations/actions
            domain: Source domain name

        Returns:
            List of extracted principles
        """
        principles = []

        # Find common patterns
        patterns = self._find_patterns(examples)

        for pattern, support in patterns.items():
            if support >= self.min_support:
                self._principle_counter += 1
                principle = self._pattern_to_principle(pattern, domain, support)
                principles.append(principle)

        return principles

    def _find_patterns(
        self,
        examples: List[Dict],
    ) -> Dict[str, int]:
        """Find recurring patterns in examples."""
        patterns = {}

        for example in examples:
            # Extract condition-action patterns
            if "condition" in example and "action" in example:
                pattern_key = f"{self._normalize(example['condition'])}=>{self._normalize(example['action'])}"
                patterns[pattern_key] = patterns.get(pattern_key, 0) + 1

            # Extract feature patterns
            if "features" in example:
                for feat, value in example["features"].items():
                    pattern_key = f"has_{feat}:{type(value).__name__}"
                    patterns[pattern_key] = patterns.get(pattern_key, 0) + 1

        return patterns

    def _normalize(self, condition: Any) -> str:
        """Normalize condition to pattern string."""
        if isinstance(condition, dict):
            return ",".join(sorted(condition.keys()))
        return str(type(condition).__name__)

    def _pattern_to_principle(
        self,
        pattern: str,
        domain: str,
        support: int,
    ) -> DomainPrinciple:
        """Convert pattern to principle."""
        # Parse pattern
        if "=>" in pattern:
            cond, action = pattern.split("=>", 1)
            name = f"if_{cond}_then_{action}"
        else:
            name = pattern

        return DomainPrinciple(
            id=f"principle_{self._principle_counter}",
            name=name,
            description=f"Principle extracted from {domain} with {support} examples",
            conditions=[lambda d, p=pattern: True],  # Placeholder
            actions=[lambda d, p=pattern: None],     # Placeholder
            source_domains={domain},
            generality_score=min(1.0, support / 10),
        )


class AbstractionEngine:
    """
    Complete abstraction engine for transfer learning.

    Extracts hierarchical abstractions and domain-general principles.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
    ):
        self.embedding_dim = embedding_dim

        self.relation_extractor = RelationExtractor()
        self.structure_mapper = StructureMapper()
        self.principle_extractor = PrincipleExtractor()

        # Concept store
        self._concepts: Dict[str, AbstractConcept] = {}
        self._principles: Dict[str, DomainPrinciple] = {}
        self._concept_counter = 0

    def abstract(
        self,
        examples: List[Dict[str, Any]],
        domain: str,
        target_level: AbstractionLevel = AbstractionLevel.STRUCTURAL,
    ) -> List[AbstractConcept]:
        """
        Abstract concepts from examples.

        Args:
            examples: Concrete examples from a domain
            domain: Domain name
            target_level: Target abstraction level

        Returns:
            List of abstracted concepts
        """
        concepts = []

        # Level 1: Categorical abstraction
        categories = self._categorize(examples)
        for cat_name, cat_examples in categories.items():
            concept = self._create_concept(
                cat_name, AbstractionLevel.CATEGORICAL, cat_examples, domain
            )
            concepts.append(concept)
            self._concepts[concept.id] = concept

        # Level 2: Structural abstraction
        if target_level.value in [
            AbstractionLevel.STRUCTURAL.value,
            AbstractionLevel.PRINCIPLED.value,
            AbstractionLevel.UNIVERSAL.value,
        ]:
            relations = self.relation_extractor.extract(examples, domain)
            structural_concepts = self._abstract_structure(relations, domain)
            concepts.extend(structural_concepts)

        # Level 3: Principled abstraction
        if target_level.value in [
            AbstractionLevel.PRINCIPLED.value,
            AbstractionLevel.UNIVERSAL.value,
        ]:
            principles = self.principle_extractor.extract(examples, domain)
            for principle in principles:
                self._principles[principle.id] = principle

        return concepts

    def _categorize(
        self,
        examples: List[Dict],
    ) -> Dict[str, List[Dict]]:
        """Categorize examples by type."""
        categories = {}

        for example in examples:
            cat = example.get("type", example.get("category", "unknown"))
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(example)

        return categories

    def _create_concept(
        self,
        name: str,
        level: AbstractionLevel,
        examples: List[Dict],
        domain: str,
    ) -> AbstractConcept:
        """Create an abstract concept."""
        self._concept_counter += 1

        # Aggregate features
        features = {}
        for example in examples:
            for key, value in example.items():
                if key not in features:
                    features[key] = []
                features[key].append(value)

        # Create embedding
        embedding = np.random.randn(self.embedding_dim)
        embedding = embedding / np.linalg.norm(embedding)

        return AbstractConcept(
            id=f"concept_{self._concept_counter}",
            name=name,
            level=level,
            features=features,
            instances=[e.get("id", str(i)) for i, e in enumerate(examples)],
            embedding=embedding,
        )

    def _abstract_structure(
        self,
        relations: List[Tuple],
        domain: str,
    ) -> List[AbstractConcept]:
        """Abstract structural patterns."""
        # Group by relation type
        relation_groups = {}
        for rel, subj, obj, conf in relations:
            if rel not in relation_groups:
                relation_groups[rel] = []
            relation_groups[rel].append((subj, obj, conf))

        concepts = []
        for rel_type, instances in relation_groups.items():
            self._concept_counter += 1
            concept = AbstractConcept(
                id=f"concept_{self._concept_counter}",
                name=f"{rel_type}_pattern",
                level=AbstractionLevel.STRUCTURAL,
                features={"relation_type": rel_type, "instances": len(instances)},
                relations={rel_type: [f"{s}->{o}" for s, o, _ in instances]},
            )
            concepts.append(concept)
            self._concepts[concept.id] = concept

        return concepts

    def find_analogy(
        self,
        source_domain: str,
        target_domain: str,
        source_examples: List[Dict],
        target_examples: List[Dict],
    ) -> StructuralAnalogy:
        """
        Find structural analogy between domains.
        """
        # Extract structure from both domains
        source_rels = self.relation_extractor.extract(source_examples, source_domain)
        target_rels = self.relation_extractor.extract(target_examples, target_domain)

        # Convert to structure format
        source_struct = {}
        for rel, subj, obj, _ in source_rels:
            if subj not in source_struct:
                source_struct[subj] = []
            source_struct[subj].append((rel, obj))

        target_struct = {}
        for rel, subj, obj, _ in target_rels:
            if subj not in target_struct:
                target_struct[subj] = []
            target_struct[subj].append((rel, obj))

        # Find mapping
        analogy = self.structure_mapper.find_mapping(source_struct, target_struct)
        analogy.source_domain = source_domain
        analogy.target_domain = target_domain

        return analogy

    def get_principles(
        self,
        domain: Optional[str] = None,
    ) -> List[DomainPrinciple]:
        """Get extracted principles."""
        if domain is None:
            return list(self._principles.values())
        return [p for p in self._principles.values() if domain in p.source_domains]

    def get_concepts(
        self,
        level: Optional[AbstractionLevel] = None,
    ) -> List[AbstractConcept]:
        """Get abstracted concepts."""
        if level is None:
            return list(self._concepts.values())
        return [c for c in self._concepts.values() if c.level == level]

    def transfer_concept(
        self,
        concept_id: str,
        target_domain: str,
        mapping: Dict[str, str],
    ) -> AbstractConcept:
        """Transfer a concept to new domain using mapping."""
        source = self._concepts.get(concept_id)
        if not source:
            raise ValueError(f"Concept {concept_id} not found")

        self._concept_counter += 1

        # Apply mapping to relations
        new_relations = {}
        for rel_type, rels in source.relations.items():
            new_rels = []
            for rel_str in rels:
                for src, tgt in mapping.items():
                    rel_str = rel_str.replace(src, tgt)
                new_rels.append(rel_str)
            new_relations[rel_type] = new_rels

        return AbstractConcept(
            id=f"concept_{self._concept_counter}",
            name=f"{source.name}_{target_domain}",
            level=source.level,
            features=source.features.copy(),
            instances=[],
            relations=new_relations,
            embedding=source.embedding.copy() if source.embedding is not None else None,
        )

    def statistics(self) -> Dict[str, Any]:
        """Get abstraction engine statistics."""
        return {
            "n_concepts": len(self._concepts),
            "n_principles": len(self._principles),
            "concepts_by_level": {
                level.value: sum(1 for c in self._concepts.values() if c.level == level)
                for level in AbstractionLevel
            },
        }
