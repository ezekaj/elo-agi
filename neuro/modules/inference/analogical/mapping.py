"""
Structure Mapping: Analogical reasoning via structure mapping.

Implements Gentner's Structure Mapping Theory for analogy.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np


class RelationOrder(Enum):
    """Order of relations (first-order, higher-order)."""

    FIRST = 1  # Relations between objects (e.g., larger(sun, moon))
    SECOND = 2  # Relations between relations (e.g., cause(heat, expand))
    THIRD = 3  # Relations between second-order relations


@dataclass
class Predicate:
    """A predicate in a relational structure."""

    name: str
    arguments: List[str]  # Object or relation names
    order: RelationOrder = RelationOrder.FIRST
    properties: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.name, tuple(self.arguments)))

    def arity(self) -> int:
        return len(self.arguments)


@dataclass
class RelationalStructure:
    """
    A relational structure representing a domain.

    Contains objects and relations between them.
    """

    name: str
    objects: Set[str] = field(default_factory=set)
    predicates: List[Predicate] = field(default_factory=list)
    attributes: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_object(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        """Add an object to the structure."""
        self.objects.add(name)
        if attrs:
            self.attributes[name] = attrs

    def add_predicate(self, predicate: Predicate) -> None:
        """Add a predicate/relation."""
        self.predicates.append(predicate)
        for arg in predicate.arguments:
            self.objects.add(arg)

    def add_relation(
        self,
        name: str,
        arguments: List[str],
        order: RelationOrder = RelationOrder.FIRST,
    ) -> Predicate:
        """Convenience method to add a relation."""
        pred = Predicate(name, arguments, order)
        self.add_predicate(pred)
        return pred

    def get_predicates_involving(self, obj: str) -> List[Predicate]:
        """Get all predicates involving an object."""
        return [p for p in self.predicates if obj in p.arguments]


@dataclass
class Mapping:
    """A mapping between two objects/relations."""

    source: str
    target: str
    confidence: float = 1.0
    is_object: bool = True  # vs. relation mapping


@dataclass
class StructuralAlignment:
    """
    An alignment between two relational structures.

    Contains object and relation mappings.
    """

    source: RelationalStructure
    target: RelationalStructure
    object_mappings: Dict[str, str] = field(default_factory=dict)  # source -> target
    relation_mappings: Dict[str, str] = field(default_factory=dict)
    score: float = 0.0
    matched_predicates: List[Tuple[Predicate, Predicate]] = field(default_factory=list)

    def get_target_for(self, source_item: str) -> Optional[str]:
        """Get target mapping for a source item."""
        return self.object_mappings.get(source_item, self.relation_mappings.get(source_item))


@dataclass
class Analogy:
    """
    A complete analogy between two domains.

    Includes the alignment and any inferences drawn.
    """

    source_domain: RelationalStructure
    target_domain: RelationalStructure
    alignment: StructuralAlignment
    inferences: List[Predicate] = field(default_factory=list)  # New predicates inferred for target
    systematicity: float = 0.0  # How systematic the analogy is
    similarity: float = 0.0

    def explain(self) -> str:
        """Generate explanation of the analogy."""
        lines = [f"Analogy: {self.source_domain.name} -> {self.target_domain.name}"]
        lines.append(f"Systematicity: {self.systematicity:.2f}")
        lines.append(f"Similarity: {self.similarity:.2f}")

        lines.append("\nObject mappings:")
        for src, tgt in self.alignment.object_mappings.items():
            lines.append(f"  {src} -> {tgt}")

        if self.inferences:
            lines.append("\nInferences:")
            for pred in self.inferences:
                args = ", ".join(pred.arguments)
                lines.append(f"  {pred.name}({args})")

        return "\n".join(lines)


class StructureMapper:
    """
    Structure Mapping Engine for analogical reasoning.

    Implements Gentner's Structure Mapping Theory:
    - Systematicity principle: Prefer mappings with connected structure
    - One-to-one constraint: Each element maps to at most one
    - Parallel connectivity: If predicates match, arguments should match

    Supports:
    - Finding structural alignments
    - Drawing analogical inferences
    - Evaluating analogy quality
    """

    def __init__(
        self,
        relation_weight: float = 1.0,
        attribute_weight: float = 0.3,
        systematicity_weight: float = 2.0,
    ):
        self.relation_weight = relation_weight
        self.attribute_weight = attribute_weight
        self.systematicity_weight = systematicity_weight

    def map(
        self,
        source: RelationalStructure,
        target: RelationalStructure,
    ) -> StructuralAlignment:
        """
        Find the best structural alignment between source and target.

        Uses a greedy algorithm inspired by SME (Structure Mapping Engine).
        """
        # Find matching predicates (same name and arity)
        match_candidates = self._find_matching_predicates(source, target)

        # Build initial object mappings from predicate matches
        object_mappings, relation_mappings, matched_preds = self._build_mappings(
            match_candidates, source, target
        )

        # Score the alignment
        score = self._score_alignment(
            object_mappings, relation_mappings, matched_preds, source, target
        )

        return StructuralAlignment(
            source=source,
            target=target,
            object_mappings=object_mappings,
            relation_mappings=relation_mappings,
            score=score,
            matched_predicates=matched_preds,
        )

    def _find_matching_predicates(
        self,
        source: RelationalStructure,
        target: RelationalStructure,
    ) -> List[Tuple[Predicate, Predicate]]:
        """Find predicates that could potentially match."""
        matches = []

        for s_pred in source.predicates:
            for t_pred in target.predicates:
                # Same predicate name and arity
                if s_pred.name == t_pred.name and s_pred.arity() == t_pred.arity():
                    matches.append((s_pred, t_pred))

        return matches

    def _build_mappings(
        self,
        candidates: List[Tuple[Predicate, Predicate]],
        source: RelationalStructure,
        target: RelationalStructure,
    ) -> Tuple[Dict[str, str], Dict[str, str], List[Tuple[Predicate, Predicate]]]:
        """Build consistent mappings from predicate matches."""
        object_mappings = {}
        relation_mappings = {}
        matched = []

        # Sort by predicate order (prefer higher-order)
        candidates.sort(key=lambda x: x[0].order.value, reverse=True)

        for s_pred, t_pred in candidates:
            # Check if mapping is consistent with existing
            consistent = True
            proposed_maps = {}

            for s_arg, t_arg in zip(s_pred.arguments, t_pred.arguments):
                if s_arg in object_mappings:
                    if object_mappings[s_arg] != t_arg:
                        consistent = False
                        break
                else:
                    # Check one-to-one constraint
                    if t_arg in object_mappings.values():
                        consistent = False
                        break
                    proposed_maps[s_arg] = t_arg

            if consistent:
                object_mappings.update(proposed_maps)
                relation_mappings[s_pred.name] = t_pred.name
                matched.append((s_pred, t_pred))

        return object_mappings, relation_mappings, matched

    def _score_alignment(
        self,
        object_mappings: Dict[str, str],
        relation_mappings: Dict[str, str],
        matched_preds: List[Tuple[Predicate, Predicate]],
        source: RelationalStructure,
        target: RelationalStructure,
    ) -> float:
        """Score an alignment based on systematicity and coverage."""
        score = 0.0

        # Base score from matched predicates
        for s_pred, t_pred in matched_preds:
            # Higher-order predicates worth more
            order_bonus = s_pred.order.value * self.systematicity_weight
            score += self.relation_weight + order_bonus

        # Attribute similarity bonus
        for s_obj, t_obj in object_mappings.items():
            s_attrs = source.attributes.get(s_obj, {})
            t_attrs = target.attributes.get(t_obj, {})

            shared = set(s_attrs.keys()) & set(t_attrs.keys())
            for attr in shared:
                if s_attrs[attr] == t_attrs[attr]:
                    score += self.attribute_weight

        return score

    def make_analogy(
        self,
        source: RelationalStructure,
        target: RelationalStructure,
    ) -> Analogy:
        """
        Create a complete analogy with inferences.

        Finds alignment and projects unmapped source predicates to target.
        """
        alignment = self.map(source, target)

        # Generate inferences
        inferences = self._project_inferences(alignment, source, target)

        # Compute systematicity
        systematicity = self._compute_systematicity(alignment, source)

        # Compute overall similarity
        similarity = self._compute_similarity(alignment, source, target)

        return Analogy(
            source_domain=source,
            target_domain=target,
            alignment=alignment,
            inferences=inferences,
            systematicity=systematicity,
            similarity=similarity,
        )

    def _project_inferences(
        self,
        alignment: StructuralAlignment,
        source: RelationalStructure,
        target: RelationalStructure,
    ) -> List[Predicate]:
        """Project unmapped source predicates to target as inferences."""
        inferences = []
        mapped_source_preds = {s for s, _ in alignment.matched_predicates}

        for s_pred in source.predicates:
            if s_pred in mapped_source_preds:
                continue

            # Try to project this predicate
            new_args = []
            can_project = True

            for arg in s_pred.arguments:
                if arg in alignment.object_mappings:
                    new_args.append(alignment.object_mappings[arg])
                else:
                    can_project = False
                    break

            if can_project:
                inferred = Predicate(
                    name=s_pred.name,
                    arguments=new_args,
                    order=s_pred.order,
                )
                inferences.append(inferred)

        return inferences

    def _compute_systematicity(
        self,
        alignment: StructuralAlignment,
        source: RelationalStructure,
    ) -> float:
        """
        Compute systematicity of the mapping.

        Higher-order relations and connected structure increase systematicity.
        """
        if not alignment.matched_predicates:
            return 0.0

        total_order = sum(p[0].order.value for p in alignment.matched_predicates)
        max_order = len(alignment.matched_predicates) * 3  # Max is THIRD order

        return total_order / max_order if max_order > 0 else 0.0

    def _compute_similarity(
        self,
        alignment: StructuralAlignment,
        source: RelationalStructure,
        target: RelationalStructure,
    ) -> float:
        """Compute overall similarity between domains."""
        # Predicate coverage
        matched_count = len(alignment.matched_predicates)
        total_count = max(len(source.predicates), len(target.predicates))

        if total_count == 0:
            return 0.0

        return matched_count / total_count

    def evaluate_inference(
        self,
        inference: Predicate,
        target: RelationalStructure,
    ) -> float:
        """
        Evaluate plausibility of an analogical inference.

        Higher scores for inferences consistent with target structure.
        """
        # Check if objects exist in target
        if not all(arg in target.objects for arg in inference.arguments):
            return 0.0

        # Check if similar predicates exist
        similar = [
            p
            for p in target.predicates
            if p.name == inference.name or p.arity() == inference.arity()
        ]

        base_score = 0.5
        if similar:
            base_score += 0.3

        # Penalize if contradicts existing knowledge
        for p in target.predicates:
            if p.name == inference.name and p.arguments == inference.arguments:
                return 1.0  # Already exists, perfect match
            if p.name == inference.name and p.arity() == inference.arity():
                # Same predicate, different args - might contradict
                shared_args = set(p.arguments) & set(inference.arguments)
                if shared_args:
                    base_score -= 0.1

        return min(1.0, max(0.0, base_score))
