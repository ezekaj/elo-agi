"""
Abstraction Engine for Compositional Generalization.

Implements hierarchical abstraction and analogy:
- Abstraction extraction from examples
- Structure mapping for analogies
- Concept transfer across domains
- Principle extraction
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from enum import Enum
import numpy as np
from itertools import combinations


class AbstractionLevel(Enum):
    """Levels of abstraction."""
    INSTANCE = "instance"        # Concrete examples
    CONCEPT = "concept"          # Categories
    SCHEMA = "schema"            # Structural patterns
    PRINCIPLE = "principle"      # Abstract rules
    UNIVERSAL = "universal"      # Domain-invariant


@dataclass
class Abstraction:
    """
    An abstraction extracted from examples.

    Represents a generalized pattern that captures common structure.
    """
    id: str
    name: str
    level: AbstractionLevel
    variables: List[str]  # Abstracted slots
    structure: Dict[str, Any]  # Pattern structure
    instances: List[Dict[str, Any]] = field(default_factory=list)
    constraints: List[Callable[[Dict], bool]] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0

    def matches(self, instance: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if instance matches this abstraction.

        Returns (matches, variable_bindings)
        """
        bindings = {}

        for var in self.variables:
            if var in instance:
                bindings[var] = instance[var]

        # Check constraints
        for constraint in self.constraints:
            try:
                if not constraint(bindings):
                    return False, {}
            except Exception:
                return False, {}

        return True, bindings

    def instantiate(self, bindings: Dict[str, Any]) -> Dict[str, Any]:
        """Create concrete instance from variable bindings."""
        result = dict(self.structure)

        for var in self.variables:
            if var in bindings:
                result[var] = bindings[var]

        return result


@dataclass
class StructureMapping:
    """
    A mapping between two relational structures.

    Implements Structure-Mapping Theory for analogical reasoning.
    """
    source_domain: str
    target_domain: str
    object_mappings: Dict[str, str]  # source_obj -> target_obj
    relation_mappings: Dict[str, str]  # source_rel -> target_rel
    structural_consistency: float
    systematicity: float  # Preference for connected structures
    inferences: List[Dict[str, Any]] = field(default_factory=list)

    def map_object(self, source_obj: str) -> Optional[str]:
        """Map source object to target."""
        return self.object_mappings.get(source_obj)

    def map_relation(self, source_rel: str) -> Optional[str]:
        """Map source relation to target."""
        return self.relation_mappings.get(source_rel)

    def quality_score(self) -> float:
        """Overall quality of the mapping."""
        return (self.structural_consistency + self.systematicity) / 2


@dataclass
class Principle:
    """
    An abstract principle extracted from patterns.

    Principles are domain-general rules that apply across contexts.
    """
    id: str
    name: str
    preconditions: List[Callable[[Dict], bool]]
    effects: List[Callable[[Dict], Dict]]
    source_domains: Set[str]
    generality: float  # How broadly applicable
    confidence: float


class AbstractionEngine:
    """
    Engine for abstraction and analogical reasoning.

    Supports:
    - Hierarchical abstraction from examples
    - Structure-mapping analogies
    - Cross-domain concept transfer
    - Principle extraction
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        similarity_threshold: float = 0.5,
        random_seed: Optional[int] = None,
    ):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        self._rng = np.random.default_rng(random_seed)

        # Storage
        self._abstractions: Dict[str, Abstraction] = {}
        self._mappings: List[StructureMapping] = []
        self._principles: Dict[str, Principle] = {}

        # Counters
        self._abstraction_counter = 0
        self._principle_counter = 0

        # Statistics
        self._n_abstractions_created = 0
        self._n_analogies_found = 0
        self._n_transfers = 0

    def abstract(
        self,
        examples: List[Dict[str, Any]],
        target_level: AbstractionLevel = AbstractionLevel.SCHEMA,
        min_support: int = 2,
    ) -> List[Abstraction]:
        """
        Extract abstractions from examples.

        Args:
            examples: List of concrete examples
            target_level: Target abstraction level
            min_support: Minimum examples to support abstraction

        Returns:
            List of extracted abstractions
        """
        abstractions = []

        if target_level == AbstractionLevel.INSTANCE:
            # Just wrap examples
            for ex in examples:
                self._abstraction_counter += 1
                abs = Abstraction(
                    id=f"abs_{self._abstraction_counter}",
                    name=f"instance_{self._abstraction_counter}",
                    level=AbstractionLevel.INSTANCE,
                    variables=[],
                    structure=ex,
                    instances=[ex],
                )
                abstractions.append(abs)

        elif target_level == AbstractionLevel.CONCEPT:
            # Group by common features
            abstractions = self._abstract_to_concepts(examples, min_support)

        elif target_level == AbstractionLevel.SCHEMA:
            # Extract structural patterns
            abstractions = self._abstract_to_schemas(examples, min_support)

        elif target_level == AbstractionLevel.PRINCIPLE:
            # Extract rules
            abstractions = self._abstract_to_principles(examples, min_support)

        for abs in abstractions:
            self._abstractions[abs.id] = abs
            self._n_abstractions_created += 1

        return abstractions

    def _abstract_to_concepts(
        self,
        examples: List[Dict[str, Any]],
        min_support: int,
    ) -> List[Abstraction]:
        """Group examples into concepts based on shared features."""
        # Find common features
        if not examples:
            return []

        all_keys = set()
        for ex in examples:
            all_keys.update(ex.keys())

        # Group by feature patterns
        groups = {}
        for ex in examples:
            # Create feature signature
            sig = tuple(sorted(ex.keys()))
            if sig not in groups:
                groups[sig] = []
            groups[sig].append(ex)

        abstractions = []
        for sig, group in groups.items():
            if len(group) >= min_support:
                self._abstraction_counter += 1

                # Find variables (features that vary)
                variables = []
                constants = {}

                for key in sig:
                    values = [ex.get(key) for ex in group]
                    if len(set(str(v) for v in values)) > 1:
                        variables.append(key)
                    else:
                        constants[key] = values[0]

                abstraction = Abstraction(
                    id=f"abs_{self._abstraction_counter}",
                    name=f"concept_{self._abstraction_counter}",
                    level=AbstractionLevel.CONCEPT,
                    variables=variables,
                    structure=constants,
                    instances=group,
                    embedding=self._create_embedding(constants),
                )
                abstractions.append(abstraction)

        return abstractions

    def _abstract_to_schemas(
        self,
        examples: List[Dict[str, Any]],
        min_support: int,
    ) -> List[Abstraction]:
        """Extract structural schemas from examples."""
        # Look for relational patterns
        patterns = {}

        for ex in examples:
            # Extract relations (keys with structured values)
            for key, value in ex.items():
                if isinstance(value, dict):
                    pattern_key = (key, tuple(sorted(value.keys())))
                    if pattern_key not in patterns:
                        patterns[pattern_key] = []
                    patterns[pattern_key].append((ex, value))

        abstractions = []
        for pattern_key, instances in patterns.items():
            if len(instances) >= min_support:
                self._abstraction_counter += 1
                rel_name, struct = pattern_key

                abstraction = Abstraction(
                    id=f"abs_{self._abstraction_counter}",
                    name=f"schema_{rel_name}",
                    level=AbstractionLevel.SCHEMA,
                    variables=list(struct),
                    structure={"relation": rel_name, "slots": list(struct)},
                    instances=[inst[0] for inst in instances],
                    embedding=self._create_embedding({"rel": rel_name}),
                )
                abstractions.append(abstraction)

        return abstractions

    def _abstract_to_principles(
        self,
        examples: List[Dict[str, Any]],
        min_support: int,
    ) -> List[Abstraction]:
        """Extract principles (if-then rules) from examples."""
        # Look for condition-action patterns
        principles = []

        # Find examples with "condition" and "result" or similar
        conditional_examples = [
            ex for ex in examples
            if any(k in ex for k in ["condition", "if", "when", "given"])
        ]

        if not conditional_examples:
            return principles

        # Group by condition type
        condition_groups = {}
        for ex in conditional_examples:
            cond_key = None
            for key in ["condition", "if", "when", "given"]:
                if key in ex:
                    cond_type = type(ex[key]).__name__
                    cond_key = (key, cond_type)
                    break

            if cond_key:
                if cond_key not in condition_groups:
                    condition_groups[cond_key] = []
                condition_groups[cond_key].append(ex)

        for cond_key, group in condition_groups.items():
            if len(group) >= min_support:
                self._abstraction_counter += 1

                abstraction = Abstraction(
                    id=f"abs_{self._abstraction_counter}",
                    name=f"principle_{cond_key[0]}",
                    level=AbstractionLevel.PRINCIPLE,
                    variables=["condition", "result"],
                    structure={"type": cond_key[0]},
                    instances=group,
                )
                principles.append(abstraction)

        return principles

    def find_analogy(
        self,
        source: Dict[str, Any],
        target: Dict[str, Any],
        source_relations: Optional[List[Tuple[str, str, str]]] = None,
        target_relations: Optional[List[Tuple[str, str, str]]] = None,
    ) -> StructureMapping:
        """
        Find analogical mapping between source and target.

        Uses Structure-Mapping Theory.
        """
        self._n_analogies_found += 1

        # Extract objects
        source_objects = set(source.keys())
        target_objects = set(target.keys())

        # Build object mappings based on value similarity
        object_mappings = {}
        used_targets = set()

        for s_obj in source_objects:
            best_match = None
            best_sim = -1

            for t_obj in target_objects - used_targets:
                sim = self._value_similarity(source.get(s_obj), target.get(t_obj))
                if sim > best_sim and sim > self.similarity_threshold:
                    best_sim = sim
                    best_match = t_obj

            if best_match:
                object_mappings[s_obj] = best_match
                used_targets.add(best_match)

        # Build relation mappings
        relation_mappings = {}
        if source_relations and target_relations:
            for s_rel, s_arg1, s_arg2 in source_relations:
                for t_rel, t_arg1, t_arg2 in target_relations:
                    # Check if arguments map
                    if (object_mappings.get(s_arg1) == t_arg1 and
                        object_mappings.get(s_arg2) == t_arg2):
                        relation_mappings[s_rel] = t_rel

        # Compute consistency and systematicity
        consistency = len(object_mappings) / max(len(source_objects), 1)
        systematicity = len(relation_mappings) / max(len(source_relations or []), 1)

        mapping = StructureMapping(
            source_domain="source",
            target_domain="target",
            object_mappings=object_mappings,
            relation_mappings=relation_mappings,
            structural_consistency=consistency,
            systematicity=systematicity,
        )

        self._mappings.append(mapping)
        return mapping

    def transfer_concept(
        self,
        concept: Abstraction,
        target_domain: str,
        mapping: StructureMapping,
    ) -> Abstraction:
        """
        Transfer a concept to a new domain using a mapping.
        """
        self._n_transfers += 1
        self._abstraction_counter += 1

        # Map structure
        new_structure = {}
        for key, value in concept.structure.items():
            new_key = mapping.object_mappings.get(key, key)
            new_structure[new_key] = value

        # Map variables
        new_variables = [
            mapping.object_mappings.get(v, v)
            for v in concept.variables
        ]

        return Abstraction(
            id=f"abs_{self._abstraction_counter}",
            name=f"{concept.name}_{target_domain}",
            level=concept.level,
            variables=new_variables,
            structure=new_structure,
            instances=[],  # New domain, no instances yet
            embedding=self._create_embedding(new_structure),
        )

    def extract_principles(
        self,
        examples: List[Dict[str, Any]],
        domains: List[str],
    ) -> List[Principle]:
        """
        Extract domain-general principles from examples across domains.
        """
        principles = []

        # Find patterns that repeat across domains
        domain_patterns = {}
        for i, ex in enumerate(examples):
            domain = domains[i] if i < len(domains) else "unknown"
            pattern = self._extract_pattern(ex)

            if pattern not in domain_patterns:
                domain_patterns[pattern] = set()
            domain_patterns[pattern].add(domain)

        # Patterns appearing in multiple domains become principles
        for pattern, pattern_domains in domain_patterns.items():
            if len(pattern_domains) >= 2:
                self._principle_counter += 1

                principle = Principle(
                    id=f"principle_{self._principle_counter}",
                    name=f"cross_domain_principle_{self._principle_counter}",
                    preconditions=[],
                    effects=[],
                    source_domains=pattern_domains,
                    generality=len(pattern_domains) / len(set(domains)),
                    confidence=1.0,
                )
                principles.append(principle)
                self._principles[principle.id] = principle

        return principles

    def _extract_pattern(self, example: Dict[str, Any]) -> str:
        """Extract pattern signature from example."""
        # Use sorted keys and value types
        items = []
        for key in sorted(example.keys()):
            value_type = type(example[key]).__name__
            items.append(f"{key}:{value_type}")
        return "|".join(items)

    def _value_similarity(self, v1: Any, v2: Any) -> float:
        """Compute similarity between two values."""
        if v1 is None or v2 is None:
            return 0.0

        if type(v1) != type(v2):
            return 0.0

        if isinstance(v1, (int, float)):
            # Numerical similarity
            diff = abs(v1 - v2)
            max_val = max(abs(v1), abs(v2), 1)
            return 1.0 / (1.0 + diff / max_val)

        if isinstance(v1, str):
            # String similarity (simple)
            if v1 == v2:
                return 1.0
            # Jaccard on characters
            s1, s2 = set(v1), set(v2)
            if not s1 and not s2:
                return 1.0
            return len(s1 & s2) / len(s1 | s2)

        if isinstance(v1, dict):
            # Recursive similarity
            keys = set(v1.keys()) | set(v2.keys())
            if not keys:
                return 1.0
            sims = [
                self._value_similarity(v1.get(k), v2.get(k))
                for k in keys
            ]
            return np.mean(sims)

        if isinstance(v1, list):
            if not v1 and not v2:
                return 1.0
            if not v1 or not v2:
                return 0.0
            # Average element similarity
            sims = []
            for e1, e2 in zip(v1, v2):
                sims.append(self._value_similarity(e1, e2))
            return np.mean(sims) if sims else 0.0

        return 1.0 if v1 == v2 else 0.0

    def _create_embedding(self, structure: Dict[str, Any]) -> np.ndarray:
        """Create embedding for a structure."""
        # Simple hash-based embedding
        np.random.seed(hash(str(sorted(structure.items()))) % (2**32))
        embedding = self._rng.normal(0, 1, self.embedding_dim)
        return embedding / np.linalg.norm(embedding)

    def get_abstraction(self, abstraction_id: str) -> Optional[Abstraction]:
        """Get abstraction by ID."""
        return self._abstractions.get(abstraction_id)

    def get_principle(self, principle_id: str) -> Optional[Principle]:
        """Get principle by ID."""
        return self._principles.get(principle_id)

    def find_matching_abstraction(
        self,
        instance: Dict[str, Any],
    ) -> List[Tuple[Abstraction, Dict[str, Any]]]:
        """Find abstractions that match an instance."""
        matches = []

        for abs in self._abstractions.values():
            is_match, bindings = abs.matches(instance)
            if is_match:
                matches.append((abs, bindings))

        return matches

    def statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "n_abstractions": len(self._abstractions),
            "n_principles": len(self._principles),
            "n_mappings": len(self._mappings),
            "n_abstractions_created": self._n_abstractions_created,
            "n_analogies_found": self._n_analogies_found,
            "n_transfers": self._n_transfers,
            "abstraction_levels": {
                level.value: sum(
                    1 for a in self._abstractions.values() if a.level == level
                )
                for level in AbstractionLevel
            },
        }
