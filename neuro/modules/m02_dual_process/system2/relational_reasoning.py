"""
Relational Reasoning - HPC-PFC Circuit Simulation

Implements binding elements into structured representations.
This enables compositional thinking - combining known concepts in novel ways.

Key properties:
- Compositional - can create novel combinations
- Relational - captures relationships, not just features
- Analogical - can map structures between domains
- Hierarchical - structures can contain structures
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid


class RelationType(Enum):
    """Basic relation types"""

    IS_A = "is_a"
    HAS = "has"
    PART_OF = "part_of"
    CAUSES = "causes"
    BEFORE = "before"
    AFTER = "after"
    GREATER = "greater"
    LESS = "less"
    EQUALS = "equals"
    SIMILAR = "similar"
    OPPOSITE = "opposite"
    MODIFIER = "modifier"  # e.g., "twice" modifies "jump"
    AGENT = "agent"
    PATIENT = "patient"
    CUSTOM = "custom"


@dataclass
class Element:
    """A conceptual element that can participate in relations"""

    id: str
    content: Any
    type_tag: str = "entity"
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """A relation between elements"""

    id: str
    source: str  # Element ID
    target: str  # Element ID
    relation_type: RelationType
    strength: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Structure:
    """A structured representation (bound elements + relations)"""

    id: str
    elements: Dict[str, Element]
    relations: List[Relation]
    root: Optional[str] = None  # Root element ID
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_element(self, element_id: str) -> Optional[Element]:
        return self.elements.get(element_id)

    def get_relations(self, element_id: str) -> List[Relation]:
        return [r for r in self.relations if r.source == element_id or r.target == element_id]


class RelationalReasoning:
    """
    Binding elements into structured representations.

    Simulates HPC-PFC circuit:
    - Creates relational structures
    - Composes structures together
    - Decomposes structures into elements
    - Maps analogies between structures
    """

    def __init__(self):
        self.elements: Dict[str, Element] = {}
        self.structures: Dict[str, Structure] = {}

    def create_element(
        self,
        content: Any,
        type_tag: str = "entity",
        features: Optional[Dict[str, Any]] = None,
        element_id: Optional[str] = None,
    ) -> Element:
        """Create a new conceptual element"""
        if element_id is None:
            element_id = str(uuid.uuid4())[:8]

        element = Element(
            id=element_id, content=content, type_tag=type_tag, features=features or {}
        )

        self.elements[element_id] = element
        return element

    def bind(
        self,
        element1: Element,
        relation_type: RelationType,
        element2: Element,
        strength: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Structure:
        """
        Bind two elements with a relation to create a structure.

        This is the core compositional operation - combining elements
        into structured wholes.
        """
        structure_id = str(uuid.uuid4())[:8]
        relation_id = str(uuid.uuid4())[:8]

        relation = Relation(
            id=relation_id,
            source=element1.id,
            target=element2.id,
            relation_type=relation_type,
            strength=strength,
            metadata=metadata or {},
        )

        structure = Structure(
            id=structure_id,
            elements={element1.id: element1, element2.id: element2},
            relations=[relation],
            root=element1.id,
        )

        self.structures[structure_id] = structure
        return structure

    def compose(
        self,
        structure1: Structure,
        structure2: Structure,
        binding_relation: Optional[Relation] = None,
    ) -> Structure:
        """
        Compose two structures into a larger structure.

        This enables hierarchical composition - structures of structures.
        """
        new_id = str(uuid.uuid4())[:8]

        # Merge elements
        elements = {}
        elements.update(structure1.elements)
        elements.update(structure2.elements)

        # Merge relations
        relations = list(structure1.relations) + list(structure2.relations)

        if binding_relation:
            relations.append(binding_relation)

        structure = Structure(
            id=new_id,
            elements=elements,
            relations=relations,
            root=structure1.root,
            metadata={"composed_from": [structure1.id, structure2.id]},
        )

        self.structures[new_id] = structure
        return structure

    def decompose(self, structure: Structure) -> Tuple[Dict[str, Element], List[Relation]]:
        """
        Decompose structure into its constituent elements and relations.

        This is the inverse of binding - extracting parts from wholes.
        """
        return structure.elements.copy(), list(structure.relations)

    def bind_modifier(self, base_concept: Element, modifier: Element) -> Structure:
        """
        Bind a modifier to a base concept.

        E.g., "jump" + "twice" = "jump twice"
        This is crucial for compositional language understanding.
        """
        return self.bind(
            modifier,
            RelationType.MODIFIER,
            base_concept,
            metadata={"composition_type": "modification"},
        )

    def create_action_structure(
        self,
        action: Element,
        agent: Optional[Element] = None,
        patient: Optional[Element] = None,
        modifiers: Optional[List[Element]] = None,
    ) -> Structure:
        """
        Create an action structure with roles.

        E.g., "The dog bit the cat" has:
        - action: "bit"
        - agent: "dog"
        - patient: "cat"
        """
        structure_id = str(uuid.uuid4())[:8]
        elements = {action.id: action}
        relations = []

        if agent:
            elements[agent.id] = agent
            relations.append(
                Relation(
                    id=str(uuid.uuid4())[:8],
                    source=agent.id,
                    target=action.id,
                    relation_type=RelationType.AGENT,
                )
            )

        if patient:
            elements[patient.id] = patient
            relations.append(
                Relation(
                    id=str(uuid.uuid4())[:8],
                    source=patient.id,
                    target=action.id,
                    relation_type=RelationType.PATIENT,
                )
            )

        if modifiers:
            for mod in modifiers:
                elements[mod.id] = mod
                relations.append(
                    Relation(
                        id=str(uuid.uuid4())[:8],
                        source=mod.id,
                        target=action.id,
                        relation_type=RelationType.MODIFIER,
                    )
                )

        structure = Structure(
            id=structure_id,
            elements=elements,
            relations=relations,
            root=action.id,
            metadata={"structure_type": "action_frame"},
        )

        self.structures[structure_id] = structure
        return structure

    def analogy(
        self, source: Structure, target_elements: Dict[str, Element]
    ) -> Optional[Structure]:
        """
        Map relational structure from source to target.

        Analogical reasoning: if A:B::C:? then find the D that has
        the same relation to C as B has to A.
        """
        # Get source relations
        _, source_relations = self.decompose(source)

        if not source_relations:
            return None

        # Build mapping from source elements to target elements
        # (This is simplified - full analogy mapping is complex)
        mapping = {}
        source_elements = list(source.elements.keys())

        for i, src_id in enumerate(source_elements):
            if i < len(target_elements):
                target_ids = list(target_elements.keys())
                mapping[src_id] = target_ids[i]

        # Create new relations with mapped elements
        new_relations = []
        for rel in source_relations:
            if rel.source in mapping and rel.target in mapping:
                new_relations.append(
                    Relation(
                        id=str(uuid.uuid4())[:8],
                        source=mapping[rel.source],
                        target=mapping[rel.target],
                        relation_type=rel.relation_type,
                        strength=rel.strength * 0.8,  # Slight uncertainty in analogies
                        metadata={"analogical_source": rel.id},
                    )
                )

        if not new_relations:
            return None

        structure = Structure(
            id=str(uuid.uuid4())[:8],
            elements=target_elements,
            relations=new_relations,
            metadata={"analogical_from": source.id},
        )

        self.structures[structure.id] = structure
        return structure

    def find_similar_structures(
        self, structure: Structure, threshold: float = 0.7
    ) -> List[Tuple[Structure, float]]:
        """
        Find structures with similar relational patterns.

        This enables structure-based retrieval and generalization.
        """
        results = []

        for sid, other in self.structures.items():
            if sid == structure.id:
                continue

            similarity = self._compute_structural_similarity(structure, other)
            if similarity >= threshold:
                results.append((other, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _compute_structural_similarity(self, s1: Structure, s2: Structure) -> float:
        """Compute similarity between two structures based on relations"""
        r1_types = {r.relation_type for r in s1.relations}
        r2_types = {r.relation_type for r in s2.relations}

        if not r1_types and not r2_types:
            return 1.0

        intersection = len(r1_types & r2_types)
        union = len(r1_types | r2_types)

        return intersection / union if union > 0 else 0.0

    def query_by_relation(self, relation_type: RelationType) -> List[Structure]:
        """Find all structures containing a specific relation type"""
        results = []
        for structure in self.structures.values():
            for rel in structure.relations:
                if rel.relation_type == relation_type:
                    results.append(structure)
                    break
        return results

    def get_related_elements(
        self, element_id: str, relation_type: Optional[RelationType] = None
    ) -> List[Tuple[Element, Relation]]:
        """Get all elements related to the given element"""
        results = []

        for structure in self.structures.values():
            for rel in structure.relations:
                if rel.source == element_id:
                    if relation_type is None or rel.relation_type == relation_type:
                        target = structure.elements.get(rel.target)
                        if target:
                            results.append((target, rel))

                if rel.target == element_id:
                    if relation_type is None or rel.relation_type == relation_type:
                        source = structure.elements.get(rel.source)
                        if source:
                            results.append((source, rel))

        return results
