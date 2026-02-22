"""
Ontology: Hierarchical knowledge organization.

Implements IS-A, HAS-A, PART-OF hierarchies with
inheritance and classification capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np


class HierarchyType(Enum):
    """Types of hierarchical relationships."""

    TAXONOMIC = "taxonomic"  # IS-A hierarchy (dog IS-A animal)
    MERONOMIC = "meronomic"  # PART-OF hierarchy (wheel PART-OF car)
    COMPOSITIONAL = "compositional"  # HAS-A hierarchy (car HAS-A wheel)
    FUNCTIONAL = "functional"  # USED-FOR hierarchy


@dataclass
class OntologyNode:
    """A node in the ontology."""

    name: str
    definition: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    depth: int = 0
    is_abstract: bool = False

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, OntologyNode):
            return self.name == other.name
        return False


@dataclass
class OntologyRelation:
    """A hierarchical relation in the ontology."""

    parent: str
    child: str
    hierarchy_type: HierarchyType
    cardinality: str = "1"  # "1", "0..1", "1..*", "*"
    inherited: bool = True  # Whether properties inherit
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OntologyQuery:
    """A query against the ontology."""

    concept: Optional[str] = None
    hierarchy_type: Optional[HierarchyType] = None
    property_filter: Optional[Dict[str, Any]] = None
    depth: Optional[int] = None
    include_inherited: bool = True


class Ontology:
    """
    Ontological hierarchy for knowledge organization.

    Supports:
    - Multiple hierarchy types
    - Property inheritance
    - Classification and subsumption
    - Constraint checking
    """

    def __init__(self, name: str = "default"):
        self.name = name

        # Node storage
        self._nodes: Dict[str, OntologyNode] = {}

        # Hierarchies (parent -> children)
        self._hierarchies: Dict[HierarchyType, Dict[str, Set[str]]] = {
            ht: {} for ht in HierarchyType
        }

        # Inverse hierarchies (child -> parents)
        self._inverse: Dict[HierarchyType, Dict[str, Set[str]]] = {ht: {} for ht in HierarchyType}

        # All relations
        self._relations: List[OntologyRelation] = []

        # Root nodes per hierarchy
        self._roots: Dict[HierarchyType, Set[str]] = {ht: set() for ht in HierarchyType}

    def add_node(
        self,
        name: str,
        definition: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        constraints: Optional[List[str]] = None,
        is_abstract: bool = False,
    ) -> OntologyNode:
        """Add a node to the ontology."""
        node = OntologyNode(
            name=name,
            definition=definition,
            properties=properties or {},
            constraints=constraints or [],
            is_abstract=is_abstract,
        )
        self._nodes[name] = node
        return node

    def get_node(self, name: str) -> Optional[OntologyNode]:
        """Get a node by name."""
        return self._nodes.get(name)

    def has_node(self, name: str) -> bool:
        """Check if node exists."""
        return name in self._nodes

    def add_relation(
        self,
        parent: str,
        child: str,
        hierarchy_type: HierarchyType,
        cardinality: str = "1",
        inherited: bool = True,
    ) -> OntologyRelation:
        """Add a hierarchical relation."""
        # Ensure nodes exist
        if parent not in self._nodes:
            self.add_node(parent)
        if child not in self._nodes:
            self.add_node(child)

        relation = OntologyRelation(
            parent=parent,
            child=child,
            hierarchy_type=hierarchy_type,
            cardinality=cardinality,
            inherited=inherited,
        )

        self._relations.append(relation)

        # Update hierarchy structures
        if parent not in self._hierarchies[hierarchy_type]:
            self._hierarchies[hierarchy_type][parent] = set()
        self._hierarchies[hierarchy_type][parent].add(child)

        if child not in self._inverse[hierarchy_type]:
            self._inverse[hierarchy_type][child] = set()
        self._inverse[hierarchy_type][child].add(parent)

        # Update depth
        parent_node = self._nodes[parent]
        child_node = self._nodes[child]
        child_node.depth = max(child_node.depth, parent_node.depth + 1)

        # Update roots (remove child from roots if it was there)
        self._roots[hierarchy_type].discard(child)
        # Add parent to roots if it has no parents
        if parent not in self._inverse[hierarchy_type] or not self._inverse[hierarchy_type][parent]:
            self._roots[hierarchy_type].add(parent)

        return relation

    def add_is_a(self, child: str, parent: str) -> OntologyRelation:
        """Add IS-A (taxonomic) relation."""
        return self.add_relation(parent, child, HierarchyType.TAXONOMIC)

    def add_part_of(self, part: str, whole: str) -> OntologyRelation:
        """Add PART-OF (meronomic) relation."""
        return self.add_relation(whole, part, HierarchyType.MERONOMIC)

    def add_has_a(self, whole: str, part: str) -> OntologyRelation:
        """Add HAS-A (compositional) relation."""
        return self.add_relation(whole, part, HierarchyType.COMPOSITIONAL)

    def get_parents(
        self,
        node: str,
        hierarchy_type: HierarchyType = HierarchyType.TAXONOMIC,
    ) -> List[str]:
        """Get direct parents of a node."""
        return list(self._inverse[hierarchy_type].get(node, set()))

    def get_children(
        self,
        node: str,
        hierarchy_type: HierarchyType = HierarchyType.TAXONOMIC,
    ) -> List[str]:
        """Get direct children of a node."""
        return list(self._hierarchies[hierarchy_type].get(node, set()))

    def get_ancestors(
        self,
        node: str,
        hierarchy_type: HierarchyType = HierarchyType.TAXONOMIC,
    ) -> List[str]:
        """Get all ancestors of a node."""
        ancestors = []
        visited = set()
        queue = list(self._inverse[hierarchy_type].get(node, set()))

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            ancestors.append(current)
            queue.extend(self._inverse[hierarchy_type].get(current, set()))

        return ancestors

    def get_descendants(
        self,
        node: str,
        hierarchy_type: HierarchyType = HierarchyType.TAXONOMIC,
    ) -> List[str]:
        """Get all descendants of a node."""
        descendants = []
        visited = set()
        queue = list(self._hierarchies[hierarchy_type].get(node, set()))

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            descendants.append(current)
            queue.extend(self._hierarchies[hierarchy_type].get(current, set()))

        return descendants

    def is_subclass(
        self,
        child: str,
        parent: str,
        hierarchy_type: HierarchyType = HierarchyType.TAXONOMIC,
    ) -> bool:
        """Check if child is subclass of parent."""
        if child == parent:
            return True
        ancestors = self.get_ancestors(child, hierarchy_type)
        return parent in ancestors

    def get_inherited_properties(
        self,
        node: str,
        hierarchy_type: HierarchyType = HierarchyType.TAXONOMIC,
    ) -> Dict[str, Any]:
        """Get all properties including inherited ones."""
        properties = {}

        # Get ancestors in order (farthest first for proper overriding)
        ancestors = self.get_ancestors(node, hierarchy_type)
        ancestors.reverse()

        # Inherit from ancestors
        for ancestor in ancestors:
            ancestor_node = self._nodes.get(ancestor)
            if ancestor_node:
                properties.update(ancestor_node.properties)

        # Override with own properties
        node_obj = self._nodes.get(node)
        if node_obj:
            properties.update(node_obj.properties)

        return properties

    def lowest_common_ancestor(
        self,
        node1: str,
        node2: str,
        hierarchy_type: HierarchyType = HierarchyType.TAXONOMIC,
    ) -> Optional[str]:
        """Find lowest common ancestor of two nodes."""
        ancestors1 = set(self.get_ancestors(node1, hierarchy_type))
        ancestors1.add(node1)

        # BFS from node2
        visited = set()
        queue = [node2]

        while queue:
            current = queue.pop(0)
            if current in ancestors1:
                return current
            if current in visited:
                continue
            visited.add(current)
            queue.extend(self._inverse[hierarchy_type].get(current, set()))

        return None

    def get_siblings(
        self,
        node: str,
        hierarchy_type: HierarchyType = HierarchyType.TAXONOMIC,
    ) -> List[str]:
        """Get siblings (nodes with same parent)."""
        siblings = set()
        parents = self.get_parents(node, hierarchy_type)

        for parent in parents:
            children = self.get_children(parent, hierarchy_type)
            siblings.update(children)

        siblings.discard(node)
        return list(siblings)

    def classify(
        self,
        properties: Dict[str, Any],
        hierarchy_type: HierarchyType = HierarchyType.TAXONOMIC,
    ) -> Optional[str]:
        """Classify an instance based on properties."""
        # Find most specific matching class
        best_match = None
        best_depth = -1

        for name, node in self._nodes.items():
            if node.is_abstract:
                continue

            # Check if properties match
            inherited_props = self.get_inherited_properties(name, hierarchy_type)
            matches = all(
                inherited_props.get(k) == v for k, v in properties.items() if k in inherited_props
            )

            if matches and node.depth > best_depth:
                best_match = name
                best_depth = node.depth

        return best_match

    def validate_constraints(
        self,
        node: str,
        instance_properties: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """Validate instance against node constraints."""
        violations = []
        node_obj = self._nodes.get(node)

        if not node_obj:
            return False, ["Node not found"]

        # Check own constraints
        for constraint in node_obj.constraints:
            # Simple constraint format: "property:required" or "property:type:int"
            parts = constraint.split(":")
            prop = parts[0]

            if len(parts) >= 2:
                if parts[1] == "required" and prop not in instance_properties:
                    violations.append(f"Missing required property: {prop}")
                elif len(parts) >= 3 and parts[1] == "type":
                    expected_type = parts[2]
                    if prop in instance_properties:
                        actual = type(instance_properties[prop]).__name__
                        if actual != expected_type:
                            violations.append(
                                f"Property {prop} expected {expected_type}, got {actual}"
                            )

        # Check inherited constraints
        for ancestor in self.get_ancestors(node):
            ancestor_node = self._nodes.get(ancestor)
            if ancestor_node:
                for constraint in ancestor_node.constraints:
                    parts = constraint.split(":")
                    prop = parts[0]
                    if len(parts) >= 2 and parts[1] == "required":
                        if prop not in instance_properties:
                            violations.append(
                                f"Missing inherited required property: {prop} (from {ancestor})"
                            )

        return len(violations) == 0, violations

    def get_roots(
        self,
        hierarchy_type: HierarchyType = HierarchyType.TAXONOMIC,
    ) -> List[str]:
        """Get root nodes of hierarchy."""
        return list(self._roots[hierarchy_type])

    def query(self, query: OntologyQuery) -> List[OntologyNode]:
        """Query the ontology."""
        results = []

        for name, node in self._nodes.items():
            # Concept filter
            if query.concept and name != query.concept:
                continue

            # Depth filter
            if query.depth is not None and node.depth != query.depth:
                continue

            # Property filter
            if query.property_filter:
                if query.include_inherited:
                    props = self.get_inherited_properties(name)
                else:
                    props = node.properties

                matches = all(props.get(k) == v for k, v in query.property_filter.items())
                if not matches:
                    continue

            results.append(node)

        return results

    def get_parts(
        self,
        whole: str,
    ) -> List[str]:
        """Get all parts of a whole (via PART-OF or HAS-A)."""
        parts = set()
        parts.update(self.get_children(whole, HierarchyType.MERONOMIC))
        parts.update(self.get_children(whole, HierarchyType.COMPOSITIONAL))
        return list(parts)

    def statistics(self) -> Dict[str, Any]:
        """Get ontology statistics."""
        stats = {
            "n_nodes": len(self._nodes),
            "n_relations": len(self._relations),
            "hierarchies": {},
        }

        for ht in HierarchyType:
            n_rels = sum(len(children) for children in self._hierarchies[ht].values())
            n_roots = len(self._roots[ht])
            stats["hierarchies"][ht.value] = {
                "n_relations": n_rels,
                "n_roots": n_roots,
            }

        return stats
