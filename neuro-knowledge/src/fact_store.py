"""
Fact Store: Triple-based knowledge storage.

Implements RDF-like triple storage with efficient
indexing for fast queries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Iterator
from enum import Enum
import time
import numpy as np


@dataclass
class Triple:
    """A subject-predicate-object triple."""
    subject: str
    predicate: str
    obj: str  # 'object' is a Python builtin

    def __hash__(self):
        return hash((self.subject, self.predicate, self.obj))

    def __eq__(self, other):
        if isinstance(other, Triple):
            return (
                self.subject == other.subject and
                self.predicate == other.predicate and
                self.obj == other.obj
            )
        return False

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.obj)


@dataclass
class Fact:
    """A fact with metadata."""
    triple: Triple
    confidence: float = 1.0
    source: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    valid_from: Optional[float] = None
    valid_to: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def subject(self) -> str:
        return self.triple.subject

    @property
    def predicate(self) -> str:
        return self.triple.predicate

    @property
    def obj(self) -> str:
        return self.triple.obj

    def is_valid(self, at_time: Optional[float] = None) -> bool:
        """Check if fact is valid at given time."""
        if at_time is None:
            at_time = time.time()

        if self.valid_from and at_time < self.valid_from:
            return False
        if self.valid_to and at_time > self.valid_to:
            return False
        return True


@dataclass
class FactQuery:
    """A query against the fact store."""
    subject: Optional[str] = None
    predicate: Optional[str] = None
    obj: Optional[str] = None
    min_confidence: float = 0.0
    source: Optional[str] = None
    valid_at: Optional[float] = None


class FactIndex:
    """Index for fast triple lookups."""

    def __init__(self):
        # Primary indexes
        self._by_subject: Dict[str, Set[int]] = {}
        self._by_predicate: Dict[str, Set[int]] = {}
        self._by_object: Dict[str, Set[int]] = {}

        # Composite indexes
        self._by_sp: Dict[Tuple[str, str], Set[int]] = {}  # subject-predicate
        self._by_po: Dict[Tuple[str, str], Set[int]] = {}  # predicate-object
        self._by_so: Dict[Tuple[str, str], Set[int]] = {}  # subject-object

    def add(self, fact_id: int, triple: Triple) -> None:
        """Add a triple to the index."""
        # Primary indexes
        if triple.subject not in self._by_subject:
            self._by_subject[triple.subject] = set()
        self._by_subject[triple.subject].add(fact_id)

        if triple.predicate not in self._by_predicate:
            self._by_predicate[triple.predicate] = set()
        self._by_predicate[triple.predicate].add(fact_id)

        if triple.obj not in self._by_object:
            self._by_object[triple.obj] = set()
        self._by_object[triple.obj].add(fact_id)

        # Composite indexes
        sp = (triple.subject, triple.predicate)
        if sp not in self._by_sp:
            self._by_sp[sp] = set()
        self._by_sp[sp].add(fact_id)

        po = (triple.predicate, triple.obj)
        if po not in self._by_po:
            self._by_po[po] = set()
        self._by_po[po].add(fact_id)

        so = (triple.subject, triple.obj)
        if so not in self._by_so:
            self._by_so[so] = set()
        self._by_so[so].add(fact_id)

    def remove(self, fact_id: int, triple: Triple) -> None:
        """Remove a triple from the index."""
        if triple.subject in self._by_subject:
            self._by_subject[triple.subject].discard(fact_id)
        if triple.predicate in self._by_predicate:
            self._by_predicate[triple.predicate].discard(fact_id)
        if triple.obj in self._by_object:
            self._by_object[triple.obj].discard(fact_id)

        sp = (triple.subject, triple.predicate)
        if sp in self._by_sp:
            self._by_sp[sp].discard(fact_id)

        po = (triple.predicate, triple.obj)
        if po in self._by_po:
            self._by_po[po].discard(fact_id)

        so = (triple.subject, triple.obj)
        if so in self._by_so:
            self._by_so[so].discard(fact_id)

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> Set[int]:
        """Query the index for matching fact IDs."""
        candidates = None

        # Use best available index
        if subject and predicate:
            candidates = self._by_sp.get((subject, predicate), set())
        elif predicate and obj:
            candidates = self._by_po.get((predicate, obj), set())
        elif subject and obj:
            candidates = self._by_so.get((subject, obj), set())
        elif subject:
            candidates = self._by_subject.get(subject, set())
        elif predicate:
            candidates = self._by_predicate.get(predicate, set())
        elif obj:
            candidates = self._by_object.get(obj, set())
        else:
            # Return all
            all_ids = set()
            for ids in self._by_subject.values():
                all_ids.update(ids)
            candidates = all_ids

        # Filter by remaining criteria
        result = set(candidates)

        if subject and predicate and obj:
            # Already exact, but verify
            pass
        elif subject and predicate:
            if obj:
                result = {fid for fid in result if fid in self._by_object.get(obj, set())}
        elif predicate and obj:
            if subject:
                result = {fid for fid in result if fid in self._by_subject.get(subject, set())}
        elif subject and obj:
            if predicate:
                result = {fid for fid in result if fid in self._by_predicate.get(predicate, set())}

        return result


class FactStore:
    """
    Triple-based fact storage with efficient querying.

    Supports:
    - Adding, removing, and querying facts
    - Confidence scoring
    - Temporal validity
    - Multi-index queries
    """

    def __init__(self):
        # Fact storage
        self._facts: Dict[int, Fact] = {}
        self._next_id = 0

        # Triple to ID mapping (for deduplication)
        self._triple_to_id: Dict[Triple, int] = {}

        # Index
        self._index = FactIndex()

        # Statistics
        self._query_count = 0

    def add(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 1.0,
        source: str = "unknown",
        valid_from: Optional[float] = None,
        valid_to: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Fact:
        """Add a fact to the store."""
        triple = Triple(subject, predicate, obj)

        # Check for existing
        if triple in self._triple_to_id:
            existing_id = self._triple_to_id[triple]
            existing = self._facts[existing_id]
            # Update if higher confidence
            if confidence > existing.confidence:
                existing.confidence = confidence
                existing.source = source
                existing.timestamp = time.time()
            return existing

        # Create new fact
        fact_id = self._next_id
        self._next_id += 1

        fact = Fact(
            triple=triple,
            confidence=confidence,
            source=source,
            valid_from=valid_from,
            valid_to=valid_to,
            metadata=metadata or {},
        )

        self._facts[fact_id] = fact
        self._triple_to_id[triple] = fact_id
        self._index.add(fact_id, triple)

        return fact

    def add_triple(self, triple: Triple, **kwargs) -> Fact:
        """Add a fact from a Triple object."""
        return self.add(triple.subject, triple.predicate, triple.obj, **kwargs)

    def remove(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> int:
        """Remove matching facts. Returns count of removed facts."""
        matching_ids = self._index.query(subject, predicate, obj)
        removed = 0

        for fact_id in list(matching_ids):
            if fact_id in self._facts:
                fact = self._facts[fact_id]
                # Verify match (index might be stale)
                if subject and fact.subject != subject:
                    continue
                if predicate and fact.predicate != predicate:
                    continue
                if obj and fact.obj != obj:
                    continue

                self._index.remove(fact_id, fact.triple)
                del self._triple_to_id[fact.triple]
                del self._facts[fact_id]
                removed += 1

        return removed

    def get(self, fact_id: int) -> Optional[Fact]:
        """Get a fact by ID."""
        return self._facts.get(fact_id)

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        min_confidence: float = 0.0,
        source: Optional[str] = None,
        valid_at: Optional[float] = None,
    ) -> List[Fact]:
        """Query facts matching the pattern."""
        self._query_count += 1

        matching_ids = self._index.query(subject, predicate, obj)
        results = []

        for fact_id in matching_ids:
            fact = self._facts.get(fact_id)
            if not fact:
                continue

            # Apply filters
            if min_confidence > 0 and fact.confidence < min_confidence:
                continue
            if source and fact.source != source:
                continue
            if valid_at and not fact.is_valid(valid_at):
                continue

            results.append(fact)

        return results

    def query_object(
        self,
        subject: str,
        predicate: str,
    ) -> List[str]:
        """Get objects for subject-predicate pair."""
        facts = self.query(subject=subject, predicate=predicate)
        return [f.obj for f in facts]

    def query_subject(
        self,
        predicate: str,
        obj: str,
    ) -> List[str]:
        """Get subjects for predicate-object pair."""
        facts = self.query(predicate=predicate, obj=obj)
        return [f.subject for f in facts]

    def exists(
        self,
        subject: str,
        predicate: str,
        obj: str,
    ) -> bool:
        """Check if a specific triple exists."""
        triple = Triple(subject, predicate, obj)
        return triple in self._triple_to_id

    def count(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> int:
        """Count matching facts."""
        matching_ids = self._index.query(subject, predicate, obj)
        return len(matching_ids)

    def subjects(self) -> Set[str]:
        """Get all unique subjects."""
        return set(self._index._by_subject.keys())

    def predicates(self) -> Set[str]:
        """Get all unique predicates."""
        return set(self._index._by_predicate.keys())

    def objects(self) -> Set[str]:
        """Get all unique objects."""
        return set(self._index._by_object.keys())

    def entities(self) -> Set[str]:
        """Get all unique entities (subjects and objects)."""
        return self.subjects() | self.objects()

    def iterate(self) -> Iterator[Fact]:
        """Iterate over all facts."""
        return iter(self._facts.values())

    def bulk_add(self, triples: List[Tuple[str, str, str]], **kwargs) -> int:
        """Add multiple facts efficiently."""
        added = 0
        for s, p, o in triples:
            self.add(s, p, o, **kwargs)
            added += 1
        return added

    def merge(self, other: 'FactStore') -> int:
        """Merge another fact store into this one."""
        merged = 0
        for fact in other.iterate():
            self.add_triple(
                fact.triple,
                confidence=fact.confidence,
                source=fact.source,
                valid_from=fact.valid_from,
                valid_to=fact.valid_to,
                metadata=fact.metadata.copy(),
            )
            merged += 1
        return merged

    def export_triples(self) -> List[Tuple[str, str, str]]:
        """Export all triples as tuples."""
        return [fact.triple.as_tuple() for fact in self._facts.values()]

    def clear(self) -> None:
        """Clear all facts."""
        self._facts.clear()
        self._triple_to_id.clear()
        self._index = FactIndex()
        self._next_id = 0

    def statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        predicates = {}
        sources = {}

        for fact in self._facts.values():
            predicates[fact.predicate] = predicates.get(fact.predicate, 0) + 1
            sources[fact.source] = sources.get(fact.source, 0) + 1

        return {
            "n_facts": len(self._facts),
            "n_subjects": len(self.subjects()),
            "n_predicates": len(self.predicates()),
            "n_objects": len(self.objects()),
            "n_entities": len(self.entities()),
            "predicate_counts": predicates,
            "source_counts": sources,
            "query_count": self._query_count,
        }
