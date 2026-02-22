"""
Research Assistant: Knowledge ingestion and question answering.

A practical application built on the Neuro AGI cognitive infrastructure.
"""

import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add module paths
neuro_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(neuro_root / "neuro-knowledge" / "src"))
sys.path.insert(0, str(neuro_root / "neuro-llm" / "src"))


@dataclass
class IngestedFact:
    """A fact extracted from text."""

    subject: str
    predicate: str
    obj: str
    source: str
    confidence: float = 0.8


@dataclass
class QueryResult:
    """Result of a query with source citations."""

    answer: str
    sources: List[str]
    confidence: float
    facts_used: List[IngestedFact]


class ResearchAssistant:
    """
    Research Assistant for document ingestion and question answering.

    Uses Neuro AGI components:
    - FactStore for storing extracted facts
    - KnowledgeGraph for entity relationships
    - SemanticBridge for query understanding
    """

    def __init__(self, use_llm: bool = False):
        """
        Initialize the research assistant.

        Args:
            use_llm: If True, use real LLM for enhanced extraction.
                    If False, use rule-based extraction.
        """
        self.use_llm = use_llm
        self._facts: List[IngestedFact] = []
        self._sources: Dict[str, str] = {}  # source_name -> full_text
        self._entities: Dict[str, List[str]] = {}  # entity -> list of source names

        # Try to import Neuro components
        try:
            from fact_store import FactStore

            self._fact_store = FactStore()
            self._has_fact_store = True
        except ImportError:
            self._fact_store = None
            self._has_fact_store = False

        try:
            from knowledge_graph import KnowledgeGraph

            self._knowledge_graph = KnowledgeGraph()
            self._has_kg = True
        except ImportError:
            self._knowledge_graph = None
            self._has_kg = False

    def ingest_text(self, text: str, source: str = "document") -> int:
        """
        Ingest text, extract facts, add to knowledge base.

        Args:
            text: The text to ingest
            source: Source attribution for citations

        Returns:
            Number of facts extracted
        """
        # Store source text
        self._sources[source] = text

        # Split into sentences
        sentences = self._split_sentences(text)

        facts_extracted = 0
        for sentence in sentences:
            # Extract facts from sentence
            extracted = self._extract_facts(sentence, source)

            for fact in extracted:
                self._facts.append(fact)

                # Track entities
                for entity in [fact.subject, fact.obj]:
                    if entity not in self._entities:
                        self._entities[entity] = []
                    if source not in self._entities[entity]:
                        self._entities[entity].append(source)

                # Add to FactStore if available
                if self._has_fact_store:
                    self._fact_store.add(
                        subject=fact.subject,
                        predicate=fact.predicate,
                        obj=fact.obj,
                        confidence=fact.confidence,
                        source=fact.source,
                    )

                # Add to KnowledgeGraph if available
                if self._has_kg:
                    self._knowledge_graph.add_edge(
                        head=fact.subject, relation=fact.predicate, tail=fact.obj
                    )

                facts_extracted += 1

        return facts_extracted

    def query(self, question: str) -> QueryResult:
        """
        Answer question using ingested knowledge.

        Args:
            question: Natural language question

        Returns:
            QueryResult with answer, sources, and confidence
        """
        # Extract key terms from question
        key_terms = self._extract_key_terms(question)

        # Find relevant facts
        relevant_facts = self._find_relevant_facts(key_terms, question)

        if not relevant_facts:
            return QueryResult(
                answer="I don't have enough information to answer this question.",
                sources=[],
                confidence=0.0,
                facts_used=[],
            )

        # Generate answer from facts
        answer, confidence = self._generate_answer(question, relevant_facts)

        # Collect unique sources
        sources = list(set(f.source for f in relevant_facts))

        return QueryResult(
            answer=answer, sources=sources, confidence=confidence, facts_used=relevant_facts
        )

    def get_sources(self) -> List[str]:
        """Return all ingested sources."""
        return list(self._sources.keys())

    def get_entities(self) -> List[str]:
        """Return all known entities."""
        return list(self._entities.keys())

    def get_facts_about(self, entity: str) -> List[IngestedFact]:
        """Get all facts involving an entity."""
        return [
            f
            for f in self._facts
            if entity.lower() in f.subject.lower() or entity.lower() in f.obj.lower()
        ]

    def statistics(self) -> Dict[str, Any]:
        """Get assistant statistics."""
        return {
            "total_facts": len(self._facts),
            "total_sources": len(self._sources),
            "total_entities": len(self._entities),
            "has_fact_store": self._has_fact_store,
            "has_knowledge_graph": self._has_kg,
        }

    # --- Private methods ---

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting on . ! ?
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _extract_facts(self, sentence: str, source: str) -> List[IngestedFact]:
        """Extract facts from a sentence using pattern matching."""
        facts = []
        sentence = sentence.strip()

        if not sentence:
            return facts

        # Pattern 1: "X verb Y" patterns
        # E.g., "Einstein developed relativity"
        verb_patterns = [
            (
                r"(\w+(?:\s+\w+)?)\s+(published|developed|invented|created|discovered|introduced|wrote|proposed)\s+(.+?)(?:\s+in\s+\d+)?\.?$",
                "developed",
            ),
            (r"(\w+(?:\s+\w+)?)\s+(is|was|are|were)\s+(?:a|an|the)?\s*(.+?)\.?$", "is"),
            (r"(\w+(?:\s+\w+)?)\s+(has|have|had)\s+(.+?)\.?$", "has"),
        ]

        for pattern, default_pred in verb_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                subject = match.group(1).strip()
                predicate = match.group(2).strip().lower()
                obj = match.group(3).strip().rstrip(".")

                if subject and obj and len(subject) > 1 and len(obj) > 1:
                    facts.append(
                        IngestedFact(
                            subject=subject,
                            predicate=predicate,
                            obj=obj,
                            source=source,
                            confidence=0.8,
                        )
                    )
                    break

        # Pattern 2: Date patterns
        # E.g., "... in 1905"
        date_match = re.search(r"(.+?)\s+in\s+(\d{4})", sentence, re.IGNORECASE)
        if date_match:
            event = date_match.group(1).strip()
            year = date_match.group(2)

            # Extract subject from event
            subject_match = re.match(r"(\w+(?:\s+\w+)?)", event)
            if subject_match and len(event) > 5:
                facts.append(
                    IngestedFact(
                        subject=event,
                        predicate="occurred_in",
                        obj=year,
                        source=source,
                        confidence=0.9,
                    )
                )

        # Pattern 3: Equation/formula patterns
        # E.g., "E=mc^2" or "the equation E=mc²"
        equation_match = re.search(r"(?:equation|formula)?\s*([A-Z]=\S+)", sentence)
        if equation_match:
            equation = equation_match.group(1)
            # Find subject (often mentioned before)
            subj_match = re.search(
                r"(\w+(?:\s+\w+)?)\s+(?:introduced|famous|equation)", sentence, re.IGNORECASE
            )
            if subj_match:
                facts.append(
                    IngestedFact(
                        subject=subj_match.group(1),
                        predicate="introduced",
                        obj=equation,
                        source=source,
                        confidence=0.85,
                    )
                )

        return facts

    def _extract_key_terms(self, question: str) -> List[str]:
        """Extract key terms from a question."""
        # Remove question words
        question = question.lower()
        stop_words = {
            "what",
            "when",
            "where",
            "who",
            "which",
            "how",
            "is",
            "was",
            "are",
            "were",
            "did",
            "does",
            "do",
            "the",
            "a",
            "an",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "by",
            "?",
            ".",
        }

        words = re.findall(r"\w+", question)
        key_terms = [w for w in words if w not in stop_words and len(w) > 2]

        return key_terms

    def _find_relevant_facts(self, key_terms: List[str], question: str) -> List[IngestedFact]:
        """Find facts relevant to the query."""
        relevant = []
        question.lower()

        for fact in self._facts:
            # Check if any key term appears in fact
            fact_text = f"{fact.subject} {fact.predicate} {fact.obj}".lower()

            matches = sum(1 for term in key_terms if term in fact_text)

            if matches > 0:
                # Score by number of matches
                relevance = matches / len(key_terms) if key_terms else 0
                relevant.append((relevance, fact))

        # Sort by relevance and return top facts
        relevant.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in relevant[:5]]

    def _generate_answer(self, question: str, facts: List[IngestedFact]) -> Tuple[str, float]:
        """Generate an answer from relevant facts."""
        if not facts:
            return "No relevant information found.", 0.0

        question_lower = question.lower()

        # Check question type
        if "when" in question_lower:
            # Look for date facts
            for fact in facts:
                if fact.predicate in ["occurred_in", "in"] or fact.obj.isdigit():
                    return f"{fact.obj} (source: {fact.source})", fact.confidence

        if "who" in question_lower:
            # Return subject
            fact = facts[0]
            return f"{fact.subject} (source: {fact.source})", fact.confidence

        if "what" in question_lower:
            # Return object or full fact
            fact = facts[0]
            if "equation" in question_lower or "formula" in question_lower:
                for f in facts:
                    if "=" in f.obj:
                        return f"{f.obj} (source: {f.source})", f.confidence
            return f"{fact.obj} (source: {fact.source})", fact.confidence

        # Default: return first fact as statement
        fact = facts[0]
        answer = f"{fact.subject} {fact.predicate} {fact.obj}"
        sources = ", ".join(set(f.source for f in facts))

        return f"{answer} (source: {sources})", fact.confidence
