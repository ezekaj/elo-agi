#!/usr/bin/env python3
"""
Research Assistant Demo

Demonstrates the Research Assistant application built on Neuro AGI.
"""

import sys
from pathlib import Path

# Add neuro package to path
neuro_root = Path(__file__).parent.parent
sys.path.insert(0, str(neuro_root))

from neuro.apps import ResearchAssistant  # noqa: E402


def main():
    print("=" * 60)
    print("Research Assistant Demo")
    print("=" * 60)
    print()

    # Create assistant
    assistant = ResearchAssistant()

    # Ingest some documents
    physics_notes = """
Einstein published the theory of special relativity in 1905.
The theory introduced the famous equation E=mc².
Special relativity unified space and time into spacetime.
Einstein was born in Germany in 1879.
"""

    history_notes = """
World War I started in 1914.
The war ended in 1918 with the Treaty of Versailles.
The United States entered the war in 1917.
"""

    biology_notes = """
Darwin developed the theory of evolution by natural selection.
The Origin of Species was published in 1859.
DNA was discovered by Watson and Crick in 1953.
"""

    print("Ingesting documents...")
    n1 = assistant.ingest_text(physics_notes, source="physics_notes")
    n2 = assistant.ingest_text(history_notes, source="history_notes")
    n3 = assistant.ingest_text(biology_notes, source="biology_notes")

    print(f"  - physics_notes: {n1} facts extracted")
    print(f"  - history_notes: {n2} facts extracted")
    print(f"  - biology_notes: {n3} facts extracted")
    print()

    # Show statistics
    stats = assistant.statistics()
    print(f"Total facts: {stats['total_facts']}")
    print(f"Total sources: {stats['total_sources']}")
    print(f"Total entities: {stats['total_entities']}")
    print()

    # Query examples
    questions = [
        "When was special relativity published?",
        "Who developed the theory of evolution?",
        "When did World War I start?",
        "What did Einstein introduce?",
        "When was DNA discovered?",
    ]

    print("=" * 60)
    print("Answering Questions")
    print("=" * 60)
    print()

    for question in questions:
        print(f"Q: {question}")
        result = assistant.query(question)
        print(f"A: {result.answer}")
        print(f"   Confidence: {result.confidence:.1%}")
        print()

    # Show entity lookup
    print("=" * 60)
    print("Entity Lookup: Einstein")
    print("=" * 60)
    print()

    einstein_facts = assistant.get_facts_about("Einstein")
    for fact in einstein_facts:
        print(f"  {fact.subject} {fact.predicate} {fact.obj}")

    print()
    print("Demo complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
