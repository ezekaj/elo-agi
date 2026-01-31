"""
Neuro-Knowledge: Knowledge representation and reasoning.

Provides:
- Semantic networks with spreading activation
- Ontological hierarchies (IS-A, HAS-A, PART-OF)
- Triple-based fact storage
- Forward/backward chaining inference
- Knowledge graph embeddings
- Common sense reasoning
"""

from semantic_network import (
    SemanticNetwork,
    Concept,
    SemanticRelation,
    RelationType,
    ActivationPattern,
)
from ontology import (
    Ontology,
    OntologyNode,
    OntologyRelation,
    HierarchyType,
    OntologyQuery,
)
from fact_store import (
    FactStore,
    Fact,
    Triple,
    FactQuery,
    FactIndex,
)
from inference_engine import (
    InferenceEngine,
    Rule,
    InferenceResult,
    InferenceChain,
    InferenceMode,
)
from knowledge_graph import (
    KnowledgeGraph,
    Entity,
    Relation,
    GraphEmbedding,
    GraphQuery,
)
from common_sense import (
    CommonSenseReasoner,
    CommonSenseKB,
    PhysicsReasoner,
    SocialReasoner,
    TemporalReasoner,
)

__all__ = [
    # Semantic Network
    'SemanticNetwork',
    'Concept',
    'SemanticRelation',
    'RelationType',
    'ActivationPattern',
    # Ontology
    'Ontology',
    'OntologyNode',
    'OntologyRelation',
    'HierarchyType',
    'OntologyQuery',
    # Fact Store
    'FactStore',
    'Fact',
    'Triple',
    'FactQuery',
    'FactIndex',
    # Inference Engine
    'InferenceEngine',
    'Rule',
    'InferenceResult',
    'InferenceChain',
    'InferenceMode',
    # Knowledge Graph
    'KnowledgeGraph',
    'Entity',
    'Relation',
    'GraphEmbedding',
    'GraphQuery',
    # Common Sense
    'CommonSenseReasoner',
    'CommonSenseKB',
    'PhysicsReasoner',
    'SocialReasoner',
    'TemporalReasoner',
]
