"""
Neuro Module 08: Neural Language Processing

Implements language processing based on 2025 neuroscience research:
- Distributed language network (Broca's, Wernicke's, Arcuate fasciculus)
- Hierarchical processing (Phonological → Syntactic → Semantic → Pragmatic)
- Grammar constraint manifold (innate limits on learnable languages)
- Recursive constituent structure (human) vs linear prediction (LLM-style)
"""

from .language_hierarchy import (
    PhonologicalLayer,
    SyntacticLayer,
    SemanticLayer,
    PragmaticLayer,
    LanguageProcessingHierarchy
)
from .language_network import (
    BrocaRegion,
    WernickeRegion,
    ArcuateFasciculus,
    DistributedLanguageNetwork
)
from .grammar_manifold import (
    GrammarState,
    GrammarConstraintManifold,
    UniversalGrammar
)
from .recursive_parser import (
    Constituent,
    RecursiveGrammar,
    ConstituentParser,
    LinearPredictor
)
from .predictive_language import PredictiveLanguageProcessor

__version__ = "0.1.0"
__all__ = [
    "PhonologicalLayer",
    "SyntacticLayer",
    "SemanticLayer",
    "PragmaticLayer",
    "LanguageProcessingHierarchy",
    "BrocaRegion",
    "WernickeRegion",
    "ArcuateFasciculus",
    "DistributedLanguageNetwork",
    "GrammarState",
    "GrammarConstraintManifold",
    "UniversalGrammar",
    "Constituent",
    "RecursiveGrammar",
    "ConstituentParser",
    "LinearPredictor",
    "PredictiveLanguageProcessor",
]
