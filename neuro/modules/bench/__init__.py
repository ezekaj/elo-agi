"""
Neuro Bench: Benchmark Suite for Cognitive System Evaluation.

Provides standardized benchmarks to measure AGI capabilities across:
- Reasoning (abstract, logical, analogical)
- Memory (working, episodic, semantic)
- Language (understanding, generation, grounding)
- Planning (goal-directed, multi-step)
- Learning (adaptation, transfer)
- Causal (counterfactual, intervention, discovery)
- Abstraction (symbol binding, composition, synthesis)
- Robustness (OOD detection, calibration, adversarial)
- Consolidation (retention, interference, efficiency)
"""

from .base_benchmark import Benchmark, BenchmarkConfig, BenchmarkResult, BenchmarkSuite
from .reasoning_bench import ReasoningBenchmark, PatternCompletion, AnalogySolving
from .memory_bench import MemoryBenchmark, WorkingMemoryTest, EpisodicRecall
from .language_bench import LanguageBenchmark, TextCompletion, InstructionFollowing
from .planning_bench import PlanningBenchmark, GoalAchievement, MultiStepPlanning
from .runner import BenchmarkRunner, RunConfig, RunResult

# New AGI capability benchmarks
from .causal_bench import (
    CausalGraph,
    CausalQuery,
    CounterfactualBenchmark,
    InterventionBenchmark,
    CausalDiscoveryBenchmark,
    NestedCounterfactualBenchmark,
    create_causal_benchmark_suite,
)
from .abstraction_bench import (
    SymbolBindingTask,
    CompositionTask,
    AnalogyTask,
    SymbolBindingBenchmark,
    CompositionalBenchmark,
    ProgramSynthesisBenchmark,
    AnalogyBenchmark,
    AbstractionLevelBenchmark,
    create_abstraction_benchmark_suite,
)
from .robustness_bench import (
    OODSample,
    CalibrationSample,
    AdversarialSample,
    OODDetectionBenchmark,
    CalibrationBenchmark,
    AdversarialBenchmark,
    UncertaintyBenchmark,
    SelectivePredictionBenchmark,
    create_robustness_benchmark_suite,
)
from .consolidation_bench import (
    MemoryItem,
    InterferenceScenario,
    LearningScenario,
    RetentionBenchmark,
    InterferenceBenchmark,
    LearningEfficiencyBenchmark,
    SchemaFormationBenchmark,
    SpacedRepetitionBenchmark,
    create_consolidation_benchmark_suite,
)

__all__ = [
    # Base
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkSuite",
    # Reasoning
    "ReasoningBenchmark",
    "PatternCompletion",
    "AnalogySolving",
    # Memory
    "MemoryBenchmark",
    "WorkingMemoryTest",
    "EpisodicRecall",
    # Language
    "LanguageBenchmark",
    "TextCompletion",
    "InstructionFollowing",
    # Planning
    "PlanningBenchmark",
    "GoalAchievement",
    "MultiStepPlanning",
    # Runner
    "BenchmarkRunner",
    "RunConfig",
    "RunResult",
    # Causal
    "CausalGraph",
    "CausalQuery",
    "CounterfactualBenchmark",
    "InterventionBenchmark",
    "CausalDiscoveryBenchmark",
    "NestedCounterfactualBenchmark",
    "create_causal_benchmark_suite",
    # Abstraction
    "SymbolBindingTask",
    "CompositionTask",
    "AnalogyTask",
    "SymbolBindingBenchmark",
    "CompositionalBenchmark",
    "ProgramSynthesisBenchmark",
    "AnalogyBenchmark",
    "AbstractionLevelBenchmark",
    "create_abstraction_benchmark_suite",
    # Robustness
    "OODSample",
    "CalibrationSample",
    "AdversarialSample",
    "OODDetectionBenchmark",
    "CalibrationBenchmark",
    "AdversarialBenchmark",
    "UncertaintyBenchmark",
    "SelectivePredictionBenchmark",
    "create_robustness_benchmark_suite",
    # Consolidation
    "MemoryItem",
    "InterferenceScenario",
    "LearningScenario",
    "RetentionBenchmark",
    "InterferenceBenchmark",
    "LearningEfficiencyBenchmark",
    "SchemaFormationBenchmark",
    "SpacedRepetitionBenchmark",
    "create_consolidation_benchmark_suite",
]
