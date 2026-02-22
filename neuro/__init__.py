"""
Neuro AGI - Neuroscience-inspired cognitive architecture.

A modular AGI system with 38 cognitive modules organized into:
- Tier 1 (00-19): Cognitive modules (integration, memory, learning, etc.)
- Tier 2: Infrastructure (system core, LLM, knowledge, sensors)
- Tier 3: Support (benchmarks, perception, integration layer)
- New AGI: Advanced capabilities (causal, abstract, robust, planning, credit)

Usage:
    from neuro import CognitiveCore

    core = CognitiveCore()
    core.initialize()
    core.perceive(input_data)
    core.think()
    action = core.act()
"""

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("elo-agi")
except Exception:
    __version__ = "0.9.0"
__author__ = "Elvi Zekaj"

# Lazy imports to avoid loading everything at startup
_lazy_imports = {
    # Core (neuro.modules.system)
    "CognitiveCore": ("neuro.modules.system.cognitive_core", "CognitiveCore"),
    "SystemConfig": ("neuro.modules.system.config", "SystemConfig"),
    # Integration (neuro.modules.integrate)
    "SharedSpace": ("neuro.modules.integrate.shared_space", "SharedSpace"),
    "CrossModuleLearner": ("neuro.modules.integrate.cross_module_learning", "CrossModuleLearner"),
    # Global Workspace (neuro.modules.m00_integration)
    "GlobalWorkspace": ("neuro.modules.m00_integration.global_workspace", "GlobalWorkspace"),
    # LLM (neuro.modules.llm)
    "LLMOracle": ("neuro.modules.llm.llm_interface", "LLMOracle"),
    # Knowledge (neuro.modules.knowledge)
    "KnowledgeBase": ("neuro.modules.knowledge.knowledge_graph", "KnowledgeGraph"),
    # Causal (neuro.modules.causal)
    "DifferentiableSCM": ("neuro.modules.causal.differentiable_scm", "DifferentiableSCM"),
    "CausalDiscovery": ("neuro.modules.causal.causal_discovery", "CausalDiscovery"),
    # Abstract (neuro.modules.abstract)
    "SymbolicBinder": ("neuro.modules.abstract.symbolic_binder", "SymbolicBinder"),
    "ProgramSynthesizer": ("neuro.modules.abstract.program_synthesis", "ProgramSynthesizer"),
    # Robust (neuro.modules.robust)
    "UncertaintyQuantifier": ("neuro.modules.robust.uncertainty", "UncertaintyQuantifier"),
    "OODDetector": ("neuro.modules.robust.ood_detection", "OODDetector"),
    # Planning (neuro.modules.planning)
    "HierarchicalMCTS": ("neuro.modules.planning.planning_search", "HierarchicalMCTS"),
    "MAXQDecomposition": ("neuro.modules.planning.goal_hierarchy", "MAXQDecomposition"),
    # Credit (neuro.modules.credit)
    "EligibilityTraceManager": (
        "neuro.modules.credit.eligibility_traces",
        "EligibilityTraceManager",
    ),
    "ContributionAccountant": (
        "neuro.modules.credit.contribution_accounting",
        "ContributionAccountant",
    ),
    # Continual (neuro.modules.continual)
    "TaskInference": ("neuro.modules.continual.task_inference", "TaskInference"),
    "CatastrophicForgettingPrevention": (
        "neuro.modules.continual.forgetting_prevention",
        "CatastrophicForgettingPrevention",
    ),
    # Meta-reasoning (neuro.modules.meta_reasoning)
    "ProblemClassifier": ("neuro.modules.meta_reasoning.problem_classifier", "ProblemClassifier"),
    "StyleSelector": ("neuro.modules.meta_reasoning.style_selector", "StyleSelector"),
    "DynamicOrchestrator": ("neuro.modules.meta_reasoning.orchestrator", "DynamicOrchestrator"),
    "FallacyDetector": ("neuro.modules.meta_reasoning.fallacy_detector", "FallacyDetector"),
    # Smart Wrapper (neuro.wrapper)
    "SmartWrapper": ("neuro.wrapper", "SmartWrapper"),
    "smart_query": ("neuro.wrapper", "smart_query"),
    # Brain API (neuro.brain)
    "Brain": ("neuro.brain", "Brain"),
    "think": ("neuro.brain", "think"),
}


def __getattr__(name: str):
    """Lazy import handler."""
    if name in _lazy_imports:
        module_name, class_name = _lazy_imports[name]
        try:
            import importlib

            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Could not import {class_name} from {module_name}. "
                f"Make sure elo-agi is properly installed: {e}"
            )
    raise AttributeError(f"module 'neuro' has no attribute '{name}'")


def __dir__():
    """List available exports."""
    return list(_lazy_imports.keys()) + ["__version__", "__author__"]


# Convenience function for quick access
def create_agent(config=None):
    """
    Create a fully initialized cognitive agent.

    Args:
        config: Optional SystemConfig

    Returns:
        Initialized CognitiveCore instance
    """
    core = __getattr__("CognitiveCore")(config)
    core.initialize()
    return core


__all__ = list(_lazy_imports.keys()) + ["create_agent", "smart_query", "__version__"]
