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

__version__ = "0.9.0"
__author__ = "Elvi Zekaj"

import sys
from pathlib import Path

# Add module paths for imports
_neuro_root = Path(__file__).parent.parent
for module_dir in _neuro_root.glob("neuro-*"):
    src_dir = module_dir / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

# Lazy imports to avoid loading everything at startup
_lazy_imports = {
    # Core
    "CognitiveCore": ("cognitive_core", "CognitiveCore"),
    "SystemConfig": ("config", "SystemConfig"),

    # Integration
    "SharedSpace": ("shared_space", "SharedSpace"),
    "CrossModuleLearner": ("cross_module_learning", "CrossModuleLearner"),

    # Global Workspace
    "GlobalWorkspace": ("global_workspace", "GlobalWorkspace"),

    # LLM
    "LLMOracle": ("oracle", "LLMOracle"),

    # Knowledge
    "KnowledgeBase": ("knowledge_graph", "KnowledgeGraph"),

    # Causal
    "DifferentiableSCM": ("differentiable_scm", "DifferentiableSCM"),
    "CausalDiscovery": ("causal_discovery", "CausalDiscovery"),

    # Abstract
    "SymbolicBinder": ("symbolic_binder", "SymbolicBinder"),
    "ProgramSynthesizer": ("program_synthesis", "ProgramSynthesizer"),

    # Robust
    "UncertaintyQuantifier": ("uncertainty", "UncertaintyQuantifier"),
    "OODDetector": ("ood_detection", "OODDetector"),

    # Planning
    "HierarchicalMCTS": ("planning_search", "HierarchicalMCTS"),
    "MAXQDecomposition": ("goal_hierarchy", "MAXQDecomposition"),

    # Credit
    "EligibilityTraceManager": ("eligibility_traces", "EligibilityTraceManager"),
    "ContributionAccountant": ("contribution_accounting", "ContributionAccountant"),

    # Continual
    "TaskInference": ("task_inference", "TaskInference"),
    "CatastrophicForgettingPrevention": ("forgetting_prevention", "CatastrophicForgettingPrevention"),

    # Meta-reasoning
    "ProblemClassifier": ("problem_classifier", "ProblemClassifier"),
    "StyleSelector": ("style_selector", "StyleSelector"),
    "DynamicOrchestrator": ("orchestrator", "DynamicOrchestrator"),
    "FallacyDetector": ("fallacy_detector", "FallacyDetector"),
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
                f"Make sure neuro-agi is properly installed: {e}"
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


__all__ = list(_lazy_imports.keys()) + ["create_agent", "__version__"]
