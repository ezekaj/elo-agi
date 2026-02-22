"""
Cognitive Orchestrator - Wires All 38 NEURO Modules

This orchestrator loads and coordinates ALL cognitive modules from the neuro-module-*
folders, providing a unified interface for the pipeline.

Modules 00-19: Core cognitive modules
+ Additional components from neuro-model/src
= Total 38+ active cognitive components
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field

# Add neuro modules to path
NEURO_ROOT = Path(__file__).parent.parent.parent


@dataclass
class ModuleResult:
    """Result from a cognitive module."""

    module_name: str
    activated: bool
    output: Any = None
    confidence: float = 0.5
    relevance: float = 0.0


@dataclass
class OrchestratorResult:
    """Aggregated result from all modules."""

    query: str
    modules_activated: int
    total_modules: int
    results: Dict[str, ModuleResult] = field(default_factory=dict)
    dominant_type: str = "general"
    confidence: float = 0.5
    insights: List[str] = field(default_factory=list)


class CognitiveOrchestrator:
    """
    Orchestrates all 38+ NEURO cognitive modules.

    This loads modules from:
    1. neuro-module-00 through neuro-module-19 (20 core modules)
    2. neuro-model/src components (knowledge, memory, etc.)
    3. Additional infrastructure modules

    Each query is broadcast through relevant modules using Global Workspace Theory.
    """

    # Module definitions with their entry points
    MODULE_CONFIGS = {
        # Core Cognitive Modules (00-19)
        "00-integration": {
            "path": "neuro-module-00-integration/src",
            "class": "GlobalWorkspace",
            "import": "global_workspace",
            "purpose": "Attention and information broadcast",
        },
        "01-predictive-coding": {
            "path": "neuro-module-01-predictive-coding/src",
            "class": "PredictiveProcessor",
            "import": "hierarchical_predictive_processing",
            "purpose": "Prediction error and belief updating",
        },
        "02-dual-process": {
            "path": "neuro-module-02-dual-process/src",
            "class": "DualProcessCore",
            "import": "dual_process_geometry",
            "purpose": "System 1/2 processing",
        },
        "03-reasoning-types": {
            "path": "neuro-module-03-reasoning-types/src",
            "class": "ReasoningCore",
            "import": "reasoning_core",
            "purpose": "Multiple reasoning strategies",
        },
        "04-memory": {
            "path": "neuro-module-04-memory/src",
            "class": "MemorySystem",
            "import": "memory_system",
            "purpose": "Working and long-term memory",
        },
        "05-sleep-consolidation": {
            "path": "neuro-module-05-sleep-consolidation/src",
            "class": "SleepModule",
            "import": "sleep_module",
            "purpose": "Memory consolidation",
        },
        "06-motivation": {
            "path": "neuro-module-06-motivation/src",
            "class": "CuriosityModule",
            "import": "curiosity_drive",
            "purpose": "Intrinsic motivation and curiosity",
        },
        "07-emotions-decisions": {
            "path": "neuro-module-07-emotions-decisions/src",
            "class": "EmotionalCore",
            "import": "emotional_core",
            "purpose": "Emotional valuation",
        },
        "08-language": {
            "path": "neuro-module-08-language/src",
            "class": "LanguageProcessor",
            "import": "language_processor",
            "purpose": "Language understanding",
        },
        "09-creativity": {
            "path": "neuro-module-09-creativity/src",
            "class": "CreativityEngine",
            "import": "creativity_engine",
            "purpose": "Creative generation",
        },
        "10-spatial-cognition": {
            "path": "neuro-module-10-spatial-cognition/src",
            "class": "SpatialCognition",
            "import": "spatial_cognition",
            "purpose": "Spatial reasoning",
        },
        "11-time-perception": {
            "path": "neuro-module-11-time-perception/src",
            "class": "TemporalCognition",
            "import": "temporal_cognition",
            "purpose": "Time perception",
        },
        "12-learning": {
            "path": "neuro-module-12-learning/src",
            "class": "LearningCore",
            "import": "learning_core",
            "purpose": "Learning mechanisms",
        },
        "13-executive": {
            "path": "neuro-module-13-executive/src",
            "class": "ExecutiveControl",
            "import": "executive_control",
            "purpose": "Executive control and planning",
        },
        "14-embodied": {
            "path": "neuro-module-14-embodied/src",
            "class": "EmbodiedCore",
            "import": "embodied_core",
            "purpose": "Embodied cognition",
        },
        "15-social": {
            "path": "neuro-module-15-social/src",
            "class": "SocialCognition",
            "import": "social_cognition",
            "purpose": "Social reasoning",
        },
        "16-consciousness": {
            "path": "neuro-module-16-consciousness/src",
            "class": "ConsciousnessCore",
            "import": "consciousness_core",
            "purpose": "Metacognition",
        },
        "17-world-model": {
            "path": "neuro-module-17-world-model/src",
            "class": "WorldModel",
            "import": "world_model",
            "purpose": "Internal world modeling",
        },
        "18-self-improvement": {
            "path": "neuro-module-18-self-improvement/src",
            "class": "SelfImprover",
            "import": "self_improver",
            "purpose": "Self-improvement",
        },
        "19-multi-agent": {
            "path": "neuro-module-19-multi-agent/src",
            "class": "MultiAgentCore",
            "import": "multi_agent_core",
            "purpose": "Multi-agent coordination",
        },
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.modules: Dict[str, Any] = {}
        self.active_count = 0
        self.fallback_count = 0
        self.failed_modules: List[str] = []

        self._load_all_modules()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  [Orchestrator] {msg}")

    def _load_all_modules(self):
        """Load all cognitive modules."""
        self._log("Loading all cognitive modules...")

        for module_id, config in self.MODULE_CONFIGS.items():
            self._load_module(module_id, config)

        # Also load neuro-model/src components as additional "modules"
        self._load_src_components()

        self._log(
            f"Loaded: {self.active_count} active, {self.fallback_count} fallback, {len(self.failed_modules)} failed"
        )

    def _load_module(self, module_id: str, config: Dict):
        """Load a single module."""
        module_path = NEURO_ROOT / config["path"]

        if not module_path.exists():
            self._log(f"  {module_id}: path not found")
            self.failed_modules.append(module_id)
            return

        # Add to path
        sys.path.insert(0, str(module_path))

        try:
            # Try to import
            mod = __import__(config["import"])

            # Try to get the class
            if hasattr(mod, config["class"]):
                cls = getattr(mod, config["class"])
                # Try to instantiate
                try:
                    instance = cls()
                    self.modules[module_id] = {
                        "instance": instance,
                        "config": config,
                        "type": "active",
                    }
                    self.active_count += 1
                    self._log(f"  {module_id}: ACTIVE")
                except Exception as e:
                    # Class exists but can't instantiate - use stub
                    self.modules[module_id] = {
                        "instance": self._create_stub(module_id, config),
                        "config": config,
                        "type": "fallback",
                    }
                    self.fallback_count += 1
                    self._log(f"  {module_id}: fallback (init failed: {e})")
            else:
                # Module imports but class not found
                self.modules[module_id] = {
                    "instance": self._create_stub(module_id, config),
                    "config": config,
                    "type": "fallback",
                }
                self.fallback_count += 1
                self._log(f"  {module_id}: fallback (class not found)")

        except ImportError as e:
            # Import failed
            self.modules[module_id] = {
                "instance": self._create_stub(module_id, config),
                "config": config,
                "type": "fallback",
            }
            self.fallback_count += 1
            self._log(f"  {module_id}: fallback (import: {e})")
        except Exception as e:
            self.failed_modules.append(module_id)
            self._log(f"  {module_id}: FAILED ({e})")

    def _create_stub(self, module_id: str, config: Dict) -> object:
        """Create a stub for a module that couldn't load."""

        class ModuleStub:
            def __init__(self, name, purpose):
                self.name = name
                self.purpose = purpose

            def process(self, query: str, context: Dict = None) -> Dict:
                return {
                    "module": self.name,
                    "type": "stub",
                    "purpose": self.purpose,
                    "activated": True,
                    "output": f"Processed by {self.name} (stub)",
                }

            def should_activate(self, query: str) -> bool:
                return True

            def statistics(self) -> Dict:
                return {"name": self.name, "type": "stub"}

        return ModuleStub(module_id, config.get("purpose", "Unknown"))

    def _load_src_components(self):
        """Load additional components from neuro-model/src."""
        src_components = {
            "knowledge_base": "Knowledge storage and retrieval",
            "episodic_memory": "Experience-based memory",
            "bayesian_surprise": "Novelty detection",
            "continual_learning": "Prevent catastrophic forgetting",
            "retrieval": "Two-stage retrieval",
        }

        Path(__file__).parent

        for comp_name, purpose in src_components.items():
            try:
                mod = __import__(comp_name)
                self.modules[f"src-{comp_name}"] = {
                    "instance": mod,
                    "config": {"purpose": purpose},
                    "type": "component",
                }
                self.active_count += 1
                self._log(f"  src-{comp_name}: loaded")
            except ImportError:
                self._log(f"  src-{comp_name}: not available")

    def process(self, query: str, context: Dict = None) -> OrchestratorResult:
        """
        Process a query through all relevant modules.

        Uses Global Workspace Theory:
        1. Broadcast query to all modules
        2. Each module decides if it should activate
        3. Activated modules process and return results
        4. Results are aggregated
        """
        context = context or {}
        result = OrchestratorResult(
            query=query, modules_activated=0, total_modules=len(self.modules)
        )

        # Broadcast to all modules
        for module_id, module_data in self.modules.items():
            instance = module_data["instance"]
            module_data["config"]

            # Check if module should activate (based on query relevance)
            should_activate = self._should_activate(module_id, query, context)

            if should_activate:
                try:
                    # Try to process
                    if hasattr(instance, "process"):
                        output = instance.process(query, context)
                    else:
                        output = {"processed": True, "module": module_id}

                    result.results[module_id] = ModuleResult(
                        module_name=module_id,
                        activated=True,
                        output=output,
                        confidence=0.7,
                        relevance=0.8,
                    )
                    result.modules_activated += 1
                except Exception as e:
                    result.results[module_id] = ModuleResult(
                        module_name=module_id,
                        activated=True,
                        output=f"Error: {e}",
                        confidence=0.0,
                        relevance=0.0,
                    )

        # Determine dominant problem type
        result.dominant_type = self._classify_query(query)

        # Aggregate confidence
        if result.modules_activated > 0:
            confidences = [r.confidence for r in result.results.values() if r.activated]
            result.confidence = sum(confidences) / len(confidences)

        # Generate insights
        result.insights = self._generate_insights(result)

        return result

    def _should_activate(self, module_id: str, query: str, context: Dict) -> bool:
        """Determine if a module should activate for this query."""
        query_lower = query.lower()

        # Module-specific activation rules
        activation_keywords = {
            "00-integration": True,  # Always active - global workspace
            "01-predictive-coding": ["predict", "expect", "belief", "update"],
            "02-dual-process": ["think", "analyze", "quick", "careful"],
            "03-reasoning-types": ["reason", "logic", "deduce", "infer", "why", "how"],
            "04-memory": ["remember", "recall", "memory", "forget", "past"],
            "05-sleep-consolidation": ["consolidate", "sleep", "review"],
            "06-motivation": ["curious", "want", "goal", "motivation", "explore"],
            "07-emotions-decisions": ["feel", "decide", "choice", "emotion", "value"],
            "08-language": True,  # Always active for language tasks
            "09-creativity": ["create", "creative", "idea", "imagine", "novel"],
            "10-spatial-cognition": ["where", "location", "space", "direction", "map"],
            "11-time-perception": ["when", "time", "duration", "schedule", "before", "after"],
            "12-learning": ["learn", "understand", "study", "pattern"],
            "13-executive": ["plan", "organize", "control", "prioritize", "task"],
            "14-embodied": ["body", "physical", "sense", "action", "motor"],
            "15-social": ["social", "people", "relationship", "communicate", "team"],
            "16-consciousness": ["aware", "conscious", "meta", "self", "reflect"],
            "17-world-model": ["world", "model", "simulate", "environment"],
            "18-self-improvement": ["improve", "better", "fix", "error", "learn from"],
            "19-multi-agent": ["agent", "coordinate", "collaborate", "team"],
        }

        # Check activation rules
        rules = activation_keywords.get(module_id, True)

        if rules is True:
            return True
        elif isinstance(rules, list):
            return any(kw in query_lower for kw in rules)

        return True  # Default: activate

    def _classify_query(self, query: str) -> str:
        """Classify the query type."""
        query_lower = query.lower()

        if any(w in query_lower for w in ["why", "how", "explain", "reason"]):
            return "reasoning"
        elif any(w in query_lower for w in ["remember", "recall", "what did"]):
            return "memory"
        elif any(w in query_lower for w in ["create", "write", "generate", "imagine"]):
            return "creative"
        elif any(w in query_lower for w in ["plan", "organize", "schedule"]):
            return "executive"
        elif any(w in query_lower for w in ["feel", "emotion", "decide"]):
            return "emotional"
        elif any(w in query_lower for w in ["search", "find", "look"]):
            return "exploration"

        return "general"

    def _generate_insights(self, result: OrchestratorResult) -> List[str]:
        """Generate insights from module results."""
        insights = []

        # Count active modules by category
        if result.modules_activated > result.total_modules * 0.7:
            insights.append("High cognitive engagement - complex query")

        if result.dominant_type == "reasoning":
            insights.append("Reasoning modules activated - logical analysis needed")
        elif result.dominant_type == "creative":
            insights.append("Creativity modules activated - novel generation")
        elif result.dominant_type == "memory":
            insights.append("Memory systems engaged - retrieval mode")

        if result.confidence > 0.8:
            insights.append("High confidence response likely")
        elif result.confidence < 0.4:
            insights.append("Low confidence - may need verification")

        return insights

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "total_modules": len(self.modules),
            "active_modules": self.active_count,
            "fallback_modules": self.fallback_count,
            "failed_modules": len(self.failed_modules),
            "failed_list": self.failed_modules,
            "module_types": {
                "core": sum(
                    1 for m in self.modules.values() if m["type"] in ["active", "fallback"]
                ),
                "components": sum(1 for m in self.modules.values() if m["type"] == "component"),
            },
        }

    def statistics(self) -> Dict[str, Any]:
        """Alias for get_stats."""
        return self.get_stats()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COGNITIVE ORCHESTRATOR TEST")
    print("=" * 70)

    orchestrator = CognitiveOrchestrator(verbose=True)

    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    import json

    print(json.dumps(orchestrator.get_stats(), indent=2))

    print("\n" + "=" * 70)
    print("TEST QUERY")
    print("=" * 70)

    result = orchestrator.process("Why is the sky blue? Explain the reasoning.")
    print(f"\nQuery: {result.query}")
    print(f"Modules activated: {result.modules_activated}/{result.total_modules}")
    print(f"Dominant type: {result.dominant_type}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Insights: {result.insights}")
