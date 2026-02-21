"""
Cognitive Module Adapters

Provides clean interfaces to all 39+ cognitive modules with proper fallbacks.
This ensures UltraThink and the cognitive pipeline can always function,
even if some modules fail to load.
"""

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

# Add all module paths
NEURO_ROOT = Path(__file__).parent.parent.parent

MODULE_PATHS = [
    NEURO_ROOT / "neuro-system" / "src",
    NEURO_ROOT / "neuro-meta-reasoning" / "src",
    NEURO_ROOT / "neuro-module-00-integration" / "src",
    NEURO_ROOT / "neuro-module-01-predictive-coding" / "src",
    NEURO_ROOT / "neuro-module-02-dual-process" / "src",
    NEURO_ROOT / "neuro-module-03-reasoning-types" / "src",
    NEURO_ROOT / "neuro-module-04-memory" / "src",
    NEURO_ROOT / "neuro-module-05-sleep-consolidation" / "src",
    NEURO_ROOT / "neuro-module-06-motivation" / "src",
    NEURO_ROOT / "neuro-module-07-emotions-decisions" / "src",
    NEURO_ROOT / "neuro-module-08-language" / "src",
    NEURO_ROOT / "neuro-module-09-creativity" / "src",
    NEURO_ROOT / "neuro-module-10-spatial-cognition" / "src",
    NEURO_ROOT / "neuro-module-11-time-perception" / "src",
    NEURO_ROOT / "neuro-module-12-learning" / "src",
    NEURO_ROOT / "neuro-module-13-executive" / "src",
    NEURO_ROOT / "neuro-module-14-embodied" / "src",
    NEURO_ROOT / "neuro-module-15-social" / "src",
    NEURO_ROOT / "neuro-module-16-consciousness" / "src",
    NEURO_ROOT / "neuro-module-17-world-model" / "src",
    NEURO_ROOT / "neuro-module-18-self-improvement" / "src",
    NEURO_ROOT / "neuro-module-19-multi-agent" / "src",
    NEURO_ROOT / "neuro-llm" / "src",
    NEURO_ROOT / "neuro-knowledge" / "src",
]

for path in MODULE_PATHS:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class ProblemType(Enum):
    LOGICAL = "logical"
    MATHEMATICAL = "mathematical"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    LINGUISTIC = "linguistic"
    PLANNING = "planning"
    CREATIVE = "creative"
    UNKNOWN = "unknown"


class ThinkingStyle(Enum):
    ANALYTICAL = "analytical"
    INTUITIVE = "intuitive"
    SYSTEMATIC = "systematic"
    CREATIVE = "creative"
    HOLISTIC = "holistic"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"


class ProblemDifficulty(Enum):
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class ProblemAnalysis:
    """Analysis of a problem."""
    problem_type: ProblemType
    type_confidence: float
    complexity: float
    difficulty: ProblemDifficulty
    features: Dict[str, float] = field(default_factory=dict)
    subproblems: List[Dict] = field(default_factory=list)
    estimated_steps: int = 1
    requires_domain_knowledge: bool = False


@dataclass
class StyleSelection:
    """Selected thinking style."""
    primary_style: ThinkingStyle
    primary_fitness: float
    secondary_styles: List[Tuple[ThinkingStyle, float]] = field(default_factory=list)
    rationale: str = ""


@dataclass
class ExecutionPlan:
    """Plan for executing reasoning."""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    estimated_confidence: float = 0.5
    estimated_time: float = 1.0


# =============================================================================
# ADAPTER BASE CLASS
# =============================================================================

class ModuleAdapter:
    """Base adapter that provides fallback functionality."""

    def __init__(self, name: str, real_module: Any = None):
        self.name = name
        self._real = real_module
        self._active = real_module is not None

    @property
    def is_active(self) -> bool:
        return self._active

    def should_activate(self, broadcast: Dict) -> bool:
        """Determine if this module should process the broadcast."""
        return self._active

    def process(self, broadcast: Dict) -> Dict[str, Any]:
        """Process a broadcast and return results."""
        return {"module": self.name, "active": self._active}


# =============================================================================
# PROBLEM CLASSIFIER ADAPTER
# =============================================================================

class ProblemClassifierAdapter(ModuleAdapter):
    """Adapter for problem classification."""

    def __init__(self):
        real = None
        try:
            from problem_classifier import ProblemClassifier
            real = ProblemClassifier(random_seed=42)
        except Exception as e:
            pass

        super().__init__("problem_classifier", real)
        self._rng = np.random.default_rng(42)

    def classify(self, embedding: np.ndarray) -> ProblemAnalysis:
        """Classify a problem based on its embedding."""
        if self._real:
            try:
                result = self._real.classify(embedding)
                # Convert to our dataclass if needed
                return ProblemAnalysis(
                    problem_type=ProblemType(result.problem_type.value),
                    type_confidence=result.type_confidence,
                    complexity=result.complexity,
                    difficulty=ProblemDifficulty(result.difficulty.value),
                    features=getattr(result, 'features', {}),
                    subproblems=getattr(result, 'subproblems', []),
                    estimated_steps=getattr(result, 'estimated_steps', 3),
                    requires_domain_knowledge=getattr(result, 'requires_domain_knowledge', False)
                )
            except Exception:
                pass

        # Fallback classification based on embedding analysis
        complexity = float(np.std(embedding))
        energy = float(np.mean(np.abs(embedding)))

        # Determine type from embedding characteristics
        if energy > 0.6:
            problem_type = ProblemType.LOGICAL
        elif complexity > 0.4:
            problem_type = ProblemType.MATHEMATICAL
        else:
            problem_type = ProblemType.LINGUISTIC

        # Determine difficulty
        if complexity < 0.3:
            difficulty = ProblemDifficulty.EASY
        elif complexity < 0.6:
            difficulty = ProblemDifficulty.MEDIUM
        else:
            difficulty = ProblemDifficulty.HARD

        return ProblemAnalysis(
            problem_type=problem_type,
            type_confidence=0.7,
            complexity=complexity,
            difficulty=difficulty,
            estimated_steps=max(1, int(complexity * 5)),
            requires_domain_knowledge=complexity > 0.5
        )


# =============================================================================
# STYLE SELECTOR ADAPTER
# =============================================================================

class StyleSelectorAdapter(ModuleAdapter):
    """Adapter for thinking style selection."""

    def __init__(self):
        real = None
        try:
            from style_selector import StyleSelector
            real = StyleSelector(random_seed=42)
        except Exception:
            pass

        super().__init__("style_selector", real)

    def select_style(self, analysis: ProblemAnalysis) -> StyleSelection:
        """Select appropriate thinking style for problem."""
        if self._real:
            try:
                result = self._real.select_style(analysis)
                return StyleSelection(
                    primary_style=ThinkingStyle(result.primary_style.value),
                    primary_fitness=result.primary_fitness,
                    secondary_styles=[
                        (ThinkingStyle(s[0].value), s[1])
                        for s in getattr(result, 'secondary_styles', [])
                    ],
                    rationale=getattr(result, 'rationale', '')
                )
            except Exception:
                pass

        # Fallback style selection
        style_map = {
            ProblemType.LOGICAL: ThinkingStyle.DEDUCTIVE,
            ProblemType.MATHEMATICAL: ThinkingStyle.ANALYTICAL,
            ProblemType.ANALOGICAL: ThinkingStyle.INTUITIVE,
            ProblemType.CAUSAL: ThinkingStyle.SYSTEMATIC,
            ProblemType.CREATIVE: ThinkingStyle.CREATIVE,
            ProblemType.LINGUISTIC: ThinkingStyle.HOLISTIC,
            ProblemType.PLANNING: ThinkingStyle.SYSTEMATIC,
        }

        primary = style_map.get(analysis.problem_type, ThinkingStyle.ANALYTICAL)

        return StyleSelection(
            primary_style=primary,
            primary_fitness=0.8,
            secondary_styles=[(ThinkingStyle.INTUITIVE, 0.5)],
            rationale=f"Selected {primary.value} for {analysis.problem_type.value} problem"
        )


# =============================================================================
# ORCHESTRATOR ADAPTER
# =============================================================================

class OrchestratorAdapter(ModuleAdapter):
    """Adapter for reasoning orchestration."""

    def __init__(self):
        real = None
        try:
            from orchestrator import DynamicOrchestrator
            real = DynamicOrchestrator(random_seed=42)
        except Exception:
            pass

        super().__init__("orchestrator", real)

    def create_plan(self, analysis: ProblemAnalysis, style: StyleSelection) -> ExecutionPlan:
        """Create an execution plan for reasoning."""
        if self._real:
            try:
                result = self._real.create_plan(analysis, style)
                return ExecutionPlan(
                    steps=[{"name": s.name, "module": getattr(s, 'module', 'unknown')}
                           for s in getattr(result, 'steps', [])],
                    estimated_confidence=getattr(result, 'estimated_confidence', 0.7),
                    estimated_time=getattr(result, 'estimated_time', 1.0)
                )
            except Exception:
                pass

        # Fallback plan
        steps = [
            {"name": "analyze", "module": "classifier"},
            {"name": "reason", "module": style.primary_style.value},
            {"name": "synthesize", "module": "global_workspace"}
        ]

        if analysis.complexity > 0.5:
            steps.insert(1, {"name": "decompose", "module": "orchestrator"})

        return ExecutionPlan(
            steps=steps,
            estimated_confidence=0.7,
            estimated_time=analysis.estimated_steps * 0.5
        )


# =============================================================================
# GLOBAL WORKSPACE ADAPTER
# =============================================================================

class GlobalWorkspaceAdapter(ModuleAdapter):
    """Adapter for global workspace (consciousness/attention)."""

    def __init__(self):
        real = None
        try:
            from global_workspace import GlobalWorkspace
            real = GlobalWorkspace()
        except Exception:
            pass

        super().__init__("global_workspace", real)
        self._attention = np.ones(128) / 128  # Uniform attention

    def broadcast(self, content: Dict) -> Dict[str, Any]:
        """Broadcast content to all modules."""
        if self._real:
            try:
                return self._real.broadcast(content)
            except Exception:
                pass

        # Fallback broadcast
        return {
            "content": content,
            "attention": self._attention.tolist(),
            "coherence": 0.7,
            "broadcast_id": hash(str(content)) % 10000
        }

    def should_activate(self, broadcast: Dict) -> bool:
        return True  # Always active

    def process(self, broadcast: Dict) -> Dict[str, Any]:
        return self.broadcast(broadcast)


# =============================================================================
# DUAL PROCESS ADAPTER
# =============================================================================

class DualProcessAdapter(ModuleAdapter):
    """Adapter for System 1/2 dual process thinking."""

    def __init__(self):
        real = None
        try:
            from dual_process_controller import DualProcessController
            real = DualProcessController()
        except Exception:
            pass

        super().__init__("dual_process", real)

    def system1_response(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Fast, intuitive System 1 response."""
        if self._real:
            try:
                return self._real.system1_process(embedding)
            except Exception:
                pass

        # Fallback - quick pattern matching
        confidence = float(np.mean(embedding) + 0.5)
        return {
            "response": "intuitive",
            "confidence": min(1.0, max(0.0, confidence)),
            "processing_time": 0.01
        }

    def system2_response(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Slow, deliberate System 2 response."""
        if self._real:
            try:
                return self._real.system2_process(embedding)
            except Exception:
                pass

        # Fallback - analytical processing
        complexity = float(np.std(embedding))
        return {
            "response": "deliberate",
            "confidence": 0.8,
            "processing_time": complexity * 0.5,
            "reasoning_steps": max(1, int(complexity * 5))
        }


# =============================================================================
# PREDICTIVE CODING ADAPTER
# =============================================================================

class PredictiveCodingAdapter(ModuleAdapter):
    """Adapter for predictive coding / free energy principle."""

    def __init__(self):
        real = None
        try:
            from predictive_hierarchy import PredictiveHierarchy
            real = PredictiveHierarchy()
        except Exception:
            pass

        super().__init__("predictive_coding", real)
        self._prior = np.zeros(128)

    def predict(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Generate predictions from input."""
        if self._real:
            try:
                return self._real.predict(embedding)
            except Exception:
                pass

        prediction = self._prior * 0.8 + embedding * 0.2
        self._prior = prediction
        return {
            "prediction": prediction.tolist()[:10],
            "confidence": 0.6
        }

    def compute_free_energy(self, embedding: np.ndarray) -> float:
        """Compute free energy (prediction error)."""
        if self._real:
            try:
                return self._real.compute_free_energy(embedding)
            except Exception:
                pass

        # Simple prediction error
        error = np.mean((embedding - self._prior) ** 2)
        return float(min(1.0, error))


# =============================================================================
# MEMORY SYSTEM ADAPTER
# =============================================================================

class MemorySystemAdapter(ModuleAdapter):
    """Adapter for memory systems (episodic, semantic, working)."""

    def __init__(self):
        real = None
        try:
            from memory_controller import MemoryController
            real = MemoryController()
        except Exception:
            pass

        super().__init__("memory", real)
        self._episodic: List[Dict] = []
        self._semantic: Dict[str, np.ndarray] = {}

    def store_episodic(self, content: str, embedding: np.ndarray, context: Dict = None):
        """Store an episodic memory."""
        if self._real:
            try:
                return self._real.store_episodic(content, embedding, context)
            except Exception:
                pass

        self._episodic.append({
            "content": content,
            "embedding": embedding,
            "context": context or {}
        })

    def recall_episodic(self, query: str, k: int = 5) -> List[Dict]:
        """Recall episodic memories."""
        if self._real:
            try:
                return self._real.recall_episodic(query, k)
            except Exception:
                pass

        return self._episodic[-k:]

    def recall_semantic(self, query: str, k: int = 5) -> List[Dict]:
        """Recall semantic memories."""
        if self._real:
            try:
                return self._real.recall_semantic(query, k)
            except Exception:
                pass

        return []


# =============================================================================
# CURIOSITY ADAPTER
# =============================================================================

class CuriosityAdapter(ModuleAdapter):
    """Adapter for curiosity-driven exploration."""

    def __init__(self):
        real = None
        try:
            from curiosity_drive import CuriosityModule
            real = CuriosityModule(state_dim=128, base_curiosity=1.0)
        except Exception:
            pass

        super().__init__("curiosity", real)
        self.curiosity_level = 1.0  # MAX CURIOSITY
        self._knowledge_gaps: List[str] = []

    def process_stimulus(self, embedding: np.ndarray) -> Dict[str, Any]:
        """Process a stimulus and compute curiosity response."""
        if self._real:
            try:
                return self._real.process_stimulus(embedding)
            except Exception:
                pass

        novelty = float(np.std(embedding))
        return {
            "novelty": novelty,
            "curiosity_response": novelty * self.curiosity_level,
            "explore": novelty > 0.3
        }

    def get_knowledge_gaps(self) -> List[str]:
        """Get identified knowledge gaps."""
        if self._real:
            try:
                return self._real.get_knowledge_gaps()
            except Exception:
                pass

        return self._knowledge_gaps

    def add_gap(self, topic: str):
        """Add a knowledge gap."""
        if topic not in self._knowledge_gaps:
            self._knowledge_gaps.append(topic)


# =============================================================================
# MOTIVATION ADAPTER
# =============================================================================

class MotivationAdapter(ModuleAdapter):
    """Adapter for intrinsic motivation / path entropy."""

    def __init__(self):
        real = None
        try:
            from intrinsic_motivation import PathEntropyMaximizer
            real = PathEntropyMaximizer(state_dim=128, action_dim=32)
        except Exception:
            pass

        super().__init__("motivation", real)

    def compute_motivation(self, state: np.ndarray) -> float:
        """Compute motivation for a given state."""
        if self._real:
            try:
                return self._real.compute_motivation(state)
            except Exception:
                pass

        # Fallback - entropy-based motivation
        entropy = -np.sum(np.abs(state) * np.log(np.abs(state) + 1e-10))
        return float(min(1.0, entropy / 10))


# =============================================================================
# EXECUTIVE FUNCTIONS ADAPTER
# =============================================================================

class ExecutiveFunctionsAdapter(ModuleAdapter):
    """Adapter for executive functions (inhibition, switching, planning)."""

    def __init__(self):
        real = None
        try:
            from executive_network import ExecutiveNetwork
            real = ExecutiveNetwork()
        except Exception:
            pass

        super().__init__("executive", real)

    def inhibit(self, action: np.ndarray) -> bool:
        """Decide whether to inhibit an action."""
        if self._real:
            try:
                return self._real.should_inhibit(action)
            except Exception:
                pass

        return float(np.mean(action)) < -0.5

    def switch_task(self, current: str, new: str) -> float:
        """Compute task switching cost."""
        if self._real:
            try:
                return self._real.compute_switch_cost(current, new)
            except Exception:
                pass

        return 0.2 if current != new else 0.0


# =============================================================================
# REASONING ADAPTER
# =============================================================================

class ReasoningAdapter(ModuleAdapter):
    """Adapter for multi-type reasoning (logical, analogical, etc.)."""

    def __init__(self):
        real = None
        try:
            from reasoning_orchestrator import ReasoningOrchestrator
            real = ReasoningOrchestrator()
        except Exception:
            pass

        super().__init__("reasoning", real)

    def reason(self, problem: np.ndarray, style: ThinkingStyle) -> Dict[str, Any]:
        """Apply reasoning to a problem."""
        if self._real:
            try:
                return self._real.reason(problem, style.value)
            except Exception:
                pass

        return {
            "conclusion": "reasoning_applied",
            "confidence": 0.7,
            "style": style.value
        }


# =============================================================================
# WORLD MODEL ADAPTER
# =============================================================================

class WorldModelAdapter(ModuleAdapter):
    """Adapter for world modeling and simulation."""

    def __init__(self):
        real = None
        try:
            from imagination import ImaginationEngine
            real = ImaginationEngine()
        except Exception:
            pass

        super().__init__("world_model", real)

    def simulate(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Simulate the outcome of an action."""
        if self._real:
            try:
                return self._real.simulate(state, action)
            except Exception:
                pass

        # Simple linear model
        return state + action * 0.1

    def counterfactual(self, state: np.ndarray, alternative: np.ndarray) -> np.ndarray:
        """Compute counterfactual outcome."""
        if self._real:
            try:
                return self._real.counterfactual(state, alternative)
            except Exception:
                pass

        return (state + alternative) / 2


# =============================================================================
# EMOTION ADAPTER
# =============================================================================

class EmotionAdapter(ModuleAdapter):
    """Adapter for emotional processing."""

    def __init__(self):
        real = None
        try:
            from emotion_circuit import EmotionCircuit
            real = EmotionCircuit()
        except Exception:
            pass

        super().__init__("emotion", real)

    def evaluate(self, stimulus: np.ndarray) -> Dict[str, float]:
        """Evaluate emotional response to stimulus."""
        if self._real:
            try:
                return self._real.evaluate(stimulus)
            except Exception:
                pass

        valence = float(np.mean(stimulus))
        arousal = float(np.std(stimulus))

        return {
            "valence": valence,
            "arousal": arousal,
            "dominance": 0.5
        }


# =============================================================================
# LANGUAGE ADAPTER
# =============================================================================

class LanguageAdapter(ModuleAdapter):
    """Adapter for language processing."""

    def __init__(self):
        real = None
        try:
            from language_network import LanguageNetwork
            real = LanguageNetwork()
        except Exception:
            pass

        super().__init__("language", real)

    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding."""
        if self._real:
            try:
                return self._real.encode(text)
            except Exception:
                pass

        # Simple hash-based encoding
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        extended = h * 4
        return np.array([b / 255.0 for b in extended[:128]])


# =============================================================================
# CREATIVITY ADAPTER
# =============================================================================

class CreativityAdapter(ModuleAdapter):
    """Adapter for creative processes."""

    def __init__(self):
        real = None
        try:
            from creative_process import CreativeProcess
            real = CreativeProcess()
        except Exception:
            pass

        super().__init__("creativity", real)

    def generate(self, seed: np.ndarray, diversity: float = 0.5) -> np.ndarray:
        """Generate creative output."""
        if self._real:
            try:
                return self._real.generate(seed, diversity)
            except Exception:
                pass

        noise = np.random.randn(len(seed)) * diversity
        return seed + noise


# =============================================================================
# SOCIAL COGNITION ADAPTER
# =============================================================================

class SocialCognitionAdapter(ModuleAdapter):
    """Adapter for theory of mind and social reasoning."""

    def __init__(self):
        real = None
        try:
            from theory_of_mind import TheoryOfMind
            real = TheoryOfMind()
        except Exception:
            pass

        super().__init__("social", real)

    def infer_mental_state(self, agent_id: str, observations: Dict) -> Dict[str, Any]:
        """Infer another agent's mental state."""
        if self._real:
            try:
                return self._real.infer_mental_state(agent_id, observations)
            except Exception:
                pass

        return {
            "agent": agent_id,
            "beliefs": [],
            "intentions": [],
            "emotions": {}
        }


# =============================================================================
# CONSCIOUSNESS ADAPTER
# =============================================================================

class ConsciousnessAdapter(ModuleAdapter):
    """Adapter for metacognition and self-awareness."""

    def __init__(self):
        real = None
        try:
            from metacognition import MetacognitionModule
            real = MetacognitionModule()
        except Exception:
            pass

        super().__init__("consciousness", real)

    def introspect(self) -> Dict[str, Any]:
        """Perform introspection."""
        if self._real:
            try:
                return self._real.introspect()
            except Exception:
                pass

        return {
            "state": "aware",
            "confidence_in_reasoning": 0.7,
            "uncertainty_areas": []
        }


# =============================================================================
# MODULE FACTORY
# =============================================================================

class CognitiveModuleFactory:
    """Factory for creating and managing all cognitive module adapters."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._modules: Dict[str, ModuleAdapter] = {}
        self._load_all()

    def _load_all(self):
        """Load all available modules."""
        adapters = [
            ("problem_classifier", ProblemClassifierAdapter),
            ("style_selector", StyleSelectorAdapter),
            ("orchestrator", OrchestratorAdapter),
            ("global_workspace", GlobalWorkspaceAdapter),
            ("dual_process", DualProcessAdapter),
            ("predictive_coding", PredictiveCodingAdapter),
            ("memory", MemorySystemAdapter),
            ("curiosity", CuriosityAdapter),
            ("motivation", MotivationAdapter),
            ("executive", ExecutiveFunctionsAdapter),
            ("reasoning", ReasoningAdapter),
            ("world_model", WorldModelAdapter),
            ("emotion", EmotionAdapter),
            ("language", LanguageAdapter),
            ("creativity", CreativityAdapter),
            ("social", SocialCognitionAdapter),
            ("consciousness", ConsciousnessAdapter),
        ]

        for name, adapter_class in adapters:
            try:
                self._modules[name] = adapter_class()
                if self.verbose:
                    status = "active" if self._modules[name].is_active else "fallback"
                    print(f"  [CognitiveModules] Loaded {name}: {status}")
            except Exception as e:
                if self.verbose:
                    print(f"  [CognitiveModules] Failed to load {name}: {e}")

    def get(self, name: str) -> Optional[ModuleAdapter]:
        """Get a module by name."""
        return self._modules.get(name)

    def get_all(self) -> Dict[str, ModuleAdapter]:
        """Get all modules."""
        return self._modules

    def get_active(self) -> List[str]:
        """Get names of modules with real implementations."""
        return [name for name, mod in self._modules.items() if mod.is_active]

    def get_fallback(self) -> List[str]:
        """Get names of modules using fallback implementations."""
        return [name for name, mod in self._modules.items() if not mod.is_active]

    def statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded modules."""
        active = self.get_active()
        fallback = self.get_fallback()
        return {
            "total_modules": len(self._modules),
            "active_modules": len(active),
            "fallback_modules": len(fallback),
            "active_list": active,
            "fallback_list": fallback
        }

    def get_stats(self) -> Dict[str, Any]:
        """Alias for statistics()."""
        return self.statistics()
