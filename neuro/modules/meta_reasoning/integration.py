"""
Meta-Reasoning Integration

Unified meta-reasoning controller combining:
- Problem classification
- Style selection
- Efficiency monitoring
- Dynamic orchestration
- Fallacy detection
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

try:
    from .problem_classifier import (
        ProblemClassifier,
        ProblemClassifierConfig,
        ProblemAnalysis,
        ProblemType,
    )
    from .style_selector import (
        StyleSelector,
        StyleSelectorConfig,
        StyleSelection,
        ReasoningStyle,
        StyleFeedback,
    )
    from .efficiency_monitor import (
        EfficiencyMonitor,
        EfficiencyConfig,
        ReasoningMetrics,
        TerminationReason,
    )
    from .orchestrator import (
        DynamicOrchestrator,
        OrchestratorConfig,
        OrchestrationPlan,
        ExecutionResult,
    )
    from .fallacy_detector import (
        FallacyDetector,
        FallacyDetectorConfig,
        FallacyDetection,
        ReasoningStep,
        FallacyType,
    )
except ImportError:
    from problem_classifier import (
        ProblemClassifier,
        ProblemClassifierConfig,
        ProblemAnalysis,
        ProblemType,
    )
    from style_selector import (
        StyleSelector,
        StyleSelectorConfig,
        StyleSelection,
        ReasoningStyle,
        StyleFeedback,
    )
    from efficiency_monitor import (
        EfficiencyMonitor,
        EfficiencyConfig,
        ReasoningMetrics,
        TerminationReason,
    )
    from orchestrator import (
        DynamicOrchestrator,
        OrchestratorConfig,
        OrchestrationPlan,
        ExecutionResult,
    )
    from fallacy_detector import (
        FallacyDetector,
        FallacyDetectorConfig,
        FallacyDetection,
        ReasoningStep,
        FallacyType,
    )


@dataclass
class MetaReasoningConfig:
    """Configuration for meta-reasoning controller."""

    embedding_dim: int = 128
    exploration_rate: float = 0.1
    time_limit_seconds: float = 60.0
    enable_fallacy_detection: bool = True
    enable_dynamic_switching: bool = True
    random_seed: Optional[int] = None


@dataclass
class ReasoningSession:
    """A complete reasoning session."""

    session_id: str
    problem_analysis: ProblemAnalysis
    style_selection: StyleSelection
    plan: OrchestrationPlan
    execution_result: Optional[ExecutionResult]
    fallacies_detected: List[FallacyDetection]
    reasoning_trace: List[ReasoningStep]


class MetaReasoningController:
    """
    Unified controller for meta-reasoning.

    Orchestrates:
    - Problem analysis and classification
    - Reasoning style selection
    - Execution monitoring
    - Fallacy detection
    - Dynamic adaptation
    """

    def __init__(
        self,
        config: Optional[MetaReasoningConfig] = None,
    ):
        self.config = config or MetaReasoningConfig()

        classifier_config = ProblemClassifierConfig(
            embedding_dim=self.config.embedding_dim,
        )
        self._classifier = ProblemClassifier(
            config=classifier_config,
            random_seed=self.config.random_seed,
        )

        style_config = StyleSelectorConfig(
            exploration_rate=self.config.exploration_rate,
        )
        self._style_selector = StyleSelector(
            config=style_config,
            random_seed=self.config.random_seed,
        )

        efficiency_config = EfficiencyConfig(
            time_limit_seconds=self.config.time_limit_seconds,
        )
        self._efficiency = EfficiencyMonitor(
            config=efficiency_config,
            random_seed=self.config.random_seed,
        )

        orchestrator_config = OrchestratorConfig(
            enable_dynamic_switching=self.config.enable_dynamic_switching,
        )
        self._orchestrator = DynamicOrchestrator(
            config=orchestrator_config,
            random_seed=self.config.random_seed,
        )

        fallacy_config = FallacyDetectorConfig()
        self._fallacy_detector = FallacyDetector(
            config=fallacy_config,
            random_seed=self.config.random_seed,
        )

        self._sessions: Dict[str, ReasoningSession] = {}
        self._session_counter = 0

    def analyze_problem(
        self,
        problem_embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> ProblemAnalysis:
        """
        Analyze a problem.

        Args:
            problem_embedding: Problem embedding
            context: Optional context

        Returns:
            ProblemAnalysis
        """
        return self._classifier.classify(problem_embedding, context)

    def select_style(
        self,
        analysis: ProblemAnalysis,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> StyleSelection:
        """
        Select reasoning style.

        Args:
            analysis: Problem analysis
            constraints: Optional constraints

        Returns:
            StyleSelection
        """
        return self._style_selector.select_style(analysis, constraints)

    def create_session(
        self,
        problem_embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None,
    ) -> ReasoningSession:
        """
        Create a new reasoning session.

        Args:
            problem_embedding: Problem embedding
            context: Optional context

        Returns:
            ReasoningSession
        """
        session_id = f"session_{self._session_counter}"
        self._session_counter += 1

        analysis = self.analyze_problem(problem_embedding, context)

        style = self.select_style(analysis)

        plan = self._orchestrator.create_plan(analysis, style)

        session = ReasoningSession(
            session_id=session_id,
            problem_analysis=analysis,
            style_selection=style,
            plan=plan,
            execution_result=None,
            fallacies_detected=[],
            reasoning_trace=[],
        )

        self._sessions[session_id] = session
        return session

    def execute_session(
        self,
        session_id: str,
        problem: Dict[str, Any],
    ) -> ExecutionResult:
        """
        Execute a reasoning session.

        Args:
            session_id: Session identifier
            problem: Problem to solve

        Returns:
            ExecutionResult
        """
        if session_id not in self._sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self._sessions[session_id]

        result = self._orchestrator.execute_plan(session.plan, problem)

        session.execution_result = result

        if self.config.enable_fallacy_detection and session.reasoning_trace:
            fallacies = self._fallacy_detector.detect_fallacies(session.reasoning_trace)
            session.fallacies_detected = fallacies

        return result

    def add_reasoning_step(
        self,
        session_id: str,
        step: ReasoningStep,
    ) -> None:
        """Add a reasoning step to the session trace."""
        if session_id in self._sessions:
            self._sessions[session_id].reasoning_trace.append(step)

    def check_for_fallacies(
        self,
        session_id: str,
    ) -> List[FallacyDetection]:
        """
        Check session for fallacies.

        Args:
            session_id: Session identifier

        Returns:
            List of detected fallacies
        """
        if session_id not in self._sessions:
            return []

        session = self._sessions[session_id]

        if not session.reasoning_trace:
            return []

        fallacies = self._fallacy_detector.detect_fallacies(session.reasoning_trace)
        session.fallacies_detected.extend(fallacies)

        return fallacies

    def get_corrections(
        self,
        session_id: str,
    ) -> List[str]:
        """
        Get correction suggestions for session.

        Args:
            session_id: Session identifier

        Returns:
            List of correction suggestions
        """
        if session_id not in self._sessions:
            return []

        session = self._sessions[session_id]
        return self._fallacy_detector.suggest_corrections(session.fallacies_detected)

    def record_feedback(
        self,
        session_id: str,
        success: bool,
        efficiency: float,
        quality: float,
    ) -> None:
        """
        Record feedback on session.

        Args:
            session_id: Session identifier
            success: Whether successful
            efficiency: Efficiency score
            quality: Quality score
        """
        if session_id not in self._sessions:
            return

        session = self._sessions[session_id]

        feedback = StyleFeedback(
            style=session.style_selection.primary_style,
            problem_type=session.problem_analysis.problem_type,
            success=success,
            efficiency=efficiency,
            quality=quality,
        )
        self._style_selector.update_from_feedback(feedback)

    def register_module(self, name: str, module_fn: Callable) -> None:
        """Register a reasoning module."""
        self._orchestrator.register_module(name, module_fn)

    def get_session(self, session_id: str) -> Optional[ReasoningSession]:
        """Get session by ID."""
        return self._sessions.get(session_id)

    def get_all_sessions(self) -> List[str]:
        """Get all session IDs."""
        return list(self._sessions.keys())

    def get_style_rankings(
        self,
        problem_type: ProblemType,
    ) -> List[Tuple[ReasoningStyle, float]]:
        """Get style rankings for problem type."""
        return self._style_selector.get_style_rankings(problem_type)

    def should_terminate(
        self,
        session_id: str,
    ) -> Tuple[bool, Optional[TerminationReason]]:
        """Check if session should terminate."""
        return self._efficiency.should_terminate_early(session_id)

    def statistics(self) -> Dict[str, Any]:
        """Get controller statistics."""
        return {
            "total_sessions": len(self._sessions),
            "classifier": self._classifier.statistics(),
            "style_selector": self._style_selector.statistics(),
            "efficiency": self._efficiency.statistics(),
            "orchestrator": self._orchestrator.statistics(),
            "fallacy_detector": self._fallacy_detector.statistics(),
        }
