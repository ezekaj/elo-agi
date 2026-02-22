"""
Dual Process Controller - System 1/2 Orchestration

Manages when to use intuition (System 1) vs deliberation (System 2).
This is the central executive that decides which system to engage.

Based on research showing:
- System 1 fires first, fast and automatic
- Conflict detection triggers System 2 engagement
- System 2 can override but at cognitive cost
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

from .system1 import PatternRecognition, HabitExecutor, EmotionalValuation
from .system1.pattern_recognition import PatternMatch
from .system1.habit_executor import HabitResponse, Action
from .system1.emotional_valuation import Valence
from .system2 import WorkingMemory, CognitiveControl, RelationalReasoning
from .system2.cognitive_control import Response, ConflictLevel


@dataclass
class System1Output:
    """Bundled output from System 1 processing"""

    patterns: List[PatternMatch]
    habit_response: Optional[HabitResponse]
    emotional_valence: Valence
    confidence: float
    processing_time: float


@dataclass
class System2Output:
    """Bundled output from System 2 processing"""

    working_memory_state: Dict[str, Any]
    reasoning_result: Any
    deliberation_steps: int
    processing_time: float


@dataclass
class DualProcessOutput:
    """Final output from dual-process system"""

    response: Any
    system_used: str  # "system1", "system2", or "hybrid"
    confidence: float
    s1_output: Optional[System1Output]
    s2_output: Optional[System2Output]
    conflict_detected: bool
    override_occurred: bool


class System1Bundle:
    """Bundles all System 1 components"""

    def __init__(self):
        self.pattern_recognition = PatternRecognition()
        self.habit_executor = HabitExecutor()
        self.emotional_valuation = EmotionalValuation()

    def process(self, input_vector: np.ndarray, context: Optional[Any] = None) -> System1Output:
        """
        Fast, parallel System 1 processing.

        All components process simultaneously.
        """
        start_time = time.time()

        # Parallel processing (in reality these would be concurrent)
        patterns = self.pattern_recognition.match(input_vector, context=context)
        habit_response = self.habit_executor.execute(input_vector, context=context)
        emotional_valence = self.emotional_valuation.evaluate(input_vector)

        # Compute overall confidence
        confidences = []
        if patterns:
            confidences.append(max(p.confidence for p in patterns))
        if habit_response.triggered:
            confidences.append(habit_response.confidence)
        if emotional_valence.intensity > 0.3:
            confidences.append(emotional_valence.intensity)

        overall_confidence = np.mean(confidences) if confidences else 0.0

        processing_time = time.time() - start_time

        return System1Output(
            patterns=patterns,
            habit_response=habit_response,
            emotional_valence=emotional_valence,
            confidence=overall_confidence,
            processing_time=processing_time,
        )


class System2Bundle:
    """Bundles all System 2 components"""

    def __init__(self):
        self.working_memory = WorkingMemory()
        self.cognitive_control = CognitiveControl()
        self.relational_reasoning = RelationalReasoning()

    def deliberate(
        self, input_data: Any, s1_output: System1Output, max_steps: int = 10
    ) -> System2Output:
        """
        Slow, serial System 2 deliberation.

        Uses working memory to hold and manipulate information.
        """
        start_time = time.time()
        steps = 0

        # Load relevant info into working memory
        self.working_memory.store("input", input_data)
        self.working_memory.store("s1_patterns", s1_output.patterns)
        self.working_memory.store("s1_habit", s1_output.habit_response)
        self.working_memory.store("s1_emotion", s1_output.emotional_valence)

        result = None

        # Serial deliberation steps
        while steps < max_steps:
            steps += 1

            # Check cognitive load
            if self.working_memory.is_full:
                # Need to chunk or drop items
                self._manage_cognitive_load()

            # Try to reach conclusion
            result = self._deliberation_step(s1_output)

            if result is not None:
                break

        wm_state = {
            item_id: self.working_memory.peek(item_id)
            for item_id in self.working_memory.get_all_items()
        }

        processing_time = time.time() - start_time

        return System2Output(
            working_memory_state=wm_state,
            reasoning_result=result,
            deliberation_steps=steps,
            processing_time=processing_time,
        )

    def _deliberation_step(self, s1_output: System1Output) -> Optional[Any]:
        """Single step of deliberation"""
        # Check if S1 outputs are conflicting
        responses = []

        for pattern in s1_output.patterns:
            responses.append(
                Response(
                    id=f"pattern_{pattern.pattern_id}",
                    activation=pattern.confidence,
                    source="pattern_recognition",
                )
            )

        if s1_output.habit_response and s1_output.habit_response.triggered:
            responses.append(
                Response(
                    id=f"habit_{s1_output.habit_response.action.id}",
                    activation=s1_output.habit_response.confidence,
                    source="habit",
                )
            )

        # If no conflict, accept S1 output
        conflict = self.cognitive_control.detect_conflict(responses)
        if conflict.level == ConflictLevel.NONE:
            if responses:
                best = max(responses, key=lambda r: r.activation)
                return {"selected": best.id, "method": "s2_confirmed_s1"}

        # Conflict - need more deliberation
        # Try to resolve through relational reasoning
        # (Simplified - real implementation would be more complex)

        return None

    def _manage_cognitive_load(self):
        """Manage working memory when at capacity"""
        # Drop lowest activation items
        activations = self.working_memory.get_activations()
        if activations:
            lowest = min(activations.keys(), key=lambda k: activations[k])
            self.working_memory.slots.pop(lowest, None)


class DualProcessController:
    """
    Main controller for dual-process cognitive system.

    Orchestrates System 1 and System 2, deciding when each should
    be engaged and how to integrate their outputs.
    """

    def __init__(self, conflict_threshold: float = 0.4, s2_engagement_cost: float = 0.1):
        self.system1 = System1Bundle()
        self.system2 = System2Bundle()

        self.conflict_threshold = conflict_threshold
        self.s2_engagement_cost = s2_engagement_cost  # Cognitive effort

        # Monitoring
        self.processing_history: List[DualProcessOutput] = []
        self._s2_engagement_count = 0
        self._total_processing_count = 0

    def process(
        self, input_data: Any, context: Optional[Any] = None, force_s2: bool = False
    ) -> DualProcessOutput:
        """
        Main processing pipeline.

        1. System 1 produces fast response
        2. Check for conflict/uncertainty
        3. Engage System 2 if needed
        4. Return final response
        """
        self._total_processing_count += 1

        # Convert input to vector if needed
        if isinstance(input_data, np.ndarray):
            input_vector = input_data
        else:
            # Simple conversion - real implementation would be more sophisticated
            input_vector = np.array([hash(str(input_data)) % 1000 / 1000.0] * 10)

        # Step 1: System 1 (always runs first)
        s1_output = self.system1.process(input_vector, context)

        # Step 2: Check if System 2 needed
        conflict_detected = self._check_conflict(s1_output)
        uncertainty = 1.0 - s1_output.confidence

        engage_s2 = (
            force_s2
            or conflict_detected
            or uncertainty > self.conflict_threshold
            or s1_output.emotional_valence.threat > 0.7  # High threat triggers careful processing
        )

        # Step 3: Maybe engage System 2
        s2_output = None
        override_occurred = False

        if engage_s2:
            self._s2_engagement_count += 1
            s2_output = self.system2.deliberate(input_data, s1_output)

            # Check if S2 overrides S1
            if s2_output.reasoning_result is not None:
                override_occurred = True

        # Step 4: Determine final response
        if override_occurred and s2_output:
            response = s2_output.reasoning_result
            system_used = "system2"
            confidence = 0.8  # S2 is more conservative
        elif s1_output.habit_response and s1_output.habit_response.triggered:
            response = s1_output.habit_response.action
            system_used = "system1"
            confidence = s1_output.confidence
        elif s1_output.patterns:
            response = s1_output.patterns[0]
            system_used = "system1"
            confidence = s1_output.confidence
        else:
            response = None
            system_used = "none"
            confidence = 0.0

        output = DualProcessOutput(
            response=response,
            system_used=system_used,
            confidence=confidence,
            s1_output=s1_output,
            s2_output=s2_output,
            conflict_detected=conflict_detected,
            override_occurred=override_occurred,
        )

        self.processing_history.append(output)
        return output

    def _check_conflict(self, s1_output: System1Output) -> bool:
        """Check if System 1 outputs are conflicting"""

        # Check pattern conflicts
        if len(s1_output.patterns) > 1:
            top_patterns = sorted(s1_output.patterns, key=lambda p: p.confidence, reverse=True)[:3]
            if len(top_patterns) >= 2:
                # Close confidence scores suggest conflict
                spread = top_patterns[0].confidence - top_patterns[-1].confidence
                if spread < 0.2:
                    return True

        # Check habit vs pattern conflict
        if s1_output.habit_response.triggered and s1_output.patterns:
            habit_action = s1_output.habit_response.action.id
            pattern_id = s1_output.patterns[0].pattern_id
            if habit_action != pattern_id:
                return True

        # Check emotional conflict (approach-avoid)
        if s1_output.emotional_valence.threat > 0.3 and s1_output.emotional_valence.reward > 0.3:
            return True

        return False

    def override_intuition(self, input_data: Any) -> DualProcessOutput:
        """Force System 2 to override System 1"""
        return self.process(input_data, force_s2=True)

    def trust_gut(self, input_data: Any, context: Optional[Any] = None) -> DualProcessOutput:
        """Use System 1 response even if conflict detected"""
        # Convert input
        if isinstance(input_data, np.ndarray):
            input_vector = input_data
        else:
            input_vector = np.array([hash(str(input_data)) % 1000 / 1000.0] * 10)

        s1_output = self.system1.process(input_vector, context)

        # Determine response from S1 only
        if s1_output.habit_response and s1_output.habit_response.triggered:
            response = s1_output.habit_response.action
        elif s1_output.patterns:
            response = s1_output.patterns[0]
        else:
            response = None

        return DualProcessOutput(
            response=response,
            system_used="system1",
            confidence=s1_output.confidence,
            s1_output=s1_output,
            s2_output=None,
            conflict_detected=self._check_conflict(s1_output),
            override_occurred=False,
        )

    def get_s2_engagement_rate(self) -> float:
        """Get rate of System 2 engagement"""
        if self._total_processing_count == 0:
            return 0.0
        return self._s2_engagement_count / self._total_processing_count

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        if not self.processing_history:
            return {}

        s1_times = [h.s1_output.processing_time for h in self.processing_history if h.s1_output]
        s2_times = [h.s2_output.processing_time for h in self.processing_history if h.s2_output]

        return {
            "total_processed": self._total_processing_count,
            "s2_engagement_rate": self.get_s2_engagement_rate(),
            "avg_s1_time": np.mean(s1_times) if s1_times else 0.0,
            "avg_s2_time": np.mean(s2_times) if s2_times else 0.0,
            "conflict_rate": sum(1 for h in self.processing_history if h.conflict_detected)
            / len(self.processing_history),
            "override_rate": sum(1 for h in self.processing_history if h.override_occurred)
            / len(self.processing_history),
        }

    def train_habit(self, stimulus: np.ndarray, action: Action, repetitions: int = 10):
        """Train a habit through repetition"""
        for _ in range(repetitions):
            self.system1.habit_executor.strengthen(stimulus, action)

    def learn_pattern(self, pattern_id: str, examples: List[np.ndarray]):
        """Learn a new pattern"""
        self.system1.pattern_recognition.learn_pattern(pattern_id, examples)

    def learn_emotional_association(self, stimulus: np.ndarray, threat: float, reward: float):
        """Learn emotional association"""
        self.system1.emotional_valuation.learn_association(stimulus, threat, reward)
