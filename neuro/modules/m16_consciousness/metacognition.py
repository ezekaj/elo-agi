"""Metacognition - Thinking about thinking

Neural basis: Anterior prefrontal cortex (aPFC)
Key functions: Monitoring, control, confidence estimation
Key insight: Metacognition may be constitutive of conscious awareness
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class MetaParams:
    """Parameters for metacognition"""
    n_features: int = 50
    confidence_threshold: float = 0.5
    monitoring_sensitivity: float = 0.7
    control_strength: float = 0.5


class ConfidenceEstimator:
    """Estimate confidence in own judgments and decisions"""

    def __init__(self, params: Optional[MetaParams] = None):
        self.params = params or MetaParams()

        # Confidence calibration (learned over time)
        self.calibration = np.ones(self.params.n_features)

        # Recent confidence judgments
        self.confidence_history: List[Tuple[float, bool]] = []

    def estimate_confidence(self, evidence: np.ndarray,
                           decision: np.ndarray) -> Dict:
        """Estimate confidence in a decision given evidence"""
        if len(evidence) != self.params.n_features:
            evidence = np.resize(evidence, self.params.n_features)
        if len(decision) != self.params.n_features:
            decision = np.resize(decision, self.params.n_features)

        # Evidence strength
        evidence_strength = np.mean(np.abs(evidence))

        # Decision consistency with evidence
        consistency = np.dot(evidence, decision) / (
            np.linalg.norm(evidence) * np.linalg.norm(decision) + 1e-8
        )

        # Calibrated confidence
        raw_confidence = (evidence_strength + consistency) / 2
        calibrated_confidence = raw_confidence * np.mean(self.calibration)

        confidence = float(np.clip(calibrated_confidence, 0, 1))

        return {
            "confidence": confidence,
            "evidence_strength": evidence_strength,
            "consistency": consistency,
            "above_threshold": confidence > self.params.confidence_threshold
        }

    def receive_feedback(self, confidence: float, was_correct: bool):
        """Update calibration based on feedback"""
        self.confidence_history.append((confidence, was_correct))

        # Adjust calibration
        if was_correct and confidence < 0.5:
            # Underconfident on correct answer
            self.calibration *= 1.05
        elif not was_correct and confidence > 0.5:
            # Overconfident on incorrect answer
            self.calibration *= 0.95

        self.calibration = np.clip(self.calibration, 0.5, 2.0)

    def get_calibration_quality(self) -> float:
        """Assess how well-calibrated confidence estimates are"""
        if len(self.confidence_history) < 10:
            return 0.5

        # Compare confidence to accuracy
        high_conf = [c for c, correct in self.confidence_history if c > 0.5]
        high_conf_correct = [c for c, correct in self.confidence_history if c > 0.5 and correct]

        if not high_conf:
            return 0.5

        return len(high_conf_correct) / len(high_conf)


class PerformanceMonitor:
    """Monitor own cognitive performance"""

    def __init__(self, params: Optional[MetaParams] = None):
        self.params = params or MetaParams()

        # aPFC activation
        self.apfc_activation = np.zeros(self.params.n_features)

        # Performance tracking
        self.performance_history: List[float] = []
        self.error_history: List[float] = []

        # Current monitoring state
        self.current_effort = 0.5
        self.current_accuracy = 0.5

    def monitor_process(self, process_state: np.ndarray,
                       expected_state: np.ndarray) -> Dict:
        """Monitor ongoing cognitive process"""
        if len(process_state) != self.params.n_features:
            process_state = np.resize(process_state, self.params.n_features)
        if len(expected_state) != self.params.n_features:
            expected_state = np.resize(expected_state, self.params.n_features)

        # aPFC processing
        self.apfc_activation = np.tanh(
            self.apfc_activation * 0.3 +
            (process_state - expected_state) * 0.7 * self.params.monitoring_sensitivity
        )

        # Calculate error signal
        error = np.linalg.norm(process_state - expected_state) / np.sqrt(self.params.n_features)
        self.error_history.append(error)

        # Estimate effort needed
        self.current_effort = min(1.0, error * 2)

        return {
            "error_signal": error,
            "effort_needed": self.current_effort,
            "apfc_activity": np.mean(np.abs(self.apfc_activation)),
            "needs_adjustment": error > 0.3
        }

    def record_outcome(self, accuracy: float):
        """Record performance outcome"""
        self.current_accuracy = accuracy
        self.performance_history.append(accuracy)

        # Limit history
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)

    def get_performance_trend(self, window: int = 10) -> float:
        """Get recent performance trend"""
        if len(self.performance_history) < window:
            return 0.0

        recent = self.performance_history[-window:]
        older = self.performance_history[-2*window:-window] if len(self.performance_history) >= 2*window else self.performance_history[:window]

        return np.mean(recent) - np.mean(older)


class MetacognitiveSystem:
    """Integrated metacognitive system"""

    def __init__(self, params: Optional[MetaParams] = None):
        self.params = params or MetaParams()

        self.confidence = ConfidenceEstimator(params)
        self.monitor = PerformanceMonitor(params)

        # Control signals
        self.control_signal = np.zeros(self.params.n_features)

        # Strategy selection
        self.current_strategy = "default"
        self.strategy_effectiveness: Dict[str, float] = {"default": 0.5}

    def evaluate_decision(self, evidence: np.ndarray,
                         decision: np.ndarray) -> Dict:
        """Metacognitive evaluation of a decision"""
        conf_result = self.confidence.estimate_confidence(evidence, decision)

        # Generate control signal based on confidence
        if conf_result["confidence"] < self.params.confidence_threshold:
            # Low confidence -> need more processing
            self.control_signal = np.ones(self.params.n_features) * self.params.control_strength
        else:
            self.control_signal = np.zeros(self.params.n_features)

        return {
            "confidence": conf_result,
            "control_signal_strength": np.mean(self.control_signal),
            "recommend_continue": conf_result["confidence"] < self.params.confidence_threshold
        }

    def monitor_task(self, current_state: np.ndarray,
                    target_state: np.ndarray) -> Dict:
        """Monitor task progress"""
        monitor_result = self.monitor.monitor_process(current_state, target_state)

        # Adjust strategy if needed
        if monitor_result["needs_adjustment"]:
            self._consider_strategy_change()

        return monitor_result

    def _consider_strategy_change(self):
        """Consider changing cognitive strategy"""
        current_eff = self.strategy_effectiveness.get(self.current_strategy, 0.5)

        # If current strategy underperforming, consider switch
        if current_eff < 0.4:
            # Find better strategy
            best_strategy = max(self.strategy_effectiveness.items(), key=lambda x: x[1])[0]
            if best_strategy != self.current_strategy:
                self.current_strategy = best_strategy

    def receive_feedback(self, was_correct: bool, confidence_given: float):
        """Receive feedback on decision"""
        self.confidence.receive_feedback(confidence_given, was_correct)

        # Update strategy effectiveness
        current_eff = self.strategy_effectiveness.get(self.current_strategy, 0.5)
        if was_correct:
            self.strategy_effectiveness[self.current_strategy] = min(1.0, current_eff + 0.05)
        else:
            self.strategy_effectiveness[self.current_strategy] = max(0.0, current_eff - 0.05)

        self.monitor.record_outcome(1.0 if was_correct else 0.0)

    def introspect_confidence(self) -> Dict:
        """Introspect on own confidence calibration"""
        return {
            "calibration_quality": self.confidence.get_calibration_quality(),
            "recent_confidence": [c for c, _ in self.confidence.confidence_history[-10:]],
            "recent_accuracy": self.monitor.performance_history[-10:] if self.monitor.performance_history else [],
            "performance_trend": self.monitor.get_performance_trend()
        }

    def get_metacognitive_state(self) -> Dict:
        """Get metacognitive state"""
        return {
            "current_strategy": self.current_strategy,
            "strategy_effectiveness": self.strategy_effectiveness.copy(),
            "current_effort": self.monitor.current_effort,
            "current_accuracy": self.monitor.current_accuracy,
            "apfc_activity": np.mean(np.abs(self.monitor.apfc_activation)),
            "control_signal_strength": np.mean(self.control_signal),
            "calibration_quality": self.confidence.get_calibration_quality()
        }
