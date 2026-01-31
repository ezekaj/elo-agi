"""
Salience Network - Dynamic Network Switching

The Salience Network acts as a switch between the DMN and ECN:
- Detects when to shift from generation to evaluation
- Detects when to shift from evaluation back to generation
- Manages the creative oscillation between divergent and convergent thinking

Key insight: Creative thought requires DYNAMIC switching, not static activation.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class NetworkState(Enum):
    """Current dominant network"""
    DMN = "default_mode"      # Generating, imagining
    ECN = "executive_control"  # Evaluating, refining
    TRANSITION = "transition"  # Switching between


class SwitchTrigger(Enum):
    """What triggers a network switch"""
    IDEA_GENERATED = "idea_generated"
    EVALUATION_COMPLETE = "evaluation_complete"
    STUCK = "stuck"
    GOAL_PROXIMITY = "goal_proximity"
    TIME_BASED = "time_based"
    NOVELTY_DROP = "novelty_drop"
    QUALITY_THRESHOLD = "quality_threshold"


@dataclass
class NetworkSwitch:
    """Record of a network switch event"""
    from_network: NetworkState
    to_network: NetworkState
    trigger: SwitchTrigger
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NetworkActivity:
    """Current activity levels of each network"""
    dmn_activity: float  # 0-1
    ecn_activity: float  # 0-1
    dominant: NetworkState
    reconfiguration_level: float  # How much networks are changing


class SalienceNetwork:
    """
    Salience Network - orchestrates network switching.

    Active during:
    - Transition moments between generation and evaluation
    - Detection of salient events requiring attention shift
    - Dynamic reconfiguration of network cooperation

    Key insight from research:
    "Generating creative ideas led to significantly higher network
    reconfiguration than generating non-creative ideas"
    """

    def __init__(self,
                 switch_threshold: float = 0.6,
                 min_time_in_state: float = 2.0,  # seconds
                 oscillation_frequency: float = 0.1):  # switches per second
        self.current_state = NetworkState.DMN
        self.switch_history: List[NetworkSwitch] = []
        self.switch_threshold = switch_threshold
        self.min_time_in_state = min_time_in_state
        self.oscillation_frequency = oscillation_frequency

        self._last_switch_time = time.time()
        self._dmn_activity = 0.8
        self._ecn_activity = 0.2

        # Track metrics for switch decisions
        self._recent_novelty_scores: List[float] = []
        self._recent_quality_scores: List[float] = []
        self._ideas_since_last_switch = 0
        self._evaluations_since_last_switch = 0

    def get_network_activity(self) -> NetworkActivity:
        """Get current network activity levels"""
        return NetworkActivity(
            dmn_activity=self._dmn_activity,
            ecn_activity=self._ecn_activity,
            dominant=self.current_state,
            reconfiguration_level=self._compute_reconfiguration()
        )

    def _compute_reconfiguration(self) -> float:
        """
        Compute how much network reconfiguration is happening.

        Higher reconfiguration = more creative processing
        """
        # Recent switch frequency
        recent_switches = [
            s for s in self.switch_history[-20:]
            if time.time() - s.timestamp < 60  # Last minute
        ]
        switch_rate = len(recent_switches) / 60.0

        # Activity balance (more balanced = more reconfiguration)
        balance = 1.0 - abs(self._dmn_activity - self._ecn_activity)

        # Combine
        reconfiguration = (switch_rate / self.oscillation_frequency) * 0.5 + balance * 0.5
        return min(1.0, reconfiguration)

    def should_switch(self,
                      current_metrics: Dict[str, float]) -> Tuple[bool, Optional[SwitchTrigger]]:
        """
        Determine if networks should switch.

        The salience network monitors various signals to detect
        when a switch would be beneficial.
        """
        time_in_state = time.time() - self._last_switch_time

        # Don't switch too quickly
        if time_in_state < self.min_time_in_state:
            return False, None

        # Check various triggers based on current state
        if self.current_state == NetworkState.DMN:
            return self._should_switch_from_dmn(current_metrics)
        elif self.current_state == NetworkState.ECN:
            return self._should_switch_from_ecn(current_metrics)

        return False, None

    def _should_switch_from_dmn(self,
                                metrics: Dict[str, float]) -> Tuple[bool, Optional[SwitchTrigger]]:
        """Check if should switch from DMN to ECN"""

        # Generated enough ideas?
        if metrics.get("ideas_generated", 0) >= metrics.get("target_ideas", 3):
            return True, SwitchTrigger.IDEA_GENERATED

        # Novelty dropping (running out of new ideas)?
        novelty = metrics.get("current_novelty", 0.5)
        self._recent_novelty_scores.append(novelty)
        if len(self._recent_novelty_scores) > 5:
            self._recent_novelty_scores.pop(0)

        if len(self._recent_novelty_scores) >= 3:
            avg_recent = np.mean(self._recent_novelty_scores[-3:])
            avg_earlier = np.mean(self._recent_novelty_scores[:-3]) if len(self._recent_novelty_scores) > 3 else avg_recent
            if avg_recent < avg_earlier * 0.7:  # 30% drop
                return True, SwitchTrigger.NOVELTY_DROP

        # Stuck (no progress)?
        if metrics.get("stuck_iterations", 0) > 5:
            return True, SwitchTrigger.STUCK

        # Time-based (been generating too long)?
        time_in_state = time.time() - self._last_switch_time
        if time_in_state > 30:  # 30 seconds max in DMN
            return True, SwitchTrigger.TIME_BASED

        return False, None

    def _should_switch_from_ecn(self,
                                metrics: Dict[str, float]) -> Tuple[bool, Optional[SwitchTrigger]]:
        """Check if should switch from ECN to DMN"""

        # Evaluation complete?
        if metrics.get("evaluation_complete", False):
            return True, SwitchTrigger.EVALUATION_COMPLETE

        # Found good enough ideas?
        best_score = metrics.get("best_score", 0)
        target_score = metrics.get("target_score", 0.7)

        if best_score >= target_score:
            return True, SwitchTrigger.GOAL_PROXIMITY

        # Quality too low (need more generation)?
        if best_score < 0.3 and metrics.get("ideas_evaluated", 0) > 2:
            return True, SwitchTrigger.QUALITY_THRESHOLD

        # Time-based
        time_in_state = time.time() - self._last_switch_time
        if time_in_state > 20:  # 20 seconds max in ECN
            return True, SwitchTrigger.TIME_BASED

        return False, None

    def execute_switch(self,
                       trigger: SwitchTrigger,
                       context: Optional[Dict[str, Any]] = None) -> NetworkSwitch:
        """
        Execute a network switch.

        This represents the salience network actually triggering
        a transition between DMN and ECN.
        """
        old_state = self.current_state

        # Transition through intermediate state
        self.current_state = NetworkState.TRANSITION

        # Determine new state
        if old_state == NetworkState.DMN:
            new_state = NetworkState.ECN
            self._dmn_activity = 0.3
            self._ecn_activity = 0.8
        else:
            new_state = NetworkState.DMN
            self._dmn_activity = 0.8
            self._ecn_activity = 0.3

        self.current_state = new_state

        # Record switch
        switch = NetworkSwitch(
            from_network=old_state,
            to_network=new_state,
            trigger=trigger,
            timestamp=time.time(),
            context=context or {}
        )

        self.switch_history.append(switch)
        self._last_switch_time = time.time()

        # Reset counters
        self._ideas_since_last_switch = 0
        self._evaluations_since_last_switch = 0

        return switch

    def update_activity(self,
                        dmn_delta: float = 0.0,
                        ecn_delta: float = 0.0):
        """
        Gradually update network activity levels.

        Networks don't switch instantly - there's gradual transition.
        """
        self._dmn_activity = np.clip(self._dmn_activity + dmn_delta, 0, 1)
        self._ecn_activity = np.clip(self._ecn_activity + ecn_delta, 0, 1)

        # Normalize to some extent (total activity bounded)
        total = self._dmn_activity + self._ecn_activity
        if total > 1.5:
            factor = 1.5 / total
            self._dmn_activity *= factor
            self._ecn_activity *= factor

    def get_recommended_action(self) -> str:
        """Get recommended action based on current state"""
        if self.current_state == NetworkState.DMN:
            return "generate"
        elif self.current_state == NetworkState.ECN:
            return "evaluate"
        else:
            return "wait"

    def record_idea_generated(self, novelty_score: float):
        """Record that an idea was generated"""
        self._ideas_since_last_switch += 1
        self._recent_novelty_scores.append(novelty_score)
        if len(self._recent_novelty_scores) > 10:
            self._recent_novelty_scores.pop(0)

    def record_evaluation_complete(self, quality_score: float):
        """Record that an evaluation was completed"""
        self._evaluations_since_last_switch += 1
        self._recent_quality_scores.append(quality_score)
        if len(self._recent_quality_scores) > 10:
            self._recent_quality_scores.pop(0)

    def get_switch_statistics(self) -> Dict[str, Any]:
        """Get statistics about network switching"""
        if not self.switch_history:
            return {"total_switches": 0}

        dmn_to_ecn = sum(1 for s in self.switch_history
                        if s.from_network == NetworkState.DMN)
        ecn_to_dmn = sum(1 for s in self.switch_history
                        if s.from_network == NetworkState.ECN)

        # Time spent in each state
        dmn_time = 0.0
        ecn_time = 0.0

        for i, switch in enumerate(self.switch_history[:-1]):
            duration = self.switch_history[i + 1].timestamp - switch.timestamp
            if switch.to_network == NetworkState.DMN:
                dmn_time += duration
            elif switch.to_network == NetworkState.ECN:
                ecn_time += duration

        # Trigger frequency
        trigger_counts = {}
        for switch in self.switch_history:
            trigger_counts[switch.trigger.value] = trigger_counts.get(switch.trigger.value, 0) + 1

        return {
            "total_switches": len(self.switch_history),
            "dmn_to_ecn": dmn_to_ecn,
            "ecn_to_dmn": ecn_to_dmn,
            "dmn_time": dmn_time,
            "ecn_time": ecn_time,
            "trigger_counts": trigger_counts,
            "current_reconfiguration": self._compute_reconfiguration()
        }

    def force_switch_to(self, target: NetworkState):
        """Force switch to specific network (override automatic)"""
        if self.current_state != target:
            self.execute_switch(
                SwitchTrigger.TIME_BASED,
                context={"forced": True}
            )
