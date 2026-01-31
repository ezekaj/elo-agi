"""Introspection - Access to own mental states

Core function: Access and report on own mental processes
Enables self-knowledge and verbal report of conscious contents
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Any


@dataclass
class IntrospectionParams:
    """Parameters for introspection"""
    n_features: int = 50
    access_threshold: float = 0.4
    report_detail: float = 0.7
    reflection_depth: int = 3


class MentalStateAccessor:
    """Access representations of own mental states"""

    def __init__(self, params: Optional[IntrospectionParams] = None):
        self.params = params or IntrospectionParams()

        # Current mental states (what's accessible)
        self.accessible_states: Dict[str, np.ndarray] = {}

        # State labels
        self.state_labels: Dict[str, str] = {}

        # Access history
        self.access_history: List[Dict] = []

    def register_state(self, state_name: str, state: np.ndarray, label: str = ""):
        """Register a mental state for potential access"""
        if len(state) != self.params.n_features:
            state = np.resize(state, self.params.n_features)

        self.accessible_states[state_name] = state.copy()
        if label:
            self.state_labels[state_name] = label

    def access_state(self, state_name: str) -> Dict:
        """Access a mental state introspectively"""
        if state_name not in self.accessible_states:
            return {"success": False, "reason": "state not found"}

        state = self.accessible_states[state_name]

        # Determine accessibility (some states harder to access)
        activation_strength = np.mean(np.abs(state))

        if activation_strength < self.params.access_threshold:
            return {
                "success": False,
                "reason": "state too weak to access",
                "activation": activation_strength
            }

        # Successful access
        access_record = {
            "state_name": state_name,
            "activation": activation_strength,
            "label": self.state_labels.get(state_name, ""),
            "time": len(self.access_history)
        }
        self.access_history.append(access_record)

        return {
            "success": True,
            "state": state.copy(),
            "activation": activation_strength,
            "label": self.state_labels.get(state_name, "")
        }

    def scan_accessible_states(self) -> List[str]:
        """Scan for currently accessible states"""
        accessible = []
        for name, state in self.accessible_states.items():
            if np.mean(np.abs(state)) >= self.params.access_threshold:
                accessible.append(name)
        return accessible

    def clear_state(self, state_name: str):
        """Clear a state from accessibility"""
        if state_name in self.accessible_states:
            del self.accessible_states[state_name]


class SelfReflection:
    """Reflect on own cognitive processes"""

    def __init__(self, params: Optional[IntrospectionParams] = None):
        self.params = params or IntrospectionParams()

        # Reflection buffer
        self.reflection_buffer: List[Dict] = []

        # Insights gained
        self.insights: List[str] = []

        # Current focus of reflection
        self.reflection_focus: Optional[str] = None

    def begin_reflection(self, focus: str):
        """Begin reflecting on a topic"""
        self.reflection_focus = focus
        self.reflection_buffer = []

    def add_observation(self, observation: str, evidence: np.ndarray):
        """Add an observation to reflection"""
        if len(evidence) != self.params.n_features:
            evidence = np.resize(evidence, self.params.n_features)

        self.reflection_buffer.append({
            "observation": observation,
            "evidence": evidence.copy(),
            "depth": len(self.reflection_buffer)
        })

    def synthesize_reflection(self) -> Dict:
        """Synthesize reflection into insight"""
        if not self.reflection_buffer:
            return {"insight": None, "confidence": 0}

        # Combine observations
        observations = [r["observation"] for r in self.reflection_buffer]
        evidence = np.mean([r["evidence"] for r in self.reflection_buffer], axis=0)

        # Generate insight based on reflection depth
        depth = len(self.reflection_buffer)
        confidence = min(1.0, depth / self.params.reflection_depth)

        insight = f"Reflection on {self.reflection_focus}: {len(observations)} observations"
        self.insights.append(insight)

        return {
            "insight": insight,
            "confidence": confidence,
            "observations_count": len(observations),
            "focus": self.reflection_focus,
            "evidence_summary": np.mean(evidence)
        }

    def get_insights(self) -> List[str]:
        """Get accumulated insights"""
        return self.insights.copy()


class IntrospectionSystem:
    """Integrated introspection system"""

    def __init__(self, params: Optional[IntrospectionParams] = None):
        self.params = params or IntrospectionParams()

        self.accessor = MentalStateAccessor(params)
        self.reflection = SelfReflection(params)

        # Introspective reports
        self.reports: List[Dict] = []

    def register_mental_state(self, name: str, state: np.ndarray, label: str = ""):
        """Register a state for introspective access"""
        self.accessor.register_state(name, state, label)

    def introspect(self, state_name: str) -> Dict:
        """Perform introspection on a mental state"""
        access_result = self.accessor.access_state(state_name)

        if not access_result["success"]:
            return access_result

        # Generate report
        report = self._generate_report(state_name, access_result)
        self.reports.append(report)

        return {
            "access": access_result,
            "report": report
        }

    def _generate_report(self, state_name: str, access_result: Dict) -> Dict:
        """Generate introspective report"""
        state = access_result["state"]

        # Characterize the state
        valence = np.mean(state)  # Positive/negative
        intensity = np.mean(np.abs(state))
        complexity = np.std(state)

        report = {
            "state_name": state_name,
            "label": access_result.get("label", ""),
            "valence": "positive" if valence > 0 else "negative",
            "intensity": "high" if intensity > 0.5 else "low",
            "complexity": "complex" if complexity > 0.3 else "simple",
            "detail_level": self.params.report_detail
        }

        return report

    def reflect_on(self, topic: str) -> Dict:
        """Begin reflection on a topic"""
        self.reflection.begin_reflection(topic)

        # Gather relevant states
        accessible = self.accessor.scan_accessible_states()

        for state_name in accessible[:3]:  # Reflect on up to 3 states
            access = self.accessor.access_state(state_name)
            if access["success"]:
                self.reflection.add_observation(
                    f"Observed {state_name}",
                    access["state"]
                )

        return self.reflection.synthesize_reflection()

    def get_conscious_contents(self) -> Dict:
        """Get currently conscious (accessible) mental contents"""
        accessible = self.accessor.scan_accessible_states()

        contents = {}
        for name in accessible:
            result = self.accessor.access_state(name)
            if result["success"]:
                contents[name] = {
                    "label": result.get("label", ""),
                    "activation": result["activation"]
                }

        return {
            "accessible_states": contents,
            "count": len(contents)
        }

    def get_introspection_state(self) -> Dict:
        """Get introspection system state"""
        return {
            "registered_states": list(self.accessor.accessible_states.keys()),
            "accessible_count": len(self.accessor.scan_accessible_states()),
            "reports_generated": len(self.reports),
            "insights": self.reflection.get_insights(),
            "current_reflection_focus": self.reflection.reflection_focus
        }
