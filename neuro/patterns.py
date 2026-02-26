"""
PatternStore - Learn patterns from successful interactions.

Tracks:
- Query types and successful approaches
- Tool usage patterns
- Success rates per pattern
- Auto-matching for similar queries
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict


@dataclass
class Pattern:
    """A learned pattern from interactions."""
    
    query_type: str
    tools: List[str] = field(default_factory=list)
    approach: str = ""
    success_count: int = 0
    failure_count: int = 0
    avg_confidence: float = 0.5
    last_used: float = field(default_factory=time.time)
    examples: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total
    
    @property
    def confidence(self) -> float:
        """Calculate pattern confidence (success rate weighted by usage)."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        
        # Bayesian smoothing with prior of 0.5
        smoothed_success = self.success_count + 2
        smoothed_total = total + 4
        return smoothed_success / smoothed_total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        """Create from dictionary."""
        return cls(**data)


class PatternStore:
    """
    Learn and match patterns from interactions.
    
    Automatically:
    - Classifies queries into types
    - Tracks which tools/approaches work best
    - Updates success rates based on outcomes
    - Suggests patterns for similar queries
    """

    def __init__(self, path: str = "~/.neuro/patterns.json"):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.patterns: Dict[str, Pattern] = {}
        self.query_history: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load patterns from disk."""
        if self.path.exists():
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                    self.patterns = {
                        k: Pattern.from_dict(v) for k, v in data.get("patterns", {}).items()
                    }
                    self.query_history = data.get("history", [])[-1000:]  # Keep last 1000
            except (json.JSONDecodeError, KeyError):
                self.patterns = {}
                self.query_history = []

    def _save(self):
        """Save patterns to disk."""
        data = {
            "patterns": {k: v.to_dict() for k, v in self.patterns.items()},
            "history": self.query_history[-1000:],
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def _classify_query(self, query: str) -> str:
        """
        Classify query into a type.
        
        Uses simple heuristics for now. Can be enhanced with LLM-based classification.
        """
        query_lower = query.lower()
        
        # Code-related
        if any(word in query_lower for word in ["code", "function", "class", "import", "debug", "fix", "write"]):
            if "test" in query_lower:
                return "code_test"
            return "code"
        
        # File operations
        if any(word in query_lower for word in ["file", "read", "write", "save", "load", "path"]):
            return "file_ops"
        
        # Search/research
        if any(word in query_lower for word in ["search", "find", "look up", "research"]):
            return "search"
        
        # Math/calculation
        if any(word in query_lower for word in ["calculate", "compute", "math", "equation"]):
            return "math"
        
        # Explanation
        if any(word in query_lower for word in ["explain", "what is", "how does", "why"]):
            return "explain"
        
        # Creative
        if any(word in query_lower for word in ["create", "design", "brainstorm", "idea"]):
            return "creative"
        
        # Analysis
        if any(word in query_lower for word in ["analyze", "compare", "review", "evaluate"]):
            return "analysis"
        
        # Default
        return "general"

    def match(self, query: str) -> Optional[Pattern]:
        """
        Find matching pattern for query.
        
        Returns the best matching pattern based on query type.
        """
        query_type = self._classify_query(query)
        return self.patterns.get(query_type)

    def learn(
        self,
        query: str,
        tools_used: List[str],
        approach: str,
        success: bool,
        confidence: float = 0.5,
        response: Optional[str] = None,
    ):
        """
        Learn from an interaction.
        
        Args:
            query: User's query
            tools_used: Tools that were used
            approach: Description of approach taken
            success: Whether the interaction was successful
            confidence: Confidence in the response
            response: The actual response (for examples)
        """
        query_type = self._classify_query(query)
        
        # Get or create pattern
        if query_type not in self.patterns:
            self.patterns[query_type] = Pattern(query_type=query_type)
        
        pattern = self.patterns[query_type]
        
        # Update success/failure counts
        if success:
            pattern.success_count += 1
        else:
            pattern.failure_count += 1
        
        # Update average confidence
        n = pattern.success_count + pattern.failure_count
        pattern.avg_confidence = (
            (pattern.avg_confidence * (n - 1) + confidence) / n
        )
        
        # Update tools (keep most common)
        for tool in tools_used:
            if tool not in pattern.tools:
                pattern.tools.append(tool)
        
        # Update approach if this was successful
        if success and approach:
            pattern.approach = approach
        
        # Update last used
        pattern.last_used = time.time()
        
        # Add example (keep last 3)
        if response and len(pattern.examples) < 3:
            pattern.examples.append(response[:200])
        
        # Record in history
        self.query_history.append({
            "query": query[:100],
            "type": query_type,
            "tools": tools_used,
            "success": success,
            "timestamp": time.time(),
        })
        
        # Save periodically (every 10 interactions)
        if len(self.query_history) % 10 == 0:
            self._save()

    def get_stats(self) -> Dict[str, Any]:
        """Get pattern statistics."""
        if not self.patterns:
            return {
                "total_patterns": 0,
                "avg_success_rate": 0.0,
                "total_interactions": 0,
            }
        
        total_interactions = sum(
            p.success_count + p.failure_count for p in self.patterns.values()
        )
        avg_success_rate = sum(p.success_rate for p in self.patterns.values()) / len(self.patterns)
        
        return {
            "total_patterns": len(self.patterns),
            "avg_success_rate": round(avg_success_rate, 2),
            "total_interactions": total_interactions,
            "by_type": {
                name: {
                    "success_rate": round(p.success_rate, 2),
                    "count": p.success_count + p.failure_count,
                    "tools": p.tools,
                }
                for name, p in sorted(
                    self.patterns.items(),
                    key=lambda x: x[1].success_count + x[1].failure_count,
                    reverse=True,
                )
            },
        }

    def suggest_tools(self, query: str) -> List[str]:
        """Suggest tools based on matched pattern."""
        pattern = self.match(query)
        if pattern:
            return pattern.tools
        return []

    def clear(self):
        """Clear all patterns and history."""
        self.patterns = {}
        self.query_history = []
        self._save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._save()
