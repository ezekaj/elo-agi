"""
Coherence Checker: Ensures belief consistency across modules.

Detects logical inconsistencies, contradictions, and belief conflicts,
maintaining a coherent world model across all cognitive processes.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import numpy as np

from .shared_space import SemanticEmbedding


class InconsistencyType(Enum):
    """Types of inconsistencies."""

    CONTRADICTION = "contradiction"  # Direct logical contradiction
    TENSION = "tension"  # Soft conflict, not full contradiction
    OUTDATED = "outdated"  # Belief based on old information
    CIRCULAR = "circular"  # Circular dependency
    INCOMPLETE = "incomplete"  # Missing required beliefs
    IMPLAUSIBLE = "implausible"  # Statistically unlikely combination


class InconsistencySeverity(Enum):
    """Severity levels for inconsistencies."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Belief:
    """A belief held by the system."""

    belief_id: str
    content: SemanticEmbedding
    source_module: str
    confidence: float
    timestamp: float
    supporting_beliefs: List[str] = field(default_factory=list)
    contradicting_beliefs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def similarity(self, other: "Belief") -> float:
        """Compute similarity with another belief."""
        return self.content.similarity(other.content)


@dataclass
class Inconsistency:
    """An inconsistency detected in the belief network."""

    inconsistency_id: int
    inconsistency_type: InconsistencyType
    severity: InconsistencySeverity
    beliefs_involved: List[str]
    description: str
    suggested_resolution: Optional[str] = None
    detected_at: float = 0.0
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoherenceReport:
    """Report on overall belief coherence."""

    timestamp: float
    n_beliefs: int
    n_inconsistencies: int
    coherence_score: float  # 0-1, higher is more coherent
    inconsistencies: List[Inconsistency]
    belief_clusters: List[List[str]]
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BeliefNetwork:
    """
    Network of beliefs with dependency tracking.

    Maintains relationships between beliefs for coherence checking.
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

        # Beliefs indexed by ID
        self._beliefs: Dict[str, Belief] = {}

        # Adjacency for support/contradiction relationships
        self._supports: Dict[str, Set[str]] = {}  # A supports B
        self._contradicts: Dict[str, Set[str]] = {}  # A contradicts B

        # Belief counter
        self._belief_counter = 0

    def add_belief(
        self,
        content: SemanticEmbedding,
        source_module: str,
        confidence: float = 1.0,
        belief_id: Optional[str] = None,
    ) -> Belief:
        """Add a belief to the network."""
        if belief_id is None:
            self._belief_counter += 1
            belief_id = f"belief_{self._belief_counter}"

        belief = Belief(
            belief_id=belief_id,
            content=content,
            source_module=source_module,
            confidence=confidence,
            timestamp=float(self._belief_counter),
        )

        self._beliefs[belief_id] = belief
        self._supports[belief_id] = set()
        self._contradicts[belief_id] = set()

        # Auto-detect relationships with existing beliefs
        self._update_relationships(belief)

        return belief

    def _update_relationships(self, new_belief: Belief) -> None:
        """Update support/contradiction relationships for new belief."""
        for existing_id, existing in self._beliefs.items():
            if existing_id == new_belief.belief_id:
                continue

            similarity = new_belief.similarity(existing)

            if similarity > self.similarity_threshold:
                # High similarity = support
                self._supports[new_belief.belief_id].add(existing_id)
                self._supports[existing_id].add(new_belief.belief_id)
                new_belief.supporting_beliefs.append(existing_id)
                existing.supporting_beliefs.append(new_belief.belief_id)

            elif similarity < -self.similarity_threshold + 1:
                # Low/negative similarity might indicate contradiction
                # (using cosine, negative means opposite direction)
                pass  # Need more sophisticated contradiction detection

    def add_support(self, belief_a: str, belief_b: str) -> None:
        """Explicitly mark that belief A supports belief B."""
        if belief_a in self._beliefs and belief_b in self._beliefs:
            self._supports[belief_a].add(belief_b)
            self._beliefs[belief_a].supporting_beliefs.append(belief_b)

    def add_contradiction(self, belief_a: str, belief_b: str) -> None:
        """Explicitly mark that belief A contradicts belief B."""
        if belief_a in self._beliefs and belief_b in self._beliefs:
            self._contradicts[belief_a].add(belief_b)
            self._contradicts[belief_b].add(belief_a)
            self._beliefs[belief_a].contradicting_beliefs.append(belief_b)
            self._beliefs[belief_b].contradicting_beliefs.append(belief_a)

    def get_belief(self, belief_id: str) -> Optional[Belief]:
        """Get a belief by ID."""
        return self._beliefs.get(belief_id)

    def remove_belief(self, belief_id: str) -> bool:
        """Remove a belief from the network."""
        if belief_id not in self._beliefs:
            return False

        # Clean up relationships
        for other_id in self._supports.get(belief_id, set()):
            if other_id in self._supports:
                self._supports[other_id].discard(belief_id)

        for other_id in self._contradicts.get(belief_id, set()):
            if other_id in self._contradicts:
                self._contradicts[other_id].discard(belief_id)

        del self._beliefs[belief_id]
        del self._supports[belief_id]
        del self._contradicts[belief_id]

        return True

    def get_supporting(self, belief_id: str) -> List[Belief]:
        """Get beliefs that support the given belief."""
        supporting_ids = self._supports.get(belief_id, set())
        return [self._beliefs[bid] for bid in supporting_ids if bid in self._beliefs]

    def get_contradicting(self, belief_id: str) -> List[Belief]:
        """Get beliefs that contradict the given belief."""
        contradicting_ids = self._contradicts.get(belief_id, set())
        return [self._beliefs[bid] for bid in contradicting_ids if bid in self._beliefs]

    def find_clusters(self) -> List[List[str]]:
        """Find clusters of mutually supporting beliefs."""
        visited = set()
        clusters = []

        for belief_id in self._beliefs:
            if belief_id in visited:
                continue

            # BFS to find connected component
            cluster = []
            queue = [belief_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                cluster.append(current)

                # Add supporters
                for supporter in self._supports.get(current, set()):
                    if supporter not in visited:
                        queue.append(supporter)

            if cluster:
                clusters.append(cluster)

        return clusters

    def get_all_beliefs(self) -> List[Belief]:
        """Get all beliefs in the network."""
        return list(self._beliefs.values())

    def get_beliefs_by_module(self, module: str) -> List[Belief]:
        """Get all beliefs from a specific module."""
        return [b for b in self._beliefs.values() if b.source_module == module]


class CoherenceChecker:
    """
    Checks and maintains coherence across the belief network.

    Detects inconsistencies and provides recommendations for resolution.
    """

    def __init__(
        self,
        contradiction_threshold: float = 0.3,
        staleness_threshold: float = 1000.0,
        random_seed: Optional[int] = None,
    ):
        self.contradiction_threshold = contradiction_threshold
        self.staleness_threshold = staleness_threshold
        self._rng = np.random.default_rng(random_seed)

        # Belief network
        self.network = BeliefNetwork()

        # Detected inconsistencies
        self._inconsistencies: List[Inconsistency] = []
        self._inconsistency_counter = 0

        # Coherence history
        self._coherence_history: List[float] = []

        # Known contradictory concept pairs (learned)
        self._known_contradictions: Set[Tuple[str, str]] = set()

    def add_belief(
        self,
        content: SemanticEmbedding,
        source_module: str,
        confidence: float = 1.0,
    ) -> Tuple[Belief, List[Inconsistency]]:
        """
        Add a belief and check for inconsistencies.

        Returns:
            (added_belief, new_inconsistencies)
        """
        belief = self.network.add_belief(content, source_module, confidence)
        inconsistencies = self._check_belief_consistency(belief)

        return belief, inconsistencies

    def _check_belief_consistency(self, belief: Belief) -> List[Inconsistency]:
        """Check if a new belief is consistent with existing beliefs."""
        inconsistencies = []

        for other_id, other in list(self.network._beliefs.items()):
            if other_id == belief.belief_id:
                continue

            # Check for contradiction
            similarity = belief.similarity(other)

            if similarity < self.contradiction_threshold:
                # Potential contradiction
                # Check if both have high confidence
                if belief.confidence > 0.5 and other.confidence > 0.5:
                    self._inconsistency_counter += 1
                    inconsistency = Inconsistency(
                        inconsistency_id=self._inconsistency_counter,
                        inconsistency_type=InconsistencyType.CONTRADICTION,
                        severity=self._compute_severity(belief, other, similarity),
                        beliefs_involved=[belief.belief_id, other_id],
                        description=f"Belief from {belief.source_module} contradicts belief from {other.source_module}",
                        suggested_resolution=self._suggest_resolution(belief, other),
                        detected_at=belief.timestamp,
                    )
                    inconsistencies.append(inconsistency)
                    self._inconsistencies.append(inconsistency)

                    # Mark contradiction in network
                    self.network.add_contradiction(belief.belief_id, other_id)

            elif 0.3 < similarity < 0.6:
                # Tension (not full contradiction but not agreement)
                if belief.confidence > 0.7 and other.confidence > 0.7:
                    self._inconsistency_counter += 1
                    inconsistency = Inconsistency(
                        inconsistency_id=self._inconsistency_counter,
                        inconsistency_type=InconsistencyType.TENSION,
                        severity=InconsistencySeverity.LOW,
                        beliefs_involved=[belief.belief_id, other_id],
                        description=f"Tension between beliefs from {belief.source_module} and {other.source_module}",
                        detected_at=belief.timestamp,
                    )
                    inconsistencies.append(inconsistency)
                    self._inconsistencies.append(inconsistency)

        return inconsistencies

    def _compute_severity(
        self,
        belief1: Belief,
        belief2: Belief,
        similarity: float,
    ) -> InconsistencySeverity:
        """Compute severity of inconsistency."""
        # Higher confidence beliefs in contradiction = more severe
        confidence_factor = (belief1.confidence + belief2.confidence) / 2

        # Lower similarity = stronger contradiction
        contradiction_strength = 1 - similarity

        score = confidence_factor * contradiction_strength

        if score > 0.8:
            return InconsistencySeverity.CRITICAL
        elif score > 0.6:
            return InconsistencySeverity.HIGH
        elif score > 0.4:
            return InconsistencySeverity.MEDIUM
        else:
            return InconsistencySeverity.LOW

    def _suggest_resolution(self, belief1: Belief, belief2: Belief) -> str:
        """Suggest how to resolve contradiction."""
        if belief1.confidence > belief2.confidence + 0.2:
            return f"Reduce confidence in or remove belief from {belief2.source_module}"
        elif belief2.confidence > belief1.confidence + 0.2:
            return f"Reduce confidence in or remove belief from {belief1.source_module}"
        elif belief1.timestamp > belief2.timestamp:
            return f"Older belief from {belief2.source_module} may be outdated"
        else:
            return "Seek additional evidence to resolve contradiction"

    def check_staleness(self, current_time: float) -> List[Inconsistency]:
        """Check for stale/outdated beliefs."""
        inconsistencies = []

        for belief in self.network.get_all_beliefs():
            age = current_time - belief.timestamp

            if age > self.staleness_threshold:
                self._inconsistency_counter += 1
                inconsistency = Inconsistency(
                    inconsistency_id=self._inconsistency_counter,
                    inconsistency_type=InconsistencyType.OUTDATED,
                    severity=InconsistencySeverity.LOW,
                    beliefs_involved=[belief.belief_id],
                    description=f"Belief from {belief.source_module} may be outdated (age: {age:.0f})",
                    suggested_resolution="Refresh belief with current information",
                    detected_at=current_time,
                )
                inconsistencies.append(inconsistency)
                self._inconsistencies.append(inconsistency)

        return inconsistencies

    def check_circular_dependencies(self) -> List[Inconsistency]:
        """Check for circular belief dependencies."""
        inconsistencies = []

        # DFS to detect cycles
        visited = set()
        rec_stack = set()

        def dfs(belief_id: str, path: List[str]) -> Optional[List[str]]:
            visited.add(belief_id)
            rec_stack.add(belief_id)

            for supported in self.network._supports.get(belief_id, set()):
                if supported not in visited:
                    cycle = dfs(supported, path + [belief_id])
                    if cycle:
                        return cycle
                elif supported in rec_stack:
                    # Found cycle
                    return path + [belief_id, supported]

            rec_stack.remove(belief_id)
            return None

        for belief_id in self.network._beliefs:
            if belief_id not in visited:
                cycle = dfs(belief_id, [])
                if cycle:
                    self._inconsistency_counter += 1
                    inconsistency = Inconsistency(
                        inconsistency_id=self._inconsistency_counter,
                        inconsistency_type=InconsistencyType.CIRCULAR,
                        severity=InconsistencySeverity.MEDIUM,
                        beliefs_involved=cycle,
                        description=f"Circular dependency detected involving {len(cycle)} beliefs",
                        suggested_resolution="Break circular dependency by removing weakest link",
                    )
                    inconsistencies.append(inconsistency)
                    self._inconsistencies.append(inconsistency)

        return inconsistencies

    def compute_coherence_score(self) -> float:
        """
        Compute overall coherence score.

        Returns value between 0 (incoherent) and 1 (fully coherent).
        """
        beliefs = self.network.get_all_beliefs()
        if len(beliefs) < 2:
            return 1.0

        # Factors affecting coherence:
        # 1. Ratio of supporting vs contradicting relationships
        n_supports = sum(len(s) for s in self.network._supports.values()) / 2
        n_contradicts = sum(len(c) for c in self.network._contradicts.values()) / 2

        if n_supports + n_contradicts > 0:
            support_ratio = n_supports / (n_supports + n_contradicts)
        else:
            support_ratio = 1.0

        # 2. Number of unresolved inconsistencies
        unresolved = [i for i in self._inconsistencies if not i.resolved]
        inconsistency_penalty = len(unresolved) / max(1, len(beliefs))
        inconsistency_factor = max(0, 1 - inconsistency_penalty)

        # 3. Average belief similarity (should be moderately high)
        total_similarity = 0.0
        n_pairs = 0
        for i, b1 in enumerate(beliefs):
            for b2 in beliefs[i + 1 :]:
                total_similarity += b1.similarity(b2)
                n_pairs += 1

        if n_pairs > 0:
            avg_similarity = total_similarity / n_pairs
            # Optimal similarity is around 0.5-0.7 (not too homogeneous, not too scattered)
            similarity_factor = 1 - abs(avg_similarity - 0.6) / 0.6
        else:
            similarity_factor = 1.0

        # Combine factors
        coherence = 0.4 * support_ratio + 0.4 * inconsistency_factor + 0.2 * similarity_factor

        self._coherence_history.append(coherence)

        return float(np.clip(coherence, 0, 1))

    def generate_report(self, current_time: Optional[float] = None) -> CoherenceReport:
        """Generate comprehensive coherence report."""
        if current_time is None:
            current_time = float(self._inconsistency_counter)

        # Check for staleness
        self.check_staleness(current_time)

        # Check for circular dependencies
        self.check_circular_dependencies()

        # Get all unresolved inconsistencies
        unresolved = [i for i in self._inconsistencies if not i.resolved]

        # Compute coherence
        coherence_score = self.compute_coherence_score()

        # Find belief clusters
        clusters = self.network.find_clusters()

        # Generate recommendations
        recommendations = self._generate_recommendations(unresolved)

        return CoherenceReport(
            timestamp=current_time,
            n_beliefs=len(self.network._beliefs),
            n_inconsistencies=len(unresolved),
            coherence_score=coherence_score,
            inconsistencies=unresolved,
            belief_clusters=clusters,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        inconsistencies: List[Inconsistency],
    ) -> List[str]:
        """Generate recommendations for improving coherence."""
        recommendations = []

        # Count inconsistency types
        type_counts: Dict[InconsistencyType, int] = {}
        for inc in inconsistencies:
            type_counts[inc.inconsistency_type] = type_counts.get(inc.inconsistency_type, 0) + 1

        if type_counts.get(InconsistencyType.CONTRADICTION, 0) > 2:
            recommendations.append(
                "Multiple contradictions detected. Consider belief revision or source reliability update."
            )

        if type_counts.get(InconsistencyType.OUTDATED, 0) > 5:
            recommendations.append("Many beliefs are outdated. Schedule belief refresh cycle.")

        if type_counts.get(InconsistencyType.CIRCULAR, 0) > 0:
            recommendations.append("Circular dependencies found. Review belief support structure.")

        # Check for module-specific issues
        module_issues: Dict[str, int] = {}
        for inc in inconsistencies:
            for bid in inc.beliefs_involved:
                belief = self.network.get_belief(bid)
                if belief:
                    module_issues[belief.source_module] = (
                        module_issues.get(belief.source_module, 0) + 1
                    )

        for module, count in module_issues.items():
            if count > 3:
                recommendations.append(
                    f"Module '{module}' involved in {count} inconsistencies. Review module reliability."
                )

        return recommendations

    def resolve_inconsistency(
        self,
        inconsistency_id: int,
        action: str = "acknowledge",
    ) -> bool:
        """Mark an inconsistency as resolved."""
        for inc in self._inconsistencies:
            if inc.inconsistency_id == inconsistency_id:
                inc.resolved = True
                inc.metadata["resolution_action"] = action
                return True
        return False

    def get_module_coherence(self, module: str) -> float:
        """Get coherence score for beliefs from a specific module."""
        beliefs = self.network.get_beliefs_by_module(module)
        if len(beliefs) < 2:
            return 1.0

        # Check internal consistency
        total_similarity = 0.0
        n_pairs = 0

        for i, b1 in enumerate(beliefs):
            for b2 in beliefs[i + 1 :]:
                total_similarity += b1.similarity(b2)
                n_pairs += 1

        if n_pairs == 0:
            return 1.0

        return float(total_similarity / n_pairs)

    def statistics(self) -> Dict[str, Any]:
        """Get coherence checker statistics."""
        unresolved = [i for i in self._inconsistencies if not i.resolved]

        type_distribution = {}
        for inc in self._inconsistencies:
            t = inc.inconsistency_type.value
            type_distribution[t] = type_distribution.get(t, 0) + 1

        return {
            "n_beliefs": len(self.network._beliefs),
            "n_inconsistencies_total": len(self._inconsistencies),
            "n_inconsistencies_unresolved": len(unresolved),
            "coherence_score": self.compute_coherence_score(),
            "n_belief_clusters": len(self.network.find_clusters()),
            "inconsistency_types": type_distribution,
            "coherence_history_length": len(self._coherence_history),
        }
