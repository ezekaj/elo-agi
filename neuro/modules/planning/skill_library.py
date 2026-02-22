"""
Skill Library

Manages reusable skills (parameterized options) for transfer and composition.
Skills can be learned, stored, retrieved, composed, and transferred across domains.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Set
from enum import Enum
import numpy as np


class SkillType(Enum):
    """Types of skills."""

    PRIMITIVE = "primitive"
    COMPOSITE = "composite"
    PARAMETERIZED = "parameterized"
    ABSTRACT = "abstract"


@dataclass
class SkillMetadata:
    """Metadata about a skill."""

    created_at: float = 0.0
    last_used: float = 0.0
    use_count: int = 0
    success_count: int = 0
    average_steps: float = 0.0
    average_reward: float = 0.0
    domains: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)

    @property
    def success_rate(self) -> float:
        """Compute success rate."""
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count

    def record_use(
        self,
        success: bool,
        steps: int,
        reward: float,
        timestamp: float,
    ) -> None:
        """Record a skill usage."""
        self.use_count += 1
        if success:
            self.success_count += 1
        self.last_used = timestamp

        n = self.use_count
        self.average_steps = self.average_steps * (n - 1) / n + steps / n
        self.average_reward = self.average_reward * (n - 1) / n + reward / n


@dataclass
class Skill:
    """
    A reusable skill that can be executed, composed, and transferred.

    Skills are more abstract than options - they represent capabilities
    that can be instantiated with parameters and adapted to contexts.
    """

    name: str
    skill_type: SkillType
    preconditions: Callable[[Any], bool]
    effects: Callable[[Any], Any]
    policy: Callable[[Any, Optional[Dict[str, Any]]], Any]
    termination: Callable[[Any, Optional[Dict[str, Any]]], bool]

    parameters: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    metadata: SkillMetadata = field(default_factory=SkillMetadata)
    parent_skills: List[str] = field(default_factory=list)
    child_skills: List[str] = field(default_factory=list)

    def can_execute(self, state: Any, params: Optional[Dict[str, Any]] = None) -> bool:
        """Check if skill can be executed in state."""
        return self.preconditions(state)

    def execute_step(
        self,
        state: Any,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, bool]:
        """
        Execute one step of the skill.

        Returns (action, terminated).
        """
        merged_params = {**self.parameters}
        if params:
            merged_params.update(params)

        action = self.policy(state, merged_params)
        terminated = self.termination(state, merged_params)

        return action, terminated

    def predict_effects(self, state: Any) -> Any:
        """Predict the effects of executing the skill."""
        return self.effects(state)

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a skill parameter."""
        self.parameters[name] = value

    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get a skill parameter."""
        return self.parameters.get(name, default)

    def similarity(self, other: "Skill") -> float:
        """Compute similarity to another skill based on embeddings."""
        if self.embedding is None or other.embedding is None:
            return 0.0

        norm1 = np.linalg.norm(self.embedding)
        norm2 = np.linalg.norm(other.embedding)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        return float(np.dot(self.embedding, other.embedding) / (norm1 * norm2))


@dataclass
class SkillComposer:
    """Composes multiple skills into composite skills."""

    def __init__(self, random_seed: Optional[int] = None):
        self._rng = np.random.default_rng(random_seed)

    def sequence(self, skills: List[Skill], name: str) -> Skill:
        """
        Create a sequential composite skill.

        Executes skills in order, each to completion.
        """
        current_skill_idx = [0]
        skill_done = [False]

        def combined_preconditions(state: Any) -> bool:
            if not skills:
                return False
            return skills[0].can_execute(state)

        def combined_effects(state: Any) -> Any:
            for skill in skills:
                state = skill.effects(state)
            return state

        def combined_policy(state: Any, params: Optional[Dict] = None) -> Any:
            if current_skill_idx[0] >= len(skills):
                return None

            current = skills[current_skill_idx[0]]
            action, done = current.execute_step(state, params)

            if done:
                skill_done[0] = True

            return action

        def combined_termination(state: Any, params: Optional[Dict] = None) -> bool:
            if skill_done[0]:
                skill_done[0] = False
                current_skill_idx[0] += 1
                if current_skill_idx[0] >= len(skills):
                    current_skill_idx[0] = 0
                    return True
            return False

        composite = Skill(
            name=name,
            skill_type=SkillType.COMPOSITE,
            preconditions=combined_preconditions,
            effects=combined_effects,
            policy=combined_policy,
            termination=combined_termination,
            child_skills=[s.name for s in skills],
        )

        if all(s.embedding is not None for s in skills):
            embeddings = [s.embedding for s in skills]
            composite.embedding = np.mean(embeddings, axis=0)

        return composite

    def parallel(self, skills: List[Skill], name: str) -> Skill:
        """
        Create a parallel composite skill.

        Executes all skills simultaneously, terminates when all complete.
        """
        skill_states = {}

        def combined_preconditions(state: Any) -> bool:
            return all(s.can_execute(state) for s in skills)

        def combined_effects(state: Any) -> Any:
            for skill in skills:
                state = skill.effects(state)
            return state

        def combined_policy(state: Any, params: Optional[Dict] = None) -> List[Any]:
            actions = []
            for skill in skills:
                if skill.name not in skill_states or not skill_states[skill.name]:
                    action, done = skill.execute_step(state, params)
                    skill_states[skill.name] = done
                    actions.append(action)
            return actions

        def combined_termination(state: Any, params: Optional[Dict] = None) -> bool:
            if all(skill_states.get(s.name, False) for s in skills):
                skill_states.clear()
                return True
            return False

        composite = Skill(
            name=name,
            skill_type=SkillType.COMPOSITE,
            preconditions=combined_preconditions,
            effects=combined_effects,
            policy=combined_policy,
            termination=combined_termination,
            child_skills=[s.name for s in skills],
        )

        return composite

    def conditional(
        self,
        condition: Callable[[Any], bool],
        if_skill: Skill,
        else_skill: Skill,
        name: str,
    ) -> Skill:
        """
        Create a conditional skill that branches based on state.
        """
        active_skill = [None]

        def combined_preconditions(state: Any) -> bool:
            return if_skill.can_execute(state) or else_skill.can_execute(state)

        def combined_effects(state: Any) -> Any:
            if condition(state):
                return if_skill.effects(state)
            return else_skill.effects(state)

        def combined_policy(state: Any, params: Optional[Dict] = None) -> Any:
            if active_skill[0] is None:
                active_skill[0] = if_skill if condition(state) else else_skill

            action, _ = active_skill[0].execute_step(state, params)
            return action

        def combined_termination(state: Any, params: Optional[Dict] = None) -> bool:
            if active_skill[0] is None:
                return False

            _, done = active_skill[0].execute_step(state, params)
            if done:
                active_skill[0] = None
                return True
            return False

        composite = Skill(
            name=name,
            skill_type=SkillType.COMPOSITE,
            preconditions=combined_preconditions,
            effects=combined_effects,
            policy=combined_policy,
            termination=combined_termination,
            child_skills=[if_skill.name, else_skill.name],
        )

        return composite


class SkillLibrary:
    """
    Library for storing, retrieving, and managing skills.

    Supports skill registration, similarity-based retrieval,
    composition, and cross-domain transfer.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        random_seed: Optional[int] = None,
    ):
        self.embedding_dim = embedding_dim
        self._rng = np.random.default_rng(random_seed)

        self._skills: Dict[str, Skill] = {}
        self._domain_skills: Dict[str, Set[str]] = {}
        self._tag_skills: Dict[str, Set[str]] = {}
        self._composer = SkillComposer(random_seed)

        self._current_time = 0.0

    def register_skill(self, skill: Skill) -> str:
        """
        Register a skill in the library.

        Returns the skill name.
        """
        if skill.embedding is None:
            skill.embedding = self._rng.normal(0, 1, self.embedding_dim)
            skill.embedding = skill.embedding / np.linalg.norm(skill.embedding)

        skill.metadata.created_at = self._current_time
        self._skills[skill.name] = skill

        for domain in skill.metadata.domains:
            if domain not in self._domain_skills:
                self._domain_skills[domain] = set()
            self._domain_skills[domain].add(skill.name)

        for tag in skill.metadata.tags:
            if tag not in self._tag_skills:
                self._tag_skills[tag] = set()
            self._tag_skills[tag].add(skill.name)

        return skill.name

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def retrieve_applicable(
        self,
        state: Any,
        domain: Optional[str] = None,
        tags: Optional[Set[str]] = None,
    ) -> List[Skill]:
        """
        Retrieve skills applicable to the current state.

        Args:
            state: Current state
            domain: Optional domain filter
            tags: Optional tag filters

        Returns:
            List of applicable skills
        """
        candidates = set(self._skills.keys())

        if domain and domain in self._domain_skills:
            candidates &= self._domain_skills[domain]

        if tags:
            for tag in tags:
                if tag in self._tag_skills:
                    candidates &= self._tag_skills[tag]

        applicable = []
        for name in candidates:
            skill = self._skills[name]
            if skill.can_execute(state):
                applicable.append(skill)

        return applicable

    def retrieve_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[Skill, float]]:
        """
        Retrieve skills similar to a query embedding.

        Returns list of (skill, similarity) tuples.
        """
        similarities = []

        query_norm = np.linalg.norm(query_embedding)
        if query_norm < 1e-8:
            return []

        query_normalized = query_embedding / query_norm

        for skill in self._skills.values():
            if skill.embedding is None:
                continue

            similarity = float(np.dot(query_normalized, skill.embedding))
            if similarity >= threshold:
                similarities.append((skill, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def compose_skills(
        self,
        skill_names: List[str],
        composition_type: str = "sequence",
        result_name: Optional[str] = None,
    ) -> Optional[Skill]:
        """
        Compose multiple skills into a new skill.

        Args:
            skill_names: Names of skills to compose
            composition_type: "sequence", "parallel", or "conditional"
            result_name: Name for the composed skill

        Returns:
            Composed skill or None if composition failed
        """
        skills = [self._skills[n] for n in skill_names if n in self._skills]
        if len(skills) < 2:
            return None

        if result_name is None:
            result_name = f"composed_{'_'.join(skill_names)}"

        if composition_type == "sequence":
            composed = self._composer.sequence(skills, result_name)
        elif composition_type == "parallel":
            composed = self._composer.parallel(skills, result_name)
        else:
            return None

        for skill in skills:
            skill.child_skills.append(composed.name)

        self.register_skill(composed)
        return composed

    def transfer_skill(
        self,
        skill_name: str,
        target_domain: str,
        adapter: Optional[Callable[[Skill], Skill]] = None,
    ) -> Optional[Skill]:
        """
        Transfer a skill to a new domain.

        Args:
            skill_name: Name of skill to transfer
            target_domain: Target domain
            adapter: Optional function to adapt the skill

        Returns:
            Transferred skill or None
        """
        source_skill = self._skills.get(skill_name)
        if source_skill is None:
            return None

        if adapter:
            transferred = adapter(source_skill)
        else:
            transferred = Skill(
                name=f"{skill_name}_{target_domain}",
                skill_type=source_skill.skill_type,
                preconditions=source_skill.preconditions,
                effects=source_skill.effects,
                policy=source_skill.policy,
                termination=source_skill.termination,
                parameters=dict(source_skill.parameters),
                embedding=source_skill.embedding.copy()
                if source_skill.embedding is not None
                else None,
                parent_skills=[skill_name],
            )

        transferred.metadata.domains.add(target_domain)
        self.register_skill(transferred)

        return transferred

    def record_skill_use(
        self,
        skill_name: str,
        success: bool,
        steps: int,
        reward: float,
    ) -> None:
        """Record a skill usage for statistics."""
        skill = self._skills.get(skill_name)
        if skill:
            skill.metadata.record_use(success, steps, reward, self._current_time)

    def get_best_skill(
        self,
        state: Any,
        metric: str = "success_rate",
    ) -> Optional[Skill]:
        """
        Get the best applicable skill by a metric.

        Metrics: "success_rate", "reward", "efficiency"
        """
        applicable = self.retrieve_applicable(state)
        if not applicable:
            return None

        def get_score(skill: Skill) -> float:
            meta = skill.metadata
            if metric == "success_rate":
                return meta.success_rate
            elif metric == "reward":
                return meta.average_reward
            elif metric == "efficiency":
                if meta.average_steps == 0:
                    return 0.0
                return meta.average_reward / meta.average_steps
            return 0.0

        return max(applicable, key=get_score)

    def advance_time(self, delta: float) -> None:
        """Advance the library's internal clock."""
        self._current_time += delta

    def list_skills(
        self,
        domain: Optional[str] = None,
        skill_type: Optional[SkillType] = None,
    ) -> List[str]:
        """List skill names with optional filters."""
        result = []

        for name, skill in self._skills.items():
            if domain and domain not in skill.metadata.domains:
                continue
            if skill_type and skill.skill_type != skill_type:
                continue
            result.append(name)

        return result

    def statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        type_counts = {}
        for skill in self._skills.values():
            t = skill.skill_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        return {
            "total_skills": len(self._skills),
            "domains": len(self._domain_skills),
            "tags": len(self._tag_skills),
            "by_type": type_counts,
            "current_time": self._current_time,
        }
