"""
Skill Library: Reusable skill primitives.

Implements skill composition, execution, and reuse across tasks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from enum import Enum
import numpy as np


class SkillType(Enum):
    """Types of skills."""

    PRIMITIVE = "primitive"  # Basic atomic skills
    COMPOSITE = "composite"  # Composed from other skills
    LEARNED = "learned"  # Learned from experience
    ADAPTED = "adapted"  # Adapted from another domain


@dataclass
class SkillPrimitive:
    """A basic skill primitive."""

    id: str
    name: str
    parameters: Dict[str, Any]
    execute: Callable[[Dict[str, Any]], Any]
    preconditions: List[Callable[[Dict[str, Any]], bool]]
    effects: List[str]  # Description of effects


@dataclass
class Skill:
    """A skill that can be executed."""

    id: str
    name: str
    skill_type: SkillType
    domain: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    success_rate: float = 0.0
    execution_count: int = 0


@dataclass
class CompositeSkill:
    """A skill composed of other skills."""

    id: str
    name: str
    components: List[str]  # Skill IDs in execution order
    parallel_groups: List[List[str]] = field(
        default_factory=list
    )  # Groups that can run in parallel
    conditional_branches: Dict[str, List[str]] = field(
        default_factory=dict
    )  # condition -> skill sequence


@dataclass
class SkillExecution:
    """Result of executing a skill."""

    skill_id: str
    success: bool
    result: Any
    execution_time: float
    state_changes: Dict[str, Any]
    errors: List[str] = field(default_factory=list)


class SkillComposer:
    """
    Compose skills into higher-level skills.
    """

    def __init__(self):
        self._composition_counter = 0

    def sequence(
        self,
        name: str,
        skills: List[str],
    ) -> CompositeSkill:
        """Create a sequential composition."""
        self._composition_counter += 1

        return CompositeSkill(
            id=f"composite_{self._composition_counter}",
            name=name,
            components=skills,
        )

    def parallel(
        self,
        name: str,
        skill_groups: List[List[str]],
    ) -> CompositeSkill:
        """Create a parallel composition."""
        self._composition_counter += 1

        # Flatten for component list
        all_skills = [s for group in skill_groups for s in group]

        return CompositeSkill(
            id=f"composite_{self._composition_counter}",
            name=name,
            components=all_skills,
            parallel_groups=skill_groups,
        )

    def conditional(
        self,
        name: str,
        branches: Dict[str, List[str]],
        default: Optional[List[str]] = None,
    ) -> CompositeSkill:
        """Create a conditional composition."""
        self._composition_counter += 1

        all_skills = []
        for branch_skills in branches.values():
            all_skills.extend(branch_skills)
        if default:
            all_skills.extend(default)

        return CompositeSkill(
            id=f"composite_{self._composition_counter}",
            name=name,
            components=list(set(all_skills)),
            conditional_branches=branches,
        )


class SkillMatcher:
    """
    Match skills across domains based on similarity.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
    ):
        self.embedding_dim = embedding_dim

    def compute_similarity(
        self,
        skill1: Skill,
        skill2: Skill,
    ) -> float:
        """Compute similarity between two skills."""
        if skill1.embedding is None or skill2.embedding is None:
            # Fall back to name/description similarity
            name_sim = self._string_similarity(skill1.name, skill2.name)
            desc_sim = self._string_similarity(skill1.description, skill2.description)
            return (name_sim + desc_sim) / 2

        # Cosine similarity
        return float(np.dot(skill1.embedding, skill2.embedding))

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Compute string similarity (Jaccard on words)."""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union

    def find_similar(
        self,
        query_skill: Skill,
        candidates: List[Skill],
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> List[Tuple[Skill, float]]:
        """Find similar skills."""
        similarities = []

        for candidate in candidates:
            if candidate.id == query_skill.id:
                continue

            sim = self.compute_similarity(query_skill, candidate)
            if sim >= threshold:
                similarities.append((candidate, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])

        return similarities[:top_k]


class SkillLibrary:
    """
    Complete skill library for storing and reusing skills.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
    ):
        self.embedding_dim = embedding_dim

        self.composer = SkillComposer()
        self.matcher = SkillMatcher(embedding_dim)

        # Skill storage
        self._primitives: Dict[str, SkillPrimitive] = {}
        self._skills: Dict[str, Skill] = {}
        self._composites: Dict[str, CompositeSkill] = {}

        # Domain organization
        self._skills_by_domain: Dict[str, Set[str]] = {}

        # Execution history
        self._execution_history: List[SkillExecution] = []

        self._skill_counter = 0

    def register_primitive(
        self,
        name: str,
        execute_fn: Callable,
        preconditions: Optional[List[Callable]] = None,
        effects: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> SkillPrimitive:
        """Register a primitive skill."""
        self._skill_counter += 1

        primitive = SkillPrimitive(
            id=f"primitive_{self._skill_counter}",
            name=name,
            parameters=parameters or {},
            execute=execute_fn,
            preconditions=preconditions or [],
            effects=effects or [],
        )

        self._primitives[primitive.id] = primitive
        return primitive

    def register_skill(
        self,
        name: str,
        domain: str,
        description: str,
        skill_type: SkillType = SkillType.LEARNED,
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> Skill:
        """Register a skill."""
        self._skill_counter += 1

        # Generate embedding if not provided
        if embedding is None:
            embedding = np.random.randn(self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)

        skill = Skill(
            id=f"skill_{self._skill_counter}",
            name=name,
            skill_type=skill_type,
            domain=domain,
            description=description,
            parameters=parameters or {},
            dependencies=dependencies or [],
            embedding=embedding,
        )

        self._skills[skill.id] = skill

        # Organize by domain
        if domain not in self._skills_by_domain:
            self._skills_by_domain[domain] = set()
        self._skills_by_domain[domain].add(skill.id)

        return skill

    def compose(
        self,
        name: str,
        skill_ids: List[str],
        composition_type: str = "sequence",
    ) -> CompositeSkill:
        """Compose skills into a higher-level skill."""
        if composition_type == "sequence":
            composite = self.composer.sequence(name, skill_ids)
        elif composition_type == "parallel":
            composite = self.composer.parallel(name, [skill_ids])
        else:
            raise ValueError(f"Unknown composition type: {composition_type}")

        self._composites[composite.id] = composite
        return composite

    def execute(
        self,
        skill_id: str,
        context: Dict[str, Any],
    ) -> SkillExecution:
        """
        Execute a skill.

        Args:
            skill_id: ID of skill to execute
            context: Execution context with parameters

        Returns:
            SkillExecution result
        """
        import time

        start_time = time.time()

        errors = []
        result = None
        success = False
        state_changes = {}

        # Check if it's a primitive
        if skill_id in self._primitives:
            primitive = self._primitives[skill_id]

            # Check preconditions
            for precond in primitive.preconditions:
                if not precond(context):
                    errors.append("Precondition failed")
                    break
            else:
                try:
                    result = primitive.execute(context)
                    success = True
                    state_changes["effects"] = primitive.effects
                except Exception as e:
                    errors.append(str(e))

        # Check if it's a composite
        elif skill_id in self._composites:
            composite = self._composites[skill_id]
            sub_results = []

            for sub_skill_id in composite.components:
                sub_exec = self.execute(sub_skill_id, context)
                sub_results.append(sub_exec)

                if not sub_exec.success:
                    errors.extend(sub_exec.errors)
                    break

                # Update context with state changes
                context.update(sub_exec.state_changes)
            else:
                success = True
                result = sub_results

        # Regular skill
        elif skill_id in self._skills:
            skill = self._skills[skill_id]

            # Check dependencies
            for dep in skill.dependencies:
                if dep not in context.get("completed_skills", []):
                    errors.append(f"Dependency not met: {dep}")

            if not errors:
                # Simulate execution
                success = np.random.random() > 0.1  # 90% success rate
                if success:
                    result = f"Executed {skill.name}"
                else:
                    errors.append("Execution failed")

            # Update skill statistics
            skill.execution_count += 1
            skill.success_rate = (
                skill.success_rate * (skill.execution_count - 1) + (1 if success else 0)
            ) / skill.execution_count

        else:
            errors.append(f"Unknown skill: {skill_id}")

        execution_time = time.time() - start_time

        execution = SkillExecution(
            skill_id=skill_id,
            success=success,
            result=result,
            execution_time=execution_time,
            state_changes=state_changes,
            errors=errors,
        )

        self._execution_history.append(execution)
        return execution

    def find_similar_skills(
        self,
        skill_id: str,
        cross_domain: bool = True,
    ) -> List[Tuple[Skill, float]]:
        """Find similar skills."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return []

        candidates = list(self._skills.values())

        if not cross_domain:
            candidates = [s for s in candidates if s.domain == skill.domain]

        return self.matcher.find_similar(skill, candidates)

    def transfer_skill(
        self,
        skill_id: str,
        target_domain: str,
        adaptation_fn: Optional[Callable] = None,
    ) -> Skill:
        """Transfer a skill to a new domain."""
        original = self._skills.get(skill_id)
        if original is None:
            raise ValueError(f"Unknown skill: {skill_id}")

        # Create adapted skill
        adapted = self.register_skill(
            name=f"{original.name}_{target_domain}",
            domain=target_domain,
            description=f"Adapted from {original.domain}: {original.description}",
            skill_type=SkillType.ADAPTED,
            parameters=original.parameters.copy(),
            embedding=original.embedding.copy() if original.embedding is not None else None,
        )

        # Apply adaptation if provided
        if adaptation_fn:
            adaptation_fn(adapted, original, target_domain)

        return adapted

    def get_skills_by_domain(self, domain: str) -> List[Skill]:
        """Get all skills for a domain."""
        skill_ids = self._skills_by_domain.get(domain, set())
        return [self._skills[sid] for sid in skill_ids if sid in self._skills]

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get a skill by ID."""
        return self._skills.get(skill_id)

    def get_execution_history(
        self,
        skill_id: Optional[str] = None,
    ) -> List[SkillExecution]:
        """Get execution history."""
        if skill_id is None:
            return self._execution_history
        return [e for e in self._execution_history if e.skill_id == skill_id]

    def get_success_rate(self, skill_id: str) -> float:
        """Get skill success rate."""
        skill = self._skills.get(skill_id)
        return skill.success_rate if skill else 0.0

    def statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        return {
            "n_primitives": len(self._primitives),
            "n_skills": len(self._skills),
            "n_composites": len(self._composites),
            "n_domains": len(self._skills_by_domain),
            "total_executions": len(self._execution_history),
            "skills_by_type": {
                st.value: sum(1 for s in self._skills.values() if s.skill_type == st)
                for st in SkillType
            },
        }
