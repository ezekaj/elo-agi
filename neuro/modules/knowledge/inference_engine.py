"""
Inference Engine: Forward and backward chaining inference.

Implements rule-based reasoning with unification and
proof generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum

from .fact_store import FactStore, Fact, Triple


class InferenceMode(Enum):
    """Mode of inference."""

    FORWARD = "forward"  # Data-driven, derive all conclusions
    BACKWARD = "backward"  # Goal-driven, prove specific facts
    MIXED = "mixed"  # Combine both


@dataclass
class Pattern:
    """A pattern that can match facts."""

    subject: str  # Can be variable like ?x
    predicate: str
    obj: str

    def is_variable(self, term: str) -> bool:
        """Check if term is a variable."""
        return term.startswith("?")

    def get_variables(self) -> Set[str]:
        """Get all variables in pattern."""
        variables = set()
        if self.is_variable(self.subject):
            variables.add(self.subject)
        if self.is_variable(self.predicate):
            variables.add(self.predicate)
        if self.is_variable(self.obj):
            variables.add(self.obj)
        return variables

    def substitute(self, bindings: Dict[str, str]) -> "Pattern":
        """Apply variable bindings to create new pattern."""
        return Pattern(
            subject=bindings.get(self.subject, self.subject),
            predicate=bindings.get(self.predicate, self.predicate),
            obj=bindings.get(self.obj, self.obj),
        )

    def matches(self, fact: Fact) -> Optional[Dict[str, str]]:
        """Try to match pattern against fact, returning bindings if successful."""
        bindings = {}

        # Match subject
        if self.is_variable(self.subject):
            bindings[self.subject] = fact.subject
        elif self.subject != fact.subject:
            return None

        # Match predicate
        if self.is_variable(self.predicate):
            bindings[self.predicate] = fact.predicate
        elif self.predicate != fact.predicate:
            return None

        # Match object
        if self.is_variable(self.obj):
            bindings[self.obj] = fact.obj
        elif self.obj != fact.obj:
            return None

        return bindings

    def to_triple(self) -> Optional[Triple]:
        """Convert to triple if fully grounded."""
        if self.get_variables():
            return None
        return Triple(self.subject, self.predicate, self.obj)


@dataclass
class Rule:
    """An inference rule with antecedents and consequents."""

    name: str
    antecedents: List[Pattern]  # IF these patterns match
    consequents: List[Pattern]  # THEN derive these
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_variables(self) -> Set[str]:
        """Get all variables in the rule."""
        variables = set()
        for pattern in self.antecedents + self.consequents:
            variables.update(pattern.get_variables())
        return variables


@dataclass
class InferenceResult:
    """Result of an inference."""

    derived_facts: List[Fact]
    rule_used: Optional[Rule]
    bindings: Dict[str, str]
    confidence: float = 1.0
    proof_depth: int = 0


@dataclass
class InferenceChain:
    """A chain of inference steps."""

    goal: Pattern
    steps: List[InferenceResult]
    success: bool
    bindings: Dict[str, str]

    def explain(self) -> str:
        """Generate human-readable explanation."""
        lines = [f"Goal: {self.goal.subject} {self.goal.predicate} {self.goal.obj}"]

        if self.success:
            lines.append("Proof:")
            for i, step in enumerate(self.steps):
                if step.rule_used:
                    lines.append(f"  {i + 1}. Applied rule: {step.rule_used.name}")
                    lines.append(f"      Bindings: {step.bindings}")
                    for fact in step.derived_facts:
                        lines.append(f"      Derived: {fact.subject} {fact.predicate} {fact.obj}")
        else:
            lines.append("Failed to prove goal.")

        return "\n".join(lines)


class InferenceEngine:
    """
    Rule-based inference engine.

    Supports:
    - Forward chaining (derive all conclusions)
    - Backward chaining (prove specific goals)
    - Rule prioritization
    - Confidence propagation
    """

    def __init__(
        self,
        fact_store: Optional[FactStore] = None,
        max_depth: int = 10,
        max_iterations: int = 1000,
    ):
        self.fact_store = fact_store or FactStore()
        self.max_depth = max_depth
        self.max_iterations = max_iterations

        # Rules
        self._rules: List[Rule] = []

        # Derived facts (separate from base facts)
        self._derived: FactStore = FactStore()

        # Statistics
        self._inferences_made = 0
        self._rules_fired = 0

    def add_rule(self, rule: Rule) -> None:
        """Add an inference rule."""
        self._rules.append(rule)
        # Sort by priority
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def add_simple_rule(
        self,
        name: str,
        if_pattern: Tuple[str, str, str],
        then_pattern: Tuple[str, str, str],
        priority: int = 0,
    ) -> Rule:
        """Add a simple single-antecedent, single-consequent rule."""
        rule = Rule(
            name=name,
            antecedents=[Pattern(*if_pattern)],
            consequents=[Pattern(*then_pattern)],
            priority=priority,
        )
        self.add_rule(rule)
        return rule

    def add_transitivity_rule(
        self,
        predicate: str,
        name: Optional[str] = None,
    ) -> Rule:
        """Add a transitivity rule for a predicate."""
        rule = Rule(
            name=name or f"transitivity_{predicate}",
            antecedents=[
                Pattern("?x", predicate, "?y"),
                Pattern("?y", predicate, "?z"),
            ],
            consequents=[
                Pattern("?x", predicate, "?z"),
            ],
        )
        self.add_rule(rule)
        return rule

    def forward_chain(self) -> List[InferenceResult]:
        """Run forward chaining to derive all possible conclusions."""
        results = []
        iteration = 0
        changed = True

        while changed and iteration < self.max_iterations:
            changed = False
            iteration += 1

            for rule in self._rules:
                # Try to apply rule
                new_results = self._apply_rule_forward(rule)

                for result in new_results:
                    for fact in result.derived_facts:
                        # Check if already known
                        if not self._is_known(fact.triple):
                            self._derived.add_triple(
                                fact.triple,
                                confidence=result.confidence,
                                source=f"rule:{rule.name}",
                            )
                            changed = True
                            self._inferences_made += 1

                    if new_results:
                        results.extend(new_results)
                        self._rules_fired += 1

        return results

    def _apply_rule_forward(self, rule: Rule) -> List[InferenceResult]:
        """Apply a rule in forward chaining mode."""
        results = []

        # Find all ways to satisfy antecedents
        all_bindings = self._find_all_bindings(rule.antecedents)

        for bindings in all_bindings:
            # Generate consequents
            derived = []
            confidence = 1.0

            for consequent in rule.consequents:
                grounded = consequent.substitute(bindings)
                triple = grounded.to_triple()

                if triple:
                    fact = Fact(
                        triple=triple,
                        confidence=confidence,
                        source=f"rule:{rule.name}",
                    )
                    derived.append(fact)

            if derived:
                results.append(
                    InferenceResult(
                        derived_facts=derived,
                        rule_used=rule,
                        bindings=bindings,
                        confidence=confidence,
                    )
                )

        return results

    def _find_all_bindings(
        self,
        patterns: List[Pattern],
        initial_bindings: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """Find all variable bindings that satisfy patterns."""
        if not patterns:
            return [initial_bindings or {}]

        bindings_list = []
        current = initial_bindings or {}

        # Get first pattern
        pattern = patterns[0]
        remaining = patterns[1:]

        # Substitute known bindings
        grounded = pattern.substitute(current)

        # Query for matching facts
        query_s = None if grounded.is_variable(grounded.subject) else grounded.subject
        query_p = None if grounded.is_variable(grounded.predicate) else grounded.predicate
        query_o = None if grounded.is_variable(grounded.obj) else grounded.obj

        # Search in both base and derived facts
        base_facts = self.fact_store.query(query_s, query_p, query_o)
        derived_facts = self._derived.query(query_s, query_p, query_o)
        all_facts = base_facts + derived_facts

        for fact in all_facts:
            match = grounded.matches(fact)
            if match is not None:
                # Check consistency with existing bindings
                combined = dict(current)
                consistent = True

                for var, value in match.items():
                    if var in combined and combined[var] != value:
                        consistent = False
                        break
                    combined[var] = value

                if consistent:
                    # Recursively find bindings for remaining patterns
                    sub_bindings = self._find_all_bindings(remaining, combined)
                    bindings_list.extend(sub_bindings)

        return bindings_list

    def backward_chain(
        self,
        goal: Pattern,
        depth: int = 0,
    ) -> InferenceChain:
        """Prove a goal using backward chaining."""
        steps = []

        # Check if goal is already known
        grounded = goal.to_triple()
        if grounded and self._is_known(grounded):
            return InferenceChain(
                goal=goal,
                steps=[],
                success=True,
                bindings={},
            )

        if depth >= self.max_depth:
            return InferenceChain(
                goal=goal,
                steps=[],
                success=False,
                bindings={},
            )

        # Try to prove using rules
        for rule in self._rules:
            for consequent in rule.consequents:
                # Try to unify goal with consequent
                bindings = self._unify(goal, consequent)

                if bindings is not None:
                    # Try to prove antecedents
                    all_proved = True
                    combined_bindings = dict(bindings)

                    for antecedent in rule.antecedents:
                        subgoal = antecedent.substitute(combined_bindings)
                        subchain = self.backward_chain(subgoal, depth + 1)

                        if not subchain.success:
                            all_proved = False
                            break

                        steps.extend(subchain.steps)
                        combined_bindings.update(subchain.bindings)

                    if all_proved:
                        # Derive the consequent
                        derived_pattern = goal.substitute(combined_bindings)
                        derived_triple = derived_pattern.to_triple()

                        if derived_triple:
                            derived_fact = Fact(
                                triple=derived_triple,
                                confidence=1.0,
                                source=f"backward:{rule.name}",
                            )

                            result = InferenceResult(
                                derived_facts=[derived_fact],
                                rule_used=rule,
                                bindings=combined_bindings,
                                proof_depth=depth,
                            )
                            steps.append(result)

                            self._derived.add_triple(derived_triple, source=f"backward:{rule.name}")
                            self._inferences_made += 1
                            self._rules_fired += 1

                            return InferenceChain(
                                goal=goal,
                                steps=steps,
                                success=True,
                                bindings=combined_bindings,
                            )

        # Try to match goal directly against facts
        query_s = None if goal.is_variable(goal.subject) else goal.subject
        query_p = None if goal.is_variable(goal.predicate) else goal.predicate
        query_o = None if goal.is_variable(goal.obj) else goal.obj

        base_facts = self.fact_store.query(query_s, query_p, query_o)
        derived_facts = self._derived.query(query_s, query_p, query_o)

        for fact in base_facts + derived_facts:
            match = goal.matches(fact)
            if match is not None:
                return InferenceChain(
                    goal=goal,
                    steps=steps,
                    success=True,
                    bindings=match,
                )

        return InferenceChain(
            goal=goal,
            steps=steps,
            success=False,
            bindings={},
        )

    def _unify(
        self,
        pattern1: Pattern,
        pattern2: Pattern,
    ) -> Optional[Dict[str, str]]:
        """Unify two patterns, returning bindings if successful."""
        bindings = {}

        def unify_term(t1: str, t2: str) -> bool:
            v1 = pattern1.is_variable(t1)
            v2 = pattern2.is_variable(t2)

            if v1 and v2:
                # Both variables - bind them together
                bindings[t1] = t2
                return True
            elif v1:
                # t1 is variable, bind to t2
                if t1 in bindings and bindings[t1] != t2:
                    return False
                bindings[t1] = t2
                return True
            elif v2:
                # t2 is variable, bind to t1
                if t2 in bindings and bindings[t2] != t1:
                    return False
                bindings[t2] = t1
                return True
            else:
                # Both constants - must be equal
                return t1 == t2

        if not unify_term(pattern1.subject, pattern2.subject):
            return None
        if not unify_term(pattern1.predicate, pattern2.predicate):
            return None
        if not unify_term(pattern1.obj, pattern2.obj):
            return None

        return bindings

    def _is_known(self, triple: Triple) -> bool:
        """Check if a triple is known (in base or derived)."""
        return self.fact_store.exists(
            triple.subject, triple.predicate, triple.obj
        ) or self._derived.exists(triple.subject, triple.predicate, triple.obj)

    def prove(
        self,
        subject: str,
        predicate: str,
        obj: str,
    ) -> InferenceChain:
        """Convenience method to prove a specific fact."""
        goal = Pattern(subject, predicate, obj)
        return self.backward_chain(goal)

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
    ) -> List[Fact]:
        """Query facts including derived ones."""
        base = self.fact_store.query(subject, predicate, obj)
        derived = self._derived.query(subject, predicate, obj)
        return base + derived

    def clear_derived(self) -> None:
        """Clear derived facts."""
        self._derived.clear()

    def get_rules(self) -> List[Rule]:
        """Get all rules."""
        return self._rules.copy()

    def statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "n_rules": len(self._rules),
            "n_base_facts": len(list(self.fact_store.iterate())),
            "n_derived_facts": len(list(self._derived.iterate())),
            "inferences_made": self._inferences_made,
            "rules_fired": self._rules_fired,
        }
