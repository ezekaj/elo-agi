"""
Deductive Reasoning - General → Specific

Derive necessary conclusions from premises.
Key properties: conclusions are CERTAIN if premises are true.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum


class PropositionType(Enum):
    ATOMIC = "atomic"
    UNIVERSAL = "universal"
    EXISTENTIAL = "existential"
    CONDITIONAL = "conditional"
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"
    NEGATION = "negation"


@dataclass
class Proposition:
    """A logical proposition"""

    proposition_id: str
    prop_type: PropositionType
    content: Any
    subject: Optional[str] = None
    predicate: Optional[str] = None
    components: List["Proposition"] = field(default_factory=list)

    def __str__(self) -> str:
        if self.prop_type == PropositionType.ATOMIC:
            return f"{self.subject} is {self.predicate}"
        elif self.prop_type == PropositionType.UNIVERSAL:
            return f"All {self.subject} are {self.predicate}"
        elif self.prop_type == PropositionType.EXISTENTIAL:
            return f"Some {self.subject} are {self.predicate}"
        elif self.prop_type == PropositionType.CONDITIONAL:
            if len(self.components) >= 2:
                return f"If {self.components[0]} then {self.components[1]}"
        elif self.prop_type == PropositionType.NEGATION:
            if self.components:
                return f"Not ({self.components[0]})"
        elif self.prop_type == PropositionType.CONJUNCTION:
            return " AND ".join(str(c) for c in self.components)
        elif self.prop_type == PropositionType.DISJUNCTION:
            return " OR ".join(str(c) for c in self.components)
        return str(self.content)


@dataclass
class Syllogism:
    """A syllogistic argument"""

    major_premise: Proposition
    minor_premise: Proposition
    conclusion: Optional[Proposition] = None
    is_valid: bool = False
    validity_explanation: str = ""


@dataclass
class Inference:
    """A derived inference"""

    inference_id: str
    premises: List[Proposition]
    conclusion: Proposition
    rule_used: str
    is_valid: bool
    certainty: float = 1.0


class DeductiveReasoner:
    """
    Derive necessary conclusions from premises.
    Implements classical deductive logic.

    "All men are mortal" + "Socrates is a man" → "Socrates is mortal"
    """

    def __init__(self):
        self.premises: Dict[str, Proposition] = {}
        self.derived: Dict[str, Inference] = {}
        self.knowledge_base: Dict[str, Set[str]] = {}

    def add_premise(self, premise: Proposition):
        """Add a premise to the knowledge base"""
        self.premises[premise.proposition_id] = premise
        self._update_knowledge_base(premise)

    def _update_knowledge_base(self, prop: Proposition):
        """Update internal knowledge representation"""
        if prop.prop_type == PropositionType.UNIVERSAL:
            if prop.subject not in self.knowledge_base:
                self.knowledge_base[prop.subject] = set()
            self.knowledge_base[prop.subject].add(prop.predicate)

        elif prop.prop_type == PropositionType.ATOMIC:
            entity = prop.subject
            if entity not in self.knowledge_base:
                self.knowledge_base[entity] = set()
            self.knowledge_base[entity].add(prop.predicate)

    def derive(self, premises: List[Proposition]) -> List[Inference]:
        """Derive all possible conclusions from premises"""
        inferences = []

        inferences.extend(self._apply_modus_ponens(premises))
        inferences.extend(self._apply_modus_tollens(premises))
        inferences.extend(self._apply_syllogism(premises))
        inferences.extend(self._apply_conjunction_elimination(premises))
        inferences.extend(self._apply_disjunctive_syllogism(premises))

        for inf in inferences:
            self.derived[inf.inference_id] = inf

        return inferences

    def _apply_modus_ponens(self, premises: List[Proposition]) -> List[Inference]:
        """
        Modus Ponens: If P then Q, P, therefore Q
        """
        inferences = []

        conditionals = [p for p in premises if p.prop_type == PropositionType.CONDITIONAL]

        for cond in conditionals:
            if len(cond.components) < 2:
                continue

            antecedent = cond.components[0]
            consequent = cond.components[1]

            for premise in premises:
                if self._propositions_match(premise, antecedent):
                    inference = Inference(
                        inference_id=f"mp_{cond.proposition_id}_{premise.proposition_id}",
                        premises=[cond, premise],
                        conclusion=consequent,
                        rule_used="modus_ponens",
                        is_valid=True,
                    )
                    inferences.append(inference)

        return inferences

    def _apply_modus_tollens(self, premises: List[Proposition]) -> List[Inference]:
        """
        Modus Tollens: If P then Q, not Q, therefore not P
        """
        inferences = []

        conditionals = [p for p in premises if p.prop_type == PropositionType.CONDITIONAL]
        negations = [p for p in premises if p.prop_type == PropositionType.NEGATION]

        for cond in conditionals:
            if len(cond.components) < 2:
                continue

            antecedent = cond.components[0]
            consequent = cond.components[1]

            for neg in negations:
                if neg.components and self._propositions_match(neg.components[0], consequent):
                    negated_antecedent = Proposition(
                        proposition_id=f"not_{antecedent.proposition_id}",
                        prop_type=PropositionType.NEGATION,
                        content=f"not ({antecedent.content})",
                        components=[antecedent],
                    )

                    inference = Inference(
                        inference_id=f"mt_{cond.proposition_id}_{neg.proposition_id}",
                        premises=[cond, neg],
                        conclusion=negated_antecedent,
                        rule_used="modus_tollens",
                        is_valid=True,
                    )
                    inferences.append(inference)

        return inferences

    def _apply_syllogism(self, premises: List[Proposition]) -> List[Inference]:
        """
        Categorical Syllogism: All A are B, All B are C, therefore All A are C
        """
        inferences = []

        universals = [p for p in premises if p.prop_type == PropositionType.UNIVERSAL]

        for p1 in universals:
            for p2 in universals:
                if p1 == p2:
                    continue

                if p1.predicate == p2.subject:
                    conclusion = Proposition(
                        proposition_id=f"syl_{p1.proposition_id}_{p2.proposition_id}",
                        prop_type=PropositionType.UNIVERSAL,
                        content=f"All {p1.subject} are {p2.predicate}",
                        subject=p1.subject,
                        predicate=p2.predicate,
                    )

                    inference = Inference(
                        inference_id=f"syllogism_{p1.proposition_id}_{p2.proposition_id}",
                        premises=[p1, p2],
                        conclusion=conclusion,
                        rule_used="categorical_syllogism",
                        is_valid=True,
                    )
                    inferences.append(inference)

        for universal in universals:
            atomics = [p for p in premises if p.prop_type == PropositionType.ATOMIC]
            for atomic in atomics:
                if atomic.predicate == universal.subject:
                    conclusion = Proposition(
                        proposition_id=f"inst_{universal.proposition_id}_{atomic.proposition_id}",
                        prop_type=PropositionType.ATOMIC,
                        content=f"{atomic.subject} is {universal.predicate}",
                        subject=atomic.subject,
                        predicate=universal.predicate,
                    )

                    inference = Inference(
                        inference_id=f"instantiation_{universal.proposition_id}_{atomic.proposition_id}",
                        premises=[universal, atomic],
                        conclusion=conclusion,
                        rule_used="universal_instantiation",
                        is_valid=True,
                    )
                    inferences.append(inference)

        return inferences

    def _apply_conjunction_elimination(self, premises: List[Proposition]) -> List[Inference]:
        """
        Conjunction Elimination: P AND Q, therefore P (and therefore Q)
        """
        inferences = []

        conjunctions = [p for p in premises if p.prop_type == PropositionType.CONJUNCTION]

        for conj in conjunctions:
            for i, component in enumerate(conj.components):
                inference = Inference(
                    inference_id=f"conj_elim_{conj.proposition_id}_{i}",
                    premises=[conj],
                    conclusion=component,
                    rule_used="conjunction_elimination",
                    is_valid=True,
                )
                inferences.append(inference)

        return inferences

    def _apply_disjunctive_syllogism(self, premises: List[Proposition]) -> List[Inference]:
        """
        Disjunctive Syllogism: P OR Q, not P, therefore Q
        """
        inferences = []

        disjunctions = [p for p in premises if p.prop_type == PropositionType.DISJUNCTION]
        negations = [p for p in premises if p.prop_type == PropositionType.NEGATION]

        for disj in disjunctions:
            if len(disj.components) != 2:
                continue

            for neg in negations:
                if not neg.components:
                    continue

                negated = neg.components[0]

                if self._propositions_match(negated, disj.components[0]):
                    inference = Inference(
                        inference_id=f"ds_{disj.proposition_id}_{neg.proposition_id}",
                        premises=[disj, neg],
                        conclusion=disj.components[1],
                        rule_used="disjunctive_syllogism",
                        is_valid=True,
                    )
                    inferences.append(inference)

                elif self._propositions_match(negated, disj.components[1]):
                    inference = Inference(
                        inference_id=f"ds_{disj.proposition_id}_{neg.proposition_id}",
                        premises=[disj, neg],
                        conclusion=disj.components[0],
                        rule_used="disjunctive_syllogism",
                        is_valid=True,
                    )
                    inferences.append(inference)

        return inferences

    def _propositions_match(self, p1: Proposition, p2: Proposition) -> bool:
        """Check if two propositions are equivalent"""
        if p1.prop_type != p2.prop_type:
            return False

        if p1.subject is not None and p2.subject is not None:
            if p1.subject != p2.subject:
                return False

        if p1.predicate is not None and p2.predicate is not None:
            if p1.predicate != p2.predicate:
                return False

        if p1.subject is None and p2.subject is None:
            if p1.content != p2.content:
                return False

        return True

    def validate(self, conclusion: Proposition, premises: List[Proposition]) -> Tuple[bool, str]:
        """Validate if conclusion follows from premises"""
        inferences = self.derive(premises)

        for inf in inferences:
            if self._propositions_match(inf.conclusion, conclusion):
                return True, f"Valid via {inf.rule_used}"

        all_premises = premises.copy()
        for _ in range(5):
            new_inferences = self.derive(all_premises)
            for inf in new_inferences:
                if self._propositions_match(inf.conclusion, conclusion):
                    return True, f"Valid via chain of inferences"
                if inf.conclusion.proposition_id not in [p.proposition_id for p in all_premises]:
                    all_premises.append(inf.conclusion)

        return False, "Could not derive conclusion from premises"

    def syllogism(self, major_premise: Proposition, minor_premise: Proposition) -> Syllogism:
        """Evaluate a classical syllogism"""
        inferences = self._apply_syllogism([major_premise, minor_premise])

        if inferences:
            return Syllogism(
                major_premise=major_premise,
                minor_premise=minor_premise,
                conclusion=inferences[0].conclusion,
                is_valid=True,
                validity_explanation="Valid categorical syllogism",
            )

        return Syllogism(
            major_premise=major_premise,
            minor_premise=minor_premise,
            conclusion=None,
            is_valid=False,
            validity_explanation="No valid conclusion can be drawn",
        )

    def modus_ponens(
        self, conditional: Proposition, antecedent: Proposition
    ) -> Optional[Proposition]:
        """Apply modus ponens directly"""
        if conditional.prop_type != PropositionType.CONDITIONAL:
            return None

        if len(conditional.components) < 2:
            return None

        if self._propositions_match(antecedent, conditional.components[0]):
            return conditional.components[1]

        return None

    def modus_tollens(
        self, conditional: Proposition, negated_consequent: Proposition
    ) -> Optional[Proposition]:
        """Apply modus tollens directly"""
        if conditional.prop_type != PropositionType.CONDITIONAL:
            return None

        if negated_consequent.prop_type != PropositionType.NEGATION:
            return None

        if len(conditional.components) < 2:
            return None

        if negated_consequent.components and self._propositions_match(
            negated_consequent.components[0], conditional.components[1]
        ):
            return Proposition(
                proposition_id=f"not_{conditional.components[0].proposition_id}",
                prop_type=PropositionType.NEGATION,
                content=f"not ({conditional.components[0].content})",
                components=[conditional.components[0]],
            )

        return None

    def query(self, subject: str, predicate: str) -> Tuple[bool, float, str]:
        """Query if subject has predicate based on knowledge"""
        if subject in self.knowledge_base:
            if predicate in self.knowledge_base[subject]:
                return True, 1.0, "direct"

        for category, properties in self.knowledge_base.items():
            if category in self.knowledge_base.get(subject, set()):
                if predicate in properties:
                    return True, 1.0, f"inherited from {category}"

        return False, 0.0, "not derivable"
