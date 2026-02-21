"""
Logic Network - Frontal Pole Reasoning System

Implements dedicated abstract formal reasoning distinct from other networks.
This is the recently discovered "logic network" in bilateral frontal areas.

Key properties:
- ABSTRACT - domain-independent formal reasoning
- Distinct from language, social, and physical reasoning networks
- Handles relational processing, constraint computation, structure updating
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import copy


class PropositionType(Enum):
    """Types of logical propositions"""
    ATOMIC = "atomic"           # Simple fact: P
    NEGATION = "negation"       # Not P
    CONJUNCTION = "conjunction"  # P and Q
    DISJUNCTION = "disjunction"  # P or Q
    IMPLICATION = "implication"  # If P then Q
    BICONDITIONAL = "biconditional"  # P if and only if Q
    UNIVERSAL = "universal"     # For all X, P(X)
    EXISTENTIAL = "existential"  # There exists X such that P(X)


@dataclass
class Proposition:
    """A logical proposition"""
    id: str
    type: PropositionType
    content: Any
    components: List['Proposition'] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """A relation between entities"""
    name: str
    arity: int  # Number of arguments
    arguments: List[str]
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Inference:
    """Result of logical inference"""
    conclusion: Proposition
    premises: List[Proposition]
    rule_applied: str
    confidence: float = 1.0
    valid: bool = True


class LogicNetwork:
    """
    Frontal pole reasoning system for abstract formal logic.

    Handles:
    - Relational processing: structured relational information
    - Constraint computation: what inferences are licensed
    - Structure updating: modify mental models after inference
    """

    def __init__(self):
        self.propositions: Dict[str, Proposition] = {}
        self.relations: Dict[str, Relation] = {}
        self.inference_history: List[Inference] = []

        # Known inference rules
        self.rules = {
            "modus_ponens": self._modus_ponens,
            "modus_tollens": self._modus_tollens,
            "hypothetical_syllogism": self._hypothetical_syllogism,
            "disjunctive_syllogism": self._disjunctive_syllogism,
            "conjunction_introduction": self._conjunction_intro,
            "conjunction_elimination": self._conjunction_elim,
        }

    # ==================== Relational Processing ====================

    def represent_relation(self,
                           entities: List[str],
                           relation_name: str,
                           properties: Optional[Dict[str, Any]] = None) -> Relation:
        """
        Represent structured relational information.

        E.g., represent_relation(["Socrates", "mortal"], "is_a")
        """
        relation = Relation(
            name=relation_name,
            arity=len(entities),
            arguments=entities,
            properties=properties or {}
        )

        rel_id = f"{relation_name}({','.join(entities)})"
        self.relations[rel_id] = relation
        return relation

    def query_relation(self,
                       relation_name: str,
                       partial_args: Optional[List[Optional[str]]] = None) -> List[Relation]:
        """
        Query for relations matching pattern.

        E.g., query_relation("is_a", [None, "mortal"]) finds all mortals
        """
        results = []

        for rel in self.relations.values():
            if rel.name != relation_name:
                continue

            if partial_args is None:
                results.append(rel)
                continue

            # Check argument pattern
            if len(partial_args) != rel.arity:
                continue

            matches = True
            for i, arg in enumerate(partial_args):
                if arg is not None and rel.arguments[i] != arg:
                    matches = False
                    break

            if matches:
                results.append(rel)

        return results

    # ==================== Constraint Computation ====================

    def compute_constraints(self,
                            premises: List[Proposition]) -> Dict[str, Any]:
        """
        Compute what constraints the premises impose.

        Returns what must be true, what cannot be true, and what's undetermined.
        """
        must_be_true = set()
        cannot_be_true = set()
        undetermined = set()

        for p in premises:
            if p.type == PropositionType.ATOMIC:
                must_be_true.add(p.id)
            elif p.type == PropositionType.NEGATION:
                if p.components:
                    cannot_be_true.add(p.components[0].id)

        # Check for implications
        for p in premises:
            if p.type == PropositionType.IMPLICATION and len(p.components) >= 2:
                antecedent, consequent = p.components[0], p.components[1]
                if antecedent.id in must_be_true:
                    must_be_true.add(consequent.id)
                if consequent.id in cannot_be_true:
                    cannot_be_true.add(antecedent.id)

        return {
            "must_be_true": must_be_true,
            "cannot_be_true": cannot_be_true,
            "undetermined": undetermined
        }

    def check_consistency(self, beliefs: List[Proposition]) -> Tuple[bool, List[str]]:
        """
        Check if beliefs are logically consistent.

        Returns (is_consistent, list_of_contradictions)
        """
        contradictions = []

        # Simple check: P and not-P
        true_props = set()
        false_props = set()

        for b in beliefs:
            if b.type == PropositionType.ATOMIC:
                if b.id in false_props:
                    contradictions.append(f"Contradiction: {b.id} is both true and false")
                true_props.add(b.id)
            elif b.type == PropositionType.NEGATION and b.components:
                inner_id = b.components[0].id
                if inner_id in true_props:
                    contradictions.append(f"Contradiction: {inner_id} is both true and false")
                false_props.add(inner_id)

        return len(contradictions) == 0, contradictions

    def derive_inferences(self, premises: List[Proposition]) -> List[Inference]:
        """
        Derive all licensed inferences from premises.

        Applies all applicable inference rules.
        """
        inferences = []

        # Try each inference rule
        for rule_name, rule_func in self.rules.items():
            new_inferences = rule_func(premises)
            inferences.extend(new_inferences)

        self.inference_history.extend(inferences)
        return inferences

    # ==================== Inference Rules ====================

    def _modus_ponens(self, premises: List[Proposition]) -> List[Inference]:
        """
        If P then Q, P, therefore Q.
        """
        inferences = []

        # Find implications
        implications = [p for p in premises if p.type == PropositionType.IMPLICATION]
        atoms = {p.id: p for p in premises if p.type == PropositionType.ATOMIC}

        for impl in implications:
            if len(impl.components) < 2:
                continue

            antecedent, consequent = impl.components[0], impl.components[1]

            if antecedent.id in atoms:
                inferences.append(Inference(
                    conclusion=consequent,
                    premises=[impl, atoms[antecedent.id]],
                    rule_applied="modus_ponens"
                ))

        return inferences

    def _modus_tollens(self, premises: List[Proposition]) -> List[Inference]:
        """
        If P then Q, not Q, therefore not P.
        """
        inferences = []

        implications = [p for p in premises if p.type == PropositionType.IMPLICATION]
        negations = [p for p in premises if p.type == PropositionType.NEGATION]

        for impl in implications:
            if len(impl.components) < 2:
                continue

            antecedent, consequent = impl.components[0], impl.components[1]

            for neg in negations:
                if neg.components and neg.components[0].id == consequent.id:
                    # Found not-Q
                    not_antecedent = Proposition(
                        id=f"not_{antecedent.id}",
                        type=PropositionType.NEGATION,
                        content=f"not {antecedent.content}",
                        components=[antecedent]
                    )
                    inferences.append(Inference(
                        conclusion=not_antecedent,
                        premises=[impl, neg],
                        rule_applied="modus_tollens"
                    ))

        return inferences

    def _hypothetical_syllogism(self, premises: List[Proposition]) -> List[Inference]:
        """
        If P then Q, If Q then R, therefore If P then R.
        """
        inferences = []
        implications = [p for p in premises if p.type == PropositionType.IMPLICATION]

        for impl1 in implications:
            if len(impl1.components) < 2:
                continue

            for impl2 in implications:
                if impl1.id == impl2.id or len(impl2.components) < 2:
                    continue

                # Check if impl1's consequent matches impl2's antecedent
                if impl1.components[1].id == impl2.components[0].id:
                    new_impl = Proposition(
                        id=f"impl_{impl1.components[0].id}_{impl2.components[1].id}",
                        type=PropositionType.IMPLICATION,
                        content=f"if {impl1.components[0].content} then {impl2.components[1].content}",
                        components=[impl1.components[0], impl2.components[1]]
                    )
                    inferences.append(Inference(
                        conclusion=new_impl,
                        premises=[impl1, impl2],
                        rule_applied="hypothetical_syllogism"
                    ))

        return inferences

    def _disjunctive_syllogism(self, premises: List[Proposition]) -> List[Inference]:
        """
        P or Q, not P, therefore Q.
        """
        inferences = []

        disjunctions = [p for p in premises if p.type == PropositionType.DISJUNCTION]
        negations = [p for p in premises if p.type == PropositionType.NEGATION]

        for disj in disjunctions:
            if len(disj.components) < 2:
                continue

            for neg in negations:
                if not neg.components:
                    continue

                negated_id = neg.components[0].id

                # Check which disjunct is negated
                if disj.components[0].id == negated_id:
                    inferences.append(Inference(
                        conclusion=disj.components[1],
                        premises=[disj, neg],
                        rule_applied="disjunctive_syllogism"
                    ))
                elif disj.components[1].id == negated_id:
                    inferences.append(Inference(
                        conclusion=disj.components[0],
                        premises=[disj, neg],
                        rule_applied="disjunctive_syllogism"
                    ))

        return inferences

    def _conjunction_intro(self, premises: List[Proposition]) -> List[Inference]:
        """
        P, Q, therefore P and Q.
        """
        inferences = []
        atoms = [p for p in premises if p.type == PropositionType.ATOMIC]

        for i, p1 in enumerate(atoms):
            for p2 in atoms[i + 1:]:
                conj = Proposition(
                    id=f"and_{p1.id}_{p2.id}",
                    type=PropositionType.CONJUNCTION,
                    content=f"{p1.content} and {p2.content}",
                    components=[p1, p2]
                )
                inferences.append(Inference(
                    conclusion=conj,
                    premises=[p1, p2],
                    rule_applied="conjunction_introduction"
                ))

        return inferences

    def _conjunction_elim(self, premises: List[Proposition]) -> List[Inference]:
        """
        P and Q, therefore P (and therefore Q).
        """
        inferences = []
        conjunctions = [p for p in premises if p.type == PropositionType.CONJUNCTION]

        for conj in conjunctions:
            for component in conj.components:
                inferences.append(Inference(
                    conclusion=component,
                    premises=[conj],
                    rule_applied="conjunction_elimination"
                ))

        return inferences

    # ==================== Structure Updating ====================

    def update_model(self,
                     model: Dict[str, Any],
                     new_info: Proposition) -> Dict[str, Any]:
        """
        Update mental model with new information.

        Revises beliefs based on new proposition.
        """
        updated = copy.deepcopy(model)

        if new_info.type == PropositionType.ATOMIC:
            updated[new_info.id] = True
        elif new_info.type == PropositionType.NEGATION and new_info.components:
            updated[new_info.components[0].id] = False
        elif new_info.type == PropositionType.IMPLICATION and len(new_info.components) >= 2:
            ant, cons = new_info.components[0], new_info.components[1]
            if updated.get(ant.id) is True:
                updated[cons.id] = True

        return updated

    def propagate_implications(self,
                                model: Dict[str, Any],
                                change: Tuple[str, Any]) -> Dict[str, Any]:
        """
        Propagate implications of a change through the model.
        """
        prop_id, new_value = change
        updated = copy.deepcopy(model)
        updated[prop_id] = new_value

        # Find all implications affected
        changed = True
        while changed:
            changed = False
            for rel_id, rel in self.relations.items():
                if rel.properties.get("type") == "implication":
                    ant_id = rel.arguments[0] if len(rel.arguments) > 0 else None
                    cons_id = rel.arguments[1] if len(rel.arguments) > 1 else None

                    if ant_id and cons_id:
                        if updated.get(ant_id) is True and updated.get(cons_id) is not True:
                            updated[cons_id] = True
                            changed = True

        return updated

    # ==================== High-Level Reasoning ====================

    def syllogism(self, major: Proposition, minor: Proposition) -> Optional[Inference]:
        """
        Apply classical syllogism.

        Major: All A are B
        Minor: X is A
        Conclusion: X is B
        """
        # Check for universal quantification in major premise
        if major.type != PropositionType.UNIVERSAL:
            return None

        # Extract the implication structure
        if "all" in str(major.content).lower():
            # Parse "All A are B" structure
            all_inferences = self.derive_inferences([major, minor])
            if all_inferences:
                return all_inferences[0]

        return None

    def is_valid_argument(self,
                          premises: List[Proposition],
                          conclusion: Proposition) -> Tuple[bool, str]:
        """
        Check if conclusion validly follows from premises.

        This is the core logic network function - determining
        if an inference is licensed.
        """
        # Derive all possible conclusions
        derived = self.derive_inferences(premises)

        for inf in derived:
            if inf.conclusion.id == conclusion.id:
                return True, f"Valid via {inf.rule_applied}"

        # Check if conclusion contradicts premises
        test_beliefs = premises + [Proposition(
            id=f"not_{conclusion.id}",
            type=PropositionType.NEGATION,
            content=f"not {conclusion.content}",
            components=[conclusion]
        )]

        consistent, contradictions = self.check_consistency(test_beliefs)
        if not consistent:
            return True, "Valid by contradiction"

        return False, "Cannot derive conclusion from premises"
