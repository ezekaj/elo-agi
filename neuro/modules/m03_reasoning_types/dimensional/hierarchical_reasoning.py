"""
Hierarchical Reasoning - Prefrontal Cortex Simulation

Abstract rule application across levels of abstraction.
Handles meta-rules, context-dependent rule selection, and exception handling.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class AbstractionLevel(Enum):
    CONCRETE = 0
    BASIC = 1
    INTERMEDIATE = 2
    ABSTRACT = 3
    META = 4


@dataclass
class Rule:
    """A rule that can be applied to instances"""

    rule_id: str
    name: str
    level: AbstractionLevel
    condition: Callable[[Any], bool]
    action: Callable[[Any], Any]
    priority: int = 0
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    exceptions: List[str] = field(default_factory=list)

    def applies_to(self, instance: Any, context: Dict = None) -> bool:
        """Check if rule applies given context"""
        if context:
            for key, required in self.context_requirements.items():
                if key not in context or context[key] != required:
                    return False
        try:
            return self.condition(instance)
        except Exception:
            return False

    def apply(self, instance: Any) -> Any:
        """Apply rule to instance"""
        return self.action(instance)


@dataclass
class MetaRule:
    """A rule about rules"""

    meta_rule_id: str
    name: str
    rule_selector: Callable[[List[Rule], Any, Dict], Rule]
    description: str = ""


class RuleHierarchy:
    """Nested rule structures with meta-rules and exceptions"""

    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.meta_rules: Dict[str, MetaRule] = {}
        self.rule_hierarchy: Dict[str, List[str]] = {}
        self.default_context: Dict[str, Any] = {}

        self._initialize_default_meta_rules()

    def _initialize_default_meta_rules(self):
        """Set up default meta-rules"""
        self.meta_rules["priority"] = MetaRule(
            meta_rule_id="priority",
            name="Priority Selection",
            rule_selector=lambda rules, inst, ctx: (
                max(rules, key=lambda r: r.priority) if rules else None
            ),
            description="Select rule with highest priority",
        )

        self.meta_rules["specificity"] = MetaRule(
            meta_rule_id="specificity",
            name="Specificity Selection",
            rule_selector=lambda rules, inst, ctx: (
                max(rules, key=lambda r: len(r.context_requirements)) if rules else None
            ),
            description="Select most specific rule",
        )

        self.meta_rules["abstraction"] = MetaRule(
            meta_rule_id="abstraction",
            name="Abstraction Level Selection",
            rule_selector=lambda rules, inst, ctx: (
                min(rules, key=lambda r: r.level.value) if rules else None
            ),
            description="Prefer concrete rules over abstract",
        )

    def add_rule(self, rule: Rule, parent_rule_id: Optional[str] = None):
        """Add a rule to the hierarchy"""
        self.rules[rule.rule_id] = rule

        if parent_rule_id:
            if parent_rule_id not in self.rule_hierarchy:
                self.rule_hierarchy[parent_rule_id] = []
            self.rule_hierarchy[parent_rule_id].append(rule.rule_id)

    def add_meta_rule(self, meta_rule: MetaRule):
        """Add a meta-rule"""
        self.meta_rules[meta_rule.meta_rule_id] = meta_rule

    def get_applicable_rules(self, instance: Any, context: Dict = None) -> List[Rule]:
        """Find all rules that apply to an instance"""
        context = context or self.default_context
        applicable = []

        for rule in self.rules.values():
            if rule.applies_to(instance, context):
                is_exception = False
                for exc_id in rule.exceptions:
                    if exc_id in self.rules:
                        exc_rule = self.rules[exc_id]
                        if exc_rule.applies_to(instance, context):
                            is_exception = True
                            break
                if not is_exception:
                    applicable.append(rule)

        return applicable

    def select_rule(
        self, instance: Any, context: Dict = None, meta_rule_id: str = "priority"
    ) -> Optional[Rule]:
        """Select the best rule to apply"""
        applicable = self.get_applicable_rules(instance, context)

        if not applicable:
            return None

        if len(applicable) == 1:
            return applicable[0]

        if meta_rule_id in self.meta_rules:
            return self.meta_rules[meta_rule_id].rule_selector(applicable, instance, context or {})

        return applicable[0]

    def get_child_rules(self, rule_id: str) -> List[Rule]:
        """Get rules that specialize a parent rule"""
        child_ids = self.rule_hierarchy.get(rule_id, [])
        return [self.rules[cid] for cid in child_ids if cid in self.rules]


class HierarchicalReasoner:
    """
    Abstract rule application across levels.
    Simulates prefrontal cortex hierarchical processing.
    """

    def __init__(self):
        self.rule_hierarchy = RuleHierarchy()
        self.context_stack: List[Dict[str, Any]] = [{}]
        self.learned_rules: List[Rule] = []

    @property
    def current_context(self) -> Dict[str, Any]:
        """Get current context (top of stack)"""
        return self.context_stack[-1] if self.context_stack else {}

    def push_context(self, context: Dict[str, Any]):
        """Push new context onto stack"""
        merged = {**self.current_context, **context}
        self.context_stack.append(merged)

    def pop_context(self) -> Dict[str, Any]:
        """Pop context from stack"""
        if len(self.context_stack) > 1:
            return self.context_stack.pop()
        return self.context_stack[0]

    def apply_rule(self, rule_id: str, instance: Any) -> Tuple[Any, bool]:
        """Apply a specific rule to an instance"""
        if rule_id not in self.rule_hierarchy.rules:
            return instance, False

        rule = self.rule_hierarchy.rules[rule_id]

        if not rule.applies_to(instance, self.current_context):
            return instance, False

        result = rule.apply(instance)
        return result, True

    def reason(self, instance: Any, meta_rule_id: str = "priority") -> Tuple[Any, Optional[str]]:
        """Apply the best applicable rule to an instance"""
        rule = self.rule_hierarchy.select_rule(instance, self.current_context, meta_rule_id)

        if rule is None:
            return instance, None

        result = rule.apply(instance)
        return result, rule.rule_id

    def abstract_rule(
        self, instances: List[Any], results: List[Any], rule_name: str = "learned_rule"
    ) -> Optional[Rule]:
        """
        Generalize a rule from specific instances.
        This is inductive learning at the rule level.
        """
        if len(instances) != len(results) or not instances:
            return None

        common_features = self._find_common_features(instances)

        def learned_condition(x):
            for feature, value in common_features.items():
                if hasattr(x, feature):
                    if getattr(x, feature) != value:
                        return False
                elif isinstance(x, dict) and feature in x:
                    if x[feature] != value:
                        return False
            return True

        transformation = self._infer_transformation(instances, results)

        def learned_action(x):
            return transformation(x)

        rule = Rule(
            rule_id=f"learned_{len(self.learned_rules)}",
            name=rule_name,
            level=AbstractionLevel.INTERMEDIATE,
            condition=learned_condition,
            action=learned_action,
        )

        self.learned_rules.append(rule)
        self.rule_hierarchy.add_rule(rule)

        return rule

    def _find_common_features(self, instances: List[Any]) -> Dict[str, Any]:
        """Find features common to all instances"""
        if not instances:
            return {}

        first = instances[0]

        if isinstance(first, dict):
            common = dict(first)
            for inst in instances[1:]:
                if isinstance(inst, dict):
                    common = {k: v for k, v in common.items() if k in inst and inst[k] == v}
            return common

        if hasattr(first, "__dict__"):
            common = dict(first.__dict__)
            for inst in instances[1:]:
                if hasattr(inst, "__dict__"):
                    common = {
                        k: v
                        for k, v in common.items()
                        if hasattr(inst, k) and getattr(inst, k) == v
                    }
            return common

        return {}

    def _infer_transformation(self, inputs: List[Any], outputs: List[Any]) -> Callable[[Any], Any]:
        """Infer a transformation function from input-output pairs"""
        if all(
            isinstance(i, (int, float)) and isinstance(o, (int, float))
            for i, o in zip(inputs, outputs)
        ):
            diffs = [o - i for i, o in zip(inputs, outputs)]
            if all(abs(d - diffs[0]) < 0.001 for d in diffs):
                offset = diffs[0]
                return lambda x: x + offset

            if all(i != 0 for i in inputs):
                ratios = [o / i for i, o in zip(inputs, outputs)]
                if all(abs(r - ratios[0]) < 0.001 for r in ratios):
                    factor = ratios[0]
                    return lambda x: x * factor

        return lambda x: x

    def compose_rules(
        self, rule1_id: str, rule2_id: str, composition_name: str = "composed_rule"
    ) -> Optional[Rule]:
        """Combine two rules into a sequence"""
        if rule1_id not in self.rule_hierarchy.rules or rule2_id not in self.rule_hierarchy.rules:
            return None

        rule1 = self.rule_hierarchy.rules[rule1_id]
        rule2 = self.rule_hierarchy.rules[rule2_id]

        def composed_condition(x):
            return rule1.condition(x)

        def composed_action(x):
            intermediate = rule1.apply(x)
            if rule2.applies_to(intermediate, self.current_context):
                return rule2.apply(intermediate)
            return intermediate

        combined_context = {**rule1.context_requirements, **rule2.context_requirements}

        composed = Rule(
            rule_id=f"composed_{rule1_id}_{rule2_id}",
            name=composition_name,
            level=max(rule1.level, rule2.level, key=lambda l: l.value),
            condition=composed_condition,
            action=composed_action,
            priority=min(rule1.priority, rule2.priority),
            context_requirements=combined_context,
        )

        self.rule_hierarchy.add_rule(composed)
        return composed

    def find_applicable_sequence(
        self, instance: Any, goal: Callable[[Any], bool], max_depth: int = 5
    ) -> List[str]:
        """Find a sequence of rules that transforms instance to satisfy goal"""
        if goal(instance):
            return []

        visited: Set[str] = set()
        queue: List[Tuple[Any, List[str]]] = [(instance, [])]

        while queue and len(visited) < 1000:
            current, path = queue.pop(0)

            if len(path) >= max_depth:
                continue

            state_key = str(current)
            if state_key in visited:
                continue
            visited.add(state_key)

            applicable = self.rule_hierarchy.get_applicable_rules(current, self.current_context)

            for rule in applicable:
                try:
                    result = rule.apply(current)
                    new_path = path + [rule.rule_id]

                    if goal(result):
                        return new_path

                    queue.append((result, new_path))
                except Exception:
                    continue

        return []
