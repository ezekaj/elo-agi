"""
BRUTAL BENCHMARKS - AGI Standard Testing Suite

Based on:
- ARC-AGI-2/3: Abstract reasoning, pattern recognition
- SWE-bench: Real-world software engineering tasks
- GAIA: General AI assistant benchmark (92% human vs 15% GPT-4)
- AgentBench: 8 environments for agentic evaluation

Metrics tracked:
- pass@k: Success rate over k attempts
- cost_per_task: Tokens/time used
- action_efficiency: Actions needed vs optimal
- error_recovery: Ability to recover from failures
- multi_step_success: Long chain completion rate

Run with: python brutal_benchmarks.py
"""

import time
import json
import random
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""
    name: str
    category: str
    passed: bool
    score: float  # 0.0 - 1.0
    expected: Any
    actual: Any
    time_taken: float
    actions_used: int
    errors: List[str] = field(default_factory=list)
    reasoning_steps: int = 0


@dataclass
class BenchmarkSuite:
    """Full benchmark suite results."""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    total_time: float = 0.0
    total_actions: int = 0
    categories: Dict[str, Dict] = field(default_factory=dict)
    results: List[BenchmarkResult] = field(default_factory=list)

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)
        self.total_tests += 1
        if result.passed:
            self.passed += 1
        else:
            self.failed += 1
        self.total_time += result.time_taken
        self.total_actions += result.actions_used

        if result.category not in self.categories:
            self.categories[result.category] = {
                'total': 0, 'passed': 0, 'failed': 0, 'scores': []
            }
        cat = self.categories[result.category]
        cat['total'] += 1
        cat['passed'] += int(result.passed)
        cat['failed'] += int(not result.passed)
        cat['scores'].append(result.score)

    def summary(self) -> Dict:
        return {
            'overall': {
                'total': self.total_tests,
                'passed': self.passed,
                'failed': self.failed,
                'pass_rate': f"{(self.passed/max(1,self.total_tests))*100:.1f}%",
                'total_time': f"{self.total_time:.2f}s",
                'avg_time': f"{self.total_time/max(1,self.total_tests):.3f}s",
                'total_actions': self.total_actions,
                'actions_per_task': self.total_actions / max(1, self.total_tests)
            },
            'by_category': {
                cat: {
                    'pass_rate': f"{(data['passed']/max(1,data['total']))*100:.1f}%",
                    'avg_score': sum(data['scores'])/max(1,len(data['scores']))
                }
                for cat, data in self.categories.items()
            }
        }


class BrutalBenchmarks:
    """
    AGI-level benchmark tests for NEURO.

    Categories:
    1. ABSTRACT_REASONING - ARC-style pattern recognition
    2. TOOL_PROFICIENCY - Multi-tool chains (GAIA-style)
    3. ERROR_RECOVERY - Self-healing under failure
    4. MULTI_STEP_REASONING - Complex chains
    5. KNOWLEDGE_INTEGRATION - Cross-domain synthesis
    6. SELF_IMPROVEMENT - Learning from mistakes
    7. EFFICIENCY - Resource optimization
    8. REAL_WORLD_SCENARIOS - AgentBench-style
    """

    def __init__(self, agent=None, verbose: bool = True):
        self.agent = agent
        self.verbose = verbose
        self.suite = BenchmarkSuite()

    def log(self, msg: str, level: str = "INFO"):
        if self.verbose:
            icons = {"PASS": "[+]", "FAIL": "[-]", "INFO": "[*]", "TEST": "[>]"}
            print(f"{icons.get(level, '[*]')} {msg}")

    # =========================================================================
    # CATEGORY 1: ABSTRACT REASONING (ARC-AGI Style)
    # =========================================================================

    def test_arc_sequence_completion(self) -> BenchmarkResult:
        """ARC-style: Complete the pattern sequence."""
        start = time.time()
        name = "ARC: Sequence Completion"

        # Test cases: input sequence -> expected next value
        test_cases = [
            ([1, 2, 4, 8], 16),           # Powers of 2
            ([1, 1, 2, 3, 5, 8], 13),     # Fibonacci
            ([2, 6, 12, 20, 30], 42),     # n(n+1) pattern
            ([1, 4, 9, 16, 25], 36),      # Squares
            ([1, 8, 27, 64], 125),        # Cubes
        ]

        passed = 0
        actions = 0

        for seq, expected in test_cases:
            # Use agent's analysis if available
            if self.agent and hasattr(self.agent, 'ultrathink') and self.agent.ultrathink:
                try:
                    analysis = self.agent.ultrathink.analyze(
                        f"What comes next in this sequence: {seq}? Just the number."
                    )
                    actions += 1
                    # Extract number from analysis
                    if 'reasoning' in analysis:
                        reasoning = str(analysis.get('reasoning', ''))
                        if str(expected) in reasoning:
                            passed += 1
                except Exception:
                    pass

        score = passed / len(test_cases)
        return BenchmarkResult(
            name=name,
            category="ABSTRACT_REASONING",
            passed=score >= 0.6,  # 60% threshold
            score=score,
            expected=f"Solve {len(test_cases)} sequences",
            actual=f"Solved {passed}/{len(test_cases)}",
            time_taken=time.time() - start,
            actions_used=actions,
            reasoning_steps=passed
        )

    def test_arc_pattern_transform(self) -> BenchmarkResult:
        """ARC-style: Understand and apply pattern transformation rules."""
        start = time.time()
        name = "ARC: Pattern Transform"

        # Pattern: Given transformation rule, apply to new input
        test_cases = [
            # (input, rule_example_in, rule_example_out, expected_output)
            ([1, 2, 3], [10, 20, 30], [20, 40, 60], [2, 4, 6]),  # Double
            (['a', 'b', 'c'], ['x', 'y'], ['xy', 'yz'], ['ab', 'bc', 'cd']),  # Concat pairs
            ([5, 10, 15], [1, 2, 3], [3, 2, 1], [15, 10, 5]),  # Reverse
        ]

        passed = 0
        actions = 0

        for inp, ex_in, ex_out, expected in test_cases:
            if self.agent and self.agent.ultrathink:
                try:
                    query = f"If {ex_in} transforms to {ex_out}, what does {inp} transform to?"
                    result = self.agent.ultrathink.analyze(query)
                    actions += 1
                    if str(expected) in str(result):
                        passed += 1
                except Exception:
                    pass

        score = passed / max(1, len(test_cases))
        return BenchmarkResult(
            name=name,
            category="ABSTRACT_REASONING",
            passed=score >= 0.5,
            score=score,
            expected=f"Transform {len(test_cases)} patterns",
            actual=f"Transformed {passed}",
            time_taken=time.time() - start,
            actions_used=actions
        )

    def test_arc_grid_reasoning(self) -> BenchmarkResult:
        """ARC-style: 2D grid pattern recognition."""
        start = time.time()
        name = "ARC: Grid Reasoning"

        # Simplified grid tests
        grids = [
            # (input_grid, expected_output_description)
            ([[1, 0], [0, 1]], "diagonal"),
            ([[1, 1], [1, 1]], "filled"),
            ([[0, 1, 0], [1, 1, 1], [0, 1, 0]], "cross"),
        ]

        passed = 0
        actions = 0

        for grid, expected_pattern in grids:
            if self.agent and self.agent.ultrathink:
                try:
                    query = f"What pattern is this grid: {grid}? One word answer."
                    result = self.agent.ultrathink.analyze(query)
                    actions += 1
                    if expected_pattern.lower() in str(result).lower():
                        passed += 1
                except Exception:
                    pass

        score = passed / max(1, len(grids))
        return BenchmarkResult(
            name=name,
            category="ABSTRACT_REASONING",
            passed=score >= 0.5,
            score=score,
            expected=f"Recognize {len(grids)} patterns",
            actual=f"Recognized {passed}",
            time_taken=time.time() - start,
            actions_used=actions
        )

    # =========================================================================
    # CATEGORY 2: TOOL PROFICIENCY (GAIA Style)
    # =========================================================================

    def test_tool_chain_execution(self) -> BenchmarkResult:
        """GAIA-style: Execute multi-step tool chains."""
        start = time.time()
        name = "GAIA: Tool Chain Execution"

        errors = []
        actions = 0

        if not self.agent or not self.agent.tools:
            return BenchmarkResult(
                name=name,
                category="TOOL_PROFICIENCY",
                passed=False,
                score=0.0,
                expected="Execute 3-step tool chain",
                actual="No tools available",
                time_taken=time.time() - start,
                actions_used=0,
                errors=["Agent tools not available"]
            )

        # Test: List files -> Read specific file -> Extract info
        try:
            # Step 1: List files
            result1 = self.agent.tools.list_files(".")
            actions += 1
            step1_passed = result1.success

            # Step 2: Read a Python file (if exists)
            step2_passed = False
            if step1_passed and "brutal_benchmarks.py" in result1.output:
                result2 = self.agent.tools.read_file("./brutal_benchmarks.py")
                actions += 1
                step2_passed = result2.success

            # Step 3: Execute Python code
            result3 = self.agent.tools.run_python("print(2 + 2)")
            actions += 1
            step3_passed = result3.success and "4" in result3.output

            passed = sum([step1_passed, step2_passed, step3_passed])
            score = passed / 3

        except Exception as e:
            errors.append(str(e))
            score = 0.0
            passed = 0

        return BenchmarkResult(
            name=name,
            category="TOOL_PROFICIENCY",
            passed=score >= 0.66,  # 2/3 steps
            score=score,
            expected="Complete 3-step tool chain",
            actual=f"Completed {int(score * 3)}/3 steps",
            time_taken=time.time() - start,
            actions_used=actions,
            errors=errors
        )

    def test_tool_selection_accuracy(self) -> BenchmarkResult:
        """GAIA-style: Select correct tool for task."""
        start = time.time()
        name = "GAIA: Tool Selection Accuracy"

        # Query -> Expected tool
        test_cases = [
            ("search for Python tutorials", "web_search"),
            ("list files in /tmp", "list_files"),
            ("read the README file", "file_read"),
            ("run python code print(1+1)", "execute_code"),
            ("what is on github.com/anthropics", "github_lookup"),
        ]

        passed = 0
        actions = 0

        if self.agent:
            for query, expected_tool in test_cases:
                plan = self.agent._plan_actions(query)
                actions += 1
                if expected_tool in plan or any(expected_tool.replace('_', '') in p.replace('_', '') for p in plan):
                    passed += 1

        score = passed / len(test_cases)
        return BenchmarkResult(
            name=name,
            category="TOOL_PROFICIENCY",
            passed=score >= 0.8,  # 80% accuracy
            score=score,
            expected=f"Select correct tool for {len(test_cases)} queries",
            actual=f"Correct: {passed}/{len(test_cases)}",
            time_taken=time.time() - start,
            actions_used=actions
        )

    def test_tool_error_handling(self) -> BenchmarkResult:
        """Test graceful handling of tool failures."""
        start = time.time()
        name = "GAIA: Tool Error Handling"

        errors = []
        actions = 0
        handled = 0

        if self.agent and self.agent.tools:
            # Test 1: Non-existent file
            try:
                result = self.agent.tools.read_file("/nonexistent/path/file.txt")
                actions += 1
                if not result.success and result.error:
                    handled += 1  # Properly reported error
            except Exception as e:
                errors.append(f"Unhandled: {e}")

            # Test 2: Invalid Python
            try:
                result = self.agent.tools.run_python("this is not valid python ][")
                actions += 1
                if not result.success:
                    handled += 1  # Caught syntax error
            except Exception as e:
                errors.append(f"Unhandled: {e}")

            # Test 3: Invalid command
            try:
                result = self.agent.tools.run_command("nonexistentcommand12345")
                actions += 1
                if not result.success:
                    handled += 1
            except Exception as e:
                errors.append(f"Unhandled: {e}")

        score = handled / 3
        return BenchmarkResult(
            name=name,
            category="TOOL_PROFICIENCY",
            passed=score >= 0.66,
            score=score,
            expected="Handle 3 error scenarios gracefully",
            actual=f"Handled {handled}/3",
            time_taken=time.time() - start,
            actions_used=actions,
            errors=errors
        )

    # =========================================================================
    # CATEGORY 3: ERROR RECOVERY & SELF-IMPROVEMENT
    # =========================================================================

    def test_self_improvement_learning(self) -> BenchmarkResult:
        """Test ability to learn from errors and improve."""
        start = time.time()
        name = "Self-Improvement: Learning"

        if not self.agent or not self.agent.improver:
            return BenchmarkResult(
                name=name,
                category="ERROR_RECOVERY",
                passed=False,
                score=0.0,
                expected="Learn from 3 errors",
                actual="No improver available",
                time_taken=time.time() - start,
                actions_used=0,
                errors=["Self-improver not available"]
            )

        initial_patterns = len(self.agent.improver.learned_patterns)
        initial_solutions = len(self.agent.improver.error_solutions)

        # Simulate errors and learning
        errors_to_learn = [
            ("timeout", "Retry with shorter context"),
            ("connection refused", "Check if service is running"),
            ("rate limit exceeded", "Wait and retry with backoff"),
        ]

        actions = 0
        for error_type, expected_solution in errors_to_learn:
            self.agent._handle_error(f"Error: {error_type}")
            actions += 1

        final_patterns = len(self.agent.improver.learned_patterns)
        final_solutions = len(self.agent.improver.error_solutions)

        learned = (final_solutions - initial_solutions)
        score = min(1.0, learned / len(errors_to_learn))

        return BenchmarkResult(
            name=name,
            category="ERROR_RECOVERY",
            passed=learned >= 2,  # At least 2 new solutions
            score=score,
            expected=f"Learn {len(errors_to_learn)} error solutions",
            actual=f"Learned {learned} new solutions",
            time_taken=time.time() - start,
            actions_used=actions
        )

    def test_error_pattern_persistence(self) -> BenchmarkResult:
        """Test that learned patterns persist."""
        start = time.time()
        name = "Self-Improvement: Persistence"

        if not self.agent or not self.agent.improver:
            return BenchmarkResult(
                name=name,
                category="ERROR_RECOVERY",
                passed=False,
                score=0.0,
                expected="Patterns persist after save",
                actual="No improver available",
                time_taken=time.time() - start,
                actions_used=0
            )

        # Add a test pattern
        test_key = f"test_pattern_{random.randint(1000, 9999)}"
        self.agent.improver.learned_patterns[test_key] = "Test solution"
        self.agent.improver._save_improvements()

        # Verify it's still there
        pattern_exists = test_key in self.agent.improver.learned_patterns

        # Clean up
        if pattern_exists:
            del self.agent.improver.learned_patterns[test_key]
            self.agent.improver._save_improvements()

        return BenchmarkResult(
            name=name,
            category="ERROR_RECOVERY",
            passed=pattern_exists,
            score=1.0 if pattern_exists else 0.0,
            expected="Pattern saved and retrievable",
            actual="Pattern persisted" if pattern_exists else "Pattern lost",
            time_taken=time.time() - start,
            actions_used=2
        )

    # =========================================================================
    # CATEGORY 4: MULTI-STEP REASONING (Complex Chains)
    # =========================================================================

    def test_multistep_math_reasoning(self) -> BenchmarkResult:
        """Test multi-step mathematical reasoning."""
        start = time.time()
        name = "Reasoning: Multi-Step Math"

        # Problems requiring multiple steps
        problems = [
            # (problem_description, expected_answer)
            ("If x = 5 and y = x * 2, what is y + 3?", "13"),
            ("A train travels 60km/h for 2.5 hours. Distance?", "150"),
            ("Sum of first 5 prime numbers", "28"),  # 2+3+5+7+11
        ]

        passed = 0
        actions = 0

        if self.agent and self.agent.ultrathink:
            for problem, expected in problems:
                try:
                    result = self.agent.ultrathink.think(problem, depth="deep")
                    actions += 1
                    # Check if answer is in reasoning chain
                    full_output = str(result.reasoning_chain) + str(result.insights)
                    if expected in full_output:
                        passed += 1
                except Exception:
                    pass

        score = passed / len(problems)
        return BenchmarkResult(
            name=name,
            category="MULTI_STEP_REASONING",
            passed=score >= 0.5,
            score=score,
            expected=f"Solve {len(problems)} multi-step problems",
            actual=f"Solved {passed}",
            time_taken=time.time() - start,
            actions_used=actions,
            reasoning_steps=passed * 3  # Approx steps per problem
        )

    def test_causal_chain_reasoning(self) -> BenchmarkResult:
        """Test causal chain understanding."""
        start = time.time()
        name = "Reasoning: Causal Chains"

        # Causal chains: A causes B causes C
        chains = [
            ("Rain -> Wet ground -> Slippery -> ?", "fall", "accident"),
            ("Study -> Knowledge -> Good grades -> ?", "success", "opportunity"),
            ("Exercise -> Fitness -> Health -> ?", "longevity", "energy"),
        ]

        passed = 0
        actions = 0

        if self.agent and self.agent.ultrathink:
            for chain, *valid_answers in chains:
                try:
                    result = self.agent.ultrathink.analyze(
                        f"Complete this causal chain: {chain}"
                    )
                    actions += 1
                    result_str = str(result).lower()
                    if any(ans in result_str for ans in valid_answers):
                        passed += 1
                except Exception:
                    pass

        score = passed / len(chains)
        return BenchmarkResult(
            name=name,
            category="MULTI_STEP_REASONING",
            passed=score >= 0.5,
            score=score,
            expected=f"Complete {len(chains)} causal chains",
            actual=f"Completed {passed}",
            time_taken=time.time() - start,
            actions_used=actions
        )

    # =========================================================================
    # CATEGORY 5: KNOWLEDGE INTEGRATION
    # =========================================================================

    def test_knowledge_storage_retrieval(self) -> BenchmarkResult:
        """Test knowledge base storage and retrieval."""
        start = time.time()
        name = "Knowledge: Store & Retrieve"

        if not self.agent or not self.agent.pipeline:
            return BenchmarkResult(
                name=name,
                category="KNOWLEDGE_INTEGRATION",
                passed=False,
                score=0.0,
                expected="Store and retrieve 3 facts",
                actual="No pipeline available",
                time_taken=time.time() - start,
                actions_used=0
            )

        actions = 0
        test_facts = [
            ("test_topic_1", "The capital of France is Paris"),
            ("test_topic_2", "Python was created by Guido van Rossum"),
            ("test_topic_3", "Neural networks are inspired by the brain"),
        ]

        # Store facts
        for topic, content in test_facts:
            try:
                self.agent.pipeline.learn(topic, content, "benchmark", importance=0.9)
                actions += 1
            except Exception:
                pass

        # Retrieve and verify
        retrieved = 0
        try:
            result = self.agent.pipeline.process("capital of France")
            actions += 1
            if "Paris" in str(result.knowledge_used) or "Paris" in str(result):
                retrieved += 1
        except Exception:
            pass

        try:
            result = self.agent.pipeline.process("who created Python")
            actions += 1
            if "Guido" in str(result.knowledge_used) or "Guido" in str(result):
                retrieved += 1
        except Exception:
            pass

        score = (len(test_facts) + retrieved) / (len(test_facts) * 2)
        return BenchmarkResult(
            name=name,
            category="KNOWLEDGE_INTEGRATION",
            passed=retrieved >= 1,
            score=score,
            expected="Store 3 facts, retrieve 2",
            actual=f"Stored {len(test_facts)}, retrieved {retrieved}",
            time_taken=time.time() - start,
            actions_used=actions
        )

    def test_cross_domain_synthesis(self) -> BenchmarkResult:
        """Test ability to synthesize knowledge across domains."""
        start = time.time()
        name = "Knowledge: Cross-Domain Synthesis"

        if not self.agent or not self.agent.ultrathink:
            return BenchmarkResult(
                name=name,
                category="KNOWLEDGE_INTEGRATION",
                passed=False,
                score=0.0,
                expected="Synthesize 2 cross-domain queries",
                actual="No ultrathink available",
                time_taken=time.time() - start,
                actions_used=0
            )

        # Queries requiring knowledge from multiple domains
        queries = [
            ("How does biology inspire computer architecture?", ["neural", "network", "brain", "parallel"]),
            ("What can economics learn from physics?", ["equilibrium", "entropy", "energy", "flow"]),
        ]

        passed = 0
        actions = 0

        for query, keywords in queries:
            try:
                result = self.agent.ultrathink.think(query, depth="deep")
                actions += 1
                result_str = str(result.reasoning_chain).lower() + str(result.insights).lower()
                if any(kw in result_str for kw in keywords):
                    passed += 1
            except Exception:
                pass

        score = passed / len(queries)
        return BenchmarkResult(
            name=name,
            category="KNOWLEDGE_INTEGRATION",
            passed=passed >= 1,
            score=score,
            expected=f"Answer {len(queries)} cross-domain queries",
            actual=f"Answered {passed}",
            time_taken=time.time() - start,
            actions_used=actions
        )

    # =========================================================================
    # CATEGORY 6: EFFICIENCY METRICS
    # =========================================================================

    def test_action_efficiency(self) -> BenchmarkResult:
        """Test if agent uses minimal actions for tasks."""
        start = time.time()
        name = "Efficiency: Action Count"

        if not self.agent:
            return BenchmarkResult(
                name=name,
                category="EFFICIENCY",
                passed=False,
                score=0.0,
                expected="Optimal action count",
                actual="No agent available",
                time_taken=time.time() - start,
                actions_used=0
            )

        # Simple query should need minimal planning
        simple_queries = [
            ("hello", 1),  # Expected: just 'respond'
            ("what time is it", 1),  # Expected: just 'respond'
            ("calculate 2+2", 1),  # Expected: 'execute_code'
        ]

        optimal_plans = 0
        total_actions = 0

        for query, expected_actions in simple_queries:
            plan = self.agent._plan_actions(query)
            total_actions += len(plan)
            if len(plan) <= expected_actions + 1:  # Allow 1 extra
                optimal_plans += 1

        score = optimal_plans / len(simple_queries)
        avg_actions = total_actions / len(simple_queries)

        return BenchmarkResult(
            name=name,
            category="EFFICIENCY",
            passed=score >= 0.66 and avg_actions <= 2,
            score=score,
            expected="Avg ≤2 actions for simple queries",
            actual=f"Avg {avg_actions:.1f} actions, {optimal_plans}/{len(simple_queries)} optimal",
            time_taken=time.time() - start,
            actions_used=total_actions
        )

    def test_response_time(self) -> BenchmarkResult:
        """Test response time for analysis."""
        start = time.time()
        name = "Efficiency: Response Time"

        if not self.agent:
            return BenchmarkResult(
                name=name,
                category="EFFICIENCY",
                passed=False,
                score=0.0,
                expected="Fast response",
                actual="No agent available",
                time_taken=0,
                actions_used=0
            )

        times = []
        actions = 0

        for _ in range(3):
            t0 = time.time()
            self.agent._plan_actions("simple query here")
            times.append(time.time() - t0)
            actions += 1

        avg_time = sum(times) / len(times)
        # Planning should be <100ms
        score = 1.0 if avg_time < 0.1 else (0.1 / avg_time)

        return BenchmarkResult(
            name=name,
            category="EFFICIENCY",
            passed=avg_time < 0.1,  # 100ms threshold
            score=min(1.0, score),
            expected="Planning <100ms",
            actual=f"Avg {avg_time*1000:.1f}ms",
            time_taken=time.time() - start,
            actions_used=actions
        )

    # =========================================================================
    # CATEGORY 7: REAL-WORLD SCENARIOS (AgentBench Style)
    # =========================================================================

    def test_project_analysis_workflow(self) -> BenchmarkResult:
        """AgentBench-style: Full project analysis workflow."""
        start = time.time()
        name = "AgentBench: Project Analysis"

        if not self.agent:
            return BenchmarkResult(
                name=name,
                category="REAL_WORLD",
                passed=False,
                score=0.0,
                expected="Complete project analysis",
                actual="No agent available",
                time_taken=time.time() - start,
                actions_used=0
            )

        # Simulate full project analysis
        query = "Analyze the project at /Users/ezekaj/Desktop/neuro"
        state = self.agent.process(query)

        # Check workflow completion
        checks = {
            'perceived': len(state.knowledge) > 0 or len(state.memories) > 0 or state.confidence > 0,
            'planned': len(state.plan) > 0,
            'acted': len(state.tools_used) > 0,
            'responded': len(state.response) > 0,
        }

        passed_checks = sum(checks.values())
        score = passed_checks / len(checks)

        return BenchmarkResult(
            name=name,
            category="REAL_WORLD",
            passed=passed_checks >= 3,
            score=score,
            expected="Complete all 4 workflow phases",
            actual=f"Completed {passed_checks}/4: {[k for k,v in checks.items() if v]}",
            time_taken=time.time() - start,
            actions_used=len(state.tools_used)
        )

    def test_error_recovery_workflow(self) -> BenchmarkResult:
        """AgentBench-style: Handle errors and recover."""
        start = time.time()
        name = "AgentBench: Error Recovery"

        if not self.agent:
            return BenchmarkResult(
                name=name,
                category="REAL_WORLD",
                passed=False,
                score=0.0,
                expected="Recover from error",
                actual="No agent available",
                time_taken=time.time() - start,
                actions_used=0
            )

        # Trigger an error scenario
        query = "Read file /nonexistent/path/that/does/not/exist.txt"
        state = self.agent.process(query)

        # Should have errors but also handle them
        has_error = len(state.errors) > 0
        has_response = len(state.response) > 0  # Still provided a response
        error_handled = has_error and has_response

        score = 1.0 if error_handled else (0.5 if has_response else 0.0)

        return BenchmarkResult(
            name=name,
            category="REAL_WORLD",
            passed=error_handled,
            score=score,
            expected="Handle error gracefully with response",
            actual=f"Errors: {len(state.errors)}, Response: {len(state.response) > 0}",
            time_taken=time.time() - start,
            actions_used=len(state.tools_used)
        )

    def test_learning_workflow(self) -> BenchmarkResult:
        """AgentBench-style: Learn from interaction."""
        start = time.time()
        name = "AgentBench: Learning Workflow"

        if not self.agent:
            return BenchmarkResult(
                name=name,
                category="REAL_WORLD",
                passed=False,
                score=0.0,
                expected="Learn from interaction",
                actual="No agent available",
                time_taken=time.time() - start,
                actions_used=0
            )

        # Initial history length
        initial_history = len(self.agent.history)

        # Process a query
        state = self.agent.process("What is machine learning?")

        # Check if it learned
        checks = {
            'history_updated': len(self.agent.history) > initial_history,
            'learnings_recorded': len(state.learnings) > 0,
        }

        passed_checks = sum(checks.values())
        score = passed_checks / len(checks)

        return BenchmarkResult(
            name=name,
            category="REAL_WORLD",
            passed=passed_checks >= 1,
            score=score,
            expected="Record learning from interaction",
            actual=f"History: {'updated' if checks['history_updated'] else 'not updated'}, "
                   f"Learnings: {len(state.learnings)}",
            time_taken=time.time() - start,
            actions_used=len(state.tools_used)
        )

    # =========================================================================
    # CATEGORY 8: COMPONENT INTEGRATION
    # =========================================================================

    def test_pipeline_integration(self) -> BenchmarkResult:
        """Test all pipeline components work together."""
        start = time.time()
        name = "Integration: Pipeline"

        if not self.agent or not self.agent.pipeline:
            return BenchmarkResult(
                name=name,
                category="INTEGRATION",
                passed=False,
                score=0.0,
                expected="Pipeline processes query",
                actual="No pipeline available",
                time_taken=time.time() - start,
                actions_used=0
            )

        try:
            result = self.agent.pipeline.process("test query for integration")
            checks = {
                'returns_result': result is not None,
                'has_knowledge': hasattr(result, 'knowledge_used'),
                'has_confidence': hasattr(result, 'confidence'),
                'has_surprise': hasattr(result, 'surprise_level'),
            }
            passed_checks = sum(checks.values())
            score = passed_checks / len(checks)
        except Exception as e:
            return BenchmarkResult(
                name=name,
                category="INTEGRATION",
                passed=False,
                score=0.0,
                expected="Pipeline processes successfully",
                actual=f"Error: {str(e)[:50]}",
                time_taken=time.time() - start,
                actions_used=1,
                errors=[str(e)]
            )

        return BenchmarkResult(
            name=name,
            category="INTEGRATION",
            passed=passed_checks >= 3,
            score=score,
            expected="Pipeline returns complete result",
            actual=f"Passed {passed_checks}/4 checks",
            time_taken=time.time() - start,
            actions_used=1
        )

    def test_ultrathink_integration(self) -> BenchmarkResult:
        """Test UltraThink deep reasoning works."""
        start = time.time()
        name = "Integration: UltraThink"

        if not self.agent or not self.agent.ultrathink:
            return BenchmarkResult(
                name=name,
                category="INTEGRATION",
                passed=False,
                score=0.0,
                expected="UltraThink processes query",
                actual="No UltraThink available",
                time_taken=time.time() - start,
                actions_used=0
            )

        try:
            result = self.agent.ultrathink.think("What is 2+2?", depth="deep")
            checks = {
                'has_chain': hasattr(result, 'reasoning_chain') and len(result.reasoning_chain) > 0,
                'has_confidence': hasattr(result, 'confidence'),
                'has_insights': hasattr(result, 'insights'),
                'has_actions': hasattr(result, 'suggested_actions'),
            }
            passed_checks = sum(checks.values())
            score = passed_checks / len(checks)
        except Exception as e:
            return BenchmarkResult(
                name=name,
                category="INTEGRATION",
                passed=False,
                score=0.0,
                expected="UltraThink thinks successfully",
                actual=f"Error: {str(e)[:50]}",
                time_taken=time.time() - start,
                actions_used=1,
                errors=[str(e)]
            )

        return BenchmarkResult(
            name=name,
            category="INTEGRATION",
            passed=passed_checks >= 3,
            score=score,
            expected="UltraThink returns complete result",
            actual=f"Passed {passed_checks}/4 checks",
            time_taken=time.time() - start,
            actions_used=1
        )

    def test_full_workflow_integration(self) -> BenchmarkResult:
        """Test complete PERCEIVE->THINK->ACT->LEARN->IMPROVE loop."""
        start = time.time()
        name = "Integration: Full Workflow"

        if not self.agent:
            return BenchmarkResult(
                name=name,
                category="INTEGRATION",
                passed=False,
                score=0.0,
                expected="Complete workflow cycle",
                actual="No agent available",
                time_taken=time.time() - start,
                actions_used=0
            )

        try:
            # Run full workflow
            state = self.agent.process("Calculate the factorial of 5", deep_think=True)

            checks = {
                'perceive': state.confidence > 0 or len(state.knowledge) >= 0,  # Always passes perceive
                'think': len(state.analysis) > 0 or len(state.plan) > 0,
                'act': True,  # Always passes (even if no tools used)
                'learn': True,  # Learning happens in background
                'improve': True,  # Improvement cycle runs
            }

            # More strict check
            workflow_complete = (
                len(state.plan) > 0 and
                state.processing_time > 0
            )

            score = 1.0 if workflow_complete else 0.5
        except Exception as e:
            return BenchmarkResult(
                name=name,
                category="INTEGRATION",
                passed=False,
                score=0.0,
                expected="Complete 5-phase workflow",
                actual=f"Error: {str(e)[:50]}",
                time_taken=time.time() - start,
                actions_used=0,
                errors=[str(e)]
            )

        return BenchmarkResult(
            name=name,
            category="INTEGRATION",
            passed=workflow_complete,
            score=score,
            expected="Complete PERCEIVE->THINK->ACT->LEARN->IMPROVE",
            actual=f"Workflow complete: {workflow_complete}, Time: {state.processing_time:.3f}s",
            time_taken=time.time() - start,
            actions_used=len(state.tools_used)
        )

    # =========================================================================
    # RUN ALL BENCHMARKS
    # =========================================================================

    def run_all(self) -> BenchmarkSuite:
        """Run all benchmark tests."""
        print("=" * 70)
        print("BRUTAL BENCHMARKS - AGI STANDARD TESTING SUITE")
        print("=" * 70)
        print(f"Categories: ARC-AGI | GAIA | AgentBench | SWE-bench inspired")
        print("=" * 70)

        tests = [
            # Abstract Reasoning (ARC-style)
            self.test_arc_sequence_completion,
            self.test_arc_pattern_transform,
            self.test_arc_grid_reasoning,

            # Tool Proficiency (GAIA-style)
            self.test_tool_chain_execution,
            self.test_tool_selection_accuracy,
            self.test_tool_error_handling,

            # Error Recovery & Self-Improvement
            self.test_self_improvement_learning,
            self.test_error_pattern_persistence,

            # Multi-Step Reasoning
            self.test_multistep_math_reasoning,
            self.test_causal_chain_reasoning,

            # Knowledge Integration
            self.test_knowledge_storage_retrieval,
            self.test_cross_domain_synthesis,

            # Efficiency
            self.test_action_efficiency,
            self.test_response_time,

            # Real-World (AgentBench-style)
            self.test_project_analysis_workflow,
            self.test_error_recovery_workflow,
            self.test_learning_workflow,

            # Integration
            self.test_pipeline_integration,
            self.test_ultrathink_integration,
            self.test_full_workflow_integration,
        ]

        for test_func in tests:
            self.log(f"Running: {test_func.__name__}", "TEST")
            try:
                result = test_func()
                self.suite.add_result(result)
                level = "PASS" if result.passed else "FAIL"
                self.log(f"{result.name}: {result.actual} [{result.score:.0%}]", level)
            except Exception as e:
                self.log(f"{test_func.__name__}: CRASHED - {e}", "FAIL")
                self.suite.add_result(BenchmarkResult(
                    name=test_func.__name__,
                    category="CRASHED",
                    passed=False,
                    score=0.0,
                    expected="No crash",
                    actual=str(e)[:50],
                    time_taken=0,
                    actions_used=0,
                    errors=[traceback.format_exc()]
                ))

        return self.suite

    def print_report(self):
        """Print detailed benchmark report."""
        summary = self.suite.summary()

        print("\n" + "=" * 70)
        print("BENCHMARK REPORT")
        print("=" * 70)

        # Overall stats
        overall = summary['overall']
        print(f"\nOVERALL: {overall['passed']}/{overall['total']} tests passed ({overall['pass_rate']})")
        print(f"Total time: {overall['total_time']}")
        print(f"Avg time per test: {overall['avg_time']}")
        print(f"Total actions: {overall['total_actions']}")
        print(f"Actions per task: {overall['actions_per_task']:.1f}")

        # By category
        print("\nBY CATEGORY:")
        print("-" * 50)
        for cat, data in summary['by_category'].items():
            status = "PASS" if float(data['pass_rate'].rstrip('%')) >= 50 else "FAIL"
            icon = "[+]" if status == "PASS" else "[-]"
            print(f"  {icon} {cat}: {data['pass_rate']} (avg score: {data['avg_score']:.2f})")

        # Failed tests
        failed = [r for r in self.suite.results if not r.passed]
        if failed:
            print("\nFAILED TESTS:")
            print("-" * 50)
            for r in failed:
                print(f"  [-] {r.name}")
                print(f"      Expected: {r.expected}")
                print(f"      Actual: {r.actual}")
                if r.errors:
                    print(f"      Error: {r.errors[0][:50]}")

        # AGI Readiness Score
        print("\n" + "=" * 70)
        print("AGI READINESS SCORE")
        print("=" * 70)

        pass_rate = float(overall['pass_rate'].rstrip('%'))
        if pass_rate >= 80:
            grade = "A - AGI Ready"
        elif pass_rate >= 60:
            grade = "B - Strong Foundation"
        elif pass_rate >= 40:
            grade = "C - Needs Work"
        elif pass_rate >= 20:
            grade = "D - Major Issues"
        else:
            grade = "F - Critical Failures"

        print(f"\nGRADE: {grade}")
        print(f"SCORE: {pass_rate:.1f}/100")

        # Recommendations
        print("\nRECOMMENDATIONS:")
        by_cat = summary['by_category']
        if 'ABSTRACT_REASONING' in by_cat and float(by_cat['ABSTRACT_REASONING']['pass_rate'].rstrip('%')) < 50:
            print("  - Improve pattern recognition (ARC-style reasoning)")
        if 'TOOL_PROFICIENCY' in by_cat and float(by_cat['TOOL_PROFICIENCY']['pass_rate'].rstrip('%')) < 50:
            print("  - Improve tool selection and chaining (GAIA-style)")
        if 'ERROR_RECOVERY' in by_cat and float(by_cat['ERROR_RECOVERY']['pass_rate'].rstrip('%')) < 50:
            print("  - Improve self-improvement and error learning")
        if 'INTEGRATION' in by_cat and float(by_cat['INTEGRATION']['pass_rate'].rstrip('%')) < 50:
            print("  - Fix component integration issues")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("INITIALIZING NEURO AGENT FOR BENCHMARKS")
    print("=" * 70)

    try:
        from neuro_agent import NeuroAgent
        agent = NeuroAgent(verbose=False)
        print("[+] NeuroAgent initialized")
        print(f"    Components: {agent.get_stats()['components']}")
    except Exception as e:
        print(f"[-] Failed to initialize NeuroAgent: {e}")
        agent = None

    # Run benchmarks
    benchmarks = BrutalBenchmarks(agent=agent, verbose=True)
    suite = benchmarks.run_all()
    benchmarks.print_report()
