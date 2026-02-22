"""
UltraThink - Maximum Intelligence Orchestrator

Coordinates all cognitive modules for deep reasoning and complex problem solving.
This is the "thinking hard" mode that uses the full cognitive architecture.
"""

import time
import json
import numpy as np
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

# Import cognitive adapters (handles all module loading)
from .cognitive_adapters import (
    CognitiveModuleFactory,
    ProblemType,
    ThinkingStyle,
    ProblemDifficulty,
    ProblemAnalysis,
    StyleSelection,
    ExecutionPlan,
)


@dataclass
class ThoughtStep:
    """A single step in the thinking process."""

    module: str
    thought: str
    confidence: float
    duration: float
    insights: List[str] = field(default_factory=list)


@dataclass
class UltraThinkResult:
    """Result from deep thinking."""

    final_answer: str
    reasoning_chain: List[ThoughtStep]
    modules_used: List[str]
    total_time: float
    confidence: float
    insights: List[str]
    suggested_actions: List[str]


class UltraThink:
    """
    Maximum Intelligence Orchestrator.

    Coordinates all cognitive modules for complex reasoning:
    1. Perception & Analysis
    2. Multi-style Reasoning
    3. Memory Integration
    4. Predictive Coding
    5. Executive Control
    6. Action Planning
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._log("Initializing cognitive modules...")
        self.factory = CognitiveModuleFactory(verbose=verbose)
        self.modules = self.factory.get_all()

        stats = self.factory.statistics()
        self._log(
            f"Loaded {stats['total_modules']} modules ({stats['active_modules']} active, {stats['fallback_modules']} fallback)"
        )

    def _text_to_embedding(self, text: str, dim: int = 128) -> np.ndarray:
        """Convert text to embedding vector."""
        hash_bytes = hashlib.sha256(text.encode()).digest()
        extended = hash_bytes * (dim // len(hash_bytes) + 1)
        embedding = np.array([b / 255.0 for b in extended[:dim]])
        return embedding

    def _log(self, message: str) -> None:
        """Log a message if verbose."""
        if self.verbose:
            print(f"  [UltraThink] {message}")

    def think(
        self,
        problem: str,
        context: Optional[str] = None,
        depth: str = "deep",  # "quick", "normal", "deep", "exhaustive"
        focus_modules: Optional[List[str]] = None,
    ) -> UltraThinkResult:
        """
        Think deeply about a problem using all cognitive modules.

        Args:
            problem: The problem or question to think about
            context: Optional additional context
            depth: How deeply to think ("quick", "normal", "deep", "exhaustive")
            focus_modules: Optional list of specific modules to prioritize

        Returns:
            UltraThinkResult with reasoning chain and answer
        """
        start_time = time.time()
        reasoning_chain = []
        insights = []
        modules_used = []

        self._log(f"Starting {depth} analysis...")

        # Create embedding for the problem
        embedding = self._text_to_embedding(problem)
        full_context = f"{problem}\n\nContext: {context}" if context else problem

        # === PHASE 1: PERCEPTION & CLASSIFICATION ===
        self._log("Phase 1: Perceiving and classifying problem...")

        analysis = None
        classifier = self.modules.get("problem_classifier")
        if classifier:
            step_start = time.time()
            analysis = classifier.classify(embedding)

            insights_list = []
            if analysis.requires_domain_knowledge:
                insights_list.append("Requires domain-specific knowledge")
            if analysis.complexity > 0.7:
                insights_list.append(f"High complexity: {analysis.complexity:.0%}")

            step = ThoughtStep(
                module="problem_classifier",
                thought=f"Problem type: {analysis.problem_type.value}, Difficulty: {analysis.difficulty.value}",
                confidence=analysis.type_confidence,
                duration=time.time() - step_start,
                insights=insights_list,
            )
            reasoning_chain.append(step)
            modules_used.append("problem_classifier")

        # === PHASE 2: STYLE SELECTION ===
        self._log("Phase 2: Selecting reasoning style...")

        style = None
        style_selector = self.modules.get("style_selector")
        if style_selector and analysis:
            step_start = time.time()
            style = style_selector.select_style(analysis)

            secondary = style.secondary_styles[:2] if style.secondary_styles else []
            secondary_str = (
                ", ".join([f"{s[0].value}({s[1]:.0%})" for s in secondary]) if secondary else "none"
            )

            step = ThoughtStep(
                module="style_selector",
                thought=f"Primary style: {style.primary_style.value} ({style.primary_fitness:.0%} fit)",
                confidence=style.primary_fitness,
                duration=time.time() - step_start,
                insights=[f"Secondary styles: {secondary_str}"] if secondary else [],
            )
            reasoning_chain.append(step)
            modules_used.append("style_selector")

        # === PHASE 3: DUAL PROCESS THINKING ===
        self._log("Phase 3: Dual process thinking (System 1 & 2)...")

        dual_process = self.modules.get("dual_process")
        if dual_process:
            step_start = time.time()
            try:
                intuition = dual_process.system1_response(embedding)
                deliberate = dual_process.system2_response(embedding)

                step = ThoughtStep(
                    module="dual_process",
                    thought=f"System 1 (intuition): {intuition.get('response', 'activated')}, System 2 (deliberate): {deliberate.get('response', 'engaged')}",
                    confidence=(
                        intuition.get("confidence", 0.5) + deliberate.get("confidence", 0.5)
                    )
                    / 2,
                    duration=time.time() - step_start,
                )
                reasoning_chain.append(step)
                modules_used.append("dual_process")
            except Exception as e:
                self._log(f"Dual process error: {e}")

        # === PHASE 4: PREDICTIVE CODING ===
        self._log("Phase 4: Predictive modeling...")

        predictive = self.modules.get("predictive_coding")
        if predictive:
            step_start = time.time()
            try:
                prediction = predictive.predict(embedding)
                free_energy = predictive.compute_free_energy(embedding)

                step = ThoughtStep(
                    module="predictive_coding",
                    thought=f"Free energy: {free_energy:.3f}, Prediction confidence: {prediction.get('confidence', 0.5):.0%}",
                    confidence=1 - min(free_energy, 1.0),
                    duration=time.time() - step_start,
                    insights=["Low free energy = good model fit"]
                    if free_energy < 0.5
                    else ["High uncertainty detected"],
                )
                reasoning_chain.append(step)
                modules_used.append("predictive_coding")
            except Exception as e:
                self._log(f"Predictive coding error: {e}")

        # === PHASE 5: MEMORY INTEGRATION ===
        self._log("Phase 5: Consulting memory...")

        memory = self.modules.get("memory")
        if memory:
            step_start = time.time()
            try:
                episodic = memory.recall_episodic(problem)
                semantic = memory.recall_semantic(problem)

                memory_insights = []
                if episodic:
                    memory_insights.append(f"Found {len(episodic)} relevant past experiences")
                if semantic:
                    memory_insights.append(f"Retrieved {len(semantic)} related concepts")

                step = ThoughtStep(
                    module="memory",
                    thought=f"Memory search complete: {len(memory_insights)} relevant memories found",
                    confidence=0.6 if memory_insights else 0.4,
                    duration=time.time() - step_start,
                    insights=memory_insights,
                )
                reasoning_chain.append(step)
                modules_used.append("memory")
            except Exception as e:
                self._log(f"Memory error: {e}")

        # === PHASE 6: CURIOSITY CHECK ===
        self._log("Phase 6: Checking curiosity and knowledge gaps...")

        curiosity = self.modules.get("curiosity")
        if curiosity:
            step_start = time.time()
            try:
                curiosity_event = curiosity.process_stimulus(embedding)
                gaps = curiosity.get_knowledge_gaps()

                step = ThoughtStep(
                    module="curiosity",
                    thought=f"Curiosity level: {curiosity.curiosity_level:.0%}, Novelty: {curiosity_event.get('novelty', 0):.0%}",
                    confidence=curiosity.curiosity_level,
                    duration=time.time() - step_start,
                    insights=[f"Knowledge gaps: {', '.join(gaps[:3])}"] if gaps else [],
                )
                reasoning_chain.append(step)
                modules_used.append("curiosity")

                if gaps:
                    insights.extend([f"Need to learn about: {g}" for g in gaps[:3]])
            except Exception as e:
                self._log(f"Curiosity error: {e}")

        # === PHASE 7: MOTIVATION ===
        self._log("Phase 7: Computing motivation...")

        motivation = self.modules.get("motivation")
        if motivation:
            step_start = time.time()
            try:
                motivation_level = motivation.compute_motivation(embedding)

                step = ThoughtStep(
                    module="motivation",
                    thought=f"Intrinsic motivation: {motivation_level:.0%}",
                    confidence=motivation_level,
                    duration=time.time() - step_start,
                )
                reasoning_chain.append(step)
                modules_used.append("motivation")
            except Exception as e:
                self._log(f"Motivation error: {e}")

        # === PHASE 8: GLOBAL WORKSPACE BROADCAST ===
        self._log("Phase 8: Broadcasting to global workspace...")

        global_workspace = self.modules.get("global_workspace")
        if global_workspace:
            step_start = time.time()
            try:
                broadcast_result = global_workspace.broadcast(
                    {
                        "problem": problem,
                        "embedding": embedding.tolist(),
                        "reasoning_so_far": [s.thought for s in reasoning_chain],
                    }
                )

                step = ThoughtStep(
                    module="global_workspace",
                    thought=f"Broadcast complete, coherence: {broadcast_result.get('coherence', 0.7):.0%}",
                    confidence=broadcast_result.get("coherence", 0.7),
                    duration=time.time() - step_start,
                )
                reasoning_chain.append(step)
                modules_used.append("global_workspace")
            except Exception as e:
                self._log(f"Global workspace error: {e}")

        # === PHASE 9: REASONING ===
        self._log("Phase 9: Applying reasoning...")

        reasoning = self.modules.get("reasoning")
        if reasoning and style:
            step_start = time.time()
            try:
                result = reasoning.reason(embedding, style.primary_style)

                step = ThoughtStep(
                    module="reasoning",
                    thought=f"Applied {style.primary_style.value} reasoning",
                    confidence=result.get("confidence", 0.7),
                    duration=time.time() - step_start,
                )
                reasoning_chain.append(step)
                modules_used.append("reasoning")
            except Exception as e:
                self._log(f"Reasoning error: {e}")

        # === PHASE 10: ORCHESTRATION ===
        self._log("Phase 10: Creating execution plan...")

        orchestrator = self.modules.get("orchestrator")
        if orchestrator and analysis and style:
            step_start = time.time()
            try:
                plan = orchestrator.create_plan(analysis, style)

                step = ThoughtStep(
                    module="orchestrator",
                    thought=f"Created plan with {len(plan.steps)} steps",
                    confidence=plan.estimated_confidence,
                    duration=time.time() - step_start,
                    insights=[f"Step 1: {plan.steps[0]['name']}"] if plan.steps else [],
                )
                reasoning_chain.append(step)
                modules_used.append("orchestrator")
            except Exception as e:
                self._log(f"Orchestrator error: {e}")

        # === PHASE 11: EXECUTIVE CONTROL ===
        self._log("Phase 11: Executive oversight...")

        executive = self.modules.get("executive")
        if executive:
            step_start = time.time()
            try:
                # Check if we should inhibit any responses
                should_inhibit = executive.inhibit(embedding)

                step = ThoughtStep(
                    module="executive",
                    thought=f"Executive control: {'inhibiting' if should_inhibit else 'proceeding'}",
                    confidence=0.8,
                    duration=time.time() - step_start,
                )
                reasoning_chain.append(step)
                modules_used.append("executive")
            except Exception as e:
                self._log(f"Executive error: {e}")

        # === PHASE 12: EMOTIONAL EVALUATION ===
        if depth in ["deep", "exhaustive"]:
            self._log("Phase 12: Emotional evaluation...")

            emotion = self.modules.get("emotion")
            if emotion:
                step_start = time.time()
                try:
                    emotional_response = emotion.evaluate(embedding)

                    step = ThoughtStep(
                        module="emotion",
                        thought=f"Emotional valence: {emotional_response.get('valence', 0):.2f}, arousal: {emotional_response.get('arousal', 0):.2f}",
                        confidence=0.6,
                        duration=time.time() - step_start,
                    )
                    reasoning_chain.append(step)
                    modules_used.append("emotion")
                except Exception as e:
                    self._log(f"Emotion error: {e}")

        # === PHASE 13: CONSCIOUSNESS/METACOGNITION ===
        if depth == "exhaustive":
            self._log("Phase 13: Metacognitive reflection...")

            consciousness = self.modules.get("consciousness")
            if consciousness:
                step_start = time.time()
                try:
                    introspection = consciousness.introspect()

                    step = ThoughtStep(
                        module="consciousness",
                        thought=f"Metacognitive state: {introspection.get('state', 'aware')}",
                        confidence=introspection.get("confidence_in_reasoning", 0.7),
                        duration=time.time() - step_start,
                    )
                    reasoning_chain.append(step)
                    modules_used.append("consciousness")
                except Exception as e:
                    self._log(f"Consciousness error: {e}")

        # === SYNTHESIS ===
        self._log("Final: Synthesizing...")

        # Gather all insights
        all_insights = []
        for step in reasoning_chain:
            all_insights.extend(step.insights)
        insights.extend(all_insights)

        # Calculate overall confidence
        confidences = [s.confidence for s in reasoning_chain]
        overall_confidence = np.mean(confidences) if confidences else 0.5

        # Generate suggested actions based on analysis
        suggested_actions = self._generate_actions(reasoning_chain, problem)

        # Create the final synthesis
        final_answer = self._synthesize(problem, reasoning_chain, insights)

        total_time = time.time() - start_time

        self._log(f"Complete! {len(modules_used)} modules used in {total_time:.2f}s")

        return UltraThinkResult(
            final_answer=final_answer,
            reasoning_chain=reasoning_chain,
            modules_used=modules_used,
            total_time=total_time,
            confidence=overall_confidence,
            insights=list(set(insights)),
            suggested_actions=suggested_actions,
        )

    def _generate_actions(self, chain: List[ThoughtStep], problem: str) -> List[str]:
        """Generate suggested actions based on reasoning."""
        actions = []

        # Check if we need external information
        low_confidence_steps = [s for s in chain if s.confidence < 0.5]
        if low_confidence_steps:
            actions.append("Search the web for more information")

        # Check for knowledge gaps
        for step in chain:
            if "knowledge gap" in " ".join(step.insights).lower():
                actions.append("Use tools to gather missing information")
                break

        # Standard actions based on problem keywords
        problem_lower = problem.lower()
        if any(w in problem_lower for w in ["github", "repo", "code", "repository"]):
            actions.append("Use github_user or github_repos tool")
        if any(w in problem_lower for w in ["file", "read", "write", "directory"]):
            actions.append("Use file system tools")
        if any(w in problem_lower for w in ["search", "find", "look up", "what is", "who is"]):
            actions.append("Use web_search tool")
        if any(w in problem_lower for w in ["calculate", "compute", "python", "run", "execute"]):
            actions.append("Use run_python tool")
        if any(w in problem_lower for w in ["browse", "website", "page", "url"]):
            actions.append("Use browse_web tool")

        return actions[:5]

    def _synthesize(self, problem: str, chain: List[ThoughtStep], insights: List[str]) -> str:
        """Synthesize the final answer from all reasoning steps."""
        parts = []

        # Problem analysis
        parts.append(f"## Analysis of: {problem[:100]}{'...' if len(problem) > 100 else ''}\n")

        # Key findings
        if chain:
            parts.append("### Reasoning Process:")
            for i, step in enumerate(chain, 1):
                conf_bar = "█" * int(step.confidence * 10) + "░" * (10 - int(step.confidence * 10))
                parts.append(f"{i}. **{step.module}**: {step.thought} [{conf_bar}]")

        # Insights
        if insights:
            parts.append("\n### Key Insights:")
            for insight in insights[:5]:
                parts.append(f"- {insight}")

        # Confidence
        avg_conf = np.mean([s.confidence for s in chain]) if chain else 0.5
        parts.append(f"\n### Overall Confidence: {avg_conf:.0%}")

        return "\n".join(parts)

    def quick_think(self, problem: str) -> str:
        """Fast thinking for simple problems."""
        result = self.think(problem, depth="quick")
        return result.final_answer

    def deep_think(self, problem: str, context: Optional[str] = None) -> UltraThinkResult:
        """Deep thinking for complex problems."""
        return self.think(problem, context=context, depth="deep")

    def exhaustive_think(self, problem: str, context: Optional[str] = None) -> UltraThinkResult:
        """Maximum depth thinking with all modules."""
        return self.think(problem, context=context, depth="exhaustive")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded modules."""
        stats = self.factory.statistics()
        return {
            "modules_loaded": stats["total_modules"],
            "active_modules": stats["active_modules"],
            "fallback_modules": stats["fallback_modules"],
            "module_names": stats["active_list"] + stats["fallback_list"],
            "active_list": stats["active_list"],
            "fallback_list": stats["fallback_list"],
        }

    def analyze(self, problem: str, cognitive_context: Dict = None) -> Dict[str, Any]:
        """
        Quick analysis for integration with chat loop.

        Returns analysis dict suitable for system prompt injection.
        """
        embedding = self._text_to_embedding(problem)

        result = {
            "type": "unknown",
            "style": "analytical",
            "confidence": 0.5,
            "complexity": 0.5,
            "suggested_actions": [],
        }

        # Classify problem
        classifier = self.modules.get("problem_classifier")
        if classifier:
            analysis = classifier.classify(embedding)
            result["type"] = analysis.problem_type.value
            result["complexity"] = analysis.complexity
            result["confidence"] = analysis.type_confidence

        # Select style
        style_selector = self.modules.get("style_selector")
        if style_selector and "type" in result:
            # Create minimal analysis for style selection
            from cognitive_adapters import ProblemAnalysis, ProblemType, ProblemDifficulty

            minimal_analysis = ProblemAnalysis(
                problem_type=ProblemType(result["type"]),
                type_confidence=result["confidence"],
                complexity=result["complexity"],
                difficulty=ProblemDifficulty.MEDIUM,
            )
            style = style_selector.select_style(minimal_analysis)
            result["style"] = style.primary_style.value

        # Generate actions
        result["suggested_actions"] = self._generate_actions([], problem)

        # Add cognitive context if provided
        if cognitive_context:
            result["cognitive_context"] = cognitive_context

        return result


def main():
    """Test UltraThink."""
    print("\n" + "=" * 60)
    print("ULTRATHINK TEST")
    print("=" * 60)

    ultra = UltraThink(verbose=True)

    print(f"\nModule Statistics:")
    stats = ultra.get_stats()
    print(f"  Total: {stats['modules_loaded']}")
    print(f"  Active: {stats['active_modules']}")
    print(f"  Fallback: {stats['fallback_modules']}")
    print(f"  Active modules: {', '.join(stats['active_list'])}")

    # Test problems
    problems = [
        "How do I build a web scraper in Python?",
        "What is the relationship between consciousness and information integration?",
        "Calculate the factorial of 10",
    ]

    for problem in problems:
        print(f"\n{'=' * 60}")
        print(f"PROBLEM: {problem}")
        print("=" * 60)

        result = ultra.think(problem, depth="deep")

        print(f"\n{result.final_answer}")
        print(f"\nModules used: {', '.join(result.modules_used)}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Time: {result.total_time:.2f}s")

        if result.suggested_actions:
            print(f"Suggested actions: {', '.join(result.suggested_actions)}")


if __name__ == "__main__":
    main()
