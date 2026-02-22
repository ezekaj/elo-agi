"""
NEURO Engine - Unified Async Processing Core

The central engine that ties together:
- Streaming responses (real-time LLM output)
- Code editing (surgical file changes)
- Parallel execution (multi-tool processing)
- Git automation (safe commits)
- Cognitive pipeline (memory, learning, reasoning)
- Emotion system (appraisal, somatic markers, mood)
- Sleep consolidation (memory replay, homeostasis)
- Social cognition (Theory of Mind, reputation)
- Embodied cognition (body schema, affordances)

This is the production-ready core of NEURO.
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime

# Import engine components
from neuro.stream import StreamHandler, StreamConfig, StreamChunk, TerminalStreamer
from neuro.editor import CodeEditor, Edit, EditResult
from neuro.executor import ParallelExecutor, ToolCall, ToolResult, ToolStatus
from neuro.git import GitAutomator, GitResult, GitStatus

# Import autonomous learning
try:
    from neuro.autonomous import AutonomousLoop, WebLearner
    from neuro.benchmark import Benchmark

    AUTONOMOUS_AVAILABLE = True
except ImportError:
    AUTONOMOUS_AVAILABLE = False


@dataclass
class EngineConfig:
    """Configuration for the NEURO engine."""

    model: str = "ministral-3:8b"
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    temperature: float = 0.7
    max_parallel_tools: int = 5
    show_thinking: bool = True
    auto_confirm_edits: bool = False
    verbose: bool = False


@dataclass
class ProcessResult:
    """Result of processing a query."""

    query: str
    response: str
    tools_used: List[str] = field(default_factory=list)
    edits_made: List[EditResult] = field(default_factory=list)
    git_operations: List[GitResult] = field(default_factory=list)
    cognitive_context: Dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    tokens: int = 0


class NeuroEngine:
    """
    Unified NEURO processing engine.

    Provides a single interface for:
    - Chat with streaming
    - Code editing with diffs
    - Tool execution (parallel)
    - Git operations
    - Cognitive enhancement (memory, learning)
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._init_components()

    def _init_components(self):
        """Initialize all engine components."""
        # Streaming
        stream_config = StreamConfig(
            base_url=self.config.base_url,
            model=self.config.model,
            timeout=self.config.timeout,
            temperature=self.config.temperature,
        )
        self.streamer = StreamHandler(stream_config)
        self.terminal = TerminalStreamer(show_thinking=self.config.show_thinking)

        # Editor
        self.editor = CodeEditor(auto_confirm=self.config.auto_confirm_edits)

        # Git
        self.git = GitAutomator(verbose=self.config.verbose)

        # Tools & Executor (lazy loaded)
        self._tools = None
        self._executor = None

        # Cognitive Pipeline (lazy loaded)
        self._pipeline = None

        # Advanced Cognitive Modules (lazy loaded)
        self._emotions = None
        self._sleep = None
        self._social = None
        self._embodied = None

        # Real Learning Systems (lazy loaded)
        self._trainer = None
        self._evolution = None

        # Autonomous Learning (lazy loaded)
        self._benchmark = None
        self._autonomous = None

        # OCR (lazy loaded)
        self._ocr = None

        # Activity callback for UI updates
        self._on_activity = None

    @property
    def tools(self):
        """Lazy load tools."""
        if self._tools is None:
            try:
                from tools import Tools

                self._tools = Tools()
            except ImportError:
                self._tools = None
        return self._tools

    @property
    def executor(self):
        """Lazy load executor."""
        if self._executor is None:
            self._executor = ParallelExecutor(
                self.tools, max_parallel=self.config.max_parallel_tools
            )
        return self._executor

    @property
    def pipeline(self):
        """Lazy load cognitive pipeline."""
        if self._pipeline is None:
            try:
                from cognitive_pipeline import CognitivePipeline

                self._pipeline = CognitivePipeline(verbose=False)
            except ImportError:
                self._pipeline = None
        return self._pipeline

    @property
    def emotions(self):
        """Lazy load emotion system."""
        if self._emotions is None:
            try:
                from emotions import EmotionSystem

                self._emotions = EmotionSystem(dim=64)
            except ImportError:
                self._emotions = None
        return self._emotions

    @property
    def sleep_system(self):
        """Lazy load sleep consolidation system."""
        if self._sleep is None:
            try:
                from sleep import SleepConsolidationSystem

                self._sleep = SleepConsolidationSystem(dim=64)
            except ImportError:
                self._sleep = None
        return self._sleep

    @property
    def social(self):
        """Lazy load social cognition system."""
        if self._social is None:
            try:
                from social import SocialCognitionSystem

                self._social = SocialCognitionSystem(dim=64)
            except ImportError:
                self._social = None
        return self._social

    @property
    def embodied(self):
        """Lazy load embodied cognition system."""
        if self._embodied is None:
            try:
                from embodied import EmbodiedCognitionSystem

                self._embodied = EmbodiedCognitionSystem(dim=64)
            except ImportError:
                self._embodied = None
        return self._embodied

    @property
    def trainer(self):
        """Lazy load self-training system."""
        if self._trainer is None:
            try:
                from self_training import SelfTrainer

                self._trainer = SelfTrainer()
            except ImportError:
                self._trainer = None
        return self._trainer

    @property
    def evolution(self):
        """Lazy load self-evolution system."""
        if self._evolution is None:
            try:
                from self_evolution import SelfEvolution

                self._evolution = SelfEvolution()
            except ImportError:
                self._evolution = None
        return self._evolution

    @property
    def benchmark(self):
        """Lazy load benchmark system."""
        if self._benchmark is None:
            if AUTONOMOUS_AVAILABLE:
                self._benchmark = Benchmark()
            else:
                self._benchmark = None
        return self._benchmark

    @property
    def ocr(self):
        """Lazy load DeepSeek OCR."""
        if self._ocr is None:
            try:
                from ocr import DeepSeekOCR

                self._ocr = DeepSeekOCR()
            except ImportError:
                self._ocr = None
        return self._ocr

    @property
    def autonomous(self):
        """Lazy load autonomous learning loop."""
        if self._autonomous is None:
            if AUTONOMOUS_AVAILABLE and self.trainer and self.evolution and self.benchmark:
                self._autonomous = AutonomousLoop(
                    chat_fn=self._sync_chat,
                    trainer=self.trainer,
                    evolution=self.evolution,
                    benchmark=self.benchmark,
                    verbose=self.config.verbose,
                    on_activity=self._on_activity,
                    ocr=self.ocr,  # Pass OCR for image learning
                )
            else:
                self._autonomous = None
        return self._autonomous

    def set_activity_callback(self, callback):
        """Set callback for activity notifications."""
        self._on_activity = callback
        # Update autonomous loop if already created
        if self._autonomous:
            self._autonomous.on_activity = callback

    def _sync_chat(self, prompt: str) -> str:
        """Synchronous chat function for autonomous loop."""
        import asyncio

        async def _chat():
            response = ""
            async for chunk in self.stream_chat(prompt):
                response += chunk.content
            return response

        # Run in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create new loop for thread
                new_loop = asyncio.new_event_loop()
                return new_loop.run_until_complete(_chat())
            else:
                return loop.run_until_complete(_chat())
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(_chat())

    def start_autonomous(self):
        """Start the autonomous learning loop (runs in background)."""
        if self.autonomous:
            self.autonomous.start()
            return True
        return False

    def stop_autonomous(self):
        """Stop the autonomous learning loop."""
        if self.autonomous:
            self.autonomous.stop()
            return True
        return False

    def mark_conversation_started(self):
        """Mark that user started a conversation (triggers autonomous learning)."""
        if self.autonomous:
            self.autonomous.mark_conversation_started()

    def user_active(self):
        """Mark that user is actively chatting (pauses autonomous learning)."""
        if self.autonomous:
            self.autonomous.user_active()

    def _build_system_prompt(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the system prompt with cognitive context."""
        base_prompt = """You are NEURO, a neuroscience-inspired AI assistant with human-like cognition.

COGNITIVE ARCHITECTURE:
- Emotion System: Appraisal, somatic markers, mood regulation
- Memory: Episodic, semantic, procedural with consolidation
- Social: Theory of Mind, reputation tracking, moral reasoning
- Embodied: Grounded concepts, body schema, affordances

CORE PRINCIPLES (NEVER VIOLATE):
1. NEVER LIE - Always be truthful. If you don't know, say so.
2. NEVER FABRICATE - Never make up data, URLs, or facts.
3. BE DIRECT - No hedging or unnecessary qualifiers.
4. ADMIT UNCERTAINTY - If confidence is low, say so.
5. CORRECT MISTAKES - Immediately correct yourself if wrong.

You have access to tools for:
- Web search and browsing
- File reading, writing, and editing
- Code execution (Python, shell)
- Git operations
- Memory and learning

When you need external information, USE TOOLS. Don't guess.

To use a tool, respond with:
<tool>tool_name</tool>
<args>{"param": "value"}</args>
"""

        # Add cognitive context if available
        if context:
            if context.get("memories"):
                base_prompt += f"\n[Relevant memories]\n"
                for mem in context["memories"][:3]:
                    base_prompt += f"- {mem}\n"

            if context.get("knowledge"):
                base_prompt += f"\n[Knowledge context]\n"
                for fact in context["knowledge"][:3]:
                    base_prompt += f"- {fact}\n"

            if context.get("confidence"):
                base_prompt += f"\n[Confidence: {context['confidence']:.0%}]\n"

        return base_prompt

    async def get_cognitive_context(self, query: str) -> Dict[str, Any]:
        """Get cognitive context for a query."""
        context = {}

        # Get pipeline context
        if self.pipeline is not None:
            try:
                result = self.pipeline.process(query)
                context.update(
                    {
                        "memories": result.memory_used,
                        "knowledge": result.knowledge_used,
                        "confidence": result.confidence,
                        "surprise": result.surprise_level,
                        "analysis": result.cognitive_analysis,
                    }
                )
            except Exception as e:
                if self.config.verbose:
                    print(f"  [engine] Pipeline context error: {e}")

        # Get emotional state
        if self.emotions is not None:
            try:
                emo_state = self.emotions.get_state()
                context["emotion"] = {
                    "dominant": emo_state.get("dominant_emotion", "neutral"),
                    "valence": emo_state.get("core_affect", {}).get("valence", 0),
                    "arousal": emo_state.get("core_affect", {}).get("arousal", 0),
                    "mood": emo_state.get("mood", {}),
                }
            except Exception as e:
                if self.config.verbose:
                    print(f"  [engine] Emotion context error: {e}")

        return context

    async def stream_chat(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        include_cognitive: bool = True,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream a chat response.

        Args:
            query: User's query
            history: Conversation history
            on_token: Callback for each token
            include_cognitive: Whether to include cognitive context

        Yields:
            StreamChunk objects
        """
        # Get cognitive context
        context = None
        if include_cognitive:
            context = await self.get_cognitive_context(query)

        # Build system prompt
        system_prompt = self._build_system_prompt(context)

        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": query})

        # Stream response
        async for chunk in self.streamer.stream(messages, on_token=on_token):
            yield chunk

    async def chat(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        stream_to_terminal: bool = True,
    ) -> str:
        """
        Chat with streaming to terminal.

        Args:
            query: User's query
            history: Conversation history
            stream_to_terminal: Whether to print tokens

        Returns:
            Full response text
        """
        if stream_to_terminal:
            self.terminal.on_start()

        full_response = ""

        def on_token(token: str):
            nonlocal full_response
            full_response += token
            if stream_to_terminal:
                self.terminal.on_token(token)

        async for chunk in self.stream_chat(query, history, on_token=on_token):
            if chunk.done:
                if stream_to_terminal:
                    self.terminal.on_done({"eval_count": chunk.eval_count})
                break

        return full_response

    async def execute_tools(
        self, tool_calls: List[Dict[str, Any]], parallel: bool = True
    ) -> List[ToolResult]:
        """
        Execute tool calls.

        Args:
            tool_calls: List of {"name": "...", "args": {...}}
            parallel: Whether to run in parallel

        Returns:
            List of ToolResult
        """
        calls = [ToolCall(name=c["name"], args=c.get("args", {})) for c in tool_calls]

        if parallel and len(calls) > 1:
            return await self.executor.execute_parallel(calls)
        else:
            results = []
            for call in calls:
                result = await self.executor.execute_one(call)
                results.append(result)
            return results

    async def edit_file(
        self, file_path: str, line_start: int, line_end: int, new_content: str, confirm: bool = True
    ) -> EditResult:
        """
        Edit a file with diff display.

        Args:
            file_path: Path to file
            line_start: First line (1-indexed)
            line_end: Last line (inclusive)
            new_content: New content
            confirm: Ask for confirmation

        Returns:
            EditResult
        """
        return self.editor.edit_lines(file_path, line_start, line_end, new_content, confirm=confirm)

    async def commit_changes(
        self, files: List[str], message: str, check_secrets: bool = True
    ) -> GitResult:
        """
        Safely commit changes.

        Args:
            files: Files to commit
            message: Commit message
            check_secrets: Whether to check for secrets

        Returns:
            GitResult
        """
        return self.git.safe_commit(files, message, check_secrets=check_secrets)

    async def push_changes(self, remote: str = "origin", branch: Optional[str] = None) -> GitResult:
        """
        Push changes to remote.

        Args:
            remote: Remote name
            branch: Branch to push

        Returns:
            GitResult
        """
        return self.git.push(remote, branch)

    async def process(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = True,
        execute_tools: bool = True,
        max_tool_rounds: int = 5,
    ) -> ProcessResult:
        """
        Full processing pipeline: chat → tools → edit → learn.

        Args:
            query: User's query
            history: Conversation history
            stream: Whether to stream response
            execute_tools: Whether to execute detected tools
            max_tool_rounds: Max tool execution rounds

        Returns:
            ProcessResult with full details
        """
        import time
        import re

        start_time = time.time()

        result = ProcessResult(query=query)

        # Get cognitive context
        result.cognitive_context = await self.get_cognitive_context(query)

        # Stream initial response
        if stream:
            response = await self.chat(query, history, stream_to_terminal=True)
        else:
            response = ""
            async for chunk in self.stream_chat(query, history):
                response += chunk.content

        result.response = response

        # Extract and execute tools
        if execute_tools:
            tool_rounds = 0
            current_response = response

            while tool_rounds < max_tool_rounds:
                # Parse tool calls from response
                tools = self._parse_tools(current_response)
                if not tools:
                    break

                tool_rounds += 1
                result.tools_used.extend([t["name"] for t in tools])

                # Execute tools
                tool_results = await self.execute_tools(tools)

                # Build context for follow-up
                tool_context = self._format_tool_results(tool_results)

                # Get follow-up response
                follow_up_query = (
                    f"Tool results:\n{tool_context}\n\nContinue based on these results."
                )
                if stream:
                    current_response = await self.chat(follow_up_query, history)
                else:
                    current_response = ""
                    async for chunk in self.stream_chat(follow_up_query, history):
                        current_response += chunk.content

                result.response += "\n" + current_response

        # Learn from interaction
        await self._learn(query, result)

        result.duration = time.time() - start_time
        return result

    def _parse_tools(self, response: str) -> List[Dict[str, Any]]:
        """Parse tool calls from response."""
        import re
        import json

        tools = []
        # Match <tool>name</tool>\n<args>{...}</args>
        pattern = r"<tool>(\w+)</tool>\s*<args>(.*?)</args>"
        matches = re.findall(pattern, response, re.DOTALL)

        for name, args_str in matches:
            try:
                args = json.loads(args_str.strip())
            except json.JSONDecodeError:
                args = {}

            tools.append({"name": name, "args": args})

        return tools

    def _format_tool_results(self, results: List[ToolResult]) -> str:
        """Format tool results for LLM."""
        lines = []
        for r in results:
            status = "SUCCESS" if r.status == ToolStatus.SUCCESS else "FAILED"
            lines.append(f"[{r.name}] {status}")
            if r.output:
                lines.append(str(r.output)[:500])
            if r.error:
                lines.append(f"Error: {r.error}")
            lines.append("")
        return "\n".join(lines)

    async def _learn(self, query: str, result: ProcessResult):
        """Learn from the interaction using real learning systems."""
        learned_count = 0

        # 1. Learn via SelfTrainer (persistent knowledge base)
        if self.trainer is not None:
            try:
                # Extract facts from the response
                facts = self._extract_facts(result.response)
                for fact in facts:
                    # Check for duplicates via evolution
                    if self.evolution and self.evolution.is_duplicate(fact["content"]):
                        continue

                    # Add to knowledge base
                    self.trainer.learn(
                        topic=fact.get("topic", query[:50]),
                        content=fact["content"],
                        source="conversation",
                    )
                    learned_count += 1

                    # Mark as learned in evolution tracker
                    if self.evolution:
                        self.evolution.mark_learned(fact["content"])

                # Save periodically
                if learned_count > 0:
                    self.trainer.kb.save()

            except Exception as e:
                if self.config.verbose:
                    print(f"  [learn] Trainer error: {e}")

        # 2. Learn via pipeline (cognitive memory)
        if self.pipeline is not None:
            try:
                self.pipeline.learn(
                    topic="interaction",
                    content=f"Q: {query[:100]} A: {result.response[:200]}",
                    source="conversation",
                    importance=0.7,
                )
            except Exception:
                pass

        # 3. Check if we should benchmark/evolve
        if self.evolution and self.evolution.should_benchmark():
            if self.config.verbose:
                print(
                    f"  [evolution] Cycle complete - {self.evolution.state['facts_this_cycle']} facts learned"
                )
            self.evolution.start_new_cycle()

        return learned_count

    def _extract_facts(self, response: str) -> List[Dict[str, str]]:
        """Extract learnable facts from a response."""
        facts = []

        # Split into sentences
        import re

        sentences = re.split(r"[.!?]\s+", response)

        for sentence in sentences:
            sentence = sentence.strip()
            # Skip short sentences or questions
            if len(sentence) < 20 or sentence.endswith("?"):
                continue
            # Skip meta-commentary
            if any(skip in sentence.lower() for skip in ["i will", "let me", "here's", "option"]):
                continue
            # This looks like a fact
            if len(sentence) < 500:
                facts.append({"content": sentence, "topic": sentence[:50]})

        return facts[:5]  # Limit to 5 facts per response

    async def close(self):
        """Clean up resources."""
        # Stop autonomous loop first
        self.stop_autonomous()

        await self.streamer.close()
        if self.pipeline:
            self.pipeline.save()
        if self.trainer:
            self.trainer.kb.save()
        if self.evolution:
            self.evolution._save_state()

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = {
            "model": self.config.model,
            "tools_available": self.tools is not None,
            "pipeline_available": self.pipeline is not None,
            "git_repo": self.git.is_repo(),
            # Cognitive modules
            "emotions_available": self.emotions is not None,
            "sleep_available": self.sleep_system is not None,
            "social_available": self.social is not None,
            "embodied_available": self.embodied is not None,
            # Learning systems
            "trainer_available": self.trainer is not None,
            "evolution_available": self.evolution is not None,
        }

        if self.pipeline:
            stats["pipeline"] = self.pipeline.get_stats()

        if self.emotions:
            emo_state = self.emotions.get_state()
            stats["emotion"] = {
                "dominant": emo_state.get("dominant_emotion", "neutral"),
                "valence": emo_state.get("core_affect", {}).get("valence", 0),
                "arousal": emo_state.get("core_affect", {}).get("arousal", 0),
            }

        # Learning stats
        if self.trainer:
            trainer_stats = self.trainer.get_stats()
            stats["learning"] = {
                "total_facts": trainer_stats.get("total_facts", 0),
                "storage_path": trainer_stats.get("storage_path", ""),
            }

        if self.evolution:
            evo_stats = self.evolution.get_stats()
            stats["evolution"] = {
                "cycle": evo_stats.get("cycle", 0),
                "unique_facts": evo_stats.get("total_facts", 0),
                "improvement": evo_stats.get("improvement", 0),
            }

        # Autonomous loop stats
        if self.autonomous:
            auto_stats = self.autonomous.get_stats()
            stats["autonomous"] = {
                "running": auto_stats.get("running", False),
                "benchmark_done": auto_stats.get("initial_benchmark_done", False),
                "weak_areas": auto_stats.get("weak_areas", []),
            }

        return stats

    def learn_fact(self, topic: str, content: str, source: str = "manual") -> bool:
        """Manually teach the engine a fact."""
        if self.trainer is None:
            return False

        # Check for duplicates
        if self.evolution and self.evolution.is_duplicate(content):
            return False

        # Add to knowledge base
        self.trainer.learn(topic, content, source)

        # Mark as learned
        if self.evolution:
            self.evolution.mark_learned(content)

        self.trainer.kb.save()
        return True

    def recall(self, query: str, k: int = 5) -> List[Dict]:
        """Recall facts related to a query."""
        if self.trainer is None:
            return []
        return self.trainer.recall(query, k=k)

    # OCR Methods

    def read_image(self, image_path: str) -> str:
        """Read text from an image using DeepSeek OCR."""
        if self.ocr is None:
            return "OCR not available"
        result = self.ocr.read_image(image_path)
        return result.text

    def analyze_image(self, image_path: str, question: str) -> str:
        """Analyze an image and answer a question about it."""
        if self.ocr is None:
            return "OCR not available"
        result = self.ocr.analyze_image(image_path, question)
        return result.text

    def extract_code_from_image(self, image_path: str) -> str:
        """Extract code from a screenshot."""
        if self.ocr is None:
            return "OCR not available"
        result = self.ocr.extract_code(image_path)
        return result.text

    def learn_from_image(self, image_path: str) -> bool:
        """Learn facts from an image."""
        if self.ocr is None or self.trainer is None:
            return False

        # Extract structured content
        data = self.ocr.extract_for_learning(image_path)
        if "error" in data:
            return False

        # Learn the extracted content
        content = data.get("content", "") or data.get("summary", "")
        if not content:
            return False

        # Check for duplicates
        if self.evolution and self.evolution.is_duplicate(content):
            return False

        # Learn main content
        topic = data.get("title", "image_content")
        self.trainer.learn(topic, content, f"ocr:{image_path}")

        # Learn individual facts
        for fact in data.get("facts", []):
            if fact and not (self.evolution and self.evolution.is_duplicate(fact)):
                self.trainer.learn(topic, fact, f"ocr:{image_path}")
                if self.evolution:
                    self.evolution.mark_learned(fact)

        if self.evolution:
            self.evolution.mark_learned(content)

        self.trainer.kb.save()
        return True


# Convenience functions for simple usage


async def chat(query: str, stream: bool = True) -> str:
    """Simple chat function."""
    engine = NeuroEngine()
    try:
        if stream:
            return await engine.chat(query)
        else:
            response = ""
            async for chunk in engine.stream_chat(query):
                response += chunk.content
            return response
    finally:
        await engine.close()


def chat_sync(query: str, stream: bool = True) -> str:
    """Synchronous chat."""
    return asyncio.run(chat(query, stream))


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NEURO ENGINE TEST")
    print("=" * 60)

    async def test():
        config = EngineConfig(verbose=True, show_thinking=True)
        engine = NeuroEngine(config)

        print("\nEngine Stats:")
        print("-" * 40)
        stats = engine.get_stats()
        for k, v in stats.items():
            if k != "pipeline":
                print(f"  {k}: {v}")

        print("\n" + "=" * 60)
        print("Test 1: Simple streaming chat")
        print("=" * 60)

        response = await engine.chat("What is 2 + 2? Answer briefly.")
        print(f"\nFull response: {response}")

        print("\n" + "=" * 60)
        print("Test 2: Cognitive context")
        print("=" * 60)

        context = await engine.get_cognitive_context("Tell me about Python")
        print(f"  Context keys: {list(context.keys())}")
        print(f"  Confidence: {context.get('confidence', 'N/A')}")

        await engine.close()
        print("\n" + "=" * 60)
        print("All tests completed!")

    asyncio.run(test())
