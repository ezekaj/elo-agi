"""
Cognitive Pipeline - Unified Processing System

Combines all components into a single coherent pipeline:
- Knowledge Base (semantic retrieval)
- Episodic Memory (experience-based retrieval)
- Two-Stage Retrieval (efficient search)
- Bayesian Surprise (novelty detection)
- Continual Learning (prevent forgetting)
- UltraThink (deep reasoning)
- Tools (external actions)
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PipelineResponse:
    """Response from the cognitive pipeline."""

    content: str
    cognitive_analysis: Dict[str, Any]
    knowledge_used: List[str]
    memory_used: List[str]
    surprise_level: float
    confidence: float
    suggested_actions: List[str]
    processing_time: float


class CognitivePipeline:
    """
    Unified cognitive processing pipeline.

    Processes queries through all cognitive modules and produces
    coherent, knowledge-enhanced responses.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._log("Initializing Cognitive Pipeline...")

        # Initialize components
        self._init_components()

        self._log(f"Pipeline ready with {len(self._active_components)} active components")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [Pipeline] {msg}")

    def _init_components(self) -> None:
        """Initialize all cognitive components."""
        self._active_components = []
        self._cognitive_modules = 0

        # Cognitive Orchestrator (loads ALL 38+ modules)
        try:
            from cognitive_orchestrator import CognitiveOrchestrator

            self.orchestrator = CognitiveOrchestrator(verbose=False)
            stats = self.orchestrator.get_stats()
            self._cognitive_modules = stats["total_modules"]
            self._active_components.append("cognitive_orchestrator")
            self._log(
                f"Cognitive Orchestrator: {stats['active_modules']} active, {stats['fallback_modules']} fallback"
            )
        except Exception as e:
            self._log(f"Cognitive Orchestrator: failed ({e})")
            self.orchestrator = None

        # Knowledge Base
        try:
            from knowledge_base import SelfTrainer

            self.knowledge = SelfTrainer()
            self._active_components.append("knowledge_base")
            self._log("Knowledge Base: loaded")
        except Exception as e:
            self._log(f"Knowledge Base: failed ({e})")
            self.knowledge = None

        # Episodic Memory
        try:
            from episodic_memory import EpisodicMemoryStore

            self.episodic = EpisodicMemoryStore()
            self._active_components.append("episodic_memory")
            self._log("Episodic Memory: loaded")
        except Exception as e:
            self._log(f"Episodic Memory: failed ({e})")
            self.episodic = None

        # Two-Stage Retrieval
        try:
            from retrieval import TwoStageRetriever

            self.retriever = TwoStageRetriever()
            self._active_components.append("retrieval")
            self._log("Two-Stage Retrieval: loaded")
        except Exception as e:
            self._log(f"Two-Stage Retrieval: failed ({e})")
            self.retriever = None

        # Bayesian Surprise
        try:
            from bayesian_surprise import BayesianSurprise, Observation

            self.surprise = BayesianSurprise(surprise_threshold=0.3)
            self._active_components.append("bayesian_surprise")
            self._log("Bayesian Surprise: loaded")
        except Exception as e:
            self._log(f"Bayesian Surprise: failed ({e})")
            self.surprise = None

        # Continual Learning
        try:
            from continual_learning import SimpleContinualLearner

            self.learner = SimpleContinualLearner()
            self._active_components.append("continual_learning")
            self._log("Continual Learning: loaded")
        except Exception as e:
            self._log(f"Continual Learning: failed ({e})")
            self.learner = None

        # UltraThink
        try:
            from ultrathink import UltraThink

            self.ultrathink = UltraThink(verbose=False)
            self._active_components.append("ultrathink")
            self._log("UltraThink: loaded")
        except Exception as e:
            self._log(f"UltraThink: failed ({e})")
            self.ultrathink = None

        # Tools
        try:
            from tools import Tools

            self.tools = Tools()
            self._active_components.append("tools")
            self._log("Tools: loaded")
        except Exception as e:
            self._log(f"Tools: failed ({e})")
            self.tools = None

    def process(
        self, query: str, context: Dict[str, Any] = None, use_deep_thinking: bool = False
    ) -> PipelineResponse:
        """
        Process a query through the full cognitive pipeline.

        Args:
            query: User's query or input
            context: Additional context
            use_deep_thinking: Whether to use full UltraThink analysis

        Returns:
            PipelineResponse with all results
        """
        start_time = time.time()
        context = context or {}

        knowledge_used = []
        memory_used = []
        surprise_level = 0.0
        confidence = 0.5
        suggested_actions = []

        # 1. Retrieve relevant knowledge
        if self.knowledge:
            knowledge_context = self.knowledge.get_knowledge_for_prompt(query, k=3)
            if knowledge_context:
                knowledge_used = [knowledge_context]

        # 2. Retrieve relevant episodic memories
        if self.episodic:
            episodes = self.episodic.retrieve(query=query, k=3)
            for ep, score in episodes:
                memory_used.append(f"[{ep.topic}] {ep.content[:100]}...")

        # 3. Compute surprise for this query
        if self.surprise:
            try:
                from bayesian_surprise import Observation

                obs = Observation(type="query", value=query[:50])
                result = self.surprise.compute_surprise(obs)
                surprise_level = result.surprise
            except Exception:
                pass

        # 4. Cognitive analysis
        cognitive_analysis = {}
        if self.ultrathink:
            if use_deep_thinking:
                # Full UltraThink analysis
                ultra_result = self.ultrathink.think(query, depth="deep")
                cognitive_analysis = {
                    "type": "deep",
                    "modules_used": ultra_result.modules_used,
                    "confidence": ultra_result.confidence,
                    "insights": ultra_result.insights,
                    "reasoning_steps": len(ultra_result.reasoning_chain),
                }
                confidence = ultra_result.confidence
                suggested_actions = ultra_result.suggested_actions
            else:
                # Quick analysis
                analysis = self.ultrathink.analyze(query)
                cognitive_analysis = {
                    "type": analysis.get("type", "unknown"),
                    "style": analysis.get("style", "analytical"),
                    "complexity": analysis.get("complexity", 0.5),
                    "confidence": analysis.get("confidence", 0.5),
                }
                confidence = analysis.get("confidence", 0.5)
                suggested_actions = analysis.get("suggested_actions", [])

        # 5. Build enhanced response context
        response_context = self._build_context(
            query=query,
            knowledge=knowledge_used,
            memories=memory_used,
            analysis=cognitive_analysis,
            surprise=surprise_level,
        )

        processing_time = time.time() - start_time

        return PipelineResponse(
            content=response_context,
            cognitive_analysis=cognitive_analysis,
            knowledge_used=knowledge_used,
            memory_used=memory_used,
            surprise_level=surprise_level,
            confidence=confidence,
            suggested_actions=suggested_actions,
            processing_time=processing_time,
        )

    def _build_context(
        self,
        query: str,
        knowledge: List[str],
        memories: List[str],
        analysis: Dict[str, Any],
        surprise: float,
    ) -> str:
        """Build enhanced context for response generation."""
        parts = []

        # Cognitive analysis
        if analysis:
            parts.append("[Cognitive Analysis]")
            parts.append(f"Problem type: {analysis.get('type', 'general')}")
            parts.append(f"Reasoning style: {analysis.get('style', 'analytical')}")
            parts.append(f"Confidence: {analysis.get('confidence', 0.5):.0%}")

            if surprise > 0.3:
                parts.append(f"Novelty detected: {surprise:.0%}")

        # Knowledge
        if knowledge:
            parts.append("")
            parts.extend(knowledge)

        # Memories
        if memories:
            parts.append("\n[Relevant past experiences]")
            for mem in memories[:3]:
                parts.append(f"- {mem}")

        return "\n".join(parts) if parts else ""

    def learn(
        self, topic: str, content: str, source: str = "conversation", importance: float = 0.5
    ) -> None:
        """Learn new knowledge from an interaction."""
        # Add to knowledge base
        if self.knowledge:
            self.knowledge.learn(topic, content, source, importance)

        # Add to episodic memory
        if self.episodic:
            self.episodic.store(content=content, topic=topic, importance=importance)

        # Add to continual learner
        if self.learner:
            self.learner.learn(topic, content, importance)

        # Add to retriever
        if self.retriever:
            doc_id = f"doc_{int(datetime.now().timestamp())}"
            self.retriever.add_document(doc_id, content, metadata={"topic": topic})

        # Check if surprising
        if self.surprise:
            try:
                from bayesian_surprise import Observation

                obs = Observation(type="topic_learned", value=topic)
                self.surprise.compute_surprise(obs)
            except Exception:
                pass

    def save(self) -> None:
        """Save all component states."""
        if self.knowledge:
            self.knowledge.save()
        if self.episodic:
            self.episodic.save()

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        # Count total cognitive modules
        cognitive_count = getattr(self, "_cognitive_modules", 0)
        if self.orchestrator:
            orch_stats = self.orchestrator.get_stats()
            cognitive_count = orch_stats.get("total_modules", 0)

        stats = {
            "active_components": self._active_components,
            "num_components": len(self._active_components) + cognitive_count,
            "cognitive_modules": cognitive_count,
            "pipeline_components": len(self._active_components),
        }

        if self.orchestrator:
            stats["orchestrator"] = self.orchestrator.get_stats()

        if self.knowledge:
            stats["knowledge"] = self.knowledge.get_stats()

        if self.episodic:
            stats["episodic"] = self.episodic.get_stats()

        if self.surprise:
            stats["surprise"] = self.surprise.get_stats()

        if self.ultrathink:
            stats["ultrathink"] = self.ultrathink.get_stats()

        return stats

    def get_system_prompt_enhancement(self, query: str) -> str:
        """
        Get enhancement text to add to LLM system prompt.

        This is the key integration point - injects cognitive
        context into the LLM's reasoning.
        """
        result = self.process(query)
        return result.content


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("COGNITIVE PIPELINE TEST")
    print("=" * 60)

    pipeline = CognitivePipeline(verbose=True)

    print(f"\nStats: {pipeline.get_stats()}")

    # Learn some things
    print("\nLearning...")
    pipeline.learn("AI", "Artificial intelligence enables machines to think", "test", 0.8)
    pipeline.learn("Python", "Python is the most popular AI programming language", "test", 0.7)

    # Process a query
    print("\nProcessing query: 'How does AI work?'")
    response = pipeline.process("How does AI work?")

    print(f"\nCognitive Analysis: {response.cognitive_analysis}")
    print(f"Knowledge Used: {response.knowledge_used}")
    print(f"Memory Used: {response.memory_used}")
    print(f"Surprise Level: {response.surprise_level:.2f}")
    print(f"Confidence: {response.confidence:.0%}")
    print(f"Suggested Actions: {response.suggested_actions}")
    print(f"Processing Time: {response.processing_time:.3f}s")

    print("\nEnhanced Context:")
    print(response.content)

    # Save
    pipeline.save()
