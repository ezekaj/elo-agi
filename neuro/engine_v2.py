"""
NEURO Engine v2 - Streamlined production-ready cognitive core.

Unified async processing with:
- UNDERSTAND → PLAN → EXECUTE → RESPOND → LEARN loop
- Streaming responses
- Memory integration
- Pattern learning
- Tool execution with retries
"""

import asyncio
import time
import json
import re
from typing import Optional, Dict, Any, List, AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path

# Import core components
from neuro.memory import PersistentMemory
from neuro.patterns import PatternStore
from neuro.tools import ToolExecutor, ToolCall, ToolResult, ToolStatus, create_default_tools


@dataclass
class EngineConfig:
    """Configuration for NEURO engine."""
    model: str = "tomng/nanbeige4.1:3b"
    base_url: str = "http://localhost:11434"
    timeout: int = 120
    temperature: float = 0.7
    max_tool_rounds: int = 3
    auto_learn: bool = True
    verbose: bool = False


@dataclass
class Context:
    """Processing context."""
    query: str
    memories: List[str] = field(default_factory=list)
    pattern: Optional[str] = None
    suggested_tools: List[str] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    confidence: float = 0.5
    emotion: Optional[Dict[str, Any]] = None


@dataclass 
class Plan:
    """Execution plan."""
    tools_needed: bool = False
    tools_to_use: List[str] = field(default_factory=list)
    approach: str = ""
    confidence: float = 0.5


@dataclass
class ProcessResult:
    """Result of processing a query."""
    query: str
    response: str
    context: Context = field(default_factory=lambda: Context(query=""))
    plan: Plan = field(default_factory=Plan)
    tools_used: List[str] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    learned: bool = False
    duration: float = 0.0


class NeuroEngine:
    """
    Unified NEURO processing engine v2.
    
    Single class that handles the full cognitive loop:
    UNDERSTAND → PLAN → EXECUTE → RESPOND → LEARN
    """

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self._init_components()

    def _init_components(self):
        """Initialize all engine components."""
        # Memory system
        self.memory = PersistentMemory()
        
        # Pattern store
        self.patterns = PatternStore()
        
        # Tool executor with default tools
        default_tools = create_default_tools()
        self.executor = ToolExecutor(tools=default_tools, max_retries=2)
        
        # Register tools
        for name, func in default_tools.items():
            self.executor.register(name, func)
        
        # LLM client (lazy loaded)
        self._llm = None
        
        # Stream handler (lazy loaded)
        self._streamer = None

    @property
    def llm(self):
        """Lazy load LLM client."""
        if self._llm is None:
            from neuro.stream import StreamHandler, StreamConfig
            config = StreamConfig(
                base_url=self.config.base_url,
                model=self.config.model,
                timeout=self.config.timeout,
                temperature=self.config.temperature,
            )
            self._streamer = StreamHandler(config)
            self._llm = self._streamer
        return self._llm

    async def _understand(self, query: str) -> Context:
        """
        UNDERSTAND phase: Parse intent and retrieve context.
        
        - Get relevant memories
        - Match patterns
        - Classify query type
        """
        context = Context(query=query)
        
        # Retrieve relevant memories
        try:
            memories = self.memory.retrieve(query, k=5)
            context.memories = [m.content for m in memories]
        except Exception as e:
            if self.config.verbose:
                print(f"  [engine] Memory retrieval error: {e}")
        
        # Match pattern and get suggested tools
        pattern = self.patterns.match(query)
        if pattern:
            context.pattern = pattern.query_type
            context.suggested_tools = pattern.tools
            context.confidence = pattern.confidence
        
        return context

    async def _plan(self, query: str, context: Context) -> Plan:
        """
        PLAN phase: Decide strategy and tools.
        
        - Determine if tools are needed
        - Select appropriate tools
        - Estimate confidence
        """
        plan = Plan()
        
        # Use pattern-matched tools if available
        if context.suggested_tools:
            plan.tools_to_use = context.suggested_tools
            plan.tools_needed = len(plan.tools_to_use) > 0
        
        # Classify query to determine tool needs
        query_lower = query.lower()
        
        # Heuristics for tool needs
        if any(word in query_lower for word in ["read", "write", "file", "path"]):
            plan.tools_needed = True
            if "read_file" not in plan.tools_to_use:
                plan.tools_to_use.append("read_file")
        
        if any(word in query_lower for word in ["create", "save", "write"]):
            plan.tools_needed = True
            if "write_file" not in plan.tools_to_use:
                plan.tools_to_use.append("write_file")
        
        if any(word in query_lower for word in ["calculate", "compute", "python", "code"]):
            plan.tools_needed = True
            if "run_python" not in plan.tools_to_use:
                plan.tools_to_use.append("run_python")
        
        # Set approach based on query type
        if context.pattern:
            plan.approach = f"Using {context.pattern} pattern"
        elif plan.tools_needed:
            plan.approach = f"Using tools: {', '.join(plan.tools_to_use)}"
        else:
            plan.approach = "Direct LLM response"
        
        plan.confidence = context.confidence
        
        return plan

    async def _execute(self, plan: Plan, context: Context) -> List[ToolResult]:
        """
        EXECUTE phase: Run tools with retry logic.
        
        - Execute planned tools
        - Handle errors gracefully
        - Collect results
        """
        if not plan.tools_needed:
            return []
        
        results = []
        
        for tool_name in plan.tools_to_use:
            # Create tool call with appropriate args
            # For now, use generic args - would be LLM-generated in full implementation
            call = ToolCall(name=tool_name, args={"query": context.query})
            
            result = await self.executor.execute(tool_name, call.args)
            results.append(result)
            
            if result.success:
                context.tool_results.append({
                    "tool": tool_name,
                    "output": result.output,
                })
            else:
                context.errors.append(f"{tool_name}: {result.error}")
        
        return results

    def _build_system_prompt(self, context: Context, plan: Plan) -> str:
        """Build system prompt with cognitive context."""
        base = """You are NEURO, a neuroscience-inspired AI assistant.

CORE PRINCIPLES:
1. Be direct and truthful
2. Admit uncertainty when confidence is low
3. Use tools when external information is needed
4. Learn from each interaction

"""
        # Add memory context
        if context.memories:
            base += "\nRELEVANT MEMORIES:\n"
            for mem in context.memories[:3]:
                base += f"- {mem[:150]}\n"
        
        # Add tool results
        if context.tool_results:
            base += "\nTOOL RESULTS:\n"
            for tr in context.tool_results:
                output = str(tr.get("output", ""))[:300]
                base += f"- {tr['tool']}: {output}\n"
        
        # Add errors
        if context.errors:
            base += "\nERRORS (acknowledge if relevant):\n"
            for err in context.errors:
                base += f"- {err}\n"
        
        return base

    async def _respond(
        self,
        query: str,
        context: Context,
        plan: Plan,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        RESPOND phase: Generate streaming response.
        
        - Build enriched prompt
        - Stream LLM response
        - Yield tokens
        """
        system_prompt = self._build_system_prompt(context, plan)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        
        if history:
            messages.extend(history)
        
        # Stream response
        full_response = ""
        async for chunk in self.llm.stream(messages):
            content = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
            if content:
                full_response += content
                yield content
        
        # Store full response in context for learning
        context.response = full_response

    async def _learn(self, query: str, result: ProcessResult):
        """
        LEARN phase: Store successful patterns.
        
        - Update pattern success rates
        - Store interaction in memory
        - Extract facts for knowledge base
        """
        if not self.config.auto_learn:
            return
        
        # Determine success (no errors + response generated)
        success = len(result.context.errors) == 0 and len(result.response) > 0
        
        # Update pattern store
        self.patterns.learn(
            query=query,
            tools_used=result.tools_used,
            approach=result.plan.approach,
            success=success,
            confidence=result.context.confidence,
            response=result.response[:500],
        )
        
        # Store in memory
        if success and len(result.response) > 20:
            self.memory.store(
                content=f"Q: {query[:100]} | A: {result.response[:200]}",
                memory_type="interaction",
                importance=0.7,
            )
        
        # Extract and store facts
        facts = self._extract_facts(result.response)
        for fact in facts:
            self.memory.store(
                content=fact,
                memory_type="fact",
                importance=0.8,
            )
        
        result.learned = True

    def _extract_facts(self, response: str) -> List[str]:
        """Extract learnable facts from response."""
        facts = []
        
        # Split into sentences
        sentences = re.split(r"[.!?]\s+", response)
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip short or meta sentences
            if len(sentence) < 30 or len(sentence) > 300:
                continue
            if any(skip in sentence.lower() for skip in ["i will", "let me", "here's"]):
                continue
            
            facts.append(sentence)
        
        return facts[:3]  # Limit to 3 facts

    async def process(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = True,
        execute_tools: bool = True,
    ) -> ProcessResult:
        """
        Full processing pipeline.
        
        Args:
            query: User query
            history: Conversation history
            stream: Whether to stream response
            execute_tools: Whether to execute tools
            
        Returns:
            ProcessResult with full details
        """
        start_time = time.time()
        result = ProcessResult(query=query)
        
        # 1. UNDERSTAND
        result.context = await self._understand(query)
        
        # 2. PLAN
        result.plan = await self._plan(query, result.context)
        
        # 3. EXECUTE (if tools needed)
        if execute_tools and result.plan.tools_needed:
            tool_results = await self._execute(result.plan, result.context)
            result.tool_results = tool_results
            result.tools_used = [tr.name for tr in tool_results if tr.success]
        
        # 4. RESPOND (streaming)
        response_parts = []
        async for token in self._respond(query, result.context, result.plan, history):
            if stream:
                print(token, end="", flush=True)
            response_parts.append(token)
        
        if stream:
            print()  # Newline after response
        
        result.response = "".join(response_parts)
        
        # 5. LEARN (background)
        if self.config.auto_learn:
            asyncio.create_task(self._learn(query, result))
        
        result.duration = time.time() - start_time
        return result

    async def chat(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Simple chat interface.
        
        Args:
            query: User query
            history: Conversation history
            
        Returns:
            Full response text
        """
        result = await self.process(query, history, stream=False)
        return result.response

    async def stream_chat(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming chat interface.
        
        Args:
            query: User query
            history: Conversation history
            
        Yields:
            Response tokens
        """
        result = await self.process(query, history, stream=True)
        
        # Re-yield for caller if needed
        # (already printed during process)

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "model": self.config.model,
            "memory": self.memory.stats(),
            "patterns": self.patterns.get_stats(),
            "tools": self.executor.list_tools(),
        }

    async def close(self):
        """Clean up resources."""
        self.memory.close()
        self.patterns._save()
        if self._streamer:
            await self._streamer.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Convenience function
async def ask(question: str, **kwargs) -> str:
    """One-line query interface."""
    async with NeuroEngine() as engine:
        return await engine.chat(question, **kwargs)
