# NEURO v3.0 - Production Refactor Plan

## Goal
Transform NEURO from a demo into a production-ready OpenCode competitor.

## Current State: DEMO
- 75% benchmark pass rate
- 0% abstract reasoning (stubs)
- 19/25 modules are fallback stubs
- Memory exists but doesn't influence responses
- Curiosity runs but is invisible
- No streaming, slow startup

## Target State: PRODUCTION
- 95%+ benchmark pass rate
- Real LLM-powered reasoning at every step
- Sub-second startup
- Streaming responses
- Actual learning from interactions
- Self-healing on errors

---

## Architecture Redesign

### Current (Fragmented)
```
User Input
    ↓
CLI (monolithic 900+ lines)
    ↓
Multiple disconnected components:
- Pipeline (7 components)
- Orchestrator (25 modules, 19 stubs)
- UltraThink (17 modules, 8 active)
- Tools (13 tools)
- Curiosity (background, invisible)
- Memory (exists, unused)
    ↓
Ollama (single call at end)
    ↓
Response
```

### Target (Unified)
```
User Input
    ↓
┌─────────────────────────────────────────────┐
│              NEURO CORE ENGINE              │
├─────────────────────────────────────────────┤
│  1. UNDERSTAND                              │
│     - Parse intent (LLM call #1)            │
│     - Retrieve relevant memory              │
│     - Check for similar past interactions   │
│                                             │
│  2. PLAN                                    │
│     - Decide action strategy (LLM call #2)  │
│     - Select tools if needed                │
│     - Estimate confidence                   │
│                                             │
│  3. EXECUTE                                 │
│     - Run tools with retry logic            │
│     - Stream partial results                │
│     - Handle errors gracefully              │
│                                             │
│  4. RESPOND                                 │
│     - Generate response (LLM call #3)       │
│     - Include tool results                  │
│     - Stream to user                        │
│                                             │
│  5. LEARN                                   │
│     - Store successful patterns             │
│     - Update memory with new facts          │
│     - Adjust confidence models              │
└─────────────────────────────────────────────┘
    ↓
Streaming Response to User
```

---

## Core Components (Simplified)

### 1. NeuroEngine (replaces everything)
Single class that does everything:

```python
class NeuroEngine:
    """Production-ready NEURO core."""

    def __init__(self, model: str = "ministral-3:8b"):
        self.llm = OllamaClient(model)
        self.memory = PersistentMemory()
        self.tools = ToolExecutor()
        self.patterns = PatternStore()

    async def process(self, query: str) -> AsyncGenerator[str, None]:
        """Process query with streaming response."""

        # 1. UNDERSTAND
        context = await self._understand(query)

        # 2. PLAN
        plan = await self._plan(query, context)

        # 3. EXECUTE (if tools needed)
        if plan.tools_needed:
            results = await self._execute(plan)
            context.tool_results = results

        # 4. RESPOND (streaming)
        async for chunk in self._respond(query, context, plan):
            yield chunk

        # 5. LEARN (background)
        asyncio.create_task(self._learn(query, context, plan))
```

### 2. PersistentMemory (replaces 5 memory systems)
Single unified memory with:
- Semantic search (embeddings)
- Recency weighting
- Importance scoring
- Automatic pruning

```python
class PersistentMemory:
    """Unified memory system."""

    def __init__(self, path: str = "~/.neuro/memory.db"):
        self.db = sqlite3.connect(path)
        self._init_schema()

    def store(self, content: str, type: str, importance: float = 0.5):
        """Store with embedding for retrieval."""
        embedding = self._embed(content)
        self.db.execute(
            "INSERT INTO memories (content, type, embedding, importance, timestamp) VALUES (?, ?, ?, ?, ?)",
            (content, type, embedding, importance, time.time())
        )

    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant memories."""
        query_emb = self._embed(query)
        # Combine semantic similarity + recency + importance
        return self._ranked_search(query_emb, k)
```

### 3. ToolExecutor (replaces fragmented tools)
Robust tool execution with:
- Retry logic
- Timeout handling
- Error recovery
- Result validation

```python
class ToolExecutor:
    """Robust tool execution."""

    async def execute(self, tool: str, args: dict, retries: int = 3) -> ToolResult:
        """Execute tool with retries and error handling."""
        for attempt in range(retries):
            try:
                result = await self._run_tool(tool, args)
                if self._validate_result(result):
                    return result
            except TimeoutError:
                if attempt < retries - 1:
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                return ToolResult(success=False, error=str(e))

        return ToolResult(success=False, error="Max retries exceeded")
```

### 4. PatternStore (replaces self-improver)
Learn from successful interactions:

```python
class PatternStore:
    """Learn patterns from successful interactions."""

    def __init__(self, path: str = "~/.neuro/patterns.json"):
        self.patterns = self._load(path)

    def match(self, query: str) -> Optional[Pattern]:
        """Find matching pattern for query type."""
        query_type = self._classify(query)
        return self.patterns.get(query_type)

    def learn(self, query: str, plan: Plan, success: bool):
        """Learn from interaction outcome."""
        if success:
            query_type = self._classify(query)
            self.patterns[query_type] = Pattern(
                tools=plan.tools_used,
                approach=plan.approach,
                success_rate=self._update_rate(query_type, True)
            )
```

---

## File Structure (Simplified)

```
neuro/
├── engine.py          # NeuroEngine - the core
├── memory.py          # PersistentMemory
├── tools.py           # ToolExecutor
├── patterns.py        # PatternStore
├── llm.py             # OllamaClient with streaming
├── cli.py             # Thin CLI wrapper (~200 lines)
└── __main__.py        # Entry point
```

**From 20+ files to 6.**

---

## Key Changes

### 1. Remove Stubs - Use Real LLM
Every "cognitive module" becomes an LLM prompt:

```python
# OLD: Stub that returns fake analysis
class ProblemClassifier:
    def classify(self, problem):
        return {"type": "general", "confidence": 0.5}  # FAKE

# NEW: Real LLM classification
async def classify_problem(llm, problem: str) -> dict:
    response = await llm.chat([
        {"role": "system", "content": "Classify this problem type. Return JSON: {type, confidence, reasoning}"},
        {"role": "user", "content": problem}
    ])
    return json.loads(response)
```

### 2. Streaming Responses
User sees progress immediately:

```python
async def respond(self, query: str, context: dict):
    async for chunk in self.llm.stream([
        {"role": "system", "content": self._build_system_prompt(context)},
        {"role": "user", "content": query}
    ]):
        yield chunk
```

### 3. Real Memory Integration
Memory influences every response:

```python
async def _understand(self, query: str) -> Context:
    # Get relevant memories
    memories = self.memory.retrieve(query, k=5)

    # Get matching patterns
    pattern = self.patterns.match(query)

    # Build rich context
    return Context(
        query=query,
        memories=memories,
        pattern=pattern,
        timestamp=time.time()
    )
```

### 4. Actual Self-Improvement
Track what works:

```python
async def _learn(self, query: str, context: Context, plan: Plan):
    # Was this successful?
    success = not context.errors and context.user_satisfied

    # Update pattern store
    self.patterns.learn(query, plan, success)

    # Store in memory
    self.memory.store(
        content=f"Q: {query}\nA: {context.response[:200]}",
        type="interaction",
        importance=0.7 if success else 0.3
    )
```

---

## Migration Path

### Phase 1: Core Engine (Day 1)
1. Create `engine.py` with NeuroEngine
2. Implement UNDERSTAND → PLAN → EXECUTE → RESPOND → LEARN loop
3. Add streaming support
4. Test basic queries work

### Phase 2: Memory & Learning (Day 2)
1. Create `memory.py` with SQLite-backed storage
2. Create `patterns.py` for pattern learning
3. Integrate into engine
4. Test memory retrieval improves responses

### Phase 3: Tool Execution (Day 3)
1. Create robust `tools.py` with retry logic
2. Add timeout handling
3. Add result validation
4. Test tool chains work reliably

### Phase 4: CLI Polish (Day 4)
1. Rewrite CLI to use engine (~200 lines)
2. Add streaming output
3. Add progress indicators
4. Remove all legacy code

### Phase 5: Testing & Benchmarks (Day 5)
1. Run brutal benchmarks
2. Target 95%+ pass rate
3. Fix any failures
4. Performance optimization

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Benchmark Pass Rate | 75% | 95%+ |
| Abstract Reasoning | 0% | 80%+ |
| Startup Time | ~3s | <0.5s |
| Response Latency | ~5s | <2s (streaming) |
| Memory Influence | 0% | 100% |
| Active Modules | 6/25 | N/A (unified) |
| Code Lines | 10,000+ | <2,000 |

---

## Competitive Analysis

### What OpenCode Does Well
- Fast, streaming responses
- Excellent tool use
- Context-aware suggestions
- Clean UI

### Where NEURO Can Win
- **Local-first**: No API costs, no data leaving machine
- **True learning**: Patterns from YOUR interactions
- **Customizable**: Swap models, add tools, extend
- **Transparent**: See exactly what it's thinking

---

## Decision Point

Two paths forward:

### Path A: Incremental Fix
- Keep current architecture
- Fix modules one by one
- Slower, lower risk
- Result: Maybe 85% benchmark

### Path B: Clean Rewrite
- New unified architecture
- 6 files instead of 20+
- Faster, higher risk
- Result: 95%+ benchmark, production-ready

**Recommendation: Path B**

The current architecture is fundamentally fragmented. Fixing it piecemeal will result in technical debt. A clean rewrite with the unified engine will be:
- Faster to develop
- Easier to maintain
- More performant
- Actually production-ready

---

## Next Steps

1. Confirm Path B approach
2. Create `engine.py` skeleton
3. Implement core loop
4. Add streaming
5. Integrate memory
6. Benchmark and iterate

**Ready to build the OpenCode competitor.**
