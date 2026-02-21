"""
NEURO Agent - Unified Agentic Workflow

This is the brain of NEURO AGI. It orchestrates all components into
a coherent think-act-learn loop:

1. PERCEIVE  - Understand the input, retrieve relevant context
2. THINK     - Analyze with UltraThink, plan approach
3. ACT       - Execute tools, generate response
4. LEARN     - Store knowledge, update patterns
5. IMPROVE   - Fix errors, evolve capabilities

All components flow together in a unified workflow.
"""

import time
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path


@dataclass
class AgentState:
    """Current state of the agent."""
    query: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    knowledge: List[str] = field(default_factory=list)
    memories: List[str] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    plan: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    response: str = ""
    errors: List[str] = field(default_factory=list)
    learnings: List[str] = field(default_factory=list)
    confidence: float = 0.5
    surprise: float = 0.0
    processing_time: float = 0.0


@dataclass
class ActionResult:
    """Result of an action."""
    success: bool
    output: str
    tool: str = ""
    error: str = ""


class NeuroAgent:
    """
    Unified NEURO AGI Agent with coherent workflow.

    The agent follows a continuous loop:

    ┌─────────────────────────────────────────────────────────────┐
    │                                                             │
    │   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
    │   │ PERCEIVE │───▶│  THINK   │───▶│   ACT    │             │
    │   └──────────┘    └──────────┘    └──────────┘             │
    │        ▲                               │                    │
    │        │                               ▼                    │
    │   ┌──────────┐                   ┌──────────┐              │
    │   │ IMPROVE  │◀──────────────────│  LEARN   │              │
    │   └──────────┘                   └──────────┘              │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, model: str = "ministral-3:8b", verbose: bool = False):
        self.model = model
        self.verbose = verbose
        self.base_url = "http://localhost:11434"

        # Initialize all components
        self._init_components()

        # Agent state
        self.state = AgentState()
        self.history: List[Dict] = []
        self.session_learnings: List[str] = []

    def _log(self, phase: str, msg: str):
        """Log with phase prefix."""
        if self.verbose:
            print(f"  [{phase}] {msg}")

    def _init_components(self):
        """Initialize all cognitive components."""
        self._log("INIT", "Loading NEURO components...")

        # Cognitive Pipeline (knowledge, memory, surprise, etc.)
        try:
            from cognitive_pipeline import CognitivePipeline
            self.pipeline = CognitivePipeline(verbose=False)
            self._log("INIT", f"Pipeline: {self.pipeline.get_stats()['num_components']} components")
        except Exception as e:
            self._log("INIT", f"Pipeline failed: {e}")
            self.pipeline = None

        # UltraThink (deep reasoning)
        try:
            from ultrathink import UltraThink
            self.ultrathink = UltraThink(verbose=False)
            self._log("INIT", f"UltraThink: {self.ultrathink.get_stats()['modules_loaded']} modules")
        except Exception as e:
            self._log("INIT", f"UltraThink failed: {e}")
            self.ultrathink = None

        # Self-Improver (error recovery, learning)
        try:
            from self_improving_agent import SelfImprovingAgent
            self.improver = SelfImprovingAgent()
            stats = self.improver.get_stats()
            self._log("INIT", f"Self-Improver: {stats['learned_patterns']} patterns, {stats['error_solutions']} solutions")
        except Exception as e:
            self._log("INIT", f"Self-Improver failed: {e}")
            self.improver = None

        # Tools
        try:
            from tools import Tools
            self.tools = Tools()
            self._log("INIT", "Tools: 13 available")
        except Exception as e:
            self._log("INIT", f"Tools failed: {e}")
            self.tools = None

        self._log("INIT", "Components loaded")

    # =========================================================================
    # PHASE 1: PERCEIVE - Understand input and gather context
    # =========================================================================

    def perceive(self, query: str, context: Dict = None) -> AgentState:
        """
        PERCEIVE phase: Understand the input and gather relevant context.

        - Parse the query
        - Retrieve relevant knowledge
        - Recall related memories
        - Detect novelty/surprise
        """
        self._log("PERCEIVE", f"Processing: {query[:50]}...")
        start_time = time.time()

        self.state = AgentState(query=query, context=context or {})

        # 1. Process through cognitive pipeline for full context
        if self.pipeline:
            try:
                result = self.pipeline.process(query)
                self.state.knowledge = result.knowledge_used
                self.state.memories = result.memory_used
                self.state.surprise = result.surprise_level
                self.state.confidence = result.confidence
                self._log("PERCEIVE", f"Retrieved {len(self.state.knowledge)} knowledge, {len(self.state.memories)} memories")
                self._log("PERCEIVE", f"Surprise: {self.state.surprise:.2f}, Confidence: {self.state.confidence:.0%}")
            except Exception as e:
                self._log("PERCEIVE", f"Pipeline error: {e}")

        # 2. Check if we have learned patterns for this type of query
        if self.improver:
            for pattern, solution in self.improver.learned_patterns.items():
                if pattern.lower() in query.lower():
                    self.state.context['learned_pattern'] = solution
                    self._log("PERCEIVE", f"Found learned pattern: {pattern}")
                    break

        self.state.processing_time = time.time() - start_time
        return self.state

    # =========================================================================
    # PHASE 2: THINK - Analyze and plan
    # =========================================================================

    def think(self, deep: bool = False) -> AgentState:
        """
        THINK phase: Analyze the problem and create an action plan.

        - Classify problem type
        - Select reasoning style
        - Determine what tools/actions are needed
        - Create execution plan
        """
        self._log("THINK", "Analyzing problem...")
        start_time = time.time()

        query = self.state.query

        # 1. Deep analysis with UltraThink
        if self.ultrathink:
            try:
                if deep:
                    result = self.ultrathink.think(query, depth="deep")
                    self.state.analysis = {
                        'type': 'deep',
                        'reasoning_steps': len(result.reasoning_chain),
                        'modules_used': result.modules_used,
                        'confidence': result.confidence,
                        'insights': result.insights[:3],
                        'suggested_actions': result.suggested_actions
                    }
                    self.state.plan = result.suggested_actions
                else:
                    analysis = self.ultrathink.analyze(query)
                    self.state.analysis = analysis

                self._log("THINK", f"Analysis: type={self.state.analysis.get('type')}, confidence={self.state.analysis.get('confidence', 0):.0%}")
            except Exception as e:
                self._log("THINK", f"UltraThink error: {e}")

        # 2. Determine what tools might be needed
        self.state.plan = self._plan_actions(query)
        self._log("THINK", f"Plan: {self.state.plan}")

        self.state.processing_time += time.time() - start_time
        return self.state

    def _plan_actions(self, query: str) -> List[str]:
        """Determine what actions/tools are needed."""
        query_lower = query.lower()
        plan = []

        # Check if this is a project/directory analysis request
        is_project_analysis = any(w in query_lower for w in ['analyze', 'project', 'about', 'what is'])
        has_path = '/' in query or '~' in query

        if is_project_analysis and has_path:
            # For project analysis: list files then read README
            plan.append("list_files")
            plan.append("read_readme")
            return plan

        # Detect intent and plan accordingly
        if any(w in query_lower for w in ['search', 'find', 'look up']):
            plan.append("web_search")

        if any(w in query_lower for w in ['what is', 'who is', 'explain']) and not has_path:
            plan.append("web_search")

        if any(w in query_lower for w in ['github', 'repo', 'repository']):
            plan.append("github_lookup")

        if any(w in query_lower for w in ['file', 'read', 'open', 'show', 'content']):
            plan.append("file_read")

        if any(w in query_lower for w in ['list', 'directory', 'folder']):
            plan.append("list_files")

        if any(w in query_lower for w in ['run', 'execute', 'command', 'python', 'calculate']):
            plan.append("execute_code")

        if any(w in query_lower for w in ['browse', 'website', 'webpage', 'url']):
            plan.append("browse_web")

        if any(w in query_lower for w in ['remember', 'store', 'save', 'note']):
            plan.append("remember")

        if any(w in query_lower for w in ['recall', 'what did', 'retrieve']):
            plan.append("recall")

        # Default: just respond
        if not plan:
            plan.append("respond")

        return plan

    # =========================================================================
    # PHASE 3: ACT - Execute tools and generate response
    # =========================================================================

    def act(self, llm_callback=None) -> AgentState:
        """
        ACT phase: Execute the plan using tools and generate response.

        - Execute planned tools
        - Handle errors with self-improvement
        - Generate final response
        """
        self._log("ACT", f"Executing plan: {self.state.plan}")
        start_time = time.time()

        results = []

        # Execute each planned action
        for action in self.state.plan:
            result = self._execute_action(action)
            results.append(result)
            self.state.tools_used.append(action)

            if result.error:
                self.state.errors.append(result.error)
                # Try to self-improve
                self._handle_error(result.error)

        # Build context for response generation
        tool_context = "\n".join([
            f"[{r.tool}]: {r.output[:500]}"
            for r in results if r.success
        ])

        # Generate response with LLM if callback provided
        if llm_callback:
            try:
                self.state.response = llm_callback(
                    query=self.state.query,
                    context=tool_context,
                    analysis=self.state.analysis,
                    knowledge=self.state.knowledge,
                    memories=self.state.memories
                )
            except Exception as e:
                self.state.errors.append(str(e))
                self._handle_error(str(e))
                self.state.response = f"I encountered an error: {e}"
        else:
            # No LLM, just summarize tool results
            self.state.response = self._summarize_results(results)

        self.state.processing_time += time.time() - start_time
        return self.state

    def _execute_action(self, action: str) -> ActionResult:
        """Execute a single action."""
        if not self.tools:
            return ActionResult(False, "", action, "Tools not available")

        query = self.state.query

        try:
            if action == "web_search":
                # Extract search query from user input
                result = self.tools.web_search(query)
                return ActionResult(result.success, result.output, "web_search", result.error or "")

            elif action == "github_lookup":
                # Extract username/repo from query
                words = query.split()
                for word in words:
                    if '/' in word:  # repo format
                        parts = word.split('/')
                        if len(parts) == 2:
                            result = self.tools.github_repo_info(parts[0], parts[1])
                            return ActionResult(result.success, result.output, "github_repo", result.error or "")
                    elif word.startswith('@'):
                        result = self.tools.github_user(word[1:])
                        return ActionResult(result.success, result.output, "github_user", result.error or "")
                # Default: search for github in query
                result = self.tools.web_search(f"github {query}")
                return ActionResult(result.success, result.output, "github_search", result.error or "")

            elif action == "list_files":
                # Extract path from query or use current dir
                path = "."
                for word in query.split():
                    if "/" in word or word.startswith("~"):
                        path = word
                        break
                result = self.tools.list_files(path)
                return ActionResult(result.success, result.output, "list_files", result.error or "")

            elif action == "file_read":
                # Extract filename from query
                for word in query.split():
                    if "." in word and "/" in word:
                        result = self.tools.read_file(word)
                        return ActionResult(result.success, result.output, "read_file", result.error or "")
                return ActionResult(False, "", "file_read", "No file path found in query")

            elif action == "execute_code":
                # Extract code or command
                if "python" in query.lower():
                    # Try to extract Python code
                    code = query.split("python")[-1].strip()
                    result = self.tools.run_python(code)
                else:
                    # Run as command
                    result = self.tools.run_command(query)
                return ActionResult(result.success, result.output, "execute", result.error or "")

            elif action == "browse_web":
                # Extract URL
                import re
                urls = re.findall(r'https?://\S+', query)
                if urls:
                    result = self.tools.browse_web(url=urls[0], action="goto")
                    return ActionResult(result.success, result.output, "browse", result.error or "")
                return ActionResult(False, "", "browse", "No URL found")

            elif action == "remember":
                # Store information
                key = f"memory_{int(time.time())}"
                result = self.tools.remember(key, query)
                return ActionResult(result.success, f"Stored as {key}", "remember", result.error or "")

            elif action == "recall":
                # Try to recall relevant memories
                result = self.tools.recall("last")
                return ActionResult(result.success, result.output, "recall", result.error or "")

            elif action == "read_readme":
                # Read README from a project directory
                import os
                path = None
                for word in query.split():
                    if "/" in word or word.startswith("~"):
                        path = os.path.expanduser(word)
                        break

                if path:
                    # Try common README locations
                    readme_names = ["README.md", "README.txt", "README", "readme.md"]
                    for readme in readme_names:
                        readme_path = os.path.join(path, readme)
                        if os.path.exists(readme_path):
                            result = self.tools.read_file(readme_path)
                            if result.success:
                                return ActionResult(True, result.output, "read_readme", "")

                    # Also try to read key documentation files
                    for doc in ["PROJECT_SUMMARY.md", "STATUS_REPORT.md", "CLAUDE.md"]:
                        doc_path = os.path.join(path, doc)
                        if os.path.exists(doc_path):
                            result = self.tools.read_file(doc_path)
                            if result.success:
                                return ActionResult(True, result.output, f"read_{doc}", "")

                    return ActionResult(False, "", "read_readme", "No README found")
                return ActionResult(False, "", "read_readme", "No path specified")

            else:
                return ActionResult(True, "", action, "")

        except Exception as e:
            return ActionResult(False, "", action, str(e))

    def _summarize_results(self, results: List[ActionResult]) -> str:
        """Summarize tool results into a response."""
        successful = [r for r in results if r.success and r.output]

        if not successful:
            return "I wasn't able to find the information you requested."

        parts = []
        for r in successful:
            parts.append(f"**{r.tool}**:\n{r.output[:1000]}")

        return "\n\n".join(parts)

    def _handle_error(self, error: str):
        """Handle an error using self-improvement."""
        if not self.improver:
            return

        self._log("IMPROVE", f"Handling error: {error[:50]}...")

        # Check for known solution
        for known_error, solution in self.improver.error_solutions.items():
            if known_error.lower() in error.lower():
                self._log("IMPROVE", f"Known fix: {solution}")
                return solution

        # Classify and learn
        if "timeout" in error.lower():
            fix = "Retry with shorter context"
            self.improver.error_solutions["timeout"] = fix
            self.improver._save_improvements()
            self._log("IMPROVE", f"Learned: {fix}")

        elif "connection" in error.lower():
            fix = "Check if service is running"
            self.improver.error_solutions["connection"] = fix
            self.improver._save_improvements()
            self._log("IMPROVE", f"Learned: {fix}")

        else:
            # Search for solution
            if self.tools:
                result = self.tools.web_search(f"fix: {error[:50]}")
                if result.success and result.output:
                    self.improver.error_solutions[error[:30]] = result.output[:100]
                    self.improver._save_improvements()
                    self._log("IMPROVE", f"Found solution online")

    # =========================================================================
    # PHASE 4: LEARN - Store knowledge and update patterns
    # =========================================================================

    def learn(self) -> AgentState:
        """
        LEARN phase: Store knowledge and update patterns.

        - Store new knowledge in knowledge base
        - Update episodic memory
        - Consolidate learnings
        """
        self._log("LEARN", "Storing learnings...")

        query = self.state.query
        response = self.state.response

        # 1. Store in knowledge base
        if self.pipeline:
            try:
                # Learn the Q&A pair
                topic = self.state.analysis.get('type', 'general')
                self.pipeline.learn(
                    topic=topic,
                    content=f"Q: {query[:100]} A: {response[:200]}",
                    source="conversation",
                    importance=self.state.confidence
                )
                self.state.learnings.append(f"Stored knowledge: {topic}")
                self._log("LEARN", f"Stored knowledge for topic: {topic}")
            except Exception as e:
                self._log("LEARN", f"Knowledge store failed: {e}")

        # 2. Learn patterns from successful tool usage
        if self.improver and self.state.tools_used and not self.state.errors:
            pattern_key = self._classify_query_type(query)
            if pattern_key:
                tools_str = ", ".join(set(self.state.tools_used))
                self.improver.learned_patterns[pattern_key] = f"Use {tools_str}"
                self.improver._save_improvements()
                self.state.learnings.append(f"Learned pattern: {pattern_key} -> {tools_str}")
                self._log("LEARN", f"Learned pattern: {pattern_key}")

        # 3. Add to history
        self.history.append({
            'query': query,
            'response': response[:500],
            'tools': self.state.tools_used,
            'confidence': self.state.confidence,
            'timestamp': datetime.now().isoformat()
        })

        return self.state

    def _classify_query_type(self, query: str) -> Optional[str]:
        """Classify query into a pattern type."""
        query_lower = query.lower()

        if 'github' in query_lower:
            return 'github_queries'
        elif 'search' in query_lower or 'find' in query_lower:
            return 'search_queries'
        elif 'file' in query_lower or 'read' in query_lower:
            return 'file_queries'
        elif 'project' in query_lower or 'analyze' in query_lower:
            return 'analysis_queries'
        elif 'python' in query_lower or 'code' in query_lower:
            return 'code_queries'

        return None

    # =========================================================================
    # PHASE 5: IMPROVE - Self-improvement and evolution
    # =========================================================================

    def improve(self) -> Dict[str, Any]:
        """
        IMPROVE phase: Analyze performance and improve.

        - Analyze errors from this session
        - Update strategies
        - Consolidate patterns
        """
        self._log("IMPROVE", "Self-improvement cycle...")

        improvements = {
            'errors_handled': len(self.state.errors),
            'patterns_learned': 0,
            'solutions_added': 0
        }

        if self.improver:
            stats = self.improver.get_stats()
            improvements['total_patterns'] = stats['learned_patterns']
            improvements['total_solutions'] = stats['error_solutions']

        # Consolidate if we have a continual learner
        if self.pipeline and hasattr(self.pipeline, 'learner') and self.pipeline.learner:
            consolidated = self.pipeline.learner.consolidate()
            improvements['consolidated'] = consolidated
            self._log("IMPROVE", f"Consolidated {consolidated} frequently-accessed items")

        return improvements

    # =========================================================================
    # MAIN WORKFLOW - The unified loop
    # =========================================================================

    def process(self, query: str, context: Dict = None,
                deep_think: bool = False, llm_callback=None) -> AgentState:
        """
        Main workflow: PERCEIVE -> THINK -> ACT -> LEARN -> IMPROVE

        This is the unified agentic loop that ties everything together.
        """
        self._log("WORKFLOW", "=" * 50)
        self._log("WORKFLOW", f"Processing: {query[:50]}...")

        # Phase 1: PERCEIVE
        self.perceive(query, context)

        # Phase 2: THINK
        self.think(deep=deep_think)

        # Phase 3: ACT
        self.act(llm_callback=llm_callback)

        # Phase 4: LEARN
        self.learn()

        # Phase 5: IMPROVE
        self.improve()

        self._log("WORKFLOW", f"Complete in {self.state.processing_time:.3f}s")
        self._log("WORKFLOW", "=" * 50)

        return self.state

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = {
            'history_length': len(self.history),
            'session_learnings': len(self.session_learnings),
            'components': {
                'pipeline': self.pipeline is not None,
                'ultrathink': self.ultrathink is not None,
                'improver': self.improver is not None,
                'tools': self.tools is not None
            }
        }

        if self.pipeline:
            stats['pipeline_stats'] = self.pipeline.get_stats()

        if self.improver:
            stats['improver_stats'] = self.improver.get_stats()

        return stats


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEURO AGENT - UNIFIED WORKFLOW TEST")
    print("=" * 70)

    agent = NeuroAgent(verbose=True)

    print("\n" + "=" * 70)
    print("TEST 1: Project Analysis Query")
    print("=" * 70)

    state = agent.process(
        "Analyze the project structure at /Users/ezekaj/Desktop/forex_2026"
    )

    print(f"\nResults:")
    print(f"  Query: {state.query[:50]}...")
    print(f"  Tools used: {state.tools_used}")
    print(f"  Errors: {state.errors}")
    print(f"  Learnings: {state.learnings}")
    print(f"  Response: {state.response[:200]}...")

    print("\n" + "=" * 70)
    print("TEST 2: Web Search Query")
    print("=" * 70)

    state = agent.process(
        "Search for the latest news about artificial intelligence"
    )

    print(f"\nResults:")
    print(f"  Tools used: {state.tools_used}")
    print(f"  Response: {state.response[:300]}...")

    print("\n" + "=" * 70)
    print("AGENT STATS")
    print("=" * 70)
    print(json.dumps(agent.get_stats(), indent=2, default=str))
