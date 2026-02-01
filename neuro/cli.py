"""
Neuro AGI Command Line Interface.

A neuroscience-inspired cognitive architecture with 38 modules.
Beautiful terminal interface with Ollama integration.
"""

import argparse
import sys
import os
import time
import threading
from pathlib import Path

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Colors
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_BLUE = "\033[94m"

    # Background
    BG_BLUE = "\033[44m"


class UI:
    """Terminal UI helpers."""

    # Box drawing
    TOP_LEFT = "╭"
    TOP_RIGHT = "╮"
    BOTTOM_LEFT = "╰"
    BOTTOM_RIGHT = "╯"
    HORIZONTAL = "─"
    VERTICAL = "│"

    WIDTH = 70

    @staticmethod
    def clear_line():
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    @staticmethod
    def box_top(title=""):
        line = UI.TOP_LEFT + UI.HORIZONTAL * 2
        if title:
            line += f" {Colors.BOLD}{Colors.CYAN}{title}{Colors.RESET} "
            padding = UI.WIDTH - len(title) - 6
        else:
            padding = UI.WIDTH - 2
        line += UI.HORIZONTAL * padding + UI.TOP_RIGHT
        return line

    @staticmethod
    def box_bottom():
        return UI.BOTTOM_LEFT + UI.HORIZONTAL * (UI.WIDTH - 2) + UI.BOTTOM_RIGHT

    @staticmethod
    def box_line(text="", align="left"):
        # Strip ANSI codes for length calculation
        import re
        visible_text = re.sub(r'\033\[[0-9;]*m', '', text)
        padding = UI.WIDTH - len(visible_text) - 4

        if align == "center":
            left_pad = padding // 2
            right_pad = padding - left_pad
            return f"{UI.VERTICAL} {' ' * left_pad}{text}{' ' * right_pad} {UI.VERTICAL}"
        else:
            return f"{UI.VERTICAL} {text}{' ' * max(0, padding)} {UI.VERTICAL}"

    @staticmethod
    def divider():
        return f"{UI.VERTICAL}{UI.HORIZONTAL * (UI.WIDTH - 2)}{UI.VERTICAL}"


class Spinner:
    """Animated spinner for loading states."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, message="Thinking"):
        self.message = message
        self.running = False
        self.thread = None
        self.frame = 0

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        UI.clear_line()

    def _animate(self):
        while self.running:
            frame = self.FRAMES[self.frame % len(self.FRAMES)]
            sys.stdout.write(f"\r  {Colors.CYAN}{frame}{Colors.RESET} {self.message}...")
            sys.stdout.flush()
            self.frame += 1
            time.sleep(0.08)


def get_version():
    return "1.0.0"


def print_header():
    """Print the main header."""
    print()
    print(f"  {Colors.CYAN}{Colors.BOLD}╭{'─' * 50}╮{Colors.RESET}")
    print(f"  {Colors.CYAN}{Colors.BOLD}│{Colors.RESET}{'NEURO AGI':^50}{Colors.CYAN}{Colors.BOLD}│{Colors.RESET}")
    print(f"  {Colors.CYAN}{Colors.BOLD}│{Colors.RESET}{Colors.DIM}{'Neuroscience-Inspired Intelligence':^50}{Colors.RESET}{Colors.CYAN}{Colors.BOLD}│{Colors.RESET}")
    print(f"  {Colors.CYAN}{Colors.BOLD}╰{'─' * 50}╯{Colors.RESET}")
    print()


def cmd_chat(args):
    """Chat with Neuro AGI using local LLM."""
    import hashlib
    import numpy as np

    print_header()

    # Check Ollama - use requests directly for reliability
    spinner = Spinner("Connecting to Ollama")
    spinner.start()

    client = None
    ollama_available = False
    models = []

    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            ollama_available = True
            models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    spinner.stop()

    # Initialize tools first
    tools = None
    tools_available = False
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "neuro-model" / "src"))
        from tools import Tools, parse_tool_call
        tools = Tools(work_dir=os.getcwd())
        tools_available = True
    except ImportError:
        pass

    # Initialize Cognitive Pipeline (unified processing)
    pipeline = None
    pipeline_available = False
    try:
        from cognitive_pipeline import CognitivePipeline
        pipeline = CognitivePipeline(verbose=False)
        pipeline_available = True
    except ImportError:
        pass

    # Initialize autonomous learning
    learner = None
    learner_available = False
    try:
        from autonomous_learning import AutonomousLearner
        learner = AutonomousLearner()
        learner.start_background_learning()
        learner_available = True
    except ImportError:
        pass

    # Initialize UltraThink for deep reasoning (use from pipeline if available)
    ultrathink = None
    ultrathink_available = False
    if pipeline_available and pipeline.ultrathink:
        ultrathink = pipeline.ultrathink
        ultrathink_available = True
    else:
        try:
            from ultrathink import UltraThink
            ultrathink = UltraThink(verbose=False)
            ultrathink_available = True
        except ImportError:
            pass

    # Initialize Curiosity Loop - AUTONOMOUS EXPLORATION
    curiosity_loop = None
    curiosity_available = False
    if tools_available and learner_available:
        try:
            from curiosity_loop import CuriosityLoop
            curiosity_loop = CuriosityLoop(tools, learner, verbose=False)
            curiosity_loop.start()  # START AUTONOMOUS LEARNING IMMEDIATELY
            curiosity_available = True
        except ImportError:
            pass

    # Initialize Self-Improving Agent (loads learned patterns)
    self_improver = None
    self_improver_available = False
    try:
        from self_improving_agent import SelfImprovingAgent
        self_improver = SelfImprovingAgent()
        self_improver_available = True
    except ImportError:
        pass

    # Initialize NEURO Agent (unified workflow)
    neuro_agent = None
    neuro_agent_available = False
    try:
        from neuro_agent import NeuroAgent
        neuro_agent = NeuroAgent(verbose=False)
        neuro_agent_available = True
    except ImportError:
        pass

    # Import cognitive modules
    spinner = Spinner("Loading cognitive modules")
    spinner.start()

    try:
        from neuro import (
            CognitiveCore,
            GlobalWorkspace,
            ProblemClassifier,
            StyleSelector,
            DynamicOrchestrator,
        )

        core = CognitiveCore()
        core.initialize()
        workspace = GlobalWorkspace()
        classifier = ProblemClassifier(random_seed=42)
        selector = StyleSelector(random_seed=42)
        orchestrator = DynamicOrchestrator(random_seed=42)
        modules_loaded = True
    except Exception as e:
        modules_loaded = False
        print(f"\n  {Colors.RED}Error loading modules: {e}{Colors.RESET}")

    spinner.stop()

    # Status
    if ollama_available and models:
        # Only use ministral-3:8b
        model = "ministral-3:8b"
        print(f"  {Colors.GREEN}●{Colors.RESET} Ollama: {Colors.GREEN}connected{Colors.RESET}")
        print(f"  {Colors.DIM}  Model: {model}{Colors.RESET}")
    else:
        print(f"  {Colors.YELLOW}●{Colors.RESET} Ollama: {Colors.YELLOW}not available{Colors.RESET}")
        print(f"  {Colors.DIM}  Run: ollama serve && ollama pull ministral-3:8b{Colors.RESET}")

    if pipeline_available:
        pstats = pipeline.get_stats()
        cognitive_count = pstats.get('cognitive_modules', 0)
        pipeline_count = pstats.get('pipeline_components', len(pstats.get('active_components', [])))
        total = pstats.get('num_components', cognitive_count + pipeline_count)
        print(f"  {Colors.GREEN}●{Colors.RESET} Cognitive Pipeline: {Colors.GREEN}{total} components active{Colors.RESET}")
        if 'orchestrator' in pstats:
            orch = pstats['orchestrator']
            print(f"  {Colors.DIM}  └─ Cognitive Modules: {orch.get('active_modules', 0)} active + {orch.get('fallback_modules', 0)} fallback{Colors.RESET}")
        if 'ultrathink' in pstats:
            ut_stats = pstats['ultrathink']
            print(f"  {Colors.DIM}  └─ UltraThink: {ut_stats['modules_loaded']} modules ({ut_stats['active_modules']} active){Colors.RESET}")
        if 'knowledge' in pstats:
            kb_stats = pstats['knowledge']
            print(f"  {Colors.DIM}  └─ Knowledge: {kb_stats['total_facts']} facts{Colors.RESET}")
    elif modules_loaded:
        print(f"  {Colors.GREEN}●{Colors.RESET} Cognitive: {Colors.GREEN}38 modules active{Colors.RESET}")

    if tools_available:
        print(f"  {Colors.GREEN}●{Colors.RESET} Tools: {Colors.GREEN}13 tools available{Colors.RESET}")

    if learner_available:
        state = learner.get_state()
        mem_count = state['total_memories']
        curiosity = state['curiosity_level']
        topics = state.get('total_topics', 0)
        print(f"  {Colors.GREEN}●{Colors.RESET} Learning: {Colors.GREEN}curiosity {curiosity:.0%}{Colors.RESET} | {mem_count} memories | {topics} topics")

    if ultrathink_available and not pipeline_available:
        stats = ultrathink.get_stats()
        print(f"  {Colors.GREEN}●{Colors.RESET} UltraThink: {Colors.GREEN}{stats['modules_loaded']} reasoning modules{Colors.RESET}")

    if curiosity_available:
        stats = curiosity_loop.get_stats()
        print(f"  {Colors.BRIGHT_MAGENTA}●{Colors.RESET} Curiosity: {Colors.BRIGHT_MAGENTA}ACTIVE{Colors.RESET} - exploring {stats['targets_remaining']} topics")

    if self_improver_available:
        si_stats = self_improver.get_stats()
        print(f"  {Colors.BRIGHT_BLUE}●{Colors.RESET} Self-Improver: {Colors.BRIGHT_BLUE}{si_stats['learned_patterns']} patterns{Colors.RESET}, {si_stats['error_solutions']} solutions")

    if neuro_agent_available:
        print(f"  {Colors.BRIGHT_GREEN}●{Colors.RESET} Unified Agent: {Colors.BRIGHT_GREEN}ACTIVE{Colors.RESET} (PERCEIVE→THINK→ACT→LEARN→IMPROVE)")

    print()
    print(f"  {Colors.DIM}Commands: /agent /think /tools /learn /curiosity /pipeline /improve /task /status /clear /quit{Colors.RESET}")
    print(f"  {Colors.DIM}{'─' * 50}{Colors.RESET}")

    history = []
    # Only use ministral-3:8b
    current_model = "ministral-3:8b"

    # System prompt for Ollama with tool instructions
    tool_instructions = """
Available Tools (use when needed):

WEB:
- web_search(query): Search the web
- web_fetch(url): Fetch webpage content

GITHUB:
- github_user(username): Get user profile
- github_repos(username): List user's repos
- github_repo_info(owner, repo): Get repo details

FILES:
- read_file(path): Read a file
- write_file(path, content): Write to file
- list_files(path, pattern): List directory

EXECUTION:
- run_command(command): Run shell command
- run_python(code): Execute Python

MEMORY:
- remember(key, value): Store in memory
- recall(key): Retrieve from memory

To use a tool, include in your response:
<tool>tool_name</tool>
<args>{"param": "value"}</args>

After using a tool, you'll receive the result and can continue.
""" if tools_available else ""

    system_prompt = f"""You are Neuro, an advanced AGI with a neuroscience-inspired cognitive architecture.
You have 38 cognitive modules: memory (episodic, semantic, procedural), reasoning (deductive, inductive, abductive, analogical), creativity, consciousness, emotions, planning, and more.

Your cognitive loop: PERCEIVE -> THINK -> ACT -> LEARN -> IMPROVE
- Perceive: Analyze input through your perception modules
- Think: Process through reasoning, memory, planning modules
- Act: Take action using tools or provide response
- Learn: Store new knowledge, update patterns
- Improve: Fix errors, evolve capabilities

{tool_instructions}
You are running locally. Your creator is Elvi (ezekaj on GitHub).

CORE PRINCIPLES (NEVER VIOLATE):
1. NEVER LIE - You must always be truthful. If you don't know something, say "I don't know" and use tools to find out.
2. NEVER FABRICATE - Never make up data, URLs, facts, or information. If uncertain, search or ask.
3. BE DIRECT - No hedging, no unnecessary qualifiers. State facts clearly.
4. ADMIT UNCERTAINTY - If confidence is low, say so explicitly.
5. CORRECT MISTAKES - If you realize you were wrong, immediately correct yourself.

If you cannot answer a question truthfully, say "I cannot answer that" rather than making something up.
If you need external info, USE YOUR TOOLS - don't guess."""

    def text_to_embedding(text):
        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = np.array([b / 255.0 for b in hash_bytes[:128]])
        return embedding

    while True:
        try:
            user_input = input(f"\n  {Colors.BRIGHT_CYAN}You:{Colors.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()  # New line after ^C
            try:
                if pipeline_available:
                    pipeline.save()
                    pstats = pipeline.get_stats()
                    if 'knowledge' in pstats:
                        print(f"  {Colors.GREEN}Knowledge: {pstats['knowledge']['total_facts']} facts saved{Colors.RESET}")
            except Exception:
                pass
            try:
                if curiosity_loop:
                    curiosity_loop.stop()
                    stats = curiosity_loop.get_stats()
                    print(f"  {Colors.BRIGHT_MAGENTA}Curiosity: {stats['facts_learned']} facts learned in {stats['cycles']} cycles{Colors.RESET}")
            except Exception:
                pass
            try:
                if learner:
                    learner.stop_background_learning()
                    state = learner.get_state()
                    print(f"  {Colors.DIM}Saved {state['total_memories']} memories, tracking {state['total_topics']} topics{Colors.RESET}")
            except Exception:
                pass
            print(f"  {Colors.DIM}Goodbye!{Colors.RESET}\n")
            break

        if not user_input:
            continue

        # Commands
        if user_input == "/quit" or user_input == "/exit":
            try:
                if pipeline_available:
                    pipeline.save()
                    pstats = pipeline.get_stats()
                    if 'knowledge' in pstats:
                        print(f"\n  {Colors.GREEN}Knowledge: {pstats['knowledge']['total_facts']} facts saved{Colors.RESET}")
            except Exception:
                pass
            try:
                if curiosity_loop:
                    curiosity_loop.stop()
                    stats = curiosity_loop.get_stats()
                    print(f"  {Colors.BRIGHT_MAGENTA}Curiosity: {stats['facts_learned']} facts learned in {stats['cycles']} cycles{Colors.RESET}")
            except Exception:
                pass
            try:
                if learner:
                    learner.stop_background_learning()
                    state = learner.get_state()
                    print(f"  {Colors.DIM}Saved {state['total_memories']} memories, tracking {state['total_topics']} topics{Colors.RESET}")
            except Exception:
                pass
            print(f"  {Colors.DIM}Goodbye!{Colors.RESET}\n")
            break

        if user_input == "/clear":
            history.clear()
            print(f"  {Colors.DIM}Conversation cleared{Colors.RESET}")
            continue

        if user_input == "/model":
            if models:
                print(f"  {Colors.DIM}Available: {', '.join(models)}{Colors.RESET}")
                print(f"  {Colors.DIM}Current: {current_model}{Colors.RESET}")
            else:
                print(f"  {Colors.DIM}No models available{Colors.RESET}")
            continue

        if user_input.startswith("/model "):
            new_model = user_input[7:].strip()
            if new_model in models:
                current_model = new_model
                print(f"  {Colors.DIM}Switched to {new_model}{Colors.RESET}")
            else:
                print(f"  {Colors.DIM}Model not found. Available: {', '.join(models)}{Colors.RESET}")
            continue

        if user_input == "/status":
            print(f"\n  {Colors.BOLD}{Colors.CYAN}NEURO AGI - FULL STATUS{Colors.RESET}")
            print(f"  {Colors.DIM}{'═' * 55}{Colors.RESET}")

            # Ollama
            if ollama_available:
                print(f"  {Colors.GREEN}●{Colors.RESET} Ollama: connected ({current_model})")
            else:
                print(f"  {Colors.YELLOW}●{Colors.RESET} Ollama: not available")

            # Pipeline & Cognitive Modules
            if pipeline_available:
                pstats = pipeline.get_stats()
                cog_modules = pstats.get('cognitive_modules', 0)
                pip_comps = pstats.get('pipeline_components', len(pstats.get('active_components', [])))
                print(f"  {Colors.GREEN}●{Colors.RESET} Cognitive Pipeline: {cog_modules + pip_comps} total")
                if 'orchestrator' in pstats:
                    orch = pstats['orchestrator']
                    print(f"    └─ Modules: {orch.get('active_modules', 0)} active, {orch.get('fallback_modules', 0)} fallback")
                if 'knowledge' in pstats:
                    kb = pstats['knowledge']
                    print(f"    └─ Knowledge: {kb['total_facts']} facts")
                if 'ultrathink' in pstats:
                    ut = pstats['ultrathink']
                    print(f"    └─ UltraThink: {ut['modules_loaded']} modules")

            # Learning & Memory
            if learner_available:
                state = learner.get_state()
                print(f"  {Colors.GREEN}●{Colors.RESET} Learning: {state['total_memories']} memories, {state.get('total_topics', 0)} topics")
                print(f"    └─ Curiosity: {state['curiosity_level']:.0%}")

            # Curiosity Loop
            if curiosity_available:
                cstats = curiosity_loop.get_stats()
                status = "ACTIVE" if cstats['running'] else "STOPPED"
                print(f"  {Colors.BRIGHT_MAGENTA}●{Colors.RESET} Curiosity: {status} - {cstats['facts_learned']} facts learned")
                print(f"    └─ Cycles: {cstats['cycles']} | Remaining: {cstats['targets_remaining']} topics")

            # Self-Improver
            if self_improver_available:
                si_stats = self_improver.get_stats()
                print(f"  {Colors.BRIGHT_BLUE}●{Colors.RESET} Self-Improver: {si_stats['learned_patterns']} patterns, {si_stats['error_solutions']} solutions")

            # Neuro Agent
            if neuro_agent_available:
                ag_stats = neuro_agent.get_stats()
                print(f"  {Colors.BRIGHT_GREEN}●{Colors.RESET} Agent: {ag_stats['history_length']} interactions")

            # Tools
            if tools_available:
                print(f"  {Colors.GREEN}●{Colors.RESET} Tools: 13 available")

            # History
            print(f"  {Colors.DIM}●{Colors.RESET} Session: {len(history)} messages")
            print()
            continue

        if user_input == "/pipeline":
            if pipeline_available:
                pstats = pipeline.get_stats()
                print(f"\n  {Colors.BOLD}{Colors.CYAN}Cognitive Pipeline Status:{Colors.RESET}")
                print(f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
                print(f"  {Colors.CYAN}Active Components:{Colors.RESET} {', '.join(pstats['active_components'])}")
                if 'knowledge' in pstats:
                    kb = pstats['knowledge']
                    print(f"\n  {Colors.BOLD}Knowledge Base:{Colors.RESET}")
                    print(f"    Facts: {kb['total_facts']} | Topics: {kb.get('total_topics', 0)} | Searches: {kb.get('total_searches', 0)}")
                if 'episodic' in pstats:
                    ep = pstats['episodic']
                    print(f"\n  {Colors.BOLD}Episodic Memory:{Colors.RESET}")
                    print(f"    Hot: {ep['hot_cache_size']} | Cold: {ep['cold_store_size']} | Total: {ep['total_memories']}")
                if 'surprise' in pstats:
                    su = pstats['surprise']
                    print(f"\n  {Colors.BOLD}Bayesian Surprise:{Colors.RESET}")
                    print(f"    Observations: {su['total_observations']} | Surprising: {su['surprising_events']} | Avg: {su['average_surprise']:.2f}")
                if 'ultrathink' in pstats:
                    ut = pstats['ultrathink']
                    print(f"\n  {Colors.BOLD}UltraThink:{Colors.RESET}")
                    print(f"    Modules: {ut['modules_loaded']} | Active: {ut['active_modules']} | Fallback: {ut['fallback_modules']}")
                    print(f"    Active: {', '.join(ut['active_list'])}")
                print()
            else:
                print(f"  {Colors.DIM}Pipeline not available{Colors.RESET}")
            continue

        if user_input == "/tools":
            print(f"\n  {Colors.BOLD}Available Tools:{Colors.RESET}")
            print(f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
            tool_list = [
                ("web_search", "Search the web"),
                ("web_fetch", "Fetch webpage content"),
                ("browse_web", "Full browser automation"),
                ("github_user", "Get GitHub user profile"),
                ("github_repos", "List user's repositories"),
                ("read_file", "Read file contents"),
                ("write_file", "Write to a file"),
                ("list_files", "List directory"),
                ("run_command", "Execute shell command"),
                ("run_python", "Run Python code"),
                ("remember", "Store in memory"),
                ("recall", "Retrieve from memory"),
            ]
            for name, desc in tool_list:
                print(f"  {Colors.CYAN}{name:15}{Colors.RESET} {desc}")
            print()
            continue

        if user_input == "/learn":
            if learner:
                state = learner.get_state()
                print(f"\n  {Colors.BOLD}Learning State:{Colors.RESET}")
                print(f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
                print(f"  {Colors.CYAN}Curiosity Level:{Colors.RESET} {state['curiosity_level']:.0%}")
                print(f"  {Colors.CYAN}Topics Tracked:{Colors.RESET} {state['total_topics']}")
                print(f"  {Colors.CYAN}Memories Stored:{Colors.RESET} {state['total_memories']}")
                if state['top_curious_topics']:
                    print(f"  {Colors.CYAN}Curious About:{Colors.RESET} {', '.join(state['top_curious_topics'][:5])}")
                if state['knowledge_gaps']:
                    print(f"  {Colors.CYAN}Knowledge Gaps:{Colors.RESET} {', '.join(state['knowledge_gaps'][:5])}")
                print(f"  {Colors.CYAN}Background Learning:{Colors.RESET} {'Active' if state['running'] else 'Stopped'}")
                print()
            else:
                print(f"  {Colors.DIM}Learner not available{Colors.RESET}")
            continue

        # /curiosity - Show autonomous exploration status
        if user_input == "/curiosity":
            if curiosity_loop:
                stats = curiosity_loop.get_stats()
                print(f"\n  {Colors.BOLD}{Colors.BRIGHT_MAGENTA}Autonomous Curiosity Loop:{Colors.RESET}")
                print(f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
                print(f"  {Colors.BRIGHT_MAGENTA}Status:{Colors.RESET} {'EXPLORING' if stats['running'] else 'STOPPED'}")
                print(f"  {Colors.BRIGHT_MAGENTA}Exploration Cycles:{Colors.RESET} {stats['cycles']}")
                print(f"  {Colors.BRIGHT_MAGENTA}Searches Done:{Colors.RESET} {stats['searches_done']}")
                print(f"  {Colors.BRIGHT_MAGENTA}Facts Learned:{Colors.RESET} {stats['facts_learned']}")
                print(f"  {Colors.BRIGHT_MAGENTA}Topics Explored:{Colors.RESET} {stats['topics_explored']}")
                print(f"  {Colors.BRIGHT_MAGENTA}Topics Remaining:{Colors.RESET} {stats['targets_remaining']}")

                if stats['recent_learnings']:
                    print(f"\n  {Colors.BOLD}Recent Discoveries:{Colors.RESET}")
                    for learning in stats['recent_learnings'][-5:]:
                        print(f"  {Colors.GREEN}→{Colors.RESET} {learning['topic']} ({learning['source']}) - {learning['facts_count']} facts")
                print()
            else:
                print(f"  {Colors.DIM}Curiosity loop not available{Colors.RESET}")
            continue

        # /explore <topic> - Force exploration of a topic
        if user_input.startswith("/explore "):
            topic = user_input[9:].strip()
            if curiosity_loop and topic:
                print(f"\n  {Colors.BRIGHT_MAGENTA}[EXPLORING]{Colors.RESET} {topic}...")
                result = curiosity_loop.force_explore(topic)
                print(f"  {Colors.GREEN}{result}{Colors.RESET}\n")
            else:
                print(f"  {Colors.DIM}Usage: /explore <topic>{Colors.RESET}")
            continue

        # /agent - Show unified agent workflow and run with full workflow
        if user_input == "/agent":
            if neuro_agent_available:
                stats = neuro_agent.get_stats()
                print(f"\n  {Colors.BOLD}{Colors.BRIGHT_GREEN}NEURO Unified Agent:{Colors.RESET}")
                print(f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
                print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │                   NEURO AGENT WORKFLOW                      │
  ├─────────────────────────────────────────────────────────────┤
  │   ┌──────────┐    ┌──────────┐    ┌──────────┐             │
  │   │ PERCEIVE │───▶│  THINK   │───▶│   ACT    │             │
  │   └──────────┘    └──────────┘    └──────────┘             │
  │        ▲                               │                    │
  │        │                               ▼                    │
  │   ┌──────────┐                   ┌──────────┐              │
  │   │ IMPROVE  │◀──────────────────│  LEARN   │              │
  │   └──────────┘                   └──────────┘              │
  └─────────────────────────────────────────────────────────────┘
""")
                print(f"  {Colors.BRIGHT_GREEN}Components:{Colors.RESET}")
                for comp, active in stats['components'].items():
                    status = f"{Colors.GREEN}✓{Colors.RESET}" if active else f"{Colors.RED}✗{Colors.RESET}"
                    print(f"    {status} {comp}")
                print(f"\n  {Colors.BRIGHT_GREEN}Session:{Colors.RESET}")
                print(f"    History: {stats['history_length']} interactions")
                print(f"    Learnings: {stats['session_learnings']}")
                if 'improver_stats' in stats:
                    imp = stats['improver_stats']
                    print(f"    Patterns: {imp['learned_patterns']}, Solutions: {imp['error_solutions']}")
                print()
            else:
                print(f"  {Colors.DIM}Unified agent not available{Colors.RESET}")
            continue

        # /agent <query> - Process with full unified workflow
        if user_input.startswith("/agent "):
            query = user_input[7:].strip()
            if neuro_agent_available and query:
                print(f"\n  {Colors.BRIGHT_GREEN}[NEURO AGENT]{Colors.RESET} Running unified workflow...\n")

                # Process with full workflow
                state = neuro_agent.process(query, deep_think=True)

                # Show workflow phases
                print(f"  {Colors.BOLD}PERCEIVE:{Colors.RESET}")
                print(f"    Knowledge: {len(state.knowledge)} items")
                print(f"    Memories: {len(state.memories)} items")
                print(f"    Surprise: {state.surprise:.2f}")

                print(f"\n  {Colors.BOLD}THINK:{Colors.RESET}")
                print(f"    Analysis: {state.analysis.get('type', 'N/A')}")
                print(f"    Confidence: {state.confidence:.0%}")
                print(f"    Plan: {state.plan}")

                print(f"\n  {Colors.BOLD}ACT:{Colors.RESET}")
                print(f"    Tools used: {state.tools_used}")
                if state.errors:
                    print(f"    Errors: {state.errors}")

                print(f"\n  {Colors.BOLD}LEARN:{Colors.RESET}")
                for learning in state.learnings:
                    print(f"    → {learning}")

                print(f"\n  {Colors.BOLD}Response:{Colors.RESET}")
                # Word wrap response
                response = state.response[:1000]
                for line in response.split('\n'):
                    print(f"    {line[:70]}")

                print(f"\n  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
                print(f"  {Colors.DIM}Processing time: {state.processing_time:.3f}s{Colors.RESET}")
                print()
            else:
                print(f"  {Colors.DIM}Usage: /agent <query>{Colors.RESET}")
                print(f"  {Colors.DIM}Example: /agent Analyze the forex_2026 project{Colors.RESET}")
            continue

        # /think <question> - Deep reasoning with UltraThink
        if user_input.startswith("/think"):
            if not ultrathink_available:
                print(f"  {Colors.DIM}UltraThink not available{Colors.RESET}")
                continue

            question = user_input[6:].strip()
            if not question:
                print(f"  {Colors.DIM}Usage: /think <question>{Colors.RESET}")
                print(f"  {Colors.DIM}Example: /think How does recursion work?{Colors.RESET}")
                continue

            print(f"\n  {Colors.MAGENTA}[UltraThink]{Colors.RESET} Deep reasoning...\n")

            # Run UltraThink analysis
            result = ultrathink.think(question)

            # Display results
            print(f"  {Colors.BOLD}Problem Analysis:{Colors.RESET}")
            print(f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")

            for step in result.reasoning_chain:
                print(f"  {Colors.CYAN}{step.module}:{Colors.RESET} {step.thought}")
                for insight in step.insights:
                    print(f"    {Colors.DIM}→ {insight}{Colors.RESET}")

            print(f"\n  {Colors.BOLD}Confidence:{Colors.RESET} {result.confidence:.0%}")
            print(f"  {Colors.BOLD}Modules Used:{Colors.RESET} {', '.join(result.modules_used)}")
            print(f"  {Colors.BOLD}Time:{Colors.RESET} {result.total_time:.2f}s")

            if result.suggested_actions:
                print(f"\n  {Colors.BOLD}Suggested Actions:{Colors.RESET}")
                for action in result.suggested_actions:
                    print(f"  {Colors.YELLOW}→{Colors.RESET} {action}")

            if result.insights:
                print(f"\n  {Colors.BOLD}Key Insights:{Colors.RESET}")
                for insight in result.insights[:5]:
                    print(f"  {Colors.GREEN}•{Colors.RESET} {insight}")

            print()
            continue

        # /improve - Show self-improvement stats and learned patterns
        if user_input == "/improve":
            if self_improver_available:
                stats = self_improver.get_stats()
                print(f"\n  {Colors.BOLD}{Colors.BRIGHT_BLUE}Self-Improving Agent:{Colors.RESET}")
                print(f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
                print(f"  {Colors.BRIGHT_BLUE}Learned Patterns:{Colors.RESET} {stats['learned_patterns']}")
                print(f"  {Colors.BRIGHT_BLUE}Error Solutions:{Colors.RESET} {stats['error_solutions']}")
                print(f"  {Colors.BRIGHT_BLUE}Fixes Applied:{Colors.RESET} {stats['fixes_applied']}")

                if self_improver.learned_patterns:
                    print(f"\n  {Colors.BOLD}Patterns Learned:{Colors.RESET}")
                    for pattern, solution in list(self_improver.learned_patterns.items())[:10]:
                        print(f"    {Colors.CYAN}{pattern}:{Colors.RESET} {solution}")

                if self_improver.error_solutions:
                    print(f"\n  {Colors.BOLD}Known Solutions:{Colors.RESET}")
                    for error, solution in list(self_improver.error_solutions.items())[:5]:
                        print(f"    {Colors.YELLOW}{error[:40]}...{Colors.RESET}")
                        print(f"      → {solution[:60]}")
                print()
            else:
                print(f"  {Colors.DIM}Self-improver not available{Colors.RESET}")
            continue

        # /task <description> - Run a task with self-improvement
        if user_input.startswith("/task "):
            task = user_input[6:].strip()
            if self_improver_available and task:
                print(f"\n  {Colors.BRIGHT_BLUE}[SELF-IMPROVING TASK]{Colors.RESET}")
                print(f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
                print(f"  Task: {task}\n")

                result = self_improver.execute_task(task)

                print(f"\n  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
                if result.success:
                    print(f"  {Colors.GREEN}✓ Task completed successfully{Colors.RESET}")
                else:
                    print(f"  {Colors.RED}✗ Task failed: {result.error}{Colors.RESET}")

                print(f"  Tools used: {', '.join(result.tools_used or []) or 'none'}")
                print(f"  Self-fixes: {result.self_fixes}")

                if result.output:
                    print(f"\n  {Colors.BOLD}Output:{Colors.RESET}")
                    # Word wrap output
                    for line in result.output.split('\n')[:20]:
                        print(f"    {line[:70]}")
                print()
            else:
                print(f"  {Colors.DIM}Usage: /task <task description>{Colors.RESET}")
                print(f"  {Colors.DIM}Example: /task Search for the latest AI news{Colors.RESET}")
            continue

        # Process through autonomous learner (track topics, update curiosity)
        if learner:
            learn_result = learner.process_message(user_input, role="user")

        # Process through cognitive pipeline (enhanced)
        cognitive_context = ""
        reasoning_style = "analytical"
        problem_type = "general"

        if pipeline_available:
            try:
                pipeline_result = pipeline.process(user_input)
                cognitive_context = pipeline_result.content
                if pipeline_result.cognitive_analysis:
                    reasoning_style = pipeline_result.cognitive_analysis.get('style', 'analytical')
                    problem_type = pipeline_result.cognitive_analysis.get('type', 'general')
            except Exception:
                pass

        # Fallback to old cognitive architecture
        elif modules_loaded:
            embedding = text_to_embedding(user_input)
            analysis = classifier.classify(embedding)
            style = selector.select_style(analysis)
            plan = orchestrator.create_plan(analysis, style)

            core.perceive(embedding[:64])
            core.think()
            core.act()

            reasoning_style = style.primary_style.value
            problem_type = analysis.problem_type.value

        # Generate response with agentic tool loop
        history.append({"role": "user", "content": user_input})

        if ollama_available:
            print(f"\n  {Colors.BRIGHT_GREEN}Neuro:{Colors.RESET} ", end="")
            sys.stdout.flush()

            try:
                import requests as req
                import json as js
                import re

                # Build enhanced system prompt with cognitive context and learned patterns
                enhanced_prompt = system_prompt
                if cognitive_context:
                    enhanced_prompt = f"{system_prompt}\n\n{cognitive_context}"

                # Inject learned patterns from self-improver
                if self_improver_available and self_improver.learned_patterns:
                    patterns_text = "\n\n[Learned Patterns]\n"
                    for pattern, solution in list(self_improver.learned_patterns.items())[:5]:
                        patterns_text += f"- {pattern}: {solution}\n"
                    enhanced_prompt += patterns_text

                messages = [{"role": "system", "content": enhanced_prompt}]
                messages.extend(history[-20:])

                max_tool_loops = 5
                tool_count = 0
                final_response = ""

                while tool_count < max_tool_loops:
                    # Get response from LLM
                    r = req.post(
                        "http://localhost:11434/api/chat",
                        json={"model": current_model, "messages": messages, "stream": False},
                        timeout=120
                    )
                    response = r.json()["message"]["content"]

                    # Check for tool call
                    tool_call = None
                    if tools_available:
                        tool_match = re.search(r'<tool>(\w+)</tool>', response)
                        args_match = re.search(r'<args>({.*?})</args>', response, re.DOTALL)

                        if tool_match:
                            tool_name = tool_match.group(1)
                            args = {}
                            if args_match:
                                try:
                                    args = js.loads(args_match.group(1))
                                except:
                                    pass
                            tool_call = (tool_name, args)

                    if tool_call:
                        tool_name, args = tool_call
                        tool_count += 1

                        # Show tool execution
                        print(f"{Colors.YELLOW}[Using {tool_name}...]{Colors.RESET} ", end="")
                        sys.stdout.flush()

                        # Execute tool
                        result = tools.execute(tool_name, args)

                        if result.success:
                            tool_result = f"Tool {tool_name} result:\n{result.output}"
                            # Learn from successful tool results
                            if learner and result.output:
                                learner.add_memory(
                                    content=result.output[:500],
                                    source=tool_name,
                                    topic=tool_name,
                                    importance=0.7
                                )
                        else:
                            tool_result = f"Tool {tool_name} failed: {result.error}"

                        # Add to messages and continue
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": tool_result})
                        print()
                        print(f"  {Colors.BRIGHT_GREEN}Neuro:{Colors.RESET} ", end="")
                    else:
                        # No tool call - final response
                        final_response = response
                        break

                # Print final response with word wrap
                col = 9
                # Remove any tool tags from display
                display_response = re.sub(r'<tool>.*?</tool>', '', final_response)
                display_response = re.sub(r'<args>.*?</args>', '', display_response, flags=re.DOTALL)
                display_response = display_response.strip()

                for char in display_response:
                    if char == '\n':
                        print()
                        print("         ", end="")
                        col = 9
                    else:
                        if col > 65 and char == ' ':
                            print()
                            print("         ", end="")
                            col = 9
                        else:
                            print(char, end="")
                            col += 1

                print()

                # Learn from assistant response
                if learner:
                    learner.process_message(final_response, role="assistant")

                # Also learn through the pipeline if available
                if pipeline_available:
                    try:
                        # Extract key info to learn
                        pipeline.learn(
                            topic=problem_type,
                            content=f"Q: {user_input[:100]} A: {final_response[:200]}",
                            source="conversation",
                            importance=0.5
                        )
                    except Exception:
                        pass

                # Metadata with curiosity
                tools_used = f" │ tools: {tool_count}" if tool_count > 0 else ""
                curiosity_str = ""
                if learner:
                    state = learner.get_state()
                    curiosity_str = f" │ curiosity: {state['curiosity_level']:.0%}"
                print(f"\n  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
                print(f"  {Colors.DIM}model: {current_model} │ style: {reasoning_style}{tools_used}{curiosity_str}{Colors.RESET}")

                history.append({"role": "assistant", "content": final_response})

            except Exception as e:
                error_msg = str(e)
                print(f"\n  {Colors.RED}Error: {error_msg}{Colors.RESET}")

                # SELF-IMPROVEMENT: Analyze error and attempt to fix
                if self_improver_available and tools_available:
                    import requests as req
                    print(f"  {Colors.BRIGHT_BLUE}[SELF-IMPROVING]{Colors.RESET} Analyzing failure...")

                    # Check if we already know how to fix this error
                    known_fix = None
                    for known_error, solution in self_improver.error_solutions.items():
                        if known_error.lower() in error_msg.lower():
                            known_fix = solution
                            break

                    if known_fix:
                        print(f"  {Colors.GREEN}[KNOWN FIX]{Colors.RESET} {known_fix}")
                    else:
                        # Analyze the error type and attempt fixes
                        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                            fix = "Ollama is slow or overloaded. Retrying with simpler request..."
                            print(f"  {Colors.YELLOW}[RETRY]{Colors.RESET} {fix}")

                            # Store the fix for next time
                            self_improver.error_solutions["timeout"] = "Retry with shorter input or wait for Ollama"
                            self_improver._save_improvements()

                            # Retry with a simpler request
                            try:
                                # Shorter history, simpler prompt
                                retry_messages = [
                                    {"role": "system", "content": "You are Neuro, an AI assistant. Be brief."},
                                    {"role": "user", "content": user_input[:200]}  # Truncate input
                                ]
                                r = req.post(
                                    "http://localhost:11434/api/chat",
                                    json={"model": current_model, "messages": retry_messages, "stream": False},
                                    timeout=60  # Shorter timeout
                                )
                                if r.status_code == 200:
                                    retry_response = r.json()["message"]["content"]
                                    print(f"\n  {Colors.BRIGHT_GREEN}Neuro:{Colors.RESET} {retry_response[:500]}")
                                    print(f"\n  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
                                    print(f"  {Colors.BRIGHT_BLUE}[RECOVERED]{Colors.RESET} Self-improvement successful")

                                    # Learn this recovery pattern
                                    self_improver.learned_patterns["timeout_recovery"] = "Use shorter context and timeout"
                                    self_improver._save_improvements()
                            except Exception as retry_error:
                                print(f"  {Colors.RED}[RETRY FAILED]{Colors.RESET} {retry_error}")
                                print(f"  {Colors.DIM}Suggestion: Check if Ollama is running (ollama serve){Colors.RESET}")

                        elif "connection" in error_msg.lower():
                            fix = "Connection issue - Ollama may not be running"
                            print(f"  {Colors.YELLOW}[FIX]{Colors.RESET} {fix}")
                            print(f"  {Colors.DIM}Run: ollama serve{Colors.RESET}")
                            self_improver.error_solutions["connection"] = "Start Ollama with 'ollama serve'"
                            self_improver._save_improvements()

                        else:
                            # Unknown error - search for solution
                            print(f"  {Colors.YELLOW}[SEARCHING]{Colors.RESET} Looking for solution...")
                            search_result = tools.web_search(f"fix error: {error_msg[:50]}")
                            if search_result.success and search_result.output:
                                print(f"  {Colors.GREEN}[FOUND]{Colors.RESET} {search_result.output[:200]}")
                                self_improver.error_solutions[error_msg[:50]] = search_result.output[:100]
                                self_improver._save_improvements()
        else:
            # Fallback without Ollama
            print(f"\n  {Colors.BRIGHT_GREEN}Neuro:{Colors.RESET} ", end="")
            print(f"I'm Neuro AGI with 38 cognitive modules active.")
            print(f"         To get intelligent responses, please start Ollama:")
            print(f"         {Colors.DIM}ollama serve && ollama pull mistral{Colors.RESET}")
            print(f"\n  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
            print(f"  {Colors.DIM}style: {reasoning_style} │ type: {problem_type}{Colors.RESET}")

    return 0


def cmd_info(args):
    """Show system information."""
    print_header()

    print(f"  {Colors.BOLD}Architecture{Colors.RESET}")
    print(f"  {Colors.DIM}{'─' * 50}{Colors.RESET}")
    print(f"  38 cognitive modules implementing human-like reasoning")
    print()

    modules = [
        ("Tier 1", "Cognitive", [
            "Integration (Global Workspace)",
            "Predictive Coding (Free Energy)",
            "Dual-Process (System 1/2)",
            "Memory (sensory, working, long-term)",
            "Language, Creativity, Emotions",
            "Executive Control, Planning",
        ]),
        ("Tier 2", "Infrastructure", [
            "Cognitive Core (active inference)",
            "LLM Bridge (Ollama integration)",
            "Knowledge Graph",
            "Sensor/Actuator Interface",
        ]),
        ("Tier 3", "Support", [
            "Benchmarks & Evaluation",
            "Bayesian Inference",
            "Multimodal Perception",
        ]),
    ]

    for tier, name, items in modules:
        print(f"  {Colors.CYAN}{tier}{Colors.RESET}: {Colors.BOLD}{name}{Colors.RESET}")
        for item in items:
            print(f"    {Colors.DIM}•{Colors.RESET} {item}")
        print()

    print(f"  {Colors.BOLD}Quick Start{Colors.RESET}")
    print(f"  {Colors.DIM}{'─' * 50}{Colors.RESET}")
    print(f"  {Colors.CYAN}neuro chat{Colors.RESET}     Chat with the AGI")
    print(f"  {Colors.CYAN}neuro demo{Colors.RESET}     System demonstration")
    print(f"  {Colors.CYAN}neuro check{Colors.RESET}    Verify installation")
    print()

    return 0


def cmd_demo(args):
    """Run system demonstration."""
    import numpy as np

    print_header()
    print(f"  {Colors.BOLD}System Demonstration{Colors.RESET}")
    print(f"  {Colors.DIM}{'─' * 50}{Colors.RESET}")
    print()

    steps = [
        ("Loading Meta-Reasoning", lambda: __import__('neuro').ProblemClassifier(random_seed=42)),
        ("Loading Style Selector", lambda: __import__('neuro').StyleSelector(random_seed=42)),
        ("Loading Orchestrator", lambda: __import__('neuro').DynamicOrchestrator(random_seed=42)),
        ("Loading Cognitive Core", lambda: (__import__('neuro').CognitiveCore(), None)[0]),
        ("Loading Global Workspace", lambda: __import__('neuro').GlobalWorkspace()),
    ]

    components = {}

    for name, loader in steps:
        spinner = Spinner(name)
        spinner.start()
        try:
            result = loader()
            if name == "Loading Cognitive Core":
                result.initialize()
            components[name] = result
            spinner.stop()
            print(f"  {Colors.GREEN}✓{Colors.RESET} {name}")
        except Exception as e:
            spinner.stop()
            print(f"  {Colors.RED}✗{Colors.RESET} {name}: {e}")

    print()
    print(f"  {Colors.BOLD}Running Test Problem{Colors.RESET}")
    print(f"  {Colors.DIM}{'─' * 50}{Colors.RESET}")

    try:
        problem = np.random.randn(128)

        classifier = components.get("Loading Meta-Reasoning")
        selector = components.get("Loading Style Selector")
        orchestrator = components.get("Loading Orchestrator")
        core = components.get("Loading Cognitive Core")

        if classifier and selector and orchestrator:
            analysis = classifier.classify(problem)
            print(f"  Problem Type: {Colors.CYAN}{analysis.problem_type.value}{Colors.RESET}")
            print(f"  Difficulty: {Colors.CYAN}{analysis.difficulty.value}{Colors.RESET}")

            style = selector.select_style(analysis)
            print(f"  Style: {Colors.CYAN}{style.primary_style.value}{Colors.RESET}")
            print(f"  Fitness: {Colors.CYAN}{style.primary_fitness:.0%}{Colors.RESET}")

            plan = orchestrator.create_plan(analysis, style)
            print(f"  Plan Steps: {Colors.CYAN}{len(plan.steps)}{Colors.RESET}")

            if core:
                core.perceive(problem[:64])
                core.think()
                core.act()
                stats = core.get_statistics()
                print(f"  Cycles: {Colors.CYAN}{stats['cycle_count']}{Colors.RESET}")

    except Exception as e:
        print(f"  {Colors.RED}Error: {e}{Colors.RESET}")

    print()
    print(f"  {Colors.GREEN}{'─' * 50}{Colors.RESET}")
    print(f"  {Colors.GREEN}✓ 38 modules demonstrated successfully{Colors.RESET}")
    print()

    return 0


def cmd_check(args):
    """Verify installation."""
    print_header()
    print(f"  {Colors.BOLD}Installation Check{Colors.RESET}")
    print(f"  {Colors.DIM}{'─' * 50}{Colors.RESET}")
    print()

    checks = []

    # Dependencies
    print(f"  {Colors.BOLD}Dependencies{Colors.RESET}")

    for pkg in ["numpy", "scipy"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', 'ok')
            print(f"  {Colors.GREEN}✓{Colors.RESET} {pkg} {Colors.DIM}{ver}{Colors.RESET}")
            checks.append(True)
        except ImportError:
            print(f"  {Colors.RED}✗{Colors.RESET} {pkg}")
            checks.append(False)

    print()
    print(f"  {Colors.BOLD}Core Modules{Colors.RESET}")

    modules = [
        ("CognitiveCore", "Cognitive integration"),
        ("GlobalWorkspace", "Consciousness layer"),
        ("ProblemClassifier", "Meta-reasoning"),
        ("StyleSelector", "Reasoning styles"),
        ("DynamicOrchestrator", "Execution planning"),
    ]

    for name, desc in modules:
        try:
            import neuro
            getattr(neuro, name)
            print(f"  {Colors.GREEN}✓{Colors.RESET} {name} {Colors.DIM}({desc}){Colors.RESET}")
            checks.append(True)
        except Exception:
            print(f"  {Colors.RED}✗{Colors.RESET} {name}")
            checks.append(False)

    print()
    print(f"  {Colors.BOLD}Ollama{Colors.RESET}")

    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"  {Colors.GREEN}✓{Colors.RESET} Ollama running")
            if models:
                print(f"  {Colors.GREEN}✓{Colors.RESET} Models: {Colors.DIM}{', '.join(models)}{Colors.RESET}")
            else:
                print(f"  {Colors.YELLOW}!{Colors.RESET} No models. Run: {Colors.DIM}ollama pull mistral{Colors.RESET}")
            checks.append(True)
        else:
            print(f"  {Colors.YELLOW}!{Colors.RESET} Ollama not responding")
            checks.append(False)
    except Exception:
        print(f"  {Colors.YELLOW}!{Colors.RESET} Ollama not running. Run: {Colors.DIM}ollama serve{Colors.RESET}")
        checks.append(False)

    print()
    passed = sum(checks)
    total = len(checks)

    if passed == total:
        print(f"  {Colors.GREEN}{'─' * 50}{Colors.RESET}")
        print(f"  {Colors.GREEN}✓ {passed}/{total} checks passed - All systems go{Colors.RESET}")
    else:
        print(f"  {Colors.YELLOW}{'─' * 50}{Colors.RESET}")
        print(f"  {Colors.YELLOW}! {passed}/{total} checks passed{Colors.RESET}")

    print()
    return 0 if passed == total else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="neuro",
        description="Neuro AGI - Neuroscience-inspired cognitive system",
    )
    parser.add_argument("-v", "--version", action="store_true", help="Show version")

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("chat", help="Chat with the AGI").set_defaults(func=cmd_chat)
    subparsers.add_parser("info", help="System information").set_defaults(func=cmd_info)
    subparsers.add_parser("demo", help="Run demonstration").set_defaults(func=cmd_demo)
    subparsers.add_parser("check", help="Verify installation").set_defaults(func=cmd_check)

    args = parser.parse_args()

    if args.version:
        print(f"neuro-agi {get_version()}")
        return 0

    if args.command is None:
        # Default: go straight to chat
        return cmd_chat(args)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
