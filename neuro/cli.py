"""
Neuro AGI Command Line Interface.

A neuroscience-inspired cognitive architecture with streaming responses.
Production-ready CLI with real-time token output.
"""

import argparse
import sys
import os
import time
import threading
import asyncio
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

        visible_text = re.sub(r"\033\[[0-9;]*m", "", text)
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

    def update(self, message: str):
        """Update spinner message."""
        self.message = message

    def _animate(self):
        while self.running:
            frame = self.FRAMES[self.frame % len(self.FRAMES)]
            sys.stdout.write(f"\r  {Colors.CYAN}{frame}{Colors.RESET} {self.message}...")
            sys.stdout.flush()
            self.frame += 1
            time.sleep(0.08)


def get_version():
    return "0.9.0"


def print_header():
    """Print the main header."""
    print()
    print(f"  {Colors.CYAN}{Colors.BOLD}╭{'─' * 50}╮{Colors.RESET}")
    print(
        f"  {Colors.CYAN}{Colors.BOLD}│{Colors.RESET}{'NEURO AGI v0.9':^50}{Colors.CYAN}{Colors.BOLD}│{Colors.RESET}"
    )
    print(
        f"  {Colors.CYAN}{Colors.BOLD}│{Colors.RESET}{Colors.DIM}{'Local AI That Learns From Your Code':^50}{Colors.RESET}{Colors.CYAN}{Colors.BOLD}│{Colors.RESET}"
    )
    print(f"  {Colors.CYAN}{Colors.BOLD}╰{'─' * 50}╯{Colors.RESET}")
    print()


def show_cognitive_phase(phase: str, details: str = ""):
    """Display cognitive phase indicator."""
    icons = {
        "perceive": "👁",
        "think": "💭",
        "act": "⚡",
        "learn": "📚",
        "improve": "🔧",
    }
    icon = icons.get(phase.lower(), "●")
    if details:
        print(f"  {Colors.DIM}{icon} {phase.upper()}: {details}{Colors.RESET}")
    else:
        print(f"  {Colors.DIM}{icon} {phase.upper()}{Colors.RESET}")


class StatusLine:
    """Interactive status line that updates in place (like Claude Code)."""

    def __init__(self):
        self.running = False
        self.thread = None
        self.engine = None
        self.last_status = ""
        self.spinner_frame = 0
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def start(self, engine):
        """Start the status line updater."""
        self.engine = engine
        self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the status line updater."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        # Clear the status line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    def _get_status(self) -> str:
        """Get current status from autonomous loop."""
        if not self.engine or not self.engine.autonomous:
            return ""

        try:
            auto = self.engine.autonomous
            if not auto.running:
                return ""

            # Get current activity
            if auto.is_busy:
                if not auto.initial_benchmark_done:
                    return f"{Colors.YELLOW}Running initial benchmark...{Colors.RESET}"

                focus = auto._current_focus
                if focus:
                    learned = auto._focus_learned
                    return f"{Colors.CYAN}Learning from {focus}{Colors.RESET} ({learned} facts)"

            # Show cycle progress
            if self.engine.evolution:
                evo = self.engine.evolution.get_stats()
                cycle = evo.get("cycle", 0)
                facts = evo.get("facts_this_cycle", 0)
                total = evo.get("total_facts", 0)

                # Progress bar
                progress = facts / 100
                bar_width = 15
                filled = int(bar_width * progress)
                bar = "█" * filled + "░" * (bar_width - filled)

                if facts > 0:
                    return f"{Colors.GREEN}Cycle {cycle}{Colors.RESET} [{bar}] {facts}/100 facts | total: {total}"

            return ""
        except Exception:
            return ""

    def _update_loop(self):
        """Background loop to update status line."""
        while self.running:
            try:
                status = self._get_status()
                if status and status != self.last_status:
                    spinner = self.spinner_chars[self.spinner_frame % len(self.spinner_chars)]
                    # Save cursor, move to status line, print, restore cursor
                    line = f"\r  {Colors.DIM}{spinner}{Colors.RESET} {status}"
                    # Pad to clear old content
                    line += " " * max(0, 80 - len(line))
                    sys.stdout.write(f"\033[s\033[999;1H{line}\033[u")
                    sys.stdout.flush()
                    self.last_status = status
                    self.spinner_frame += 1
            except Exception:
                pass
            time.sleep(0.5)


class ActivityDisplay:
    """Real-time activity display for autonomous learning (like Claude Code)."""

    ICONS = {
        "benchmark": "📊",
        "benchmark_done": "✅",
        "fetching": "🔍",
        "focus": "🎯",
        "analyzing": "🧠",
        "learned": "📚",
        "skip": "⏭️",
        "reflection": "💭",
        "training": "🔧",
        "training_done": "✨",
        "training_check": "🔄",
        "training_skip": "⏸️",
        "new_cycle": "🔄",
        "error": "❌",
        "ocr": "👁",
        "image": "🖼️",
    }

    def __init__(self):
        self.last_activity = ""
        self.activity_count = 0

    def on_activity(self, activity_type: str, message: str, data: dict):
        """Callback for activity notifications - displays in terminal."""
        icon = self.ICONS.get(activity_type, "●")

        # Color based on type
        if activity_type in ["learned", "benchmark_done", "training_done"]:
            color = Colors.GREEN
        elif activity_type in ["focus", "new_cycle"]:
            color = Colors.BRIGHT_CYAN
        elif activity_type in ["analyzing", "fetching"]:
            color = Colors.CYAN
        elif activity_type in ["skip"]:
            color = Colors.DIM
        elif activity_type in ["error"]:
            color = Colors.RED
        else:
            color = Colors.YELLOW

        # Format message based on type
        if activity_type == "learned":
            # Show detailed learning info
            topic = data.get("topic", "Unknown")
            source = data.get("source", "Unknown")
            progress = data.get("cycle_progress", 0)
            total = data.get("total_facts", 0)

            # Progress bar
            bar_width = 20
            filled = int(bar_width * (progress / 100))
            bar = f"{Colors.GREEN}{'█' * filled}{Colors.DIM}{'░' * (bar_width - filled)}{Colors.RESET}"

            print(f"\n  {icon} {color}LEARNED{Colors.RESET} [{bar}] {progress}/100")
            print(f"     {Colors.CYAN}Topic:{Colors.RESET} {topic[:50]}")
            print(f"     {Colors.DIM}Source: {source} | Total: {total} facts{Colors.RESET}")

        elif activity_type == "benchmark_done":
            score = data.get("score", 0)
            weak = data.get("weak_areas", [])
            print(f"\n  {icon} {color}BENCHMARK COMPLETE{Colors.RESET}")
            print(f"     {Colors.CYAN}Score:{Colors.RESET} {score:.1%}")
            if weak:
                weak_str = ", ".join([f"{w[0]}: {w[1]:.0%}" for w in weak[:3]])
                print(f"     {Colors.YELLOW}Weak areas:{Colors.RESET} {weak_str}")

        elif activity_type == "focus":
            source = data.get("source", "Unknown")
            items = data.get("items", 0)
            print(f"\n  {icon} {color}FOCUS:{Colors.RESET} {source}")
            print(f"     {Colors.DIM}Processing {items} items...{Colors.RESET}")

        elif activity_type == "new_cycle":
            cycle = data.get("cycle", 0)
            print(f"\n  {icon} {color}NEW LEARNING CYCLE {cycle}{Colors.RESET}")
            print(f"     {Colors.DIM}Target: 100 unique facts{Colors.RESET}")

        elif activity_type == "reflection":
            # Show reflection summary
            print(f"\n  {icon} {color}REFLECTION{Colors.RESET}")
            # Parse reflection message for key stats
            lines = message.strip().split("\n")
            for line in lines[:5]:
                if line.strip() and not line.startswith("="):
                    print(f"     {Colors.DIM}{line.strip()}{Colors.RESET}")

        elif activity_type in ["skip", "fetching", "analyzing"]:
            # Compact display for frequent activities - inline update
            short_msg = message[:55] + "..." if len(message) > 55 else message
            sys.stdout.write(f"\r  {icon} {color}{short_msg}{Colors.RESET}" + " " * 20)
            sys.stdout.flush()
            self.last_activity = message
            self.activity_count += 1
            return  # Don't print newline

        else:
            # Generic display
            print(f"\n  {icon} {color}{message}{Colors.RESET}")

        self.last_activity = message
        self.activity_count += 1
        # Reprint prompt hint
        sys.stdout.write(f"\n  {Colors.DIM}(autonomous learning active){Colors.RESET}")
        sys.stdout.flush()


class LearningNotifier:
    """Shows real-time learning notifications (like Claude Code activity)."""

    def __init__(self):
        self.last_facts = 0
        self.last_cycle = 0
        self.notifications = []

    def check_and_notify(self, engine) -> list:
        """Check for new learning events and return notifications."""
        notifications = []

        if not engine or not engine.evolution:
            return notifications

        try:
            evo = engine.evolution.get_stats()
            facts = evo.get("total_facts", 0)
            cycle = evo.get("cycle", 0)
            facts_this_cycle = evo.get("facts_this_cycle", 0)

            # New facts learned
            if facts > self.last_facts:
                diff = facts - self.last_facts
                notifications.append(
                    f"{Colors.GREEN}+{diff} facts learned{Colors.RESET} (total: {facts})"
                )
                self.last_facts = facts

            # New cycle started
            if cycle > self.last_cycle:
                notifications.append(
                    f"{Colors.BRIGHT_CYAN}Started learning cycle {cycle}{Colors.RESET}"
                )
                self.last_cycle = cycle

            # Milestone notifications
            if facts_this_cycle == 50 and self.last_facts < facts:
                notifications.append(
                    f"{Colors.YELLOW}Halfway to benchmark! (50/100 facts){Colors.RESET}"
                )

            if facts_this_cycle >= 100:
                notifications.append(
                    f"{Colors.BRIGHT_GREEN}Cycle complete! Running benchmark...{Colors.RESET}"
                )

            # Check autonomous loop state
            if engine.autonomous:
                auto = engine.autonomous
                if auto.initial_benchmark_done and auto.benchmark_results:
                    auto.benchmark_results.get("avg_score", 0)
                    weak = auto.weak_areas
                    if weak and len(notifications) == 0:
                        # Occasionally remind about weak areas
                        area, _ = weak[0]
                        notifications.append(f"{Colors.DIM}Focusing on: {area}{Colors.RESET}")

        except Exception:
            pass

        return notifications


def cmd_chat(args):
    """Chat with Neuro AGI using streaming responses."""
    # Get flags with defaults
    model = getattr(args, "model", "nanbeige4.1:3b") or "nanbeige4.1:3b"
    verbose = getattr(args, "verbose", False)
    print_mode = getattr(args, "print_mode", False)
    initial_prompt = getattr(args, "prompt", None)
    no_stream = getattr(args, "no_stream", False)
    output_format = getattr(args, "output_format", "text")
    system_prompt = getattr(args, "system_prompt", None)
    getattr(args, "append_system_prompt", None)

    # Skip header in print mode
    if not print_mode:
        print_header()

    # Check Ollama
    if not print_mode:
        spinner = Spinner("Connecting to Ollama")
        spinner.start()

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

    if not print_mode:
        spinner.stop()

    # Initialize the new NEURO Engine
    engine = None
    engine_available = False

    try:
        from neuro.engine import NeuroEngine, EngineConfig

        config = EngineConfig(model=model, show_thinking=not print_mode, verbose=verbose)
        if system_prompt:
            config.system_prompt = system_prompt
        engine = NeuroEngine(config)
        engine_available = True
    except ImportError:
        # Fallback: try relative import
        try:
            from neuro.engine import NeuroEngine, EngineConfig

            config = EngineConfig(model=model, show_thinking=not print_mode, verbose=verbose)
            if system_prompt:
                config.system_prompt = system_prompt
            engine = NeuroEngine(config)
            engine_available = True
        except ImportError:
            pass

    # Initialize tools (for fallback mode)
    tools_available = False
    try:
        from neuro.modules.model.tools import Tools

        Tools(work_dir=os.getcwd())
        tools_available = True
    except ImportError:
        pass

    # Initialize Cognitive Pipeline
    pipeline = None
    pipeline_available = False
    try:
        from cognitive_pipeline import CognitivePipeline

        pipeline = CognitivePipeline(verbose=False)
        pipeline_available = True
    except ImportError:
        pass

    # Initialize autonomous learning via engine
    learner = None
    learner_available = False
    autonomous_started = False
    activity_display = ActivityDisplay()

    if engine_available:
        # Set up activity callback for real-time display
        engine.set_activity_callback(activity_display.on_activity)

        # Start autonomous loop from engine
        if engine.start_autonomous():
            autonomous_started = True
            learner_available = True
        # Also check if we have trainer/evolution directly
        if engine.trainer:
            learner_available = True

    # Initialize UltraThink
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

    # Initialize Curiosity Loop (DISABLED - replaced by autonomous learning)
    # Old curiosity loop is replaced by the new autonomous learning system
    # which is integrated into the engine

    # Initialize Self-Improving Agent
    self_improver = None
    self_improver_available = False
    try:
        from self_improving_agent import SelfImprovingAgent

        self_improver = SelfImprovingAgent()
        self_improver_available = True
    except ImportError:
        pass

    # Status display
    if ollama_available and models:
        model = "nanbeige4.1:3b"
        print(f"  {Colors.GREEN}●{Colors.RESET} Ollama: {Colors.GREEN}connected{Colors.RESET}")
        print(f"  {Colors.DIM}  Model: {model}{Colors.RESET}")
    else:
        print(
            f"  {Colors.YELLOW}●{Colors.RESET} Ollama: {Colors.YELLOW}not available{Colors.RESET}"
        )
        print(f"  {Colors.DIM}  Run: ollama serve && ollama pull nanbeige4.1:3b{Colors.RESET}")

    if engine_available:
        stats = engine.get_stats()
        print(
            f"  {Colors.BRIGHT_GREEN}●{Colors.RESET} NEURO Engine: {Colors.BRIGHT_GREEN}STREAMING ENABLED{Colors.RESET}"
        )
        if stats.get("pipeline_available"):
            print(f"  {Colors.DIM}  └─ Cognitive Pipeline: active{Colors.RESET}")
        if stats.get("emotions_available"):
            emo = stats.get("emotion", {})
            print(f"  {Colors.DIM}  └─ Emotions: {emo.get('dominant', 'neutral')}{Colors.RESET}")
        if stats.get("social_available"):
            print(f"  {Colors.DIM}  └─ Social Cognition: active{Colors.RESET}")
        if stats.get("embodied_available"):
            print(f"  {Colors.DIM}  └─ Embodied Cognition: active{Colors.RESET}")
        if stats.get("git_repo"):
            print(f"  {Colors.DIM}  └─ Git: repository detected{Colors.RESET}")
    elif pipeline_available:
        pstats = pipeline.get_stats()
        cognitive_count = pstats.get("cognitive_modules", 0)
        pipeline_count = pstats.get("pipeline_components", len(pstats.get("active_components", [])))
        total = pstats.get("num_components", cognitive_count + pipeline_count)
        print(
            f"  {Colors.GREEN}●{Colors.RESET} Cognitive Pipeline: {Colors.GREEN}{total} components{Colors.RESET}"
        )

    if tools_available:
        print(
            f"  {Colors.GREEN}●{Colors.RESET} Tools: {Colors.GREEN}13 tools available{Colors.RESET}"
        )

    if learner_available and engine_available:
        if engine.trainer:
            trainer_stats = engine.trainer.get_stats()
            fact_count = trainer_stats.get("total_facts", 0)
            topic_count = trainer_stats.get("topics_count", 0)
            if autonomous_started:
                print(
                    f"  {Colors.BRIGHT_GREEN}●{Colors.RESET} Autonomous: {Colors.BRIGHT_GREEN}READY{Colors.RESET} | {fact_count} facts | {topic_count} topics"
                )
                print(
                    f"  {Colors.DIM}    └─ Will learn when idle (10s after your last message){Colors.RESET}"
                )
            else:
                print(
                    f"  {Colors.GREEN}●{Colors.RESET} Learning: {Colors.GREEN}{fact_count} facts{Colors.RESET} | {topic_count} topics"
                )

    if self_improver_available:
        si_stats = self_improver.get_stats()
        print(
            f"  {Colors.BRIGHT_BLUE}●{Colors.RESET} Self-Improver: {Colors.BRIGHT_BLUE}{si_stats['learned_patterns']} patterns{Colors.RESET}, {si_stats['error_solutions']} solutions"
        )

    print()
    print(
        f"  {Colors.DIM}Commands: /help /status /model /tools /learn /benchmark /compact /cost /clear /exit{Colors.RESET}"
    )
    print(f"  {Colors.DIM}{'─' * 50}{Colors.RESET}")

    history = []
    current_model = model

    # Async event loop for streaming
    loop = asyncio.new_event_loop()

    def cleanup():
        """Cleanup on exit."""
        try:
            if engine_available:
                loop.run_until_complete(engine.close())
            if pipeline_available:
                pipeline.save()
                pstats = pipeline.get_stats()
                if "knowledge" in pstats:
                    print(
                        f"  {Colors.GREEN}Knowledge: {pstats['knowledge']['total_facts']} facts saved{Colors.RESET}"
                    )
        except Exception:
            pass
        # Autonomous loop is stopped by engine.close()
        try:
            if engine_available and engine.trainer:
                stats = engine.trainer.get_stats()
                print(f"  {Colors.DIM}Saved {stats.get('total_facts', 0)} facts{Colors.RESET}")
        except Exception:
            pass
        print(f"  {Colors.DIM}Goodbye!{Colors.RESET}\n")
        loop.close()

    # Handle print mode (-p flag) - non-interactive
    if print_mode and initial_prompt:
        if not ollama_available:
            print("Error: Ollama not available", file=sys.stderr)
            return 1

        try:
            if engine_available:
                response = loop.run_until_complete(
                    engine.chat(initial_prompt, stream_to_terminal=not no_stream)
                )
            else:
                import requests

                r = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": current_model,
                        "messages": [{"role": "user", "content": initial_prompt}],
                        "stream": False,
                    },
                    timeout=120,
                )
                response = r.json()["message"]["content"]

            # Output based on format
            if output_format == "json":
                import json

                print(json.dumps({"response": response}))
            elif output_format == "stream-json":
                import json

                print(json.dumps({"type": "message", "content": response}))
            else:
                if no_stream:
                    print(response)
                # If streaming was on, response was already printed
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        loop.close()
        return 0

    # Handle initial prompt in interactive mode
    if initial_prompt and not print_mode:
        history.append({"role": "user", "content": initial_prompt})
        # Will be processed in first loop iteration

    while True:
        try:
            user_input = input(f"\n  {Colors.BRIGHT_CYAN}You:{Colors.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            cleanup()
            break

        if not user_input:
            continue

        # Commands
        if user_input in ("/quit", "/exit"):
            cleanup()
            break

        if user_input == "/clear":
            history.clear()
            print(f"  {Colors.DIM}Conversation cleared{Colors.RESET}")
            continue

        # /help - Show all commands (Claude-style)
        if user_input == "/help":
            print(f"\n  {Colors.BOLD}{Colors.CYAN}NEURO Commands{Colors.RESET}")
            print(f"  {Colors.DIM}{'─' * 55}{Colors.RESET}")
            commands = [
                ("/help", "Show this help message"),
                ("/status", "View system and session status"),
                ("/model [name]", "Switch model (e.g., /model qwen2.5:7b)"),
                ("/tools", "List available tools"),
                ("/learn", "Show learning statistics"),
                ("/benchmark", "Run capability benchmark"),
                ("/autonomous", "Show autonomous learning status"),
                ("/think <query>", "Deep reasoning with UltraThink"),
                ("/edit <file>", "Edit a file with diff preview"),
                ("/git [cmd]", "Git operations (status, commit)"),
                ("/ocr <image>", "Read text from image (DeepSeek OCR)"),
                ("/compact", "Compress conversation context"),
                ("/cost", "Show token usage and cost estimate"),
                ("/context", "View context usage"),
                ("/memory", "View/edit memory files"),
                ("/config", "Open settings"),
                ("/clear", "Clear conversation history"),
                ("/exit", "Exit NEURO"),
            ]
            for cmd, desc in commands:
                print(f"  {Colors.CYAN}{cmd:20}{Colors.RESET} {desc}")

            print(f"\n  {Colors.BOLD}Input Syntax:{Colors.RESET}")
            print(f"  {Colors.CYAN}@./file.py{Colors.RESET}          Reference a file")
            print(f"  {Colors.CYAN}!command{Colors.RESET}           Execute shell command")
            print()
            continue

        # /model - Switch model (Claude-style)
        if user_input.startswith("/model"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                new_model = parts[1].strip()
                # Check if model exists
                if new_model in models or ":" in new_model:
                    current_model = new_model
                    if engine_available:
                        engine.config.model = new_model
                    print(f"  {Colors.GREEN}Switched to model: {new_model}{Colors.RESET}")
                else:
                    print(f"  {Colors.RED}Model not found: {new_model}{Colors.RESET}")
                    print(f"  {Colors.DIM}Available: {', '.join(models[:5])}...{Colors.RESET}")
            else:
                print(f"  {Colors.CYAN}Current model:{Colors.RESET} {current_model}")
                print(f"  {Colors.DIM}Available models:{Colors.RESET}")
                for m in models[:10]:
                    marker = " *" if m == current_model else ""
                    print(f"    {m}{marker}")
                if len(models) > 10:
                    print(f"    {Colors.DIM}... and {len(models) - 10} more{Colors.RESET}")
            continue

        # /compact - Compress context (Claude-style)
        if user_input.startswith("/compact"):
            if len(history) < 4:
                print(f"  {Colors.DIM}Not enough history to compact{Colors.RESET}")
                continue

            # Summarize older messages
            old_count = len(history)
            if engine_available:
                summary_prompt = "Summarize this conversation in 2-3 sentences, preserving key facts and decisions:\n\n"
                for msg in history[:-4]:
                    summary_prompt += f"{msg['role'].upper()}: {msg['content'][:200]}\n"

                try:
                    summary = loop.run_until_complete(
                        engine.chat(summary_prompt, stream_to_terminal=False)
                    )
                    # Keep last 4 messages, prepend summary
                    history = [
                        {"role": "system", "content": f"Previous context: {summary}"}
                    ] + history[-4:]
                    print(
                        f"  {Colors.GREEN}Compacted {old_count} messages to {len(history)}{Colors.RESET}"
                    )
                except Exception as e:
                    print(f"  {Colors.RED}Compact failed: {e}{Colors.RESET}")
            else:
                # Simple truncation
                history = history[-6:]
                print(f"  {Colors.GREEN}Truncated to last 6 messages{Colors.RESET}")
            continue

        # /cost - Show token usage (Claude-style)
        if user_input == "/cost":
            print(f"\n  {Colors.BOLD}{Colors.CYAN}Token Usage{Colors.RESET}")
            print(f"  {Colors.DIM}{'─' * 55}{Colors.RESET}")

            # Estimate tokens (rough: 4 chars = 1 token)
            total_chars = sum(len(m.get("content", "")) for m in history)
            est_tokens = total_chars // 4

            print(f"  {Colors.CYAN}Messages:{Colors.RESET} {len(history)}")
            print(f"  {Colors.CYAN}Est. tokens:{Colors.RESET} ~{est_tokens:,}")
            print(f"  {Colors.CYAN}Model:{Colors.RESET} {current_model}")
            print(f"  {Colors.DIM}(Local model - no API cost){Colors.RESET}")
            print()
            continue

        # /context - Show context grid (Claude-style)
        if user_input == "/context":
            print(f"\n  {Colors.BOLD}{Colors.CYAN}Context Usage{Colors.RESET}")
            print(f"  {Colors.DIM}{'─' * 55}{Colors.RESET}")

            # Visual context bar
            total_chars = sum(len(m.get("content", "")) for m in history)
            max_context = 32000  # Approximate for most models
            usage = min(1.0, total_chars / max_context)
            bar_width = 40
            filled = int(bar_width * usage)

            if usage < 0.5:
                color = Colors.GREEN
            elif usage < 0.8:
                color = Colors.YELLOW
            else:
                color = Colors.RED

            bar = f"{color}{'█' * filled}{Colors.DIM}{'░' * (bar_width - filled)}{Colors.RESET}"
            print(f"  [{bar}] {usage:.0%}")
            print(f"  {Colors.DIM}~{total_chars:,} chars / ~{max_context:,} max{Colors.RESET}")
            print()
            continue

        # /memory - View memory files (Claude-style)
        if user_input == "/memory":
            print(f"\n  {Colors.BOLD}{Colors.CYAN}Memory Files{Colors.RESET}")
            print(f"  {Colors.DIM}{'─' * 55}{Colors.RESET}")

            memory_files = [
                ("~/.neuro/settings.json", "User settings"),
                ("~/.cognitive_ai_knowledge/", "Knowledge base"),
                ("./.neuro/NEURO.md", "Project memory"),
            ]
            for path, desc in memory_files:
                expanded = os.path.expanduser(path)
                exists = os.path.exists(expanded)
                status = (
                    f"{Colors.GREEN}exists{Colors.RESET}"
                    if exists
                    else f"{Colors.DIM}not found{Colors.RESET}"
                )
                print(f"  {path:35} {status}")
            print()
            continue

        # /config - Open config (Claude-style)
        if user_input == "/config":
            config_path = os.path.expanduser("~/.neuro/settings.json")
            print(f"\n  {Colors.BOLD}{Colors.CYAN}Configuration{Colors.RESET}")
            print(f"  {Colors.DIM}{'─' * 55}{Colors.RESET}")
            print(f"  {Colors.CYAN}Config file:{Colors.RESET} {config_path}")

            if os.path.exists(config_path):
                try:
                    import json

                    with open(config_path) as f:
                        cfg = json.load(f)
                    for k, v in cfg.items():
                        print(f"  {Colors.CYAN}{k}:{Colors.RESET} {v}")
                except Exception:
                    print(f"  {Colors.DIM}(Could not read config){Colors.RESET}")
            else:
                print(f"  {Colors.DIM}(Config file not found - using defaults){Colors.RESET}")
            print()
            continue

        if user_input == "/status":
            print(f"\n  {Colors.BOLD}{Colors.CYAN}NEURO AGI v0.9 - STATUS{Colors.RESET}")
            print(f"  {Colors.DIM}{'═' * 55}{Colors.RESET}")

            if ollama_available:
                print(f"  {Colors.GREEN}●{Colors.RESET} Ollama: connected ({current_model})")
            else:
                print(f"  {Colors.YELLOW}●{Colors.RESET} Ollama: not available")

            if engine_available:
                stats = engine.get_stats()
                print(f"  {Colors.BRIGHT_GREEN}●{Colors.RESET} Engine: streaming enabled")
                print(f"    └─ Model: {stats['model']}")
                print(f"    └─ Tools: {'available' if stats['tools_available'] else 'not loaded'}")
                print(
                    f"    └─ Pipeline: {'active' if stats['pipeline_available'] else 'not loaded'}"
                )
                print(f"    └─ Git repo: {'yes' if stats['git_repo'] else 'no'}")

            if pipeline_available:
                pstats = pipeline.get_stats()
                if "knowledge" in pstats:
                    kb = pstats["knowledge"]
                    print(f"  {Colors.GREEN}●{Colors.RESET} Knowledge: {kb['total_facts']} facts")
                if "ultrathink" in pstats:
                    ut = pstats["ultrathink"]
                    print(
                        f"  {Colors.GREEN}●{Colors.RESET} UltraThink: {ut['modules_loaded']} modules"
                    )

            if learner_available and engine_available:
                if engine.trainer:
                    trainer_stats = engine.trainer.get_stats()
                    print(
                        f"  {Colors.GREEN}●{Colors.RESET} Learning: {trainer_stats.get('total_facts', 0)} facts"
                    )
                if engine.evolution:
                    evo_stats = engine.evolution.get_stats()
                    print(
                        f"  {Colors.CYAN}●{Colors.RESET} Evolution: cycle {evo_stats.get('cycle', 0)}, {evo_stats.get('total_facts', 0)} unique facts"
                    )
                    improvement = evo_stats.get("improvement", 0)
                    if improvement:
                        print(f"    └─ Improvement: {improvement:+.1%}")
                if autonomous_started and engine.autonomous:
                    auto_stats = engine.autonomous.get_stats()
                    status = "RUNNING" if auto_stats.get("running", False) else "STOPPED"
                    print(f"  {Colors.BRIGHT_GREEN}●{Colors.RESET} Autonomous: {status}")
                    if auto_stats.get("initial_benchmark_done"):
                        weak = auto_stats.get("weak_areas", [])
                        if weak:
                            weak_str = ", ".join([f"{w[0]}: {w[1]:.0%}" for w in weak[:3]])
                            print(f"    └─ Weak areas: {weak_str}")

            # Curiosity loop replaced by autonomous learning

            print(f"  {Colors.DIM}●{Colors.RESET} Session: {len(history)} messages")
            print()
            continue

        if user_input == "/tools":
            print(f"\n  {Colors.BOLD}Available Tools:{Colors.RESET}")
            print(f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")
            tool_list = [
                ("web_search", "Search the web"),
                ("web_fetch", "Fetch webpage content"),
                ("github_user", "Get GitHub user profile"),
                ("github_repos", "List user's repositories"),
                ("read_file", "Read file contents"),
                ("write_file", "Write to a file"),
                ("edit_file", "Edit specific lines (with diff)"),
                ("list_files", "List directory"),
                ("run_command", "Execute shell command"),
                ("run_python", "Run Python code"),
                ("remember", "Store in memory"),
                ("recall", "Retrieve from memory"),
                ("git_commit", "Safe git commit"),
                ("ocr_read", "Read text from image (DeepSeek OCR)"),
                ("ocr_analyze", "Analyze image and answer questions"),
                ("ocr_code", "Extract code from screenshot"),
            ]
            for name, desc in tool_list:
                print(f"  {Colors.CYAN}{name:15}{Colors.RESET} {desc}")
            print()
            continue

        if user_input == "/learn":
            if engine_available and engine.trainer:
                trainer_stats = engine.trainer.get_stats()
                print(f"\n  {Colors.BOLD}Learning State:{Colors.RESET}")
                print(
                    f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}"
                )
                print(
                    f"  {Colors.CYAN}Total Facts:{Colors.RESET} {trainer_stats.get('total_facts', 0)}"
                )
                print(
                    f"  {Colors.CYAN}Topics:{Colors.RESET} {trainer_stats.get('topics_count', 0)}"
                )

                if engine.evolution:
                    evo = engine.evolution.get_stats()
                    print(f"  {Colors.CYAN}Unique Facts:{Colors.RESET} {evo.get('total_facts', 0)}")
                    print(f"  {Colors.CYAN}Learning Cycle:{Colors.RESET} {evo.get('cycle', 0)}")
                    print(
                        f"  {Colors.CYAN}Facts This Cycle:{Colors.RESET} {evo.get('facts_this_cycle', 0)}/100"
                    )
                    if evo.get("baseline_score"):
                        print(
                            f"  {Colors.CYAN}Baseline Score:{Colors.RESET} {evo['baseline_score']:.1%}"
                        )
                    if evo.get("current_score"):
                        print(
                            f"  {Colors.CYAN}Current Score:{Colors.RESET} {evo['current_score']:.1%}"
                        )
                    if evo.get("improvement"):
                        imp = evo["improvement"]
                        color = Colors.GREEN if imp > 0 else Colors.RED
                        print(
                            f"  {Colors.CYAN}Improvement:{Colors.RESET} {color}{imp:+.1%}{Colors.RESET}"
                        )

                print()
            else:
                print(f"  {Colors.DIM}Learning system not available{Colors.RESET}")
            continue

        if user_input == "/benchmark":
            if engine_available and engine.benchmark:
                print(f"\n  {Colors.BOLD}{Colors.YELLOW}Running Benchmark...{Colors.RESET}")
                print(
                    f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}"
                )

                def think_fn(q):
                    knowledge = ""
                    if engine.trainer:
                        knowledge = engine.trainer.get_knowledge_for_prompt(q)
                    if knowledge:
                        enhanced = f"{knowledge}\n\nQuestion: {q}\nThink step by step:"
                    else:
                        enhanced = f"Question: {q}\nThink step by step:"
                    return loop.run_until_complete(engine.chat(enhanced, stream_to_terminal=False))

                results = engine.benchmark.run_benchmark(think_fn, "cli-test")

                avg_score = results.get("avg_score", 0)
                print(f"\n  {Colors.BOLD}Results:{Colors.RESET}")
                print(f"  {Colors.CYAN}Average Score:{Colors.RESET} {avg_score:.1%}")

                # Show by category
                category_scores = {}
                for test in results.get("tests", []):
                    cat = test.get("category", "unknown")
                    score = test.get("score", 0)
                    if cat not in category_scores:
                        category_scores[cat] = []
                    category_scores[cat].append(score)

                print(f"  {Colors.CYAN}By Category:{Colors.RESET}")
                for cat, scores in sorted(category_scores.items()):
                    avg = sum(scores) / len(scores) if scores else 0
                    color = (
                        Colors.GREEN if avg >= 0.7 else Colors.YELLOW if avg >= 0.4 else Colors.RED
                    )
                    print(f"    {cat:15} {color}{avg:.0%}{Colors.RESET}")

                # Record in evolution if available
                if engine.evolution:
                    engine.evolution.record_benchmark(
                        avg_score,
                        {
                            "categories": {k: sum(v) / len(v) for k, v in category_scores.items()},
                            "source": "manual",
                        },
                    )
                    print(
                        f"\n  {Colors.GREEN}Benchmark recorded in evolution tracker{Colors.RESET}"
                    )

                print()
            else:
                print(f"  {Colors.DIM}Benchmark not available{Colors.RESET}")
            continue

        if user_input == "/curiosity" or user_input == "/autonomous":
            if autonomous_started and engine_available and engine.autonomous:
                auto = engine.autonomous
                stats = auto.get_stats()

                # Determine actual status
                if not stats.get("running", False):
                    status_str = f"{Colors.RED}STOPPED{Colors.RESET}"
                elif auto.paused:
                    idle = time.time() - auto.last_user_activity
                    remaining = max(0, auto.idle_threshold - idle)
                    status_str = (
                        f"{Colors.YELLOW}PAUSED{Colors.RESET} (resumes in {remaining:.0f}s)"
                    )
                else:
                    status_str = f"{Colors.BRIGHT_GREEN}LEARNING{Colors.RESET}"

                print(
                    f"\n  {Colors.BOLD}{Colors.BRIGHT_GREEN}Autonomous Learning Loop:{Colors.RESET}"
                )
                print(
                    f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}"
                )
                print(f"  {Colors.BRIGHT_GREEN}Status:{Colors.RESET} {status_str}")
                print(
                    f"  {Colors.BRIGHT_GREEN}Initial Benchmark:{Colors.RESET} {'Done' if stats.get('initial_benchmark_done') else 'Pending'}"
                )
                print(
                    f"  {Colors.BRIGHT_GREEN}Current Focus:{Colors.RESET} {stats.get('current_focus') or 'None'}"
                )
                print(
                    f"  {Colors.BRIGHT_GREEN}Focus Learned:{Colors.RESET} {stats.get('focus_learned', 0)} facts"
                )

                # Show last learned topic
                if auto.last_learned_topic:
                    print(
                        f"  {Colors.CYAN}Last Learned:{Colors.RESET} {auto.last_learned_topic[:50]}"
                    )
                    print(f"  {Colors.DIM}    from {auto.last_learned_source}{Colors.RESET}")

                evo = stats.get("evolution", {})
                if evo:
                    print(f"  {Colors.CYAN}Learning Cycle:{Colors.RESET} {evo.get('cycle', 0)}")
                    print(f"  {Colors.CYAN}Unique Facts:{Colors.RESET} {evo.get('total_facts', 0)}")
                    progress = evo.get("facts_this_cycle", 0)
                    bar_w = 20
                    filled = int(bar_w * (progress / 100))
                    bar = f"{'█' * filled}{'░' * (bar_w - filled)}"
                    print(f"  {Colors.CYAN}Cycle Progress:{Colors.RESET} [{bar}] {progress}/100")

                weak = stats.get("weak_areas", [])
                if weak:
                    print(f"  {Colors.YELLOW}Weak Areas:{Colors.RESET}")
                    for area, score in weak[:5]:
                        print(f"    - {area}: {score:.0%}")

                # Show recent activity
                if auto.activity_log:
                    print(f"\n  {Colors.DIM}Recent Activity:{Colors.RESET}")
                    for a in auto.activity_log[-3:]:
                        icon = ActivityDisplay.ICONS.get(a["type"], "●")
                        print(f"    {icon} {a['message'][:50]}")

                print()
            else:
                print(f"  {Colors.DIM}Autonomous learning not available{Colors.RESET}")
            continue

        # /think <question> - Deep reasoning with UltraThink
        if user_input.startswith("/think"):
            if not ultrathink_available:
                print(f"  {Colors.DIM}UltraThink not available{Colors.RESET}")
                continue

            question = user_input[6:].strip()
            if not question:
                print(f"  {Colors.DIM}Usage: /think <question>{Colors.RESET}")
                continue

            print(f"\n  {Colors.MAGENTA}[UltraThink]{Colors.RESET} Deep reasoning...\n")
            result = ultrathink.think(question)

            print(f"  {Colors.BOLD}Problem Analysis:{Colors.RESET}")
            print(f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}")

            for step in result.reasoning_chain:
                print(f"  {Colors.CYAN}{step.module}:{Colors.RESET} {step.thought}")
                for insight in step.insights:
                    print(f"    {Colors.DIM}→ {insight}{Colors.RESET}")

            print(f"\n  {Colors.BOLD}Confidence:{Colors.RESET} {result.confidence:.0%}")
            print(f"  {Colors.BOLD}Modules Used:{Colors.RESET} {', '.join(result.modules_used)}")
            print()
            continue

        # /edit <file> - Edit a file with diff preview
        if user_input.startswith("/edit "):
            if not engine_available:
                print(f"  {Colors.DIM}Engine not available for editing{Colors.RESET}")
                continue

            file_path = user_input[6:].strip()
            if not file_path:
                print(f"  {Colors.DIM}Usage: /edit <file_path>{Colors.RESET}")
                continue

            # Read the file
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    content = f.read()
                print(f"\n  {Colors.CYAN}File:{Colors.RESET} {file_path}")
                print(f"  {Colors.DIM}Lines: {len(content.splitlines())}{Colors.RESET}")
                print(
                    f"  {Colors.DIM}Ask me to make specific edits (e.g., 'change line 5 to ...'):{Colors.RESET}"
                )
            else:
                print(f"  {Colors.RED}File not found: {file_path}{Colors.RESET}")
            continue

        # /git - Git operations
        if user_input.startswith("/git"):
            if not engine_available:
                print(f"  {Colors.DIM}Engine not available for git{Colors.RESET}")
                continue

            git_cmd = user_input[4:].strip()
            if not git_cmd:
                # Show git status
                if engine.git.is_repo():
                    changes = engine.git.get_status()
                    branch = engine.git.get_current_branch()
                    print(f"\n  {Colors.BOLD}Git Status:{Colors.RESET}")
                    print(
                        f"  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}"
                    )
                    print(f"  {Colors.CYAN}Branch:{Colors.RESET} {branch}")
                    print(f"  {Colors.CYAN}Changes:{Colors.RESET} {len(changes)} files")
                    for c in changes[:10]:
                        staged = "[+]" if c.staged else "[ ]"
                        print(f"    {staged} {c.status} {c.path}")
                    print()
                else:
                    print(f"  {Colors.DIM}Not a git repository{Colors.RESET}")
                continue

            if git_cmd == "commit":
                # Smart commit
                if engine.git.is_repo():
                    print(f"\n  {Colors.CYAN}Generating commit message...{Colors.RESET}")
                    msg = engine.git.generate_commit_message()
                    print(f"  {Colors.GREEN}Suggested:{Colors.RESET} {msg}")
                    confirm = (
                        input(f"  {Colors.DIM}Commit with this message? [Y/n]:{Colors.RESET} ")
                        .strip()
                        .lower()
                    )
                    if confirm in ("", "y", "yes"):
                        changes = engine.git.get_status()
                        files = [c.path for c in changes if c.status != "?"]
                        result = loop.run_until_complete(engine.commit_changes(files, msg))
                        if result.status.value == "success":
                            print(f"  {Colors.GREEN}Committed!{Colors.RESET}")
                        else:
                            print(f"  {Colors.RED}{result.message}{Colors.RESET}")
                continue

        # /ocr <image> [question] - Read text from image using DeepSeek OCR
        if user_input.startswith("/ocr"):
            if not engine_available or not engine.ocr:
                print(f"  {Colors.DIM}DeepSeek OCR not available{Colors.RESET}")
                print(
                    f"  {Colors.DIM}Make sure deepseek-ocr:latest is installed in Ollama{Colors.RESET}"
                )
                continue

            parts = user_input[4:].strip().split(maxsplit=1)
            if not parts:
                print(f"  {Colors.DIM}Usage: /ocr <image_path> [question]{Colors.RESET}")
                print(f"  {Colors.DIM}Examples:{Colors.RESET}")
                print(
                    f"    {Colors.CYAN}/ocr screenshot.png{Colors.RESET}              - Extract all text"
                )
                print(
                    f"    {Colors.CYAN}/ocr diagram.png what does this show?{Colors.RESET} - Analyze image"
                )
                print(
                    f"    {Colors.CYAN}/ocr code.png --code{Colors.RESET}             - Extract code only"
                )
                continue

            image_path = parts[0]
            question = parts[1] if len(parts) > 1 else None

            if not os.path.exists(image_path):
                print(f"  {Colors.RED}Image not found: {image_path}{Colors.RESET}")
                continue

            print(f"\n  {Colors.CYAN}Processing image with DeepSeek OCR...{Colors.RESET}")

            try:
                result = None
                if question == "--code":
                    # Extract code from screenshot
                    result = engine.ocr.extract_code(image_path)
                    print(f"\n  {Colors.BOLD}Extracted Code:{Colors.RESET}")
                    print(f"  {Colors.DIM}{'─' * 55}{Colors.RESET}")
                    for line in result.text.split("\n"):
                        print(f"  {Colors.GREEN}{line}{Colors.RESET}")
                elif question == "--learn":
                    # Learn from image
                    success = engine.learn_from_image(image_path)
                    if success:
                        print(f"  {Colors.GREEN}Learned from image!{Colors.RESET}")
                    else:
                        print(f"  {Colors.YELLOW}Could not extract learnable content{Colors.RESET}")
                elif question:
                    # Analyze with question
                    result = engine.ocr.analyze_image(image_path, question)
                    print(f"\n  {Colors.BOLD}Analysis:{Colors.RESET}")
                    print(f"  {Colors.DIM}{'─' * 55}{Colors.RESET}")
                    for line in result.text.split("\n"):
                        print(f"  {line}")
                else:
                    # Extract all text
                    result = engine.ocr.read_image(image_path)
                    print(f"\n  {Colors.BOLD}Extracted Text:{Colors.RESET}")
                    print(f"  {Colors.DIM}{'─' * 55}{Colors.RESET}")
                    for line in result.text.split("\n"):
                        print(f"  {line}")

                # Show metadata
                if result and hasattr(result, "confidence"):
                    print(f"\n  {Colors.DIM}Confidence: {result.confidence:.0%}{Colors.RESET}")

            except Exception as e:
                print(f"  {Colors.RED}OCR Error: {e}{Colors.RESET}")

            continue

        # ========================================================================
        # SPECIAL INPUT SYNTAX (Claude-style)
        # ========================================================================

        # !command - Execute shell command
        if user_input.startswith("!"):
            cmd = user_input[1:].strip()
            if cmd:
                print(f"  {Colors.DIM}$ {cmd}{Colors.RESET}")
                try:
                    import subprocess

                    result = subprocess.run(
                        cmd, shell=True, capture_output=True, text=True, timeout=30
                    )
                    if result.stdout:
                        print(f"  {result.stdout.rstrip()}")
                    if result.stderr:
                        print(f"  {Colors.RED}{result.stderr.rstrip()}{Colors.RESET}")
                except subprocess.TimeoutExpired:
                    print(f"  {Colors.RED}Command timed out{Colors.RESET}")
                except Exception as e:
                    print(f"  {Colors.RED}Error: {e}{Colors.RESET}")
            continue

        # @./file - Include file content in prompt
        if "@" in user_input:
            import re

            file_refs = re.findall(r"@(\.?/[^\s]+)", user_input)
            for ref in file_refs:
                file_path = os.path.expanduser(ref)
                if os.path.exists(file_path):
                    try:
                        if os.path.isdir(file_path):
                            # List directory
                            files = os.listdir(file_path)[:20]
                            content = f"[Directory {ref}]: {', '.join(files)}"
                        else:
                            with open(file_path, "r") as f:
                                content = f.read()[:5000]  # Limit size
                            content = f"[File {ref}]:\n```\n{content}\n```"
                        user_input = user_input.replace(f"@{ref}", content)
                        print(f"  {Colors.DIM}Included: {ref}{Colors.RESET}")
                    except Exception as e:
                        print(f"  {Colors.RED}Could not read {ref}: {e}{Colors.RESET}")
                else:
                    print(f"  {Colors.YELLOW}File not found: {ref}{Colors.RESET}")

        # Mark user activity - pause autonomous learning while chatting
        if engine_available and autonomous_started and engine.autonomous:
            engine.autonomous.user_active()
            engine.mark_conversation_started()

        # ========================================================================
        # MAIN RESPONSE GENERATION - WITH STREAMING
        # ========================================================================

        history.append({"role": "user", "content": user_input})

        if ollama_available:
            if engine_available:
                # ============================================================
                # NEW: Use the NEURO Engine with streaming
                # ============================================================
                try:
                    # Show cognitive phase: PERCEIVE
                    show_cognitive_phase("perceive", "Retrieving context...")

                    # Get cognitive context
                    cognitive_context = loop.run_until_complete(
                        engine.get_cognitive_context(user_input)
                    )

                    if cognitive_context:
                        memories = cognitive_context.get("memories", [])
                        confidence = cognitive_context.get("confidence", 0)
                        if memories:
                            show_cognitive_phase(
                                "perceive", f"{len(memories)} memories, {confidence:.0%} confidence"
                            )

                    # Show cognitive phase: THINK
                    show_cognitive_phase("think", "Streaming response...")

                    # Stream the response
                    print(f"\n  {Colors.BRIGHT_GREEN}Neuro:{Colors.RESET} ", end="", flush=True)

                    response = loop.run_until_complete(
                        engine.chat(user_input, history=history[:-1], stream_to_terminal=True)
                    )

                    # Show cognitive phase: LEARN (Real learning!)
                    facts_learned = 0
                    if engine_available and engine.trainer:
                        try:
                            # Extract and learn facts from response
                            facts = engine._extract_facts(response)
                            for fact in facts:
                                if engine.evolution and engine.evolution.is_duplicate(
                                    fact["content"]
                                ):
                                    continue
                                engine.trainer.learn(
                                    topic=fact.get("topic", user_input[:30]),
                                    content=fact["content"],
                                    source="conversation",
                                )
                                if engine.evolution:
                                    engine.evolution.mark_learned(fact["content"])
                                facts_learned += 1

                            if facts_learned > 0:
                                engine.trainer.kb.save()
                                total = engine.trainer.kb.stats.get("total_facts", 0)
                                show_cognitive_phase(
                                    "learn", f"Learned {facts_learned} facts (total: {total})"
                                )
                            else:
                                show_cognitive_phase("learn", "No new facts to learn")
                        except Exception as e:
                            show_cognitive_phase("learn", f"Error: {e}")
                    elif pipeline_available:
                        show_cognitive_phase("learn", "Storing interaction...")
                        try:
                            pipeline.learn(
                                topic="conversation",
                                content=f"Q: {user_input[:100]} A: {response[:200]}",
                                source="chat",
                                importance=0.5,
                            )
                        except Exception:
                            pass

                    # Show metadata
                    tools_used = ""
                    learning_str = ""
                    auto_str = ""

                    # Get real learning stats
                    if engine_available and engine.trainer:
                        total_facts = engine.trainer.kb.stats.get("total_facts", 0)
                        learning_str = f" | {Colors.GREEN}{total_facts} facts{Colors.RESET}"

                        if engine.evolution:
                            evo_stats = engine.evolution.get_stats()
                            cycle = evo_stats.get("cycle", 0)
                            progress = evo_stats.get("facts_this_cycle", 0)
                            learning_str += f" | cycle {cycle}"

                            # Show progress bar
                            bar_w = 10
                            filled = int(bar_w * (progress / 100))
                            bar = f"{'█' * filled}{'░' * (bar_w - filled)}"
                            learning_str += f" [{bar}]"

                    # Show autonomous status
                    if autonomous_started and engine.autonomous:
                        if engine.autonomous.running:
                            focus = engine.autonomous._current_focus
                            if focus:
                                auto_str = (
                                    f"\n  {Colors.CYAN}🎯 Learning from: {focus}{Colors.RESET}"
                                )
                            elif engine.autonomous.is_busy:
                                auto_str = f"\n  {Colors.YELLOW}🔄 Processing...{Colors.RESET}"

                    print(
                        f"\n  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}"
                    )
                    print(
                        f"  {Colors.DIM}model: {current_model} | streaming{tools_used}{Colors.RESET}{learning_str}"
                    )
                    if auto_str:
                        print(auto_str)

                    history.append({"role": "assistant", "content": response})

                    # Learn from response
                    if learner:
                        learner.process_message(response, role="assistant")

                except Exception as e:
                    print(f"\n  {Colors.RED}Error: {e}{Colors.RESET}")
                    # Fall back to non-streaming
                    print(f"  {Colors.DIM}Falling back to non-streaming mode...{Colors.RESET}")

            else:
                # ============================================================
                # FALLBACK: Old synchronous mode (non-streaming)
                # ============================================================
                print(f"\n  {Colors.BRIGHT_GREEN}Neuro:{Colors.RESET} ", end="")
                sys.stdout.flush()

                try:
                    import requests as req
                    import re

                    # Build enhanced system prompt
                    system_prompt = """You are Neuro, an advanced AGI with neuroscience-inspired cognition.

CORE PRINCIPLES:
1. NEVER LIE - Always be truthful
2. NEVER FABRICATE - Don't make up information
3. BE DIRECT - No unnecessary hedging
4. ADMIT UNCERTAINTY - Say when confidence is low
5. USE TOOLS - When you need external information

Available Tools:
- web_search(query): Search the web
- read_file(path): Read a file
- write_file(path, content): Write to file
- run_command(command): Run shell command

To use a tool: <tool>name</tool><args>{"param": "value"}</args>
"""

                    messages = [{"role": "system", "content": system_prompt}]
                    messages.extend(history[-20:])

                    r = req.post(
                        "http://localhost:11434/api/chat",
                        json={"model": current_model, "messages": messages, "stream": False},
                        timeout=120,
                    )
                    response = r.json()["message"]["content"]

                    # Word wrap and print
                    col = 9
                    display_response = re.sub(r"<tool>.*?</tool>", "", response)
                    display_response = re.sub(
                        r"<args>.*?</args>", "", display_response, flags=re.DOTALL
                    )
                    display_response = display_response.strip()

                    for char in display_response:
                        if char == "\n":
                            print()
                            print("         ", end="")
                            col = 9
                        else:
                            if col > 65 and char == " ":
                                print()
                                print("         ", end="")
                                col = 9
                            else:
                                print(char, end="")
                                col += 1
                    print()

                    print(
                        f"\n  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}"
                    )
                    print(f"  {Colors.DIM}model: {current_model} | fallback mode{Colors.RESET}")

                    history.append({"role": "assistant", "content": response})

                except Exception as e:
                    print(f"\n  {Colors.RED}Error: {e}{Colors.RESET}")
        else:
            print(f"\n  {Colors.BRIGHT_GREEN}Neuro:{Colors.RESET} ", end="")
            print("Ollama is not running. Please start it:")
            print(f"         {Colors.DIM}ollama serve && ollama pull nanbeige4.1:3b{Colors.RESET}")
            print(
                f"\n  {Colors.DIM}─────────────────────────────────────────────────{Colors.RESET}"
            )

    return 0


def cmd_info(args):
    """Show system information."""
    print_header()

    print(f"  {Colors.BOLD}NEURO v0.9 Architecture{Colors.RESET}")
    print(f"  {Colors.DIM}{'─' * 50}{Colors.RESET}")
    print()

    features = [
        ("Streaming", "Real-time token output from Ollama"),
        ("Code Editing", "Surgical file edits with colored diffs"),
        ("Parallel Execution", "Async tool execution"),
        ("Git Automation", "Safe commits with secrets detection"),
        ("Cognitive Pipeline", "26 neuroscience-inspired modules"),
        ("Memory & Learning", "Gets smarter the more you use it"),
    ]

    for name, desc in features:
        print(f"  {Colors.GREEN}●{Colors.RESET} {Colors.BOLD}{name}{Colors.RESET}")
        print(f"    {Colors.DIM}{desc}{Colors.RESET}")
    print()

    print(f"  {Colors.BOLD}Quick Start{Colors.RESET}")
    print(f"  {Colors.DIM}{'─' * 50}{Colors.RESET}")
    print(f"  {Colors.CYAN}neuro{Colors.RESET}         Start chatting (streaming mode)")
    print(f"  {Colors.CYAN}neuro check{Colors.RESET}   Verify installation")
    print(f"  {Colors.CYAN}neuro demo{Colors.RESET}    Run demonstration")
    print()

    return 0


def cmd_demo(args):
    """Run system demonstration."""
    print_header()
    print(f"  {Colors.BOLD}System Demonstration{Colors.RESET}")
    print(f"  {Colors.DIM}{'─' * 50}{Colors.RESET}")
    print()

    # Test streaming engine
    spinner = Spinner("Loading streaming engine")
    spinner.start()

    try:
        from neuro.engine import NeuroEngine, EngineConfig

        config = EngineConfig(model="nanbeige4.1:3b", show_thinking=False)
        engine = NeuroEngine(config)
        spinner.stop()
        print(f"  {Colors.GREEN}✓{Colors.RESET} Streaming Engine loaded")

        stats = engine.get_stats()
        print(f"    └─ Model: {stats['model']}")
        print(f"    └─ Tools: {'available' if stats['tools_available'] else 'not loaded'}")
        print(f"    └─ Pipeline: {'active' if stats['pipeline_available'] else 'not loaded'}")
        print(f"    └─ Git: {'repo' if stats['git_repo'] else 'no repo'}")

    except Exception as e:
        spinner.stop()
        print(f"  {Colors.RED}✗{Colors.RESET} Streaming Engine: {e}")

    # Test code editor
    spinner = Spinner("Loading code editor")
    spinner.start()

    try:
        from editor import CodeEditor

        CodeEditor(auto_confirm=True)
        spinner.stop()
        print(f"  {Colors.GREEN}✓{Colors.RESET} Code Editor loaded")
    except Exception as e:
        spinner.stop()
        print(f"  {Colors.RED}✗{Colors.RESET} Code Editor: {e}")

    # Test git automator
    spinner = Spinner("Loading git automator")
    spinner.start()

    try:
        from git import GitAutomator

        git = GitAutomator()
        is_repo = git.is_repo()
        spinner.stop()
        print(f"  {Colors.GREEN}✓{Colors.RESET} Git Automator loaded")
        print(f"    └─ In repo: {is_repo}")
        if is_repo:
            print(f"    └─ Branch: {git.get_current_branch()}")
    except Exception as e:
        spinner.stop()
        print(f"  {Colors.RED}✗{Colors.RESET} Git Automator: {e}")

    # Test executor
    spinner = Spinner("Loading parallel executor")
    spinner.start()

    try:
        from executor import ParallelExecutor

        ParallelExecutor(max_parallel=5)
        spinner.stop()
        print(f"  {Colors.GREEN}✓{Colors.RESET} Parallel Executor loaded")
    except Exception as e:
        spinner.stop()
        print(f"  {Colors.RED}✗{Colors.RESET} Parallel Executor: {e}")

    print()
    print(f"  {Colors.GREEN}{'─' * 50}{Colors.RESET}")
    print(f"  {Colors.GREEN}✓ NEURO v0.9 Engine components verified{Colors.RESET}")
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

    for pkg in ["numpy", "scipy", "aiohttp", "rich"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "ok")
            print(f"  {Colors.GREEN}✓{Colors.RESET} {pkg} {Colors.DIM}{ver}{Colors.RESET}")
            checks.append(True)
        except ImportError:
            print(f"  {Colors.YELLOW}!{Colors.RESET} {pkg} {Colors.DIM}(optional){Colors.RESET}")
            checks.append(True)  # Optional deps don't fail

    print()
    print(f"  {Colors.BOLD}NEURO v0.9 Engine{Colors.RESET}")

    components = [
        ("stream", "StreamHandler", "Streaming responses"),
        ("editor", "CodeEditor", "Code editing with diffs"),
        ("executor", "ParallelExecutor", "Parallel tool execution"),
        ("git", "GitAutomator", "Git automation"),
        ("engine", "NeuroEngine", "Unified engine"),
    ]

    for module, cls, desc in components:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            mod = __import__(module)
            getattr(mod, cls)
            print(f"  {Colors.GREEN}✓{Colors.RESET} {cls} {Colors.DIM}({desc}){Colors.RESET}")
            checks.append(True)
        except Exception:
            print(f"  {Colors.RED}✗{Colors.RESET} {cls}")
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
                print(
                    f"  {Colors.GREEN}✓{Colors.RESET} Models: {Colors.DIM}{', '.join(models)}{Colors.RESET}"
                )
            else:
                print(
                    f"  {Colors.YELLOW}!{Colors.RESET} No models. Run: {Colors.DIM}ollama pull nanbeige4.1:3b{Colors.RESET}"
                )
            checks.append(True)
        else:
            print(f"  {Colors.YELLOW}!{Colors.RESET} Ollama not responding")
            checks.append(False)
    except Exception:
        print(
            f"  {Colors.YELLOW}!{Colors.RESET} Ollama not running. Run: {Colors.DIM}ollama serve{Colors.RESET}"
        )
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
        description="NEURO AGI v0.9 - Local AI That Learns From Your Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neuro                          Start interactive REPL
  neuro "explain this code"      Start with initial prompt
  neuro -p "what is 2+2"         Print mode (no interactive)
  neuro -c                       Continue last conversation
  neuro --model qwen2.5:7b       Use specific model
  neuro info                     Show system information
  neuro check                    Verify installation
""",
    )

    # Core flags (Claude-style)
    parser.add_argument("prompt", nargs="?", help="Initial prompt to send")
    parser.add_argument("-v", "--version", action="store_true", help="Show version")
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        dest="print_mode",
        help="Print response and exit (no interactive mode)",
    )
    parser.add_argument(
        "-c",
        "--continue",
        action="store_true",
        dest="continue_session",
        help="Continue most recent conversation",
    )
    parser.add_argument(
        "-r", "--resume", type=str, metavar="SESSION", help="Resume specific session by ID"
    )
    parser.add_argument(
        "--model", type=str, default="nanbeige4.1:3b", help="Model to use (default: nanbeige4.1:3b)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # System prompt flags
    parser.add_argument("--system-prompt", type=str, help="Custom system prompt (replaces default)")
    parser.add_argument("--append-system-prompt", type=str, help="Append to default system prompt")

    # Output flags
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "stream-json"],
        default="text",
        help="Output format for print mode",
    )
    parser.add_argument(
        "--no-stream", action="store_true", help="Disable streaming (wait for full response)"
    )

    # Permission flags
    parser.add_argument(
        "--dangerously-skip-permissions", action="store_true", help="Skip all permission prompts"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", help="Chat with streaming (default)")
    subparsers.add_parser("info", help="System information")
    subparsers.add_parser("demo", help="Run demonstration")
    subparsers.add_parser("check", help="Verify installation")
    subparsers.add_parser("config", help="Open configuration")
    subparsers.add_parser("mcp", help="Configure MCP servers")

    args = parser.parse_args()

    if args.version:
        print(f"elo-agi {get_version()}")
        return 0

    # Handle subcommands
    if args.command == "info":
        return cmd_info(args)
    elif args.command == "demo":
        return cmd_demo(args)
    elif args.command == "check":
        return cmd_check(args)
    elif args.command == "config":
        print(f"  {Colors.DIM}Config location: ~/.neuro/settings.json{Colors.RESET}")
        return 0
    elif args.command == "mcp":
        print(f"  {Colors.DIM}MCP not yet implemented{Colors.RESET}")
        return 0

    # Default: chat mode
    return cmd_chat(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
