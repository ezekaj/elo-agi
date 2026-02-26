#!/usr/bin/env python3
"""
NEURO CLI v2 - Streamlined interface to NeuroEngine.

Usage:
    neuro              Start interactive session
    neuro --version    Show version
    neuro --stats      Show engine statistics
    neuro "query"      One-shot query
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Version from package
try:
    from importlib.metadata import version
    __version__ = version("elo-agi")
except Exception:
    __version__ = "0.9.6"


# ANSI colors
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BLUE = "\033[34m"


def print_banner():
    """Print NEURO banner."""
    banner = f"""
{Colors.BOLD}{Colors.CYAN}╭{'─' * 50}╮{Colors.RESET}
{Colors.BOLD}{Colors.CYAN}│{Colors.RESET}  {Colors.BOLD}NEURO{Colors.RESET} - Neuroscience-inspired AGI{Colors.CYAN}{' ' * 24}│{Colors.RESET}
{Colors.BOLD}{Colors.CYAN}│{Colors.RESET}  v{__version__} - Production Engine v2{Colors.BLUE}{' ' * 20}│{Colors.RESET}
{Colors.BOLD}{Colors.CYAN}╰{'─' * 50}╯{Colors.RESET}
"""
    print(banner)


async def interactive_session():
    """Run interactive chat session."""
    from neuro.engine_v2 import NeuroEngine
    
    print_banner()
    print(f"{Colors.GREEN}Ready!{Colors.RESET} Type your message (or 'quit' to exit)\n")
    print(f"{Colors.DIM}Commands: /help, /stats, /history, /sessions, /git, /clear{Colors.RESET}\n")
    
    async with NeuroEngine() as engine:
        history = []
        
        while True:
            try:
                # Get user input
                user_input = input(f"{Colors.BOLD}{Colors.BLUE}You:{Colors.RESET} ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ("quit", "exit", "q"):
                    print(f"\n{Colors.DIM}Goodbye!{Colors.RESET}")
                    break
                
                # Handle slash commands
                if user_input.startswith("/"):
                    await handle_command(engine, user_input)
                    continue
                
                # Process query
                print(f"{Colors.GREEN}NEURO:{Colors.RESET} ", end="", flush=True)
                
                response = ""
                async for token in engine.stream_chat(user_input, history):
                    response += token
                    print(token, end="", flush=True)
                
                print()  # Newline
                
                # Update history
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": response})
                
                # Keep history manageable
                if len(history) > 20:
                    history = history[-20:]
                
            except KeyboardInterrupt:
                print(f"\n{Colors.DIM}Interrupted. Type 'quit' to exit.{Colors.RESET}")
            except EOFError:
                print(f"\n{Colors.DIM}Goodbye!{Colors.RESET}")
                break
            except Exception as e:
                print(f"\n{Colors.RED}Error:{Colors.RESET} {e}")


async def handle_command(engine, command: str):
    """Handle slash commands."""
    parts = command.split()
    cmd = parts[0].lower()
    args = parts[1:]
    
    if cmd == "/help":
        print(f"""
{Colors.BOLD}Commands:{Colors.RESET}
  /help              - Show this help
  /stats             - Show engine statistics
  /history [n]       - Show last n messages (default: 10)
  /sessions          - List saved sessions
  /resume <id>       - Resume a session
  /git [status|diff] - Show git status/diff
  /clear             - Clear conversation history
  quit               - Exit session
""")
    
    elif cmd == "/stats":
        stats = engine.get_stats()
        print(f"\n{Colors.CYAN}Memory:{Colors.RESET} {stats['memory']['total']} entries")
        print(f"{Colors.CYAN}Patterns:{Colors.RESET} {stats['patterns']['total_patterns']} learned")
        print(f"{Colors.CYAN}Tools:{Colors.RESET} {', '.join(stats['tools'])}")
        print(f"{Colors.CYAN}Git Repo:{Colors.RESET} {'Yes' if stats['git_repo'] else 'No'}")
        print(f"{Colors.CYAN}Session:{Colors.RESET} {stats['session_id']} ({stats['session_history_count']} messages)\n")
    
    elif cmd == "/history":
        limit = int(args[0]) if args else 10
        history = engine.get_history(limit)
        print(f"\n{Colors.CYAN}Last {len(history)} messages:{Colors.RESET}")
        for msg in history:
            role = "You" if msg["role"] == "user" else "NEURO"
            preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
            print(f"  {Colors.BOLD}{role}:{Colors.RESET} {preview}")
        print()
    
    elif cmd == "/sessions":
        sessions = engine.list_sessions()
        if not sessions:
            print(f"\n{Colors.DIM}No saved sessions.{Colors.RESET}\n")
        else:
            print(f"\n{Colors.CYAN}Saved sessions:{Colors.RESET}")
            for s in sessions[:10]:
                print(f"  {s['session_id']} - {s['message_count']} messages")
            print()
    
    elif cmd == "/resume":
        if not args:
            print(f"\n{Colors.RED}Usage: /resume <session_id>{Colors.RESET}\n")
        else:
            session_id = args[0]
            history = engine._load_session(session_id)
            if history:
                print(f"\n{Colors.GREEN}Resumed session {session_id} ({len(history)} messages){Colors.RESET}\n")
            else:
                print(f"\n{Colors.RED}Session {session_id} not found.{Colors.RESET}\n")
    
    elif cmd == "/git":
        subcmd = args[0] if args else "status"
        if subcmd == "status":
            status = engine.git.status()
            if "error" in status:
                print(f"\n{Colors.RED}{status['error']}{Colors.RESET}\n")
            else:
                print(f"\n{Colors.CYAN}Git Status:{Colors.RESET}")
                print(f"  Branch: {status['branch']}")
                if status['has_changes']:
                    print(f"  Changed files:")
                    for f in status['changed_files'][:10]:
                        print(f"    [{f['status']}] {f['file']}")
                else:
                    print(f"  {Colors.GREEN}Working tree clean{Colors.RESET}")
                if status['ahead'] or status['behind']:
                    print(f"  Ahead: {status['ahead']}, Behind: {status['behind']}")
                print()
        elif subcmd == "diff":
            diff = engine.git.diff()
            if diff:
                print(f"\n{Colors.CYAN}Git Diff:{Colors.RESET}")
                print(diff[:2000])  # Limit output
            else:
                print(f"\n{Colors.DIM}No changes.{Colors.RESET}\n")
        elif subcmd == "log":
            log = engine.git.log(5)
            print(f"\n{Colors.CYAN}Recent commits:{Colors.RESET}")
            for commit in log:
                print(f"  {commit['hash']} - {commit['message'][:50]} ({commit['author']})")
            print()
    
    elif cmd == "/clear":
        engine.clear_history()
        print(f"\n{Colors.GREEN}Conversation history cleared.{Colors.RESET}\n")
    
    else:
        print(f"\n{Colors.RED}Unknown command: {cmd}. Type /help for available commands.{Colors.RESET}\n")


async def one_shot_query(query: str):
    """Run a single query."""
    from neuro.engine_v2 import NeuroEngine
    
    async with NeuroEngine() as engine:
        async for token in engine.stream_chat(query):
            print(token, end="", flush=True)
        print()


def show_stats():
    """Show engine statistics."""
    from neuro.memory import PersistentMemory
    from neuro.patterns import PatternStore
    
    print(f"{Colors.BOLD}NEURO Statistics{Colors.RESET}\n")
    
    # Memory stats
    with PersistentMemory() as memory:
        stats = memory.stats()
        print(f"{Colors.CYAN}Memory:{Colors.RESET}")
        print(f"  Total entries: {stats['total']}")
        for mem_type, count in stats.get('by_type', {}).items():
            print(f"  - {mem_type}: {count}")
        print(f"  Avg importance: {stats['avg_importance']:.2f}\n")
    
    # Pattern stats
    with PatternStore() as patterns:
        stats = patterns.get_stats()
        print(f"{Colors.CYAN}Patterns:{Colors.RESET}")
        print(f"  Total patterns: {stats['total_patterns']}")
        print(f"  Avg success rate: {stats['avg_success_rate']:.0%}")
        print(f"  Total interactions: {stats['total_interactions']}\n")
        
        if stats.get('by_type'):
            print(f"{Colors.CYAN}By type:{Colors.RESET}")
            for ptype, data in list(stats['by_type'].items())[:5]:
                print(f"  - {ptype}: {data['count']} ({data['success_rate']:.0%} success)")


def doctor_check():
    """Run health checks."""
    import shutil
    import sys
    
    print(f"{Colors.BOLD}NEURO Health Check{Colors.RESET}\n")
    
    checks = []
    all_passed = True
    
    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 9)
    checks.append(("Python version", py_version, py_ok))
    if not py_ok:
        all_passed = False
    
    # Check Ollama
    ollama_ok = shutil.which("ollama") is not None
    checks.append(("Ollama installed", "Yes" if ollama_ok else "No", ollama_ok))
    if not ollama_ok:
        all_passed = False
    
    # Check ripgrep
    rg_ok = shutil.which("rg") is not None
    checks.append(("Ripgrep installed", "Yes" if rg_ok else "No", rg_ok))
    # ripgrep is optional
    
    # Check git
    git_ok = shutil.which("git") is not None
    checks.append(("Git installed", "Yes" if git_ok else "No", git_ok))
    if not git_ok:
        all_passed = False
    
    # Check numpy
    try:
        import numpy
        numpy_ok = True
        numpy_version = numpy.__version__
    except ImportError:
        numpy_ok = False
        numpy_version = "Not installed"
    checks.append(("NumPy", numpy_version, numpy_ok))
    if not numpy_ok:
        all_passed = False
    
    # Check scipy
    try:
        import scipy
        scipy_ok = True
        scipy_version = scipy.__version__
    except ImportError:
        scipy_ok = False
        scipy_version = "Not installed"
    checks.append(("SciPy", scipy_version, scipy_ok))
    if not scipy_ok:
        all_passed = False
    
    # Check aiohttp
    try:
        import aiohttp
        aiohttp_ok = True
    except ImportError:
        aiohttp_ok = False
    checks.append(("aiohttp", "Installed" if aiohttp_ok else "Not installed", aiohttp_ok))
    if not aiohttp_ok:
        all_passed = False
    
    # Check neuro package
    try:
        from neuro.engine_v2 import NeuroEngine
        neuro_ok = True
    except ImportError as e:
        neuro_ok = False
        checks.append(("NEURO engine", str(e), neuro_ok))
        all_passed = False
    
    # Check memory directory
    from pathlib import Path
    memory_dir = Path("~/.neuro").expanduser()
    memory_ok = memory_dir.exists() or memory_dir.mkdir(parents=True, exist_ok=True)
    checks.append(("Memory directory", str(memory_dir), memory_ok))
    
    # Print results
    for name, value, passed in checks:
        status = f"{Colors.GREEN}✓{Colors.RESET}" if passed else f"{Colors.RED}✗{Colors.RESET}"
        print(f"  {status} {name}: {value}")
    
    print()
    if all_passed:
        print(f"{Colors.GREEN}All checks passed!{Colors.RESET}\n")
    else:
        print(f"{Colors.RED}Some checks failed. Please install missing dependencies.{Colors.RESET}\n")
        print(f"Run: pip install -e '.[all]'{Colors.RESET}\n")
    
    return all_passed


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NEURO - Neuroscience-inspired AGI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Query to process (interactive mode if omitted)",
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"neuro {__version__}",
    )
    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="Show engine statistics",
    )
    parser.add_argument(
        "--doctor", "-d",
        action="store_true",
        help="Run health checks",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    try:
        if args.doctor:
            doctor_check()
        elif args.stats:
            show_stats()
        elif args.query:
            asyncio.run(one_shot_query(args.query))
        else:
            asyncio.run(interactive_session())
    except KeyboardInterrupt:
        print(f"\n{Colors.DIM}Interrupted.{Colors.RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"{Colors.RED}Error:{Colors.RESET} {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
