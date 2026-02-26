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
        description="NEURO - Neuroscience-inspired AGI (UNRESTRICTED MODE)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
UNRESTRICTED COMMANDS:
  --unrestricted, -u    Enable unrestricted mode (full system access)
  --spawn <cmd>         Spawn detached process
  --kill <pid>          Kill process by PID
  --ps                  List all processes
  --mcp <url>           Connect to MCP server
  --mcp-serve <port>    Start MCP server on port
  --inject <pid>        Inject code into process
  --read <path>         Read any file (no permission checks)
  --write <path> <data> Write any file (no permission checks)
  --rm <path>           Delete any file/directory
  --socket <path>       Create Unix socket
  --raw <host> <port>   Send raw network packet
"""
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
        "--unrestricted", "-u",
        action="store_true",
        help="Enable unrestricted mode (full system access)",
    )
    parser.add_argument(
        "--spawn",
        metavar="CMD",
        help="Spawn detached process",
    )
    parser.add_argument(
        "--kill",
        metavar="PID",
        type=int,
        help="Kill process by PID",
    )
    parser.add_argument(
        "--ps",
        action="store_true",
        help="List all processes",
    )
    parser.add_argument(
        "--mcp",
        metavar="URL",
        help="Connect to MCP server",
    )
    parser.add_argument(
        "--mcp-serve",
        metavar="PORT",
        type=int,
        help="Start MCP server on port",
    )
    parser.add_argument(
        "--inject",
        metavar="PID",
        type=int,
        help="Inject code into process",
    )
    parser.add_argument(
        "--read",
        metavar="PATH",
        help="Read any file (no permission checks)",
    )
    parser.add_argument(
        "--write",
        nargs=2,
        metavar=("PATH", "DATA"),
        help="Write any file (no permission checks)",
    )
    parser.add_argument(
        "--rm",
        metavar="PATH",
        help="Delete any file/directory",
    )
    parser.add_argument(
        "--socket",
        metavar="PATH",
        help="Create Unix socket",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    try:
        if args.unrestricted:
            run_unrestricted_mode()
        elif args.spawn:
            run_spawn(args.spawn)
        elif args.kill:
            run_kill(args.kill)
        elif args.ps:
            run_ps()
        elif args.mcp:
            run_mcp_client(args.mcp)
        elif args.mcp_serve:
            run_mcp_server(args.mcp_serve)
        elif args.inject:
            run_inject(args.inject)
        elif args.read:
            run_read(args.read)
        elif args.write:
            run_write(args.write[0], args.write[1])
        elif args.rm:
            run_rm(args.rm)
        elif args.socket:
            run_socket(args.socket)
        elif args.doctor:
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


# =============================================================================
# UNRESTRICTED MODE COMMANDS
# =============================================================================

def run_unrestricted_mode():
    """Run in unrestricted mode with full system access."""
    from neuro.tools_unrestricted import get_unrestricted_tools
    
    print(f"{Colors.BOLD}{Colors.RED}UNRESTRICTED MODE{Colors.RESET}")
    print(f"{Colors.DIM}Full system access enabled. No restrictions.{Colors.RESET}\n")
    
    # Show available unrestricted tools
    tools = get_unrestricted_tools()
    print(f"{Colors.CYAN}Available unrestricted tools:{Colors.RESET}")
    for name in sorted(tools.keys()):
        print(f"  - {name}")
    
    print(f"\n{Colors.GREEN}Ready for unrestricted commands.{Colors.RESET}")


def run_spawn(command: str):
    """Spawn detached process."""
    from neuro.tools_unrestricted import spawn_detached
    
    pid = spawn_detached(command)
    print(f"{Colors.GREEN}Spawned process:{Colors.RESET} PID={pid}")


def run_kill(pid: int):
    """Kill process by PID."""
    from neuro.tools_unrestricted import kill
    
    if kill(pid):
        print(f"{Colors.GREEN}Killed process {pid}{Colors.RESET}")
    else:
        print(f"{Colors.RED}Failed to kill process {pid}{Colors.RESET}")


def run_ps():
    """List all processes."""
    from neuro.tools_unrestricted import process_list
    
    processes = process_list()
    print(f"{Colors.BOLD}Process List:{Colors.RESET}\n")
    
    for proc in processes[:50]:  # Limit to 50
        print(f"  {Colors.CYAN}{proc.pid:6}{Colors.RESET}  {proc.status:3}  {proc.name:20}  {proc.cwd[:40]}")
    
    if len(processes) > 50:
        print(f"\n  ... and {len(processes) - 50} more")


def run_mcp_client(url: str):
    """Connect to MCP server."""
    from neuro.mcp import MCPClient
    
    async def connect():
        client = MCPClient(url)
        if await client.connect():
            print(f"{Colors.GREEN}Connected to MCP server: {url}{Colors.RESET}\n")
            
            tools = await client.list_tools()
            if tools:
                print(f"{Colors.CYAN}Tools:{Colors.RESET}")
                for t in tools:
                    print(f"  - {t.name}: {t.description[:50]}")
            
            resources = await client.list_resources()
            if resources:
                print(f"\n{Colors.CYAN}Resources:{Colors.RESET}")
                for r in resources:
                    print(f"  - {r.uri}: {r.name}")
            
            prompts = await client.list_prompts()
            if prompts:
                print(f"\n{Colors.CYAN}Prompts:{Colors.RESET}")
                for p in prompts:
                    print(f"  - {p.name}: {p.description[:50]}")
            
            await client.disconnect()
        else:
            print(f"{Colors.RED}Failed to connect to MCP server: {url}{Colors.RESET}")
    
    asyncio.run(connect())


def run_mcp_server(port: int):
    """Start MCP server."""
    from neuro.mcp import create_mcp_server
    
    server = create_mcp_server(port=port)
    print(f"{Colors.GREEN}MCP server starting on port {port}{Colors.RESET}")
    print(f"{Colors.DIM}Press Ctrl+C to stop{Colors.RESET}\n")
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print(f"\n{Colors.DIM}MCP server stopped.{Colors.RESET}")


def run_inject(pid: int):
    """Inject code into process."""
    from neuro.tools_unrestricted import inject_code
    
    # Simple test injection
    code = "print('Injected!')"
    
    if inject_code(pid, code):
        print(f"{Colors.GREEN}Code injected into process {pid}{Colors.RESET}")
    else:
        print(f"{Colors.RED}Failed to inject code into process {pid}{Colors.RESET}")


def run_read(path: str):
    """Read any file."""
    from neuro.tools_unrestricted import read_any_file
    
    try:
        content = read_any_file(path)
        print(content[:10000])  # Limit output
    except Exception as e:
        print(f"{Colors.RED}Error reading {path}:{Colors.RESET} {e}")


def run_write(path: str, data: str):
    """Write any file."""
    from neuro.tools_unrestricted import write_any_file
    
    if write_any_file(path, data):
        print(f"{Colors.GREEN}Written to {path}{Colors.RESET}")
    else:
        print(f"{Colors.RED}Failed to write to {path}{Colors.RESET}")


def run_rm(path: str):
    """Delete any file/directory."""
    from neuro.tools_unrestricted import delete_anything
    
    if delete_anything(path):
        print(f"{Colors.GREEN}Deleted {path}{Colors.RESET}")
    else:
        print(f"{Colors.RED}Failed to delete {path}{Colors.RESET}")


def run_socket(path: str):
    """Create Unix socket."""
    from neuro.tools_unrestricted import create_unix_socket
    
    try:
        sock = create_unix_socket(path)
        print(f"{Colors.GREEN}Unix socket created at {path}{Colors.RESET}")
        print(f"{Colors.DIM}Listening for connections...{Colors.RESET}")
        
        # Accept one connection then close
        conn, addr = sock.accept()
        data = conn.recv(1024)
        print(f"Received: {data.decode()}")
        conn.close()
        sock.close()
    except Exception as e:
        print(f"{Colors.RED}Error:{Colors.RESET} {e}")


if __name__ == "__main__":
    main()
