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
                
                if user_input == "/stats":
                    stats = engine.get_stats()
                    print(f"\n{Colors.CYAN}Memory:{Colors.RESET} {stats['memory']['total']} entries")
                    print(f"{Colors.CYAN}Patterns:{Colors.RESET} {stats['patterns']['total_patterns']} learned")
                    print(f"{Colors.CYAN}Tools:{Colors.RESET} {', '.join(stats['tools'])}\n")
                    continue
                
                if user_input == "/help":
                    print(f"""
{Colors.BOLD}Commands:{Colors.RESET}
  /stats   - Show engine statistics
  /help    - Show this help
  quit     - Exit session
""")
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
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    args = parser.parse_args()
    
    try:
        if args.stats:
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
