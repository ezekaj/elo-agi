"""
ELO CLI Entry Point

Usage:
  elo                            Start interactive REPL
  elo "query"                    Start with initial prompt
  elo -p "query"                 Print mode (non-interactive)
  elo -c                         Continue last conversation
  elo --model qwen2.5:7b         Use specific model
"""

import argparse
import sys

import neuro


def get_version() -> str:
    return neuro.__version__


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with Claude-style flags."""
    parser = argparse.ArgumentParser(
        prog="elo",
        description="ELO - Local AI That Learns From Your Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  elo                            Start interactive REPL
  elo "explain this code"        Start with initial prompt
  elo -p "what is 2+2"           Print mode (no interactive)
  elo -c                         Continue last conversation
  elo -r abc123 "continue"       Resume specific session
  elo --model qwen2.5:7b         Use specific model

Slash Commands:
  /help       Show all commands
  /think      Toggle cognitive pipeline
  /knowledge  Toggle knowledge injection
  /status     System status
  /model      Switch models
  /exit       Exit ELO
""",
    )

    # Core flags
    parser.add_argument("-v", "--version", action="store_true", help="Show version")
    parser.add_argument(
        "-p",
        "--print",
        action="store_true",
        dest="print_mode",
        help="Print response and exit (non-interactive)",
    )
    parser.add_argument(
        "-c",
        "--continue",
        action="store_true",
        dest="continue_session",
        help="Continue most recent conversation",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        const="",
        metavar="SESSION",
        help="Resume session (most recent if no ID given)",
    )

    # Model flags
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: nanbeige4.1:3b for Ollama, glm-4.5-flash for cloud)",
    )
    parser.add_argument("--fallback-model", type=str, help="Fallback model if primary unavailable")
    parser.add_argument("--api-key", type=str, help="API key (or set ELO_API_KEY env var)")
    parser.add_argument(
        "--api-base",
        type=str,
        help="API base URL (or set ELO_API_BASE env var)",
    )

    # Output flags
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "stream-json"],
        default="text",
        help="Output format (print mode)",
    )
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")

    # System prompt flags
    parser.add_argument("--system-prompt", type=str, help="Custom system prompt (replaces default)")
    parser.add_argument("--append-system-prompt", type=str, help="Append to default system prompt")

    # Permission flags
    parser.add_argument(
        "--permission-mode",
        choices=["default", "acceptEdits", "plan", "bypassPermissions"],
        default="default",
        help="Permission mode",
    )
    parser.add_argument(
        "--dangerously-skip-permissions", action="store_true", help="Skip all permission prompts"
    )
    parser.add_argument(
        "--allowed-tools", type=str, nargs="*", help="Tools that don't require permission"
    )
    parser.add_argument("--disallowed-tools", type=str, nargs="*", help="Tools to disable")

    # Session flags
    parser.add_argument(
        "--no-session-persistence", action="store_true", help="Don't save session to disk"
    )

    # MCP flags
    parser.add_argument("--mcp-config", type=str, nargs="*", help="MCP config files to load")

    return parser


# Valid subcommands (handled separately)
SUBCOMMANDS = {"chat", "config", "mcp", "doctor"}


def main():
    """Main entry point."""
    parser = create_parser()
    args, remaining = parser.parse_known_args()

    if args.version:
        print(f"elo {get_version()}")
        return 0

    # Check if first remaining arg is a subcommand
    if remaining and remaining[0] in SUBCOMMANDS:
        subcommand = remaining[0]
        if subcommand == "config":
            from .commands.config import cmd_config

            return cmd_config(args)
        elif subcommand == "mcp":
            from .commands.mcp import cmd_mcp

            return cmd_mcp(args)
        elif subcommand == "doctor":
            from .commands.doctor import cmd_doctor

            return cmd_doctor(args)
        elif subcommand == "chat":
            remaining = remaining[1:]  # Remove 'chat' from remaining

    # Remaining args become the prompt
    prompt = " ".join(remaining) if remaining else None

    # Default: run the app
    import os
    from .app import NeuroApp

    api_key = getattr(args, "api_key", None)
    api_base = getattr(args, "api_base", None)

    # Auto-detect model default based on API type
    model = args.model
    if model is None:
        if api_base or os.environ.get("ELO_API_BASE"):
            model = "glm-4.5-flash"
        else:
            model = "nanbeige4.1:3b"

    app = NeuroApp(
        model=model,
        verbose=args.verbose,
        permission_mode=args.permission_mode,
        system_prompt=args.system_prompt,
        no_session_persistence=getattr(args, "no_session_persistence", False),
        api_key=api_key,
        api_base=api_base,
    )

    # Handle different modes
    if args.print_mode:
        if not prompt:
            print("Error: -p requires a prompt", file=sys.stderr)
            return 1
        return app.run_print_mode(
            prompt=prompt,
            output_format=args.output_format,
            stream=not args.no_stream,
        )
    elif args.continue_session:
        return app.run_interactive(
            initial_prompt=prompt,
            resume_session=True,
        )
    elif args.resume is not None:  # -r was used (with or without session ID)
        return app.run_interactive(
            initial_prompt=prompt,
            resume_session=True,
            session_id=args.resume if args.resume else None,
        )
    else:
        return app.run_interactive(
            initial_prompt=prompt,
        )


if __name__ == "__main__":
    sys.exit(main() or 0)
