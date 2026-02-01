"""
NEURO CLI - Claude Code-style terminal interface.

A modular, extensible CLI with:
- Streaming responses
- Permission system
- Hooks lifecycle
- MCP integration
- Subagent support
"""

from .main import main
from .app import NeuroApp

__version__ = "3.0.0"
__all__ = ["main", "NeuroApp", "__version__"]
