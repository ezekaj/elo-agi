"""MCP command."""


def cmd_mcp(args):
    """Manage MCP servers."""
    print(f"\n  MCP Server Management")
    print(f"  {'─' * 50}")
    print(f"  MCP integration coming soon.")
    print(f"\n  To configure MCP servers, create ~/.neuro/mcp.json:")
    print(f'  {{"mcpServers": {{"memory": {{"command": "npx", "args": ["-y", "@anthropic-ai/mcp-server-memory"]}}}}}}')
    print()
    return 0
