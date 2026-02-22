"""MCP command."""


def cmd_mcp(args):
    """Manage MCP servers."""
    print("\n  MCP Server Management")
    print(f"  {'─' * 50}")
    print("  MCP integration coming soon.")
    print("\n  To configure MCP servers, create ~/.neuro/mcp.json:")
    print(
        '  {"mcpServers": {"memory": {"command": "npx", "args": ["-y", "@anthropic-ai/mcp-server-memory"]}}}'
    )
    print()
    return 0
