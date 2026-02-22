"""
MCP Manager - Model Context Protocol server integration.

Connects to MCP servers for extended capabilities like:
- Memory persistence
- File system access
- Database connections
- Custom tools
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import json
import os
import asyncio
import subprocess


class MCPTransport(Enum):
    """MCP transport types."""
    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    transport: MCPTransport = MCPTransport.STDIO


@dataclass
class MCPTool:
    """A tool exposed by an MCP server."""
    name: str
    server: str
    original_name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResource:
    """A resource exposed by an MCP server."""
    uri: str
    name: str
    server: str
    description: str = ""
    mime_type: str = "text/plain"


class MCPManager:
    """
    Manages MCP server connections.

    Features:
    - Connect to MCP servers via stdio
    - Tool discovery and registration
    - Resource access
    - Request/response handling
    """

    def __init__(
        self,
        project_dir: str = ".",
    ):
        self.project_dir = os.path.abspath(project_dir)

        self._servers: Dict[str, MCPServerConfig] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._request_id = 0

        self._load_config()

    def _load_config(self):
        """Load MCP configuration from files."""
        config_paths = [
            os.path.expanduser("~/.neuro/mcp.json"),
            os.path.join(self.project_dir, ".neuro", "mcp.json"),
            os.path.join(self.project_dir, ".mcp.json"),
        ]

        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)

                    for name, config in data.get("mcpServers", {}).items():
                        self._servers[name] = MCPServerConfig(
                            name=name,
                            command=config.get("command", ""),
                            args=config.get("args", []),
                            env=config.get("env", {}),
                            transport=MCPTransport(config.get("transport", "stdio")),
                        )
                except Exception as e:
                    print(f"Error loading MCP config from {path}: {e}")

    async def connect_all(self) -> Dict[str, bool]:
        """Connect to all configured MCP servers."""
        results = {}
        for name in self._servers:
            results[name] = await self.connect(name)
        return results

    async def connect(self, server_name: str) -> bool:
        """Connect to an MCP server."""
        if server_name not in self._servers:
            return False

        config = self._servers[server_name]

        if config.transport == MCPTransport.STDIO:
            return await self._connect_stdio(config)

        return False

    async def _connect_stdio(self, config: MCPServerConfig) -> bool:
        """Connect via stdio transport."""
        try:
            env = {**os.environ, **config.env}

            process = subprocess.Popen(
                [config.command] + config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=self.project_dir,
            )

            self._processes[config.name] = process

            # Initialize
            response = await self._send_request(
                config.name,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "neuro",
                        "version": "0.9.0"
                    }
                }
            )

            if not response:
                return False

            # Send initialized notification
            await self._send_notification(config.name, "notifications/initialized", {})

            # Discover tools
            await self._discover_tools(config.name)

            # Discover resources
            await self._discover_resources(config.name)

            return True

        except Exception as e:
            print(f"Failed to connect to MCP server {config.name}: {e}")
            return False

    async def _discover_tools(self, server_name: str):
        """Discover tools from an MCP server."""
        response = await self._send_request(
            server_name,
            "tools/list",
            {}
        )

        if response and "tools" in response:
            for tool_data in response["tools"]:
                tool_name = f"mcp__{server_name}__{tool_data['name']}"
                self._tools[tool_name] = MCPTool(
                    name=tool_name,
                    server=server_name,
                    original_name=tool_data["name"],
                    description=tool_data.get("description", ""),
                    input_schema=tool_data.get("inputSchema", {}),
                )

    async def _discover_resources(self, server_name: str):
        """Discover resources from an MCP server."""
        response = await self._send_request(
            server_name,
            "resources/list",
            {}
        )

        if response and "resources" in response:
            for res_data in response["resources"]:
                self._resources[res_data["uri"]] = MCPResource(
                    uri=res_data["uri"],
                    name=res_data.get("name", res_data["uri"]),
                    server=server_name,
                    description=res_data.get("description", ""),
                    mime_type=res_data.get("mimeType", "text/plain"),
                )

    async def _send_request(
        self,
        server_name: str,
        method: str,
        params: Dict[str, Any],
    ) -> Optional[Dict]:
        """Send a JSON-RPC request to an MCP server."""
        if server_name not in self._processes:
            return None

        process = self._processes[server_name]

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        try:
            request_json = json.dumps(request) + "\n"
            process.stdin.write(request_json.encode())
            process.stdin.flush()

            response_line = process.stdout.readline()
            if not response_line:
                return None

            response = json.loads(response_line.decode())

            if "error" in response:
                print(f"MCP error: {response['error']}")
                return None

            return response.get("result")

        except Exception as e:
            print(f"MCP request failed: {e}")
            return None

    async def _send_notification(
        self,
        server_name: str,
        method: str,
        params: Dict[str, Any],
    ):
        """Send a JSON-RPC notification (no response expected)."""
        if server_name not in self._processes:
            return

        process = self._processes[server_name]

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            notification_json = json.dumps(notification) + "\n"
            process.stdin.write(notification_json.encode())
            process.stdin.flush()
        except Exception:
            pass

    def get_tools(self) -> List[MCPTool]:
        """Get all discovered MCP tools."""
        return list(self._tools.values())

    def get_tool(self, name: str) -> Optional[MCPTool]:
        """Get a specific MCP tool."""
        return self._tools.get(name)

    def get_resources(self) -> List[MCPResource]:
        """Get all discovered resources."""
        return list(self._resources.values())

    async def call_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> Any:
        """Call an MCP tool."""
        tool = self._tools.get(tool_name)
        if not tool:
            raise ValueError(f"MCP tool not found: {tool_name}")

        response = await self._send_request(
            tool.server,
            "tools/call",
            {
                "name": tool.original_name,
                "arguments": args,
            }
        )

        if response and "content" in response:
            # Return text content
            for item in response["content"]:
                if item.get("type") == "text":
                    return item.get("text", "")
            return response["content"]

        return response

    async def read_resource(self, uri: str) -> Optional[str]:
        """Read a resource by URI."""
        resource = self._resources.get(uri)
        if not resource:
            return None

        response = await self._send_request(
            resource.server,
            "resources/read",
            {"uri": uri}
        )

        if response and "contents" in response:
            for content in response["contents"]:
                if "text" in content:
                    return content["text"]

        return None

    async def disconnect_all(self):
        """Disconnect all MCP servers."""
        for name, process in self._processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                process.kill()

        self._processes.clear()
        self._tools.clear()
        self._resources.clear()

    def get_servers(self) -> List[str]:
        """Get list of configured server names."""
        return list(self._servers.keys())

    def is_connected(self, server_name: str) -> bool:
        """Check if a server is connected."""
        return server_name in self._processes
