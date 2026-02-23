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
import asyncio
import json
import logging
import os


logger = logging.getLogger(__name__)


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
    - Connect to MCP servers via stdio using async subprocesses
    - Tool discovery and registration
    - Resource access
    - Request/response correlation with futures
    - Background read loops for each server
    - Automatic reconnection with exponential backoff
    """

    def __init__(
        self,
        project_dir: str = ".",
    ):
        self.project_dir = os.path.abspath(project_dir)

        self._servers: Dict[str, MCPServerConfig] = {}
        self._processes: Dict[str, asyncio.subprocess.Process] = {}
        self._readers: Dict[str, asyncio.Task] = {}
        self._pending: Dict[int, asyncio.Future] = {}
        self._tools: Dict[str, MCPTool] = {}
        self._resources: Dict[str, MCPResource] = {}
        self._request_id: int = 0

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
                    logger.error(f"Error loading MCP config from {path}: {e}")

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
        """Connect via stdio transport using async subprocess."""
        try:
            env = {**os.environ, **config.env}

            process = await asyncio.create_subprocess_exec(
                config.command,
                *config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.project_dir,
            )

            self._processes[config.name] = process

            reader_task = asyncio.create_task(self._read_loop(config.name))
            self._readers[config.name] = reader_task

            response = await self._send_request(
                config.name,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "elo", "version": "0.9.6"},
                },
            )

            if not response:
                await self._cleanup_server(config.name)
                return False

            await self._send_notification(config.name, "notifications/initialized", {})

            await self._discover_tools(config.name)

            await self._discover_resources(config.name)

            return True

        except Exception as e:
            logger.error(f"Failed to connect to MCP server {config.name}: {e}")
            await self._cleanup_server(config.name)
            return False

    async def _read_loop(self, server_name: str):
        """Background task that reads JSON-RPC responses from a server's stdout."""
        process = self._processes.get(server_name)
        if not process or not process.stdout:
            return

        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                try:
                    message = json.loads(line.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                if "id" in message:
                    req_id = message["id"]
                    future = self._pending.pop(req_id, None)
                    if future and not future.done():
                        if "error" in message:
                            future.set_result(None)
                            logger.error(f"MCP error from {server_name}: {message['error']}")
                        else:
                            future.set_result(message.get("result"))
                else:
                    method = message.get("method", "")
                    if method == "notifications/tools/list_changed":
                        asyncio.create_task(self._discover_tools(server_name))
                    elif method == "notifications/resources/list_changed":
                        asyncio.create_task(self._discover_resources(server_name))

        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error(f"Read loop error for {server_name}: {e}")

        if server_name in self._servers:
            asyncio.create_task(self._reconnect(server_name))

    async def _send_request(
        self,
        server_name: str,
        method: str,
        params: Dict[str, Any],
        timeout: float = 30,
    ) -> Optional[Dict]:
        """Send a JSON-RPC request and await the correlated response."""
        if server_name not in self._processes:
            return None

        process = self._processes[server_name]
        if not process.stdin:
            return None

        self._request_id += 1
        req_id = self._request_id

        request = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[req_id] = future

        try:
            request_json = json.dumps(request) + "\n"
            process.stdin.write(request_json.encode())
            await process.stdin.drain()

            result = await asyncio.wait_for(future, timeout=timeout)
            return result

        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            if not future.done():
                future.cancel()
            logger.error(f"MCP request timed out: {method} on {server_name}")
            return None
        except Exception as e:
            self._pending.pop(req_id, None)
            if not future.done():
                future.cancel()
            logger.error(f"MCP request failed: {e}")
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
        if not process.stdin:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            notification_json = json.dumps(notification) + "\n"
            process.stdin.write(notification_json.encode())
            await process.stdin.drain()
        except Exception:
            pass

    async def _discover_tools(self, server_name: str):
        """Discover tools from an MCP server."""
        self._tools = {k: v for k, v in self._tools.items() if v.server != server_name}

        response = await self._send_request(server_name, "tools/list", {})

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
        self._resources = {k: v for k, v in self._resources.items() if v.server != server_name}

        response = await self._send_request(server_name, "resources/list", {})

        if response and "resources" in response:
            for res_data in response["resources"]:
                self._resources[res_data["uri"]] = MCPResource(
                    uri=res_data["uri"],
                    name=res_data.get("name", res_data["uri"]),
                    server=server_name,
                    description=res_data.get("description", ""),
                    mime_type=res_data.get("mimeType", "text/plain"),
                )

    async def _reconnect(self, server_name: str):
        """Attempt to reconnect to a server with exponential backoff."""
        await self._cleanup_server(server_name)

        if server_name not in self._servers:
            return

        config = self._servers[server_name]
        max_attempts = 3

        for attempt in range(max_attempts):
            delay = 2**attempt
            logger.info(f"Reconnecting to {server_name} in {delay}s (attempt {attempt + 1}/{max_attempts})")
            await asyncio.sleep(delay)

            success = await self._connect_stdio(config)
            if success:
                logger.info(f"Reconnected to {server_name}")
                return

        logger.error(f"Failed to reconnect to {server_name} after {max_attempts} attempts")

    async def _cleanup_server(self, server_name: str):
        """Clean up a single server's process, reader task, and pending futures."""
        reader = self._readers.pop(server_name, None)
        if reader and not reader.done():
            reader.cancel()
            try:
                await reader
            except asyncio.CancelledError:
                pass

        process = self._processes.pop(server_name, None)
        if process:
            try:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            except ProcessLookupError:
                pass

        stale_ids = [rid for rid, fut in self._pending.items() if not fut.done()]
        for rid in stale_ids:
            fut = self._pending.pop(rid, None)
            if fut and not fut.done():
                fut.cancel()

        self._tools = {k: v for k, v in self._tools.items() if v.server != server_name}
        self._resources = {k: v for k, v in self._resources.items() if v.server != server_name}

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
            },
        )

        if response and "content" in response:
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

        response = await self._send_request(resource.server, "resources/read", {"uri": uri})

        if response and "contents" in response:
            for content in response["contents"]:
                if "text" in content:
                    return content["text"]

        return None

    async def disconnect_all(self):
        """Disconnect all MCP servers."""
        server_names = list(self._processes.keys())
        for name in server_names:
            await self._cleanup_server(name)

        self._processes.clear()
        self._readers.clear()
        self._pending.clear()
        self._tools.clear()
        self._resources.clear()

    def get_servers(self) -> List[str]:
        """Get list of configured server names."""
        return list(self._servers.keys())

    def is_connected(self, server_name: str) -> bool:
        """Check if a server is connected."""
        return server_name in self._processes
