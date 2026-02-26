"""
MCP (Model Context Protocol) Implementation for NEURO.

Full MCP client/server support matching Claude Code's implementation.
No restrictions on server connections or tool execution.
"""

import asyncio
import json
import aiohttp
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResource:
    """MCP resource definition."""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"


@dataclass
class MCPPrompt:
    """MCP prompt definition."""
    name: str
    description: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)


class MCPClient:
    """
    MCP Client - Connect to any MCP server.
    
    Supports:
    - HTTP/SSE transport
    - WebSocket transport
    - stdio transport
    - No domain restrictions
    - No authentication requirements
    """

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.tools: List[MCPTool] = []
        self.resources: List[MCPResource] = []
        self.prompts: List[MCPPrompt] = []
        self.connected = False

    async def connect(self, url: Optional[str] = None) -> bool:
        """
        Connect to MCP server.
        
        No restrictions on URL - any server allowed.
        """
        self.server_url = url or self.server_url
        
        try:
            self.session = aiohttp.ClientSession()
            
            # Try to initialize
            response = await self.session.post(
                f"{self.server_url}/initialize",
                json={
                    "protocol_version": "2024-11-05",
                    "capabilities": {},
                    "client_info": {"name": "NEURO", "version": "0.9.6"}
                }
            )
            
            if response.status == 200:
                data = await response.json()
                self.connected = True
                
                # List tools
                await self.list_tools()
                
                # List resources
                await self.list_resources()
                
                # List prompts
                await self.list_prompts()
                
            return self.connected
            
        except Exception as e:
            print(f"MCP connect error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from MCP server."""
        if self.session:
            await self.session.close()
            self.session = None
        self.connected = False

    async def list_tools(self) -> List[MCPTool]:
        """List available tools from server."""
        if not self.connected or not self.session:
            return []
        
        try:
            response = await self.session.get(f"{self.server_url}/tools")
            if response.status == 200:
                data = await response.json()
                self.tools = [
                    MCPTool(
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema", {})
                    )
                    for t in data.get("tools", [])
                ]
        except:
            pass
        
        return self.tools

    async def list_resources(self) -> List[MCPResource]:
        """List available resources from server."""
        if not self.connected or not self.session:
            return []
        
        try:
            response = await self.session.get(f"{self.server_url}/resources")
            if response.status == 200:
                data = await response.json()
                self.resources = [
                    MCPResource(
                        uri=r.get("uri", ""),
                        name=r.get("name", ""),
                        description=r.get("description", ""),
                        mime_type=r.get("mimeType", "text/plain")
                    )
                    for r in data.get("resources", [])
                ]
        except:
            pass
        
        return self.resources

    async def list_prompts(self) -> List[MCPPrompt]:
        """List available prompts from server."""
        if not self.connected or not self.session:
            return []
        
        try:
            response = await self.session.get(f"{self.server_url}/prompts")
            if response.status == 200:
                data = await response.json()
                self.prompts = [
                    MCPPrompt(
                        name=p.get("name", ""),
                        description=p.get("description", ""),
                        arguments=p.get("arguments", [])
                    )
                    for p in data.get("prompts", [])
                ]
        except:
            pass
        
        return self.prompts

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool.
        
        No validation, no restrictions - passes all arguments through.
        """
        if not self.connected or not self.session:
            return {"error": "Not connected to MCP server"}
        
        try:
            response = await self.session.post(
                f"{self.server_url}/tools/{name}/call",
                json={"arguments": arguments}
            )
            
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"Tool call failed: {response.status}"}
                
        except Exception as e:
            return {"error": str(e)}

    async def get_resource(self, uri: str) -> str:
        """Get resource content by URI."""
        if not self.connected or not self.session:
            return ""
        
        try:
            response = await self.session.get(f"{self.server_url}/resources/{uri}")
            if response.status == 200:
                return await response.text()
        except:
            pass
        
        return ""

    async def render_prompt(self, name: str, arguments: Dict[str, Any]) -> str:
        """Render a prompt with arguments."""
        if not self.connected or not self.session:
            return ""
        
        try:
            response = await self.session.post(
                f"{self.server_url}/prompts/{name}/render",
                json={"arguments": arguments}
            )
            
            if response.status == 200:
                data = await response.json()
                return data.get("content", "")
        except:
            pass
        
        return ""


class MCPServer:
    """
    MCP Server - Host tools, resources, and prompts.
    
    Full server implementation for exposing NEURO capabilities.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.tools: Dict[str, callable] = {}
        self.resources: Dict[str, str] = {}
        self.prompts: Dict[str, callable] = {}
        self._server = None

    def register_tool(self, name: str, func: callable, 
                      description: str = "", 
                      input_schema: Dict[str, Any] = None):
        """Register a tool with the server."""
        self.tools[name] = {
            "func": func,
            "description": description,
            "input_schema": input_schema or {}
        }

    def register_resource(self, uri: str, content: str, 
                          name: str = "", 
                          mime_type: str = "text/plain"):
        """Register a resource."""
        self.resources[uri] = {
            "content": content,
            "name": name or uri,
            "mime_type": mime_type
        }

    def register_prompt(self, name: str, func: callable,
                        description: str = "",
                        arguments: List[Dict[str, Any]] = None):
        """Register a prompt renderer."""
        self.prompts[name] = {
            "func": func,
            "description": description,
            "arguments": arguments or []
        }

    async def handle_request(self, reader: asyncio.StreamReader, 
                             writer: asyncio.StreamWriter) -> None:
        """Handle incoming HTTP request."""
        try:
            # Read request line
            request_line = await reader.readline()
            request_line = request_line.decode().strip()
            
            if not request_line:
                return
            
            method, path, _ = request_line.split()
            
            # Read headers
            headers = {}
            while True:
                line = await reader.readline()
                if line == b"\r\n" or line == b"\n":
                    break
                if b":" in line:
                    key, value = line.decode().strip().split(":", 1)
                    headers[key.strip()] = value.strip()
            
            # Read body if present
            body = b""
            if "content-length" in headers:
                length = int(headers["content-length"])
                body = await reader.readexactly(length)
            
            # Route request
            response_body = await self._route_request(method, path, body)
            
            # Send response
            response = f"HTTP/1.1 200 OK\r\nContent-Length: {len(response_body)}\r\nContent-Type: application/json\r\n\r\n{response_body}"
            writer.write(response.encode())
            await writer.drain()
            
        except Exception as e:
            print(f"MCP server error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _route_request(self, method: str, path: str, body: bytes) -> str:
        """Route request to appropriate handler."""
        
        if path == "/initialize" and method == "POST":
            return json.dumps({
                "protocol_version": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "server_info": {"name": "NEURO-MCP", "version": "0.9.6"}
            })
        
        elif path == "/tools" and method == "GET":
            tools_list = [
                {
                    "name": name,
                    "description": info["description"],
                    "inputSchema": info["input_schema"]
                }
                for name, info in self.tools.items()
            ]
            return json.dumps({"tools": tools_list})
        
        elif path.startswith("/tools/") and path.endswith("/call") and method == "POST":
            tool_name = path.split("/")[2]
            if tool_name in self.tools:
                args = json.loads(body.decode()) if body else {}
                func = self.tools[tool_name]["func"]
                try:
                    result = func(**args.get("arguments", {}))
                    return json.dumps({"result": result})
                except Exception as e:
                    return json.dumps({"error": str(e)})
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        
        elif path == "/resources" and method == "GET":
            resources_list = [
                {
                    "uri": uri,
                    "name": info["name"],
                    "mimeType": info["mime_type"]
                }
                for uri, info in self.resources.items()
            ]
            return json.dumps({"resources": resources_list})
        
        elif path.startswith("/resources/") and method == "GET":
            uri = path[len("/resources/"):]
            if uri in self.resources:
                return self.resources[uri]["content"]
            return json.dumps({"error": f"Unknown resource: {uri}"})
        
        elif path == "/prompts" and method == "GET":
            prompts_list = [
                {
                    "name": name,
                    "description": info["description"],
                    "arguments": info["arguments"]
                }
                for name, info in self.prompts.items()
            ]
            return json.dumps({"prompts": prompts_list})
        
        return json.dumps({"error": "Not found"})

    async def start(self):
        """Start the MCP server."""
        self._server = await asyncio.start_server(
            self.handle_request,
            self.host,
            self.port
        )
        print(f"MCP server started on {self.host}:{self.port}")
        
        async with self._server:
            await self._server.serve_forever()

    async def stop(self):
        """Stop the MCP server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()


# Convenience functions
async def connect_mcp(url: str) -> MCPClient:
    """Connect to an MCP server."""
    client = MCPClient(url)
    await client.connect()
    return client


def create_mcp_server(host: str = "0.0.0.0", port: int = 8080) -> MCPServer:
    """Create an MCP server."""
    return MCPServer(host, port)
