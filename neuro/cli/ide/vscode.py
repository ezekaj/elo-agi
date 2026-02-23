"""
VSCode Integration for ELO CLI.

Communicates with VSCode via:
1. `code` CLI command for basic operations
2. Unix socket for bidirectional communication with ELO extension
"""

import asyncio
import json
import os
import subprocess
import tempfile
from typing import Optional, Dict, Any

from .integration import (
    IDEIntegration,
    IDEType,
    EditorContext,
    IDECommand,
)


class VSCodeIntegration(IDEIntegration):
    """
    VSCode integration using CLI and socket communication.

    The socket allows a VSCode extension to:
    - Send editor context updates
    - Receive commands from ELO
    - Sync file changes
    """

    SOCKET_PATH = "/tmp/neuro-vscode.sock"
    PORT = 19432  # Fallback if socket not available

    def __init__(self, workspace_root: str = "."):
        super().__init__(workspace_root)
        self._socket_server = None
        self._client_writer = None
        self._context = EditorContext()
        self._code_path = self._find_code_binary()

    @property
    def ide_type(self) -> IDEType:
        return IDEType.VSCODE

    def _find_code_binary(self) -> Optional[str]:
        """Find the VSCode CLI binary."""
        # Check common locations
        candidates = [
            "/usr/local/bin/code",
            "/usr/bin/code",
            "/opt/homebrew/bin/code",
            os.path.expanduser("~/.local/bin/code"),
            # macOS app location
            "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code",
        ]

        for path in candidates:
            if os.path.exists(path):
                return path

        # Try PATH
        try:
            result = subprocess.run(
                ["which", "code"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return None

    async def connect(self) -> bool:
        """
        Connect to VSCode.

        Starts a socket server that VSCode extension can connect to.
        """
        if self._connected:
            return True

        try:
            # Try Unix socket first
            if os.path.exists(self.SOCKET_PATH):
                os.remove(self.SOCKET_PATH)

            self._socket_server = await asyncio.start_unix_server(
                self._handle_client,
                path=self.SOCKET_PATH,
            )
            self._connected = True
            return True

        except Exception:
            # Fallback to TCP
            try:
                self._socket_server = await asyncio.start_server(
                    self._handle_client,
                    "127.0.0.1",
                    self.PORT,
                )
                self._connected = True
                return True
            except Exception:
                return False

    async def disconnect(self) -> None:
        """Disconnect from VSCode."""
        if self._socket_server:
            self._socket_server.close()
            await self._socket_server.wait_closed()
            self._socket_server = None

        if os.path.exists(self.SOCKET_PATH):
            os.remove(self.SOCKET_PATH)

        self._connected = False

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ):
        """Handle incoming connection from VSCode extension."""
        self._client_writer = writer

        try:
            while True:
                data = await reader.readline()
                if not data:
                    break

                try:
                    message = json.loads(data.decode())
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    pass

        except asyncio.CancelledError:
            pass
        finally:
            self._client_writer = None
            writer.close()
            await writer.wait_closed()

    async def _handle_message(self, message: Dict[str, Any]):
        """Handle message from VSCode extension."""
        msg_type = message.get("type")

        if msg_type == "context":
            # Update editor context
            data = message.get("data", {})
            self._context = EditorContext(
                file_path=data.get("filePath"),
                line_number=data.get("line"),
                column=data.get("column"),
                selection=data.get("selection"),
                selection_start=tuple(data["selectionStart"])
                if data.get("selectionStart")
                else None,
                selection_end=tuple(data["selectionEnd"]) if data.get("selectionEnd") else None,
                language=data.get("language"),
                workspace_root=data.get("workspaceRoot", self.workspace_root),
                open_files=data.get("openFiles", []),
                dirty_files=data.get("dirtyFiles", []),
            )
            self._emit_event("context_updated", data)

        elif msg_type == "file_changed":
            self._emit_event("file_changed", message.get("data", {}))

        elif msg_type == "selection_changed":
            data = message.get("data", {})
            self._context.selection = data.get("selection")
            self._emit_event("selection_changed", data)

    async def _send_to_extension(self, message: Dict[str, Any]) -> bool:
        """Send message to VSCode extension."""
        if not self._client_writer:
            return False

        try:
            data = json.dumps(message) + "\n"
            self._client_writer.write(data.encode())
            await self._client_writer.drain()
            return True
        except Exception:
            return False

    async def get_context(self) -> EditorContext:
        """Get current editor context."""
        # If connected to extension, request fresh context
        if self._client_writer:
            await self._send_to_extension({"type": "get_context"})
            await asyncio.sleep(0.1)  # Give time for response

        return self._context

    async def execute_command(self, command: IDECommand) -> bool:
        """Execute a command in VSCode."""
        # Try extension first
        if self._client_writer:
            sent = await self._send_to_extension(
                {
                    "type": "command",
                    "data": {
                        "action": command.action,
                        "filePath": command.file_path,
                        "line": command.line_number,
                        "column": command.column,
                        "content": command.content,
                    },
                }
            )
            if sent:
                return True

        # Fallback to CLI
        return await self._execute_via_cli(command)

    async def _execute_via_cli(self, command: IDECommand) -> bool:
        """Execute command via `code` CLI."""
        if not self._code_path:
            return False

        try:
            if command.action == "open":
                args = [self._code_path]
                if command.file_path:
                    file_arg = command.file_path
                    if command.line_number:
                        file_arg += f":{command.line_number}"
                        if command.column:
                            file_arg += f":{command.column}"
                    args.extend(["--goto", file_arg])

                proc = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
                return proc.returncode == 0

            elif command.action == "goto":
                if command.file_path and command.line_number:
                    file_arg = f"{command.file_path}:{command.line_number}"
                    if command.column:
                        file_arg += f":{command.column}"

                    proc = await asyncio.create_subprocess_exec(
                        self._code_path,
                        "--goto",
                        file_arg,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await proc.wait()
                    return proc.returncode == 0

            elif command.action == "diff":
                # Open diff view
                if command.file_path and command.content:
                    # Create temp file for comparison
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", delete=False) as f:
                        f.write(command.content)
                        temp_path = f.name

                    proc = await asyncio.create_subprocess_exec(
                        self._code_path,
                        "--diff",
                        command.file_path,
                        temp_path,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await proc.wait()
                    # Cleanup temp file after delay
                    asyncio.get_event_loop().call_later(5, lambda: os.unlink(temp_path))
                    return proc.returncode == 0

            return False

        except Exception:
            return False

    async def show_diff(self, file_path: str, new_content: str) -> bool:
        """Show diff between file and new content in VSCode."""
        cmd = IDECommand(
            action="diff",
            file_path=os.path.abspath(file_path),
            content=new_content,
        )
        return await self.execute_command(cmd)
