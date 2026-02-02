"""Session management with SQLite storage."""

import sqlite3
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table

console = Console()

DB_PATH = Path.home() / ".neuro" / "sessions.db"


def init_db() -> sqlite3.Connection:
    """Initialize the database and return connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            model TEXT,
            working_dir TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            created_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    conn.commit()
    return conn


class SessionManager:
    """Manages conversation sessions."""

    def __init__(self):
        self.conn = init_db()
        self.current_session: Optional[str] = None

    def new_session(self, model: str, working_dir: str = None) -> str:
        """Create a new session and return its ID."""
        sid = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        wd = working_dir or str(Path.cwd())

        self.conn.execute(
            "INSERT INTO sessions (id, title, model, working_dir, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (sid, "New Session", model, wd, now, now)
        )
        self.conn.commit()
        self.current_session = sid
        return sid

    def add_message(self, role: str, content: str):
        """Add a message to the current session."""
        if not self.current_session:
            return

        now = datetime.now().isoformat()
        self.conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (self.current_session, role, content, now)
        )

        # Update session timestamp and auto-generate title
        self.conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, self.current_session)
        )
        self.conn.commit()

        # Auto-title from first user message
        if role == "user":
            cur = self.conn.execute(
                "SELECT title FROM sessions WHERE id = ?",
                (self.current_session,)
            )
            title = cur.fetchone()[0]
            if title == "New Session":
                # Use first 50 chars of first user message as title
                new_title = content[:50].replace("\n", " ")
                if len(content) > 50:
                    new_title += "..."
                self.conn.execute(
                    "UPDATE sessions SET title = ? WHERE id = ?",
                    (new_title, self.current_session)
                )
                self.conn.commit()

    def get_messages(self, session_id: str = None) -> list[dict]:
        """Get all messages for a session."""
        sid = session_id or self.current_session
        if not sid:
            return []

        cur = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (sid,)
        )
        return [{"role": role, "content": content} for role, content in cur.fetchall()]

    def list_sessions(self, limit: int = 10) -> list[tuple]:
        """List recent sessions."""
        cur = self.conn.execute(
            """SELECT id, title, model, updated_at
               FROM sessions
               ORDER BY updated_at DESC
               LIMIT ?""",
            (limit,)
        )
        return cur.fetchall()

    def get_last_session_id(self) -> Optional[str]:
        """Get the ID of the most recent session."""
        cur = self.conn.execute(
            "SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        return row[0] if row else None

    def resume(self, session_id: str) -> list[dict]:
        """Resume a session and return its messages."""
        # Verify session exists
        cur = self.conn.execute(
            "SELECT id FROM sessions WHERE id = ?",
            (session_id,)
        )
        if not cur.fetchone():
            return []

        self.current_session = session_id
        return self.get_messages(session_id)

    def delete_session(self, session_id: str):
        """Delete a session and its messages."""
        self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.conn.commit()

    def show_sessions(self):
        """Display sessions in a nice table."""
        sessions = self.list_sessions()

        if not sessions:
            console.print("[dim]No sessions yet.[/dim]")
            return

        table = Table(title="[bold cyan]Recent Sessions[/bold cyan]", border_style="cyan")
        table.add_column("ID", style="cyan", width=10)
        table.add_column("Title", width=40)
        table.add_column("Model", style="dim")
        table.add_column("Last Updated", style="dim")

        for sid, title, model, updated in sessions:
            # Format date nicely
            try:
                dt = datetime.fromisoformat(updated)
                date_str = dt.strftime("%m/%d %H:%M")
            except:
                date_str = updated[:16]

            table.add_row(sid, title[:40], model, date_str)

        console.print(table)
        console.print("\n[dim]Use /resume <id> to continue a session[/dim]")


# Global session manager instance
session_manager = SessionManager()
