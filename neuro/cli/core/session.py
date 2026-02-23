"""
Session Manager - Persistent conversation sessions.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import json
import uuid
import os
import hashlib


@dataclass
class SessionMessage:
    """A message in the conversation."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SessionMessage":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    """A conversation session."""

    id: str
    project_dir: str
    created_at: datetime
    messages: List[SessionMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime state
    token_count: int = 0
    is_compacted: bool = False
    compaction_count: int = 0

    def add_message(self, role: str, content: str, **metadata):
        """Add a message to the session."""
        msg = SessionMessage(
            role=role,
            content=content,
            metadata=metadata,
        )
        self.messages.append(msg)
        self.token_count += len(content) // 4  # Rough estimate

    def get_history(self, max_messages: int = 50) -> List[Dict[str, str]]:
        """Get conversation history for LLM."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages[-max_messages:]
            if m.role in ("user", "assistant", "system")
        ]


class SessionManager:
    """
    Manages conversation sessions with persistence.

    Features:
    - Persistent sessions via transcript files (.jsonl)
    - Session resume/continue
    - Context compaction
    """

    TRANSCRIPT_EXT = ".jsonl"

    def __init__(
        self,
        project_dir: str = ".",
        persist: bool = True,
    ):
        self.project_dir = os.path.abspath(project_dir)
        self.persist = persist

        # Sessions directory
        self.sessions_path = os.path.join(
            os.path.expanduser("~"), ".neuro", "projects", self._get_project_hash(), "sessions"
        )
        if persist:
            os.makedirs(self.sessions_path, exist_ok=True)

        self._current: Optional[Session] = None

    def _get_project_hash(self) -> str:
        """Get a hash identifier for the project."""
        return hashlib.md5(self.project_dir.encode()).hexdigest()[:12]

    def new_session(self) -> Session:
        """Create a new session."""
        session = Session(
            id=str(uuid.uuid4()),
            project_dir=self.project_dir,
            created_at=datetime.now(),
        )
        self._current = session
        return session

    def get_current(self) -> Optional[Session]:
        """Get current active session."""
        return self._current

    def resume(self, session_id: Optional[str] = None) -> Optional[Session]:
        """Resume a session by ID, or most recent if None."""
        if not self.persist:
            return None

        if session_id:
            return self._load_session(session_id)

        # Find most recent
        transcripts = sorted(
            Path(self.sessions_path).glob(f"*{self.TRANSCRIPT_EXT}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if transcripts:
            session_id = transcripts[0].stem
            return self._load_session(session_id)

        return None

    def save(self, session: Optional[Session] = None):
        """Save session to transcript file."""
        if not self.persist:
            return

        session = session or self._current
        if not session:
            return

        transcript_path = os.path.join(self.sessions_path, f"{session.id}{self.TRANSCRIPT_EXT}")

        with open(transcript_path, "w") as f:
            # Write session metadata
            f.write(
                json.dumps(
                    {
                        "type": "session",
                        "id": session.id,
                        "project_dir": session.project_dir,
                        "created_at": session.created_at.isoformat(),
                        "metadata": session.metadata,
                    }
                )
                + "\n"
            )

            # Write messages
            for msg in session.messages:
                f.write(json.dumps({"type": "message", **msg.to_dict()}) + "\n")

    def _load_session(self, session_id: str) -> Optional[Session]:
        """Load session from transcript file."""
        transcript_path = os.path.join(self.sessions_path, f"{session_id}{self.TRANSCRIPT_EXT}")

        if not os.path.exists(transcript_path):
            return None

        session = None
        messages = []

        with open(transcript_path) as f:
            for line in f:
                data = json.loads(line)
                if data["type"] == "session":
                    session = Session(
                        id=data["id"],
                        project_dir=data["project_dir"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        metadata=data.get("metadata", {}),
                    )
                elif data["type"] == "message":
                    messages.append(SessionMessage.from_dict(data))

        if session:
            session.messages = messages
            session.token_count = sum(len(m.content) // 4 for m in messages)
            self._current = session

        return session

    def list_sessions(self, limit: int = 10) -> List[Dict]:
        """List recent sessions with preview."""
        if not self.persist:
            return []

        sessions = []
        transcripts = sorted(
            Path(self.sessions_path).glob(f"*{self.TRANSCRIPT_EXT}"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]

        for t in transcripts:
            try:
                with open(t) as f:
                    session_data = None
                    preview = ""
                    msg_count = 0

                    for line in f:
                        data = json.loads(line)
                        if data["type"] == "session":
                            session_data = data
                        elif data["type"] == "message":
                            msg_count += 1
                            # Get first user message as preview
                            if not preview and data.get("role") == "user":
                                preview = data.get("content", "")[:50]

                    if session_data:
                        sessions.append(
                            {
                                "id": session_data["id"],
                                "created_at": session_data["created_at"],
                                "project_dir": session_data["project_dir"],
                                "preview": preview,
                                "message_count": msg_count,
                            }
                        )
            except Exception:
                pass

        return sessions

    def microcompact(self, session: Optional[Session] = None, max_result_len: int = 500) -> int:
        """
        Phase 1: Truncate long tool results individually.

        Returns the number of messages truncated.
        Based on Claude Code's microcompaction pattern — shrink tool outputs
        before doing full conversation compaction.
        """
        session = session or self._current
        if not session:
            return 0

        truncated = 0
        for msg in session.messages:
            if msg.role in ("tool", "system") and msg.metadata.get("type") == "tool_result":
                if len(msg.content) > max_result_len:
                    keep = max_result_len - 30  # Leave room for marker
                    msg.content = msg.content[:keep] + "\n[...truncated...]"
                    truncated += 1
            # Also truncate tool results embedded in user messages from follow-up
            elif msg.role == "user" and "[Tool " in msg.content and len(msg.content) > max_result_len * 2:
                keep = max_result_len * 2 - 30
                msg.content = msg.content[:keep] + "\n[...truncated...]"
                truncated += 1

        if truncated:
            session.token_count = sum(len(m.content) // 4 for m in session.messages)
            session.compaction_count += 1

        return truncated

    async def compact(self, session: Optional[Session] = None) -> bool:
        """
        Phase 2: Full context compaction.

        First runs microcompaction on tool results, then drops older messages
        while preserving system messages and recent conversation.
        """
        session = session or self._current
        if not session or len(session.messages) < 10:
            return False

        # Phase 1: microcompact tool results first
        self.microcompact(session)

        # Phase 2: drop old messages if still too many
        system_msgs = [m for m in session.messages if m.role == "system"]
        other_msgs = [m for m in session.messages if m.role != "system"]

        if len(other_msgs) > 20:
            # Summarize what was dropped
            dropped = other_msgs[:-20]
            user_msgs = [m.content[:100] for m in dropped if m.role == "user"]
            summary_parts = ["[Previous context compacted]"]
            if user_msgs:
                topics = "; ".join(user_msgs[:5])
                summary_parts.append(f"Topics discussed: {topics}")

            session.messages = (
                system_msgs
                + [
                    SessionMessage(
                        role="system",
                        content="\n".join(summary_parts),
                        metadata={"type": "compact_boundary"},
                    )
                ]
                + other_msgs[-20:]
            )
            session.is_compacted = True
            session.compaction_count += 1
            session.token_count = sum(len(m.content) // 4 for m in session.messages)
            self.save(session)
            return True

        return False
