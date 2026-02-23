"""Subagent system."""

from .manager import SubagentManager, SubagentConfig, SubagentExecution, SubagentType
from .team import TeamManager, AgentTeam, Teammate, TeammateMailbox, TeammateStatus

__all__ = [
    "SubagentManager",
    "SubagentConfig",
    "SubagentExecution",
    "SubagentType",
    "TeamManager",
    "AgentTeam",
    "Teammate",
    "TeammateMailbox",
    "TeammateStatus",
]
