"""Base Tool class and ToolResult dataclass."""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class ToolResult:
    """Result from tool execution."""

    success: bool
    output: str
    error: str = ""
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"Error: {self.error}"


class Tool(ABC):
    """Base class for all NEURO tools."""

    name: str = ""
    description: str = ""
    requires_permission: bool = False

    # JSON Schema for parameters (used by Ollama function calling)
    parameters: dict = {"type": "object", "properties": {}, "required": []}

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass

    def to_ollama_format(self) -> dict:
        """Convert tool to Ollama function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_params(self, **kwargs) -> tuple[bool, str]:
        """Validate parameters against schema."""
        required = self.parameters.get("required", [])
        for param in required:
            if param not in kwargs:
                return False, f"Missing required parameter: {param}"
        return True, ""
