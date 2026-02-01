"""
Intent Router - Route user input to actions without relying on LLM tool parsing.

Based on autonomus-elo pattern: detect intent via keywords/patterns, then execute directly.
"""

import re
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Intent:
    """A detected intent with confidence."""
    name: str
    confidence: float
    params: Dict[str, str]
    action: Callable


class IntentRouter:
    """
    Route user input to actions using pattern matching.

    This avoids relying on LLM to output tool calls in specific formats.
    Instead, we detect intent from natural language and execute directly.
    """

    def __init__(self):
        self.intents: List[Tuple[str, List[str], Callable]] = []
        self._register_default_intents()

    def _register_default_intents(self):
        """Register default intent patterns."""
        # Self-improvement intents
        self.register(
            "improve_self",
            [
                r"improve\s*(your)?self",
                r"evolve",
                r"upgrade\s*(your)?self",
                r"make\s*(your)?self\s*better",
                r"self[- ]?improv",
                r"fix\s*(your)?self",
            ]
        )

        # Research/learning intents
        self.register(
            "research",
            [
                r"research\s+(.+)",
                r"learn\s+about\s+(.+)",
                r"study\s+(.+)",
                r"find\s+out\s+about\s+(.+)",
                r"what\s+is\s+(.+)",
            ]
        )

        # Web search intents
        self.register(
            "web_search",
            [
                r"search\s+(?:the\s+)?(?:web|internet|online)\s+(?:for\s+)?(.+)",
                r"google\s+(.+)",
                r"look\s+up\s+(.+)",
                r"find\s+online\s+(.+)",
            ]
        )

        # File operations
        self.register(
            "read_file",
            [
                r"read\s+(?:the\s+)?file\s+(.+)",
                r"show\s+(?:me\s+)?(?:the\s+)?(?:contents?\s+(?:of\s+)?)?(.+\.(?:py|js|ts|txt|md|json|yaml|yml))",
                r"cat\s+(.+)",
                r"open\s+(.+\.(?:py|js|ts|txt|md|json|yaml|yml))",
            ]
        )

        self.register(
            "list_files",
            [
                r"list\s+(?:the\s+)?files",
                r"show\s+(?:the\s+)?directory",
                r"ls\b",
                r"what\s+files",
            ]
        )

        # Code operations
        self.register(
            "analyze_code",
            [
                r"analyze\s+(?:the\s+)?code",
                r"review\s+(?:the\s+)?code",
                r"check\s+(?:the\s+)?code",
                r"code\s+review",
            ]
        )

        self.register(
            "run_tests",
            [
                r"run\s+(?:the\s+)?tests?",
                r"test\s+(?:the\s+)?code",
                r"pytest",
                r"execute\s+tests?",
            ]
        )

        # Git operations
        self.register(
            "git_status",
            [
                r"git\s+status",
                r"show\s+(?:git\s+)?changes",
                r"what\s+(?:has\s+)?changed",
            ]
        )

        self.register(
            "git_commit",
            [
                r"commit\s+(?:the\s+)?changes?",
                r"git\s+commit",
                r"save\s+(?:the\s+)?changes?",
            ]
        )

        # System commands
        self.register(
            "run_command",
            [
                r"run\s+(?:the\s+)?command\s+(.+)",
                r"execute\s+(.+)",
                r"shell\s+(.+)",
            ]
        )

    def register(self, intent_name: str, patterns: List[str], action: Callable = None):
        """Register an intent with patterns."""
        self.intents.append((intent_name, patterns, action))

    def detect(self, text: str) -> Optional[Intent]:
        """Detect intent from text."""
        text_lower = text.lower().strip()

        for intent_name, patterns, action in self.intents:
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.IGNORECASE)
                if match:
                    # Extract parameters from capture groups
                    params = {}
                    if match.groups():
                        params["query"] = match.group(1).strip() if match.group(1) else ""

                    # Calculate confidence based on pattern specificity
                    confidence = 0.8 if len(pattern) > 20 else 0.6

                    return Intent(
                        name=intent_name,
                        confidence=confidence,
                        params=params,
                        action=action
                    )

        return None

    def get_intent_examples(self) -> Dict[str, List[str]]:
        """Get example phrases for each intent (for training)."""
        examples = {}
        for intent_name, patterns, _ in self.intents:
            examples[intent_name] = [
                p.replace(r"\s+", " ").replace(r"\s*", " ")
                  .replace(r"(.+)", "<query>")
                  .replace(r"(?:the\s+)?", "")
                  .replace(r"(?:your)?", "")
                  .replace(r"\b", "")
                for p in patterns[:3]
            ]
        return examples


# Global router instance
intent_router = IntentRouter()


def detect_intent(text: str) -> Optional[Intent]:
    """Convenience function to detect intent."""
    return intent_router.detect(text)


# Test
if __name__ == "__main__":
    test_inputs = [
        "improve yourself",
        "research neural networks",
        "search the web for python best practices",
        "read the file app.py",
        "list files",
        "run the tests",
        "git status",
        "evolve",
        "what is machine learning",
    ]

    print("Intent Detection Test")
    print("=" * 50)

    for text in test_inputs:
        intent = detect_intent(text)
        if intent:
            print(f"'{text}'")
            print(f"  -> Intent: {intent.name} ({intent.confidence:.0%})")
            if intent.params:
                print(f"  -> Params: {intent.params}")
            print()
        else:
            print(f"'{text}' -> No intent detected\n")
