"""
Configuration Manager - Multi-source settings with merging.

Loads settings from (in priority order):
1. CLI arguments (highest)
2. .neuro/settings.local.json (local overrides, gitignored)
3. .neuro/settings.json (project-level)
4. ~/.neuro/settings.json (user-level, lowest)
"""

import json
import os
from typing import Any, Dict


class ConfigManager:
    """Multi-source configuration with cascading merge."""

    def __init__(self, project_dir: str = ".", cli_overrides: Dict[str, Any] = None):
        self.project_dir = os.path.abspath(project_dir)
        self._config: Dict[str, Any] = {}
        self._cli_overrides = cli_overrides or {}
        self.reload()

    def reload(self):
        """Load and merge configuration from all sources."""
        self._config = {}

        # Load in order (lower priority first, higher overwrites)
        sources = [
            os.path.expanduser("~/.neuro/settings.json"),
            os.path.join(self.project_dir, ".neuro", "settings.json"),
            os.path.join(self.project_dir, ".neuro", "settings.local.json"),
        ]

        for path in sources:
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    self._deep_merge(self._config, data)
                except Exception:
                    pass

        # CLI overrides take highest priority
        self._deep_merge(self._config, self._cli_overrides)

    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-separated key path."""
        parts = key.split(".")
        current = self._config
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, key: str, value: Any):
        """Set a config value (in memory only)."""
        parts = key.split(".")
        current = self._config
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def save_project(self):
        """Save current config to project settings."""
        path = os.path.join(self.project_dir, ".neuro", "settings.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._config, f, indent=2)

    def save_user(self):
        """Save current config to user settings."""
        path = os.path.expanduser("~/.neuro/settings.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._config, f, indent=2)

    def get_all(self) -> Dict[str, Any]:
        """Get the full merged config."""
        return dict(self._config)

    def get_sources(self) -> list:
        """List config sources that exist."""
        sources = [
            ("user", os.path.expanduser("~/.neuro/settings.json")),
            ("project", os.path.join(self.project_dir, ".neuro", "settings.json")),
            ("local", os.path.join(self.project_dir, ".neuro", "settings.local.json")),
        ]
        return [(name, path) for name, path in sources if os.path.exists(path)]
