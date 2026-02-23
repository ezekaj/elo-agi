"""
Keybindings Manager - Custom keyboard shortcuts.
"""

import json
import os
from typing import Dict, Optional


DEFAULT_KEYBINDINGS = {
    "ctrl+c": "cancel",
    "ctrl+d": "exit",
    "ctrl+l": "clear",
    "ctrl+k": "compact",
}


class KeybindingsManager:
    """Load and manage keyboard shortcuts."""

    KEYBINDINGS_FILE = "keybindings.json"

    def __init__(self, config_dir: str = "~/.neuro"):
        self.config_dir = os.path.expanduser(config_dir)
        self._bindings: Dict[str, str] = dict(DEFAULT_KEYBINDINGS)
        self._load()

    def _load(self):
        path = os.path.join(self.config_dir, self.KEYBINDINGS_FILE)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    custom = json.load(f)
                self._bindings.update(custom)
            except Exception:
                pass

    def save(self):
        path = os.path.join(self.config_dir, self.KEYBINDINGS_FILE)
        os.makedirs(self.config_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._bindings, f, indent=2)

    def get_action(self, key: str) -> Optional[str]:
        return self._bindings.get(key)

    def set_binding(self, key: str, action: str):
        self._bindings[key] = action

    def get_all(self) -> Dict[str, str]:
        return dict(self._bindings)
