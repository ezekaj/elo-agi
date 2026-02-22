"""
System Configuration for the Unified Cognitive Architecture.

Defines parameters for:
- Module loading and integration
- Processing timescales
- Active inference parameters
- Sensory/motor interfaces
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import os


@dataclass
class ModuleConfig:
    """Configuration for a single cognitive module."""

    module_id: str
    enabled: bool = True
    priority: float = 1.0  # Higher = processed earlier
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """
    Configuration for the unified cognitive system.

    Controls all aspects of the cognitive architecture including
    module loading, processing parameters, and interface settings.
    """

    # Module paths
    modules_base_path: Path = field(
        default_factory=lambda: Path(os.path.expanduser("~/Desktop/neuro"))
    )

    # Module configurations
    module_configs: Dict[str, ModuleConfig] = field(default_factory=dict)

    # Processing parameters
    timestep: float = 0.1  # Seconds per cognitive cycle
    max_cycles_per_step: int = 10  # Max internal iterations
    workspace_capacity: int = 7  # Miller's Law

    # Active inference parameters
    efe_horizon: int = 5  # Planning horizon
    efe_samples: int = 10  # Monte Carlo samples
    exploration_weight: float = 0.3  # Epistemic vs pragmatic balance
    temperature: float = 1.0  # Action selection temperature

    # Sensory interface
    input_dim: int = 64  # Standard input vector size
    input_normalization: bool = True

    # Motor interface
    output_dim: int = 32  # Standard output vector size
    action_threshold: float = 0.5  # Min activation to act

    # Learning
    learning_enabled: bool = True
    learning_rate: float = 0.01

    # Safety
    max_memory_mb: int = 1024  # Memory limit
    timeout_seconds: float = 10.0  # Max time per cycle

    def __post_init__(self):
        """Initialize default module configurations."""
        if not self.module_configs:
            self._init_default_modules()

    def _init_default_modules(self) -> None:
        """Set up default configurations for all 20 modules."""
        default_modules = [
            ("00", "integration", 10.0),  # Highest priority - hub
            ("01", "predictive-coding", 9.0),
            ("02", "dual-process", 8.0),
            ("03", "reasoning-types", 7.0),
            ("04", "memory", 8.5),
            ("05", "sleep-consolidation", 1.0),  # Low priority - offline
            ("06", "motivation", 7.5),
            ("07", "emotions-decisions", 7.0),
            ("08", "language", 6.0),
            ("09", "creativity", 5.0),
            ("10", "spatial-cognition", 6.0),
            ("11", "time-perception", 5.5),
            ("12", "learning", 8.0),
            ("13", "executive", 9.0),
            ("14", "embodied", 6.5),
            ("15", "social", 5.0),
            ("16", "consciousness", 9.5),
            ("17", "world-model", 9.0),
            ("18", "self-improvement", 2.0),  # Low priority - meta
            ("19", "multi-agent", 3.0),
        ]

        for mod_id, name, priority in default_modules:
            self.module_configs[mod_id] = ModuleConfig(
                module_id=mod_id,
                enabled=True,
                priority=priority,
            )

    def get_module_path(self, module_id: str) -> Path:
        """Get the path to a module directory."""
        # Map module IDs to directory names
        name_map = {
            "00": "neuro-module-00-integration",
            "17": "neuro-module-17-world-model",
            "18": "neuro-module-18-self-improvement",
            "19": "neuro-module-19-multi-agent",
        }

        if module_id in name_map:
            return self.modules_base_path / name_map[module_id]

        # For modules 01-16, use pattern
        module_num = int(module_id)
        module_names = {
            1: "predictive-coding",
            2: "dual-process",
            3: "reasoning-types",
            4: "memory",
            5: "sleep-consolidation",
            6: "motivation",
            7: "emotions-decisions",
            8: "language",
            9: "creativity",
            10: "spatial-cognition",
            11: "time-perception",
            12: "learning",
            13: "executive",
            14: "embodied",
            15: "social",
            16: "consciousness",
        }

        if module_num in module_names:
            return self.modules_base_path / f"neuro-module-{module_id}-{module_names[module_num]}"

        return self.modules_base_path / f"neuro-module-{module_id}"

    def get_enabled_modules(self) -> List[str]:
        """Get list of enabled module IDs, sorted by priority."""
        enabled = [(mid, cfg.priority) for mid, cfg in self.module_configs.items() if cfg.enabled]
        enabled.sort(key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in enabled]

    def enable_module(self, module_id: str) -> None:
        """Enable a module."""
        if module_id in self.module_configs:
            self.module_configs[module_id].enabled = True

    def disable_module(self, module_id: str) -> None:
        """Disable a module."""
        if module_id in self.module_configs:
            self.module_configs[module_id].enabled = False

    def set_module_priority(self, module_id: str, priority: float) -> None:
        """Set module priority."""
        if module_id in self.module_configs:
            self.module_configs[module_id].priority = priority


# Default configuration
DEFAULT_CONFIG = SystemConfig()
