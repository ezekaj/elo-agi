"""
Module Loader: Dynamically loads and manages cognitive modules.

Handles:
- Discovery and import of all 20 neuro-modules
- Registration with the Global Workspace
- Graceful handling of missing modules
- Module lifecycle management
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type
from pathlib import Path
import sys
import importlib
import importlib.util
from enum import Enum

from .config import SystemConfig, ModuleConfig


class ModuleStatus(Enum):
    """Status of a loaded module."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    DISABLED = "disabled"


@dataclass
class LoadedModule:
    """Represents a loaded cognitive module."""

    module_id: str
    name: str
    status: ModuleStatus
    module_object: Optional[Any] = None  # The actual module instance
    error: Optional[str] = None
    capabilities: List[str] = field(default_factory=list)

    def is_available(self) -> bool:
        """Check if module is loaded and available."""
        return self.status == ModuleStatus.LOADED and self.module_object is not None


class ModuleLoader:
    """
    Loads and manages all cognitive modules.

    The loader:
    1. Discovers modules in the configured path
    2. Imports and instantiates module classes
    3. Tracks module status and errors
    4. Provides access to loaded modules
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self._modules: Dict[str, LoadedModule] = {}
        self._load_order: List[str] = []

    def load_all(self) -> Dict[str, LoadedModule]:
        """
        Load all enabled modules.

        Returns:
            Dictionary of module_id -> LoadedModule
        """
        enabled = self.config.get_enabled_modules()

        for module_id in enabled:
            self.load_module(module_id)

        return self._modules

    def load_module(self, module_id: str) -> LoadedModule:
        """
        Load a single module.

        Args:
            module_id: The module ID (e.g., "00", "17")

        Returns:
            LoadedModule instance
        """
        # Check if already loaded
        if module_id in self._modules:
            return self._modules[module_id]

        # Get config
        config = self.config.module_configs.get(module_id)
        if config and not config.enabled:
            loaded = LoadedModule(
                module_id=module_id,
                name=f"module_{module_id}",
                status=ModuleStatus.DISABLED,
            )
            self._modules[module_id] = loaded
            return loaded

        # Create loading entry
        loaded = LoadedModule(
            module_id=module_id,
            name=f"module_{module_id}",
            status=ModuleStatus.LOADING,
        )

        try:
            # Get module path
            module_path = self.config.get_module_path(module_id)

            if not module_path.exists():
                loaded.status = ModuleStatus.FAILED
                loaded.error = f"Module path not found: {module_path}"
                self._modules[module_id] = loaded
                return loaded

            # Add to sys.path if needed
            src_path = module_path / "src"
            if src_path.exists() and str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))

            # Import and instantiate based on module type
            module_obj = self._import_module(module_id, src_path)

            if module_obj is not None:
                loaded.status = ModuleStatus.LOADED
                loaded.module_object = module_obj
                loaded.capabilities = self._get_capabilities(module_obj)
                loaded.name = type(module_obj).__name__
            else:
                loaded.status = ModuleStatus.FAILED
                loaded.error = "Failed to instantiate module"

        except Exception as e:
            loaded.status = ModuleStatus.FAILED
            loaded.error = str(e)

        self._modules[module_id] = loaded
        self._load_order.append(module_id)
        return loaded

    def _import_module(self, module_id: str, src_path: Path) -> Optional[Any]:
        """Import and instantiate a module."""
        # Module-specific imports
        import_map = {
            "00": ("global_workspace", "GlobalWorkspace"),
            "17": ("world_model", "WorldModel"),
            "18": ("darwin_godel", "DarwinGodelMachine"),
            "19": ("swarm", "SwarmIntelligence"),
        }

        if module_id in import_map:
            file_name, class_name = import_map[module_id]
            return self._import_class(src_path, file_name, class_name)

        # For modules 01-16, try common patterns
        # These modules may have different structures
        common_patterns = [
            ("__init__", None),  # Try getting main class from __init__
        ]

        for file_name, class_name in common_patterns:
            obj = self._import_class(src_path, file_name, class_name)
            if obj is not None:
                return obj

        # Return a stub if we can't load the actual module
        return ModuleStub(module_id)

    def _import_class(
        self, src_path: Path, file_name: str, class_name: Optional[str]
    ) -> Optional[Any]:
        """Import a class from a module file."""
        try:
            file_path = src_path / f"{file_name}.py"
            if not file_path.exists():
                return None

            # Load module
            spec = importlib.util.spec_from_file_location(file_name, file_path)
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if class_name:
                # Get specific class
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    return cls()
            else:
                # Try to find main class
                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and name != "ABC":
                        try:
                            return obj()
                        except:
                            continue

            return None

        except Exception:
            return None

    def _get_capabilities(self, module_obj: Any) -> List[str]:
        """Get list of capabilities from a module."""
        capabilities = []

        # Check for common methods
        method_caps = [
            ("propose", "proposal"),
            ("receive_broadcast", "broadcast"),
            ("process", "processing"),
            ("predict", "prediction"),
            ("imagine", "imagination"),
            ("learn", "learning"),
            ("step", "stepping"),
        ]

        for method, cap in method_caps:
            if hasattr(module_obj, method) and callable(getattr(module_obj, method)):
                capabilities.append(cap)

        return capabilities

    def get_module(self, module_id: str) -> Optional[Any]:
        """Get a loaded module's object."""
        if module_id in self._modules:
            loaded = self._modules[module_id]
            if loaded.is_available():
                return loaded.module_object
        return None

    def get_all_loaded(self) -> Dict[str, Any]:
        """Get all successfully loaded modules."""
        return {
            mid: loaded.module_object
            for mid, loaded in self._modules.items()
            if loaded.is_available()
        }

    def get_status(self) -> Dict[str, ModuleStatus]:
        """Get status of all modules."""
        return {mid: loaded.status for mid, loaded in self._modules.items()}

    def get_failed(self) -> Dict[str, str]:
        """Get failed modules and their errors."""
        return {
            mid: loaded.error or "Unknown error"
            for mid, loaded in self._modules.items()
            if loaded.status == ModuleStatus.FAILED
        }

    def unload_module(self, module_id: str) -> bool:
        """Unload a module."""
        if module_id in self._modules:
            del self._modules[module_id]
            if module_id in self._load_order:
                self._load_order.remove(module_id)
            return True
        return False

    def reload_module(self, module_id: str) -> LoadedModule:
        """Reload a module."""
        self.unload_module(module_id)
        return self.load_module(module_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get loader statistics."""
        statuses = self.get_status()
        return {
            "total_modules": len(self._modules),
            "loaded": sum(1 for s in statuses.values() if s == ModuleStatus.LOADED),
            "failed": sum(1 for s in statuses.values() if s == ModuleStatus.FAILED),
            "disabled": sum(1 for s in statuses.values() if s == ModuleStatus.DISABLED),
            "load_order": self._load_order.copy(),
        }


class ModuleStub:
    """
    Stub for modules that couldn't be loaded.

    Provides a minimal interface to prevent crashes when
    a module is unavailable.
    """

    def __init__(self, module_id: str):
        self.module_id = module_id
        self._name = f"Stub_{module_id}"

    def propose(self, *args, **kwargs) -> List:
        """Stub propose returns empty list."""
        return []

    def receive_broadcast(self, *args, **kwargs) -> None:
        """Stub receive does nothing."""
        pass

    def process(self, *args, **kwargs) -> None:
        """Stub process does nothing."""
        pass

    def step(self, *args, **kwargs) -> None:
        """Stub step does nothing."""
        pass

    def __repr__(self) -> str:
        return f"<ModuleStub {self.module_id}>"
