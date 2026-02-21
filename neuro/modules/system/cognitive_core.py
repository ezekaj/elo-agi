"""
Cognitive Core: The unified mind integrating all modules.

This is the central class that:
- Loads and connects all 20 cognitive modules
- Runs the perception-action loop
- Coordinates via Global Workspace
- Uses Active Inference for action selection

Usage:
    core = CognitiveCore()
    core.perceive(input_data)
    core.think()
    action = core.act()
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import time

from .config import SystemConfig, DEFAULT_CONFIG
from .module_loader import ModuleLoader, LoadedModule, ModuleStatus
from .sensory_interface import SensoryInterface, SensoryInput, InputType
from .motor_interface import MotorInterface, MotorOutput, OutputType
from .active_inference import ActiveInferenceController, Policy


@dataclass
class CognitiveState:
    """Current state of the cognitive system."""
    cycle_count: int = 0
    last_input: Optional[np.ndarray] = None
    last_output: Optional[np.ndarray] = None
    workspace_contents: List[Any] = field(default_factory=list)
    active_modules: List[str] = field(default_factory=list)
    goals: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CycleResult:
    """Result of a single cognitive cycle."""
    cycle_id: int
    input_processed: bool
    proposals_generated: int
    broadcast_occurred: bool
    action_generated: bool
    action: Optional[np.ndarray]
    duration: float
    errors: List[str] = field(default_factory=list)


class CognitiveCore:
    """
    The unified cognitive system.

    Integrates all 20 neuro-modules into a single coherent mind that:
    1. Perceives through sensory interface
    2. Thinks via global workspace competition
    3. Acts via active inference

    The core implements a continuous perception-action loop where
    modules compete for access to the global workspace, and the
    winning content drives action selection.
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or DEFAULT_CONFIG

        # Core components
        self.sensory = SensoryInterface(self.config)
        self.motor = MotorInterface(self.config)
        self.loader = ModuleLoader(self.config)
        self.controller = ActiveInferenceController(self.config)

        # State
        self.state = CognitiveState()
        self._initialized = False

        # Module references (populated after loading)
        self._global_workspace = None
        self._world_model = None
        self._self_improver = None
        self._modules: Dict[str, Any] = {}

        # Statistics
        self._cycle_count = 0
        self._total_time = 0.0
        self._errors: List[str] = []

    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the cognitive system by loading all modules.

        Returns:
            Status dictionary with loaded/failed modules
        """
        if self._initialized:
            return self.loader.get_statistics()

        # Load all modules
        self.loader.load_all()

        # Get key module references
        self._global_workspace = self.loader.get_module("00")
        self._world_model = self.loader.get_module("17")
        self._self_improver = self.loader.get_module("18")

        # Store all loaded modules
        self._modules = self.loader.get_all_loaded()

        # Connect world model to controller
        if self._world_model is not None:
            self.controller.set_world_model(self._world_model)

        # Register modules with global workspace if available
        if self._global_workspace is not None:
            self._register_with_workspace()

        self._initialized = True
        return self.loader.get_statistics()

    def _register_with_workspace(self) -> None:
        """Register loaded modules with the global workspace."""
        if self._global_workspace is None:
            return

        # Check if workspace has register method
        if hasattr(self._global_workspace, 'register_module'):
            for module_id, module in self._modules.items():
                if module_id != "00":  # Don't register workspace with itself
                    try:
                        self._global_workspace.register_module(module)
                    except Exception as e:
                        self._errors.append(f"Failed to register {module_id}: {e}")

    def perceive(self, data: Any, input_type: Optional[InputType] = None) -> SensoryInput:
        """
        Process incoming sensory data.

        Args:
            data: Raw sensory input (vector, text, image, etc.)
            input_type: Type of input (auto-detected if None)

        Returns:
            Processed SensoryInput
        """
        if not self._initialized:
            self.initialize()

        # Process through sensory interface
        sensory_input = self.sensory.process(data, input_type)

        # Update cognitive state
        self.state.last_input = sensory_input.processed
        self.state.timestamp = time.time()

        # Update active inference belief
        if sensory_input.processed is not None:
            self.controller.update_belief(sensory_input.processed)

        # Feed to predictive coding module if available
        pred_coding = self.loader.get_module("01")
        if pred_coding is not None and hasattr(pred_coding, 'process_input'):
            try:
                pred_coding.process_input(sensory_input.processed)
            except Exception:
                pass

        return sensory_input

    def think(self, dt: float = 0.1) -> int:
        """
        Run one cognitive cycle.

        This is the core thinking loop where:
        1. Modules generate proposals
        2. Global workspace runs competition
        3. Winners are broadcast to all modules
        4. World model is updated

        Args:
            dt: Time step duration

        Returns:
            Number of proposals generated
        """
        if not self._initialized:
            self.initialize()

        proposals_count = 0

        # Get current input state
        current_input = self.state.last_input
        if current_input is None:
            current_input = np.zeros(self.config.input_dim)

        # Run global workspace cycle if available
        if self._global_workspace is not None:
            try:
                if hasattr(self._global_workspace, 'step'):
                    result = self._global_workspace.step(current_input)
                    if hasattr(result, 'n_proposals'):
                        proposals_count = result.n_proposals
                    self.state.workspace_contents = []
                    if hasattr(result, 'winning_proposal') and result.winning_proposal:
                        self.state.workspace_contents.append(result.winning_proposal)
            except Exception as e:
                self._errors.append(f"Workspace error: {e}")

        # If no workspace, collect proposals directly from modules
        if proposals_count == 0:
            for module_id, module in self._modules.items():
                if hasattr(module, 'propose'):
                    try:
                        proposals = module.propose(current_input)
                        if proposals:
                            proposals_count += len(proposals)
                    except Exception:
                        pass

        # Update world model
        if self._world_model is not None:
            try:
                if hasattr(self._world_model, 'update'):
                    self._world_model.update(current_input)
                elif hasattr(self._world_model, 'step'):
                    self._world_model.step(current_input)
            except Exception as e:
                self._errors.append(f"World model error: {e}")

        # Process internal module steps
        for module_id, module in self._modules.items():
            if hasattr(module, 'process'):
                try:
                    module.process(dt)
                except Exception:
                    pass

        self._cycle_count += 1
        self.state.cycle_count = self._cycle_count

        return proposals_count

    def act(self, output_type: OutputType = OutputType.VECTOR) -> MotorOutput:
        """
        Generate motor output from current cognitive state.

        Uses active inference to select action that minimizes
        Expected Free Energy.

        Args:
            output_type: Desired output format

        Returns:
            Motor output
        """
        if not self._initialized:
            self.initialize()

        # Get current state for action selection
        current_state = self.state.last_input
        if current_state is None:
            current_state = np.zeros(self.config.input_dim)

        # Use active inference to select action
        action_vector = self.controller.select_action(current_state)

        # Generate motor output
        output = self.motor.generate(action_vector, output_type)

        # Update state
        self.state.last_output = output.value if isinstance(output.value, np.ndarray) else None

        return output

    def run(self, steps: int = 100, input_generator: Optional[callable] = None) -> List[CycleResult]:
        """
        Run the cognitive loop for multiple steps.

        Args:
            steps: Number of cycles to run
            input_generator: Optional function that generates input for each step

        Returns:
            List of cycle results
        """
        if not self._initialized:
            self.initialize()

        results = []

        for i in range(steps):
            start_time = time.time()
            errors = []

            # Generate or get input
            if input_generator is not None:
                try:
                    input_data = input_generator(i)
                    self.perceive(input_data)
                    input_processed = True
                except Exception as e:
                    errors.append(f"Input error: {e}")
                    input_processed = False
            else:
                input_processed = self.state.last_input is not None

            # Think
            try:
                proposals = self.think(self.config.timestep)
            except Exception as e:
                errors.append(f"Think error: {e}")
                proposals = 0

            # Act
            try:
                output = self.act()
                action_generated = True
                action = output.value if isinstance(output.value, np.ndarray) else None
            except Exception as e:
                errors.append(f"Act error: {e}")
                action_generated = False
                action = None

            duration = time.time() - start_time
            self._total_time += duration

            result = CycleResult(
                cycle_id=self._cycle_count,
                input_processed=input_processed,
                proposals_generated=proposals,
                broadcast_occurred=len(self.state.workspace_contents) > 0,
                action_generated=action_generated,
                action=action,
                duration=duration,
                errors=errors,
            )
            results.append(result)

        return results

    def set_goals(self, goals: np.ndarray) -> None:
        """Set the system's goals/preferences."""
        self.state.goals = goals
        self.controller.set_goals(goals)

    def get_state(self) -> CognitiveState:
        """Get current cognitive state."""
        self.state.active_modules = list(self._modules.keys())
        return self.state

    def get_module(self, module_id: str) -> Optional[Any]:
        """Get a specific loaded module."""
        return self._modules.get(module_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        loader_stats = self.loader.get_statistics()
        sensory_stats = self.sensory.get_statistics()
        motor_stats = self.motor.get_statistics()
        controller_stats = self.controller.get_statistics()

        return {
            'initialized': self._initialized,
            'cycle_count': self._cycle_count,
            'total_time': self._total_time,
            'avg_cycle_time': self._total_time / max(1, self._cycle_count),
            'error_count': len(self._errors),
            'recent_errors': self._errors[-5:],
            'loader': loader_stats,
            'sensory': sensory_stats,
            'motor': motor_stats,
            'controller': controller_stats,
        }

    def reset(self) -> None:
        """Reset the cognitive system to initial state."""
        self.sensory.reset()
        self.motor.reset()
        self.controller.reset()

        self.state = CognitiveState()
        self._cycle_count = 0
        self._total_time = 0.0
        self._errors = []

        # Reset modules
        for module in self._modules.values():
            if hasattr(module, 'reset'):
                try:
                    module.reset()
                except Exception:
                    pass

    def shutdown(self) -> None:
        """Shutdown the cognitive system."""
        self.reset()
        self._modules = {}
        self._global_workspace = None
        self._world_model = None
        self._self_improver = None
        self._initialized = False


# Convenience function
def create_cognitive_system(config: Optional[SystemConfig] = None) -> CognitiveCore:
    """Create and initialize a cognitive system."""
    core = CognitiveCore(config)
    core.initialize()
    return core
