"""
System Updater: Applies validated improvements to the system.

The updater is responsible for safely applying verified modifications
to the running system, maintaining rollback capabilities.

Based on:
- Hot-swapping in production systems
- Transactional updates
- Version control for neural networks
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import numpy as np
import time
import copy


from .generator import Modification, ModificationType
from .verifier import VerificationResult


class UpdateStatus(Enum):
    """Status of an update."""

    PENDING = "pending"
    APPLYING = "applying"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class UpdaterParams:
    """Parameters for the updater."""

    max_concurrent_updates: int = 1  # Max simultaneous updates
    checkpoint_interval: int = 10  # Create checkpoint every N updates
    max_rollback_history: int = 100  # Max rollback points to keep
    gradual_application: bool = True  # Apply changes gradually
    gradual_steps: int = 5  # Steps for gradual application


@dataclass
class Checkpoint:
    """A system checkpoint for rollback."""

    checkpoint_id: str
    timestamp: float
    state_snapshot: Dict[str, Any]
    modifications_applied: List[str]
    performance: float


@dataclass
class UpdateResult:
    """Result of an update operation."""

    modification: Modification
    status: UpdateStatus
    applied_at: Optional[float]
    rollback_available: bool
    checkpoint_id: Optional[str]
    performance_delta: float
    details: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class SystemUpdater:
    """
    Updater that applies modifications to the system.

    The updater ensures safe application of changes with:

    1. **Checkpointing**: Save state before updates for rollback
    2. **Gradual application**: Apply changes incrementally
    3. **Monitoring**: Track effects during application
    4. **Rollback**: Revert to previous state if needed

    This is the component that actually modifies the running system.
    """

    def __init__(self, params: Optional[UpdaterParams] = None):
        self.params = params or UpdaterParams()

        # Checkpoints for rollback
        self._checkpoints: List[Checkpoint] = []
        self._checkpoint_counter = 0

        # Currently active updates
        self._active_updates: Dict[str, UpdateResult] = {}

        # History
        self._update_history: List[UpdateResult] = []

        # Component modification handlers
        self._modification_handlers: Dict[str, Callable[[Modification, float], bool]] = {}

        # State accessor (for checkpointing)
        self._get_state: Optional[Callable[[], Dict[str, Any]]] = None
        self._set_state: Optional[Callable[[Dict[str, Any]], None]] = None

        # Performance monitor
        self._get_performance: Optional[Callable[[], float]] = None

    def register_handler(
        self,
        component: str,
        handler: Callable[[Modification, float], bool],
    ) -> None:
        """
        Register a modification handler for a component.

        Handler takes (modification, blend_factor) and returns success.
        blend_factor goes from 0 to 1 for gradual application.
        """
        self._modification_handlers[component] = handler

    def set_state_accessors(
        self,
        get_state: Callable[[], Dict[str, Any]],
        set_state: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Set functions to get/set system state."""
        self._get_state = get_state
        self._set_state = set_state

    def set_performance_monitor(
        self,
        get_performance: Callable[[], float],
    ) -> None:
        """Set function to get current performance."""
        self._get_performance = get_performance

    def create_checkpoint(self) -> Checkpoint:
        """Create a checkpoint of current system state."""
        if self._get_state is None:
            state = {}
        else:
            state = copy.deepcopy(self._get_state())

        if self._get_performance is not None:
            performance = self._get_performance()
        else:
            performance = 0.0

        self._checkpoint_counter += 1
        checkpoint = Checkpoint(
            checkpoint_id=f"ckpt_{self._checkpoint_counter}",
            timestamp=time.time(),
            state_snapshot=state,
            modifications_applied=[u.modification.mod_id for u in self._update_history],
            performance=performance,
        )

        self._checkpoints.append(checkpoint)

        # Limit checkpoint history
        if len(self._checkpoints) > self.params.max_rollback_history:
            self._checkpoints.pop(0)

        return checkpoint

    def apply(
        self,
        modification: Modification,
        verification: Optional[VerificationResult] = None,
    ) -> UpdateResult:
        """
        Apply a modification to the system.

        Args:
            modification: The modification to apply
            verification: Optional verification result

        Returns:
            UpdateResult
        """
        # Check for concurrent update limit
        active_count = sum(
            1 for u in self._active_updates.values() if u.status == UpdateStatus.APPLYING
        )
        if active_count >= self.params.max_concurrent_updates:
            return UpdateResult(
                modification=modification,
                status=UpdateStatus.PENDING,
                applied_at=None,
                rollback_available=False,
                checkpoint_id=None,
                performance_delta=0.0,
                details={"reason": "concurrent_limit"},
            )

        # Create checkpoint if interval reached
        if len(self._update_history) % self.params.checkpoint_interval == 0:
            checkpoint = self.create_checkpoint()
        else:
            checkpoint = self._checkpoints[-1] if self._checkpoints else self.create_checkpoint()

        # Get baseline performance
        if self._get_performance is not None:
            baseline = self._get_performance()
        else:
            baseline = 0.0

        # Apply the modification
        result = UpdateResult(
            modification=modification,
            status=UpdateStatus.APPLYING,
            applied_at=None,
            rollback_available=True,
            checkpoint_id=checkpoint.checkpoint_id,
            performance_delta=0.0,
            details={},
        )
        self._active_updates[modification.mod_id] = result

        try:
            if self.params.gradual_application:
                success = self._gradual_apply(modification)
            else:
                success = self._direct_apply(modification, 1.0)

            if success:
                result.status = UpdateStatus.APPLIED
                result.applied_at = time.time()

                # Measure performance delta
                if self._get_performance is not None:
                    new_performance = self._get_performance()
                    result.performance_delta = new_performance - baseline
            else:
                result.status = UpdateStatus.FAILED
                result.details["reason"] = "handler_failed"

        except Exception as e:
            result.status = UpdateStatus.FAILED
            result.details["error"] = str(e)

        # Move to history
        del self._active_updates[modification.mod_id]
        self._update_history.append(result)

        return result

    def _direct_apply(
        self,
        modification: Modification,
        blend_factor: float,
    ) -> bool:
        """Apply modification directly."""
        handler = self._modification_handlers.get(modification.target_component)

        if handler is None:
            # No specific handler - apply generically
            return self._generic_apply(modification, blend_factor)

        return handler(modification, blend_factor)

    def _gradual_apply(self, modification: Modification) -> bool:
        """Apply modification gradually over multiple steps."""
        for step in range(self.params.gradual_steps):
            blend_factor = (step + 1) / self.params.gradual_steps

            success = self._direct_apply(modification, blend_factor)
            if not success:
                return False

            # Brief pause between steps
            time.sleep(0.001)

        return True

    def _generic_apply(
        self,
        modification: Modification,
        blend_factor: float,
    ) -> bool:
        """Generic modification application."""
        # Generic handling based on modification type
        if modification.mod_type == ModificationType.WEIGHT_ADJUSTMENT:
            return self._apply_weight_adjustment(modification, blend_factor)
        elif modification.mod_type == ModificationType.HYPERPARAMETER:
            return self._apply_hyperparameter(modification, blend_factor)
        elif modification.mod_type == ModificationType.LEARNING_RATE:
            return self._apply_learning_rate(modification, blend_factor)
        else:
            # Unknown type - report as applied but do nothing
            return True

    def _apply_weight_adjustment(
        self,
        modification: Modification,
        blend_factor: float,
    ) -> bool:
        """Apply weight adjustment modification."""
        if self._get_state is None or self._set_state is None:
            return True

        state = self._get_state()
        scale = modification.changes.get("adjustment_scale", 0.1) * blend_factor

        # Apply small random adjustment (placeholder)
        if "weights" in state:
            state["weights"] = state["weights"] * (1 + scale * np.random.randn())

        self._set_state(state)
        return True

    def _apply_hyperparameter(
        self,
        modification: Modification,
        blend_factor: float,
    ) -> bool:
        """Apply hyperparameter modification."""
        if self._get_state is None or self._set_state is None:
            return True

        state = self._get_state()

        for key, value in modification.changes.items():
            if key in state and isinstance(value, (int, float)):
                # Blend toward new value
                old_value = state[key]
                state[key] = old_value + blend_factor * (value - old_value)

        self._set_state(state)
        return True

    def _apply_learning_rate(
        self,
        modification: Modification,
        blend_factor: float,
    ) -> bool:
        """Apply learning rate modification."""
        if self._get_state is None or self._set_state is None:
            return True

        state = self._get_state()

        if "learning_rate" in state:
            multiplier = modification.changes.get("multiplier", 1.0)
            state["learning_rate"] *= 1 + (multiplier - 1) * blend_factor

        self._set_state(state)
        return True

    def rollback(self, checkpoint_id: Optional[str] = None) -> bool:
        """
        Rollback to a checkpoint.

        Args:
            checkpoint_id: Specific checkpoint to rollback to (None = most recent)

        Returns:
            Success status
        """
        if not self._checkpoints:
            return False

        if checkpoint_id is None:
            checkpoint = self._checkpoints[-1]
        else:
            checkpoint = next(
                (c for c in self._checkpoints if c.checkpoint_id == checkpoint_id), None
            )
            if checkpoint is None:
                return False

        if self._set_state is not None:
            self._set_state(copy.deepcopy(checkpoint.state_snapshot))

        # Mark relevant updates as rolled back
        for update in self._update_history:
            if update.applied_at and update.applied_at > checkpoint.timestamp:
                update.status = UpdateStatus.ROLLED_BACK

        return True

    def rollback_modification(self, mod_id: str) -> bool:
        """Rollback a specific modification."""
        update = next((u for u in self._update_history if u.modification.mod_id == mod_id), None)

        if update is None or update.checkpoint_id is None:
            return False

        return self.rollback(update.checkpoint_id)

    def get_applied_modifications(self) -> List[Modification]:
        """Get list of currently applied modifications."""
        return [u.modification for u in self._update_history if u.status == UpdateStatus.APPLIED]

    def get_update_status(self, mod_id: str) -> Optional[UpdateStatus]:
        """Get status of a modification."""
        if mod_id in self._active_updates:
            return self._active_updates[mod_id].status

        for update in self._update_history:
            if update.modification.mod_id == mod_id:
                return update.status

        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get updater statistics."""
        if not self._update_history:
            return {
                "n_updates": 0,
                "success_rate": 0.0,
                "n_checkpoints": len(self._checkpoints),
            }

        applied = [u for u in self._update_history if u.status == UpdateStatus.APPLIED]
        failed = [u for u in self._update_history if u.status == UpdateStatus.FAILED]
        rolled_back = [u for u in self._update_history if u.status == UpdateStatus.ROLLED_BACK]

        return {
            "n_updates": len(self._update_history),
            "n_applied": len(applied),
            "n_failed": len(failed),
            "n_rolled_back": len(rolled_back),
            "success_rate": len(applied) / len(self._update_history),
            "avg_performance_delta": float(np.mean([u.performance_delta for u in applied]))
            if applied
            else 0.0,
            "n_checkpoints": len(self._checkpoints),
            "n_active": len(self._active_updates),
        }

    def reset(self) -> None:
        """Reset updater state."""
        self._checkpoints = []
        self._checkpoint_counter = 0
        self._active_updates = {}
        self._update_history = []
