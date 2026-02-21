"""Working Memory System - DLPFC-based active maintenance

Neural basis: Dorsolateral prefrontal cortex (DLPFC)
Key features: Capacity limits, active maintenance, updating, interference
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Any, Tuple


@dataclass
class WMParams:
    """Parameters for working memory"""
    capacity: int = 4  # Cowan's magical number
    n_units: int = 100
    decay_rate: float = 0.05
    interference_strength: float = 0.3
    maintenance_strength: float = 0.1
    noise_level: float = 0.02
    retrieval_threshold: float = 0.3


class CapacityLimitedStore:
    """Capacity-limited storage with slot-based representation"""

    def __init__(self, capacity: int = 4, item_dim: int = 20):
        self.capacity = capacity
        self.item_dim = item_dim

        # Slots for storing items (capacity x item_dim)
        self.slots = np.zeros((capacity, item_dim))
        # Activation level of each slot
        self.activation = np.zeros(capacity)
        # Whether slot is occupied
        self.occupied = np.zeros(capacity, dtype=bool)
        # Time since encoding for each slot
        self.time_since_encoding = np.zeros(capacity)

    def encode(self, item: np.ndarray) -> int:
        """Encode item into working memory

        Args:
            item: Vector representation of item

        Returns:
            Slot index where item was stored, -1 if full
        """
        # Ensure item is right size
        if len(item) != self.item_dim:
            item = np.resize(item, self.item_dim)

        # Find empty slot or replace least active
        empty_slots = np.where(~self.occupied)[0]

        if len(empty_slots) > 0:
            slot = empty_slots[0]
        else:
            # Replace least active item
            slot = np.argmin(self.activation)

        self.slots[slot] = item
        self.activation[slot] = 1.0
        self.occupied[slot] = True
        self.time_since_encoding[slot] = 0

        return slot

    def retrieve(self, cue: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Retrieve item from working memory using cue

        Args:
            cue: Partial or related cue for retrieval

        Returns:
            Retrieved item and match strength, or None
        """
        if len(cue) != self.item_dim:
            cue = np.resize(cue, self.item_dim)

        if not np.any(self.occupied):
            return None, 0.0

        # Compute similarity to all occupied slots
        similarities = np.zeros(self.capacity)
        for i in range(self.capacity):
            if self.occupied[i]:
                # Cosine similarity weighted by activation
                norm_cue = np.linalg.norm(cue)
                norm_slot = np.linalg.norm(self.slots[i])
                if norm_cue > 0 and norm_slot > 0:
                    similarities[i] = (np.dot(cue, self.slots[i]) /
                                      (norm_cue * norm_slot)) * self.activation[i]

        best_slot = np.argmax(similarities)
        if similarities[best_slot] > 0.1:  # Minimum threshold
            # Boost activation of retrieved item
            self.activation[best_slot] = min(1.0, self.activation[best_slot] + 0.2)
            return self.slots[best_slot].copy(), similarities[best_slot]

        return None, 0.0

    def update(self, dt: float = 1.0, decay_rate: float = 0.05):
        """Update working memory state (decay and time)"""
        self.time_since_encoding += dt

        # Activation decays over time
        self.activation *= (1 - decay_rate * dt)

        # Items with very low activation are forgotten
        forgotten = self.activation < 0.1
        self.occupied[forgotten] = False
        self.slots[forgotten] = 0
        self.activation[forgotten] = 0

    def get_contents(self) -> List[np.ndarray]:
        """Get all items currently in working memory"""
        return [self.slots[i].copy() for i in range(self.capacity) if self.occupied[i]]

    def clear(self):
        """Clear working memory"""
        self.slots = np.zeros((self.capacity, self.item_dim))
        self.activation = np.zeros(self.capacity)
        self.occupied = np.zeros(self.capacity, dtype=bool)
        self.time_since_encoding = np.zeros(self.capacity)

    def get_load(self) -> int:
        """Get current number of items in WM"""
        return int(np.sum(self.occupied))


class DLPFCNetwork:
    """DLPFC network for active maintenance

    Implements sustained firing to maintain representations
    """

    def __init__(self, n_units: int = 100, params: Optional[WMParams] = None):
        self.n_units = n_units
        self.params = params or WMParams()

        # Neural activity
        self.activity = np.zeros(n_units)
        # Recurrent weights for maintenance
        self.W_recurrent = np.random.randn(n_units, n_units) * 0.1
        # Make recurrent weights support stable activity
        self.W_recurrent = (self.W_recurrent + self.W_recurrent.T) / 2
        np.fill_diagonal(self.W_recurrent, 0.5)  # Self-excitation

        # Stored patterns
        self.stored_patterns = []

    def encode_pattern(self, pattern: np.ndarray):
        """Encode a pattern for active maintenance"""
        if len(pattern) != self.n_units:
            pattern = np.resize(pattern, self.n_units)

        # Add pattern to activity
        self.activity = np.clip(self.activity + pattern, 0, 1)
        self.stored_patterns.append(pattern.copy())

        # Limit stored patterns to capacity
        if len(self.stored_patterns) > self.params.capacity:
            self.stored_patterns.pop(0)

    def maintain(self, dt: float = 1.0):
        """Active maintenance through recurrent activity"""
        # Recurrent dynamics
        input_current = np.dot(self.W_recurrent, self.activity)

        # Add maintenance input for stored patterns
        maintenance_signal = np.zeros(self.n_units)
        for pattern in self.stored_patterns:
            maintenance_signal += pattern * self.params.maintenance_strength

        # Update activity
        d_activity = (-self.params.decay_rate * self.activity +
                     0.5 * np.tanh(input_current + maintenance_signal) +
                     np.random.randn(self.n_units) * self.params.noise_level)

        self.activity = np.clip(self.activity + d_activity * dt, 0, 1)

    def get_activity(self) -> np.ndarray:
        """Get current DLPFC activity"""
        return self.activity.copy()

    def clear(self):
        """Clear maintained representations"""
        self.activity = np.zeros(self.n_units)
        self.stored_patterns = []


class WorkingMemory:
    """Integrated working memory system

    Combines capacity-limited store with DLPFC maintenance
    """

    def __init__(self, params: Optional[WMParams] = None):
        self.params = params or WMParams()

        # Components
        self.store = CapacityLimitedStore(
            capacity=self.params.capacity,
            item_dim=self.params.n_units
        )
        self.dlpfc = DLPFCNetwork(self.params.n_units, self.params)

        # Focus of attention (which slot is attended)
        self.focus = 0

    def encode(self, item: np.ndarray) -> int:
        """Encode item into working memory"""
        slot = self.store.encode(item)
        if slot >= 0:
            self.dlpfc.encode_pattern(item)
        return slot

    def retrieve(self, cue: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Retrieve item using cue"""
        return self.store.retrieve(cue)

    def update_state(self, dt: float = 1.0):
        """Update working memory state"""
        self.store.update(dt, self.params.decay_rate)
        self.dlpfc.maintain(dt)

        # Add interference between items
        load = self.store.get_load()
        if load > 1:
            interference = self.params.interference_strength * (load - 1) / self.params.capacity
            self.store.activation *= (1 - interference * dt)

    def get_load(self) -> int:
        """Get current WM load"""
        return self.store.get_load()

    def is_full(self) -> bool:
        """Check if WM is at capacity"""
        return self.get_load() >= self.params.capacity

    def get_contents(self) -> List[np.ndarray]:
        """Get all items in WM"""
        return self.store.get_contents()

    def clear(self):
        """Clear working memory"""
        self.store.clear()
        self.dlpfc.clear()

    def shift_focus(self, slot: int):
        """Shift attentional focus to slot"""
        if 0 <= slot < self.params.capacity:
            self.focus = slot
            # Boost activation of focused item
            if self.store.occupied[slot]:
                self.store.activation[slot] = min(1.0, self.store.activation[slot] + 0.3)

    def get_focused_item(self) -> Optional[np.ndarray]:
        """Get currently focused item"""
        if self.store.occupied[self.focus]:
            return self.store.slots[self.focus].copy()
        return None

    def n_back_check(self, item: np.ndarray, n: int = 2) -> bool:
        """Check if item matches n-back item

        Args:
            item: Current item
            n: How many items back to compare

        Returns:
            True if match
        """
        patterns = self.dlpfc.stored_patterns
        if len(patterns) >= n + 1:
            n_back_item = patterns[-(n + 1)]
            similarity = np.dot(item, n_back_item) / (
                np.linalg.norm(item) * np.linalg.norm(n_back_item) + 1e-8
            )
            return similarity > 0.8
        return False
