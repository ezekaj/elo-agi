"""
Working Memory: Limited capacity active manipulation

Based on Miller's Law (7±2 items) and research showing ~30 second
decay without active rehearsal.

Brain region: Prefrontal cortex (dorsolateral)
"""

import time
from typing import Optional, Any, List, Callable, Tuple
from dataclasses import dataclass, field
import hashlib


@dataclass
class MemorySlot:
    """Single item in working memory"""
    content: Any
    timestamp: float
    last_rehearsal: float
    activation: float = 1.0
    slot_id: str = field(default_factory=lambda: "")

    def __post_init__(self):
        if not self.slot_id:
            # Generate unique ID from content hash
            content_str = str(self.content)
            self.slot_id = hashlib.md5(content_str.encode()).hexdigest()[:8]


class Chunk:
    """A chunk combining multiple items into one slot"""
    def __init__(self, items: List[Any], label: Optional[str] = None):
        self.items = items
        self.label = label or f"chunk_{len(items)}"

    def __repr__(self):
        return f"Chunk({self.label}: {len(self.items)} items)"

    def __len__(self):
        return len(self.items)


class WorkingMemory:
    """
    Prefrontal cortex - 7±2 items, ~30s active maintenance

    Implements Miller's Law capacity limits with decay and rehearsal.
    Supports chunking to increase effective capacity.
    """

    def __init__(
        self,
        capacity: int = 7,
        decay_time: float = 30.0,
        decay_rate: float = 0.1
    ):
        """
        Initialize working memory.

        Args:
            capacity: Maximum number of slots (default 7, range 5-9)
            decay_time: Seconds until item expires without rehearsal
            decay_rate: Rate of activation decay per second
        """
        self.capacity = max(5, min(9, capacity))  # Enforce 5-9 range
        self.decay_time = decay_time
        self.decay_rate = decay_rate
        self._slots: List[MemorySlot] = []
        self._time_fn = time.time

    def store(self, item: Any) -> bool:
        """
        Add item to working memory.

        Displaces lowest activation item if at capacity.

        Args:
            item: Content to store

        Returns:
            True if stored successfully
        """
        current_time = self._time_fn()

        # Check if item already exists (update instead)
        for slot in self._slots:
            if self._items_equal(slot.content, item):
                slot.last_rehearsal = current_time
                slot.activation = 1.0
                return True

        # Create new slot
        new_slot = MemorySlot(
            content=item,
            timestamp=current_time,
            last_rehearsal=current_time,
            activation=1.0
        )

        # If at capacity, remove lowest activation
        if len(self._slots) >= self.capacity:
            self._decay_all()  # Update activations first
            self._slots.sort(key=lambda s: s.activation)
            self._slots.pop(0)  # Remove lowest

        self._slots.append(new_slot)
        return True

    def _items_equal(self, a: Any, b: Any) -> bool:
        """Check if two items are equal (handles numpy arrays)"""
        try:
            import numpy as np
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                    return bool(np.array_equal(a, b))
                return False
        except ImportError:
            pass
        try:
            result = a == b
            # Handle case where comparison returns array
            if hasattr(result, '__iter__') and not isinstance(result, (str, bytes)):
                import numpy as np
                return bool(np.all(result))
            return bool(result)
        except Exception:
            return False

    def retrieve(self, query: Any) -> Optional[Any]:
        """
        Find matching item and boost its activation.

        Args:
            query: Item to search for (exact or partial match)

        Returns:
            Matching content or None
        """
        self._decay_all()

        for slot in self._slots:
            if self._matches(slot.content, query):
                # Retrieval boosts activation
                slot.activation = min(1.0, slot.activation + 0.3)
                slot.last_rehearsal = self._time_fn()
                return slot.content

        return None

    def _matches(self, content: Any, query: Any) -> bool:
        """Check if content matches query"""
        # Exact match
        if self._items_equal(content, query):
            return True

        # Chunk contains item
        if isinstance(content, Chunk):
            return any(self._items_equal(item, query) for item in content.items)

        # String partial match
        if isinstance(content, str) and isinstance(query, str):
            return query.lower() in content.lower()

        return False

    def rehearse(self, item: Any) -> bool:
        """
        Refresh item's decay timer and boost activation.

        Args:
            item: Item to rehearse

        Returns:
            True if item found and rehearsed
        """
        for slot in self._slots:
            if self._matches(slot.content, item):
                slot.last_rehearsal = self._time_fn()
                slot.activation = 1.0
                return True
        return False

    def manipulate(self, item: Any, operation: Callable[[Any], Any]) -> Optional[Any]:
        """
        Transform an item in place.

        Args:
            item: Item to transform
            operation: Function to apply

        Returns:
            Transformed content or None if not found
        """
        for slot in self._slots:
            if self._matches(slot.content, item):
                slot.content = operation(slot.content)
                slot.last_rehearsal = self._time_fn()
                slot.activation = 1.0
                return slot.content
        return None

    def chunk(self, items: List[Any], label: Optional[str] = None) -> bool:
        """
        Combine multiple items into one slot.

        This increases effective capacity by grouping related items.

        Args:
            items: Items to combine (must exist in WM or be new)
            label: Optional label for the chunk

        Returns:
            True if chunking successful
        """
        # Remove individual items that will be chunked
        items_to_remove = []
        for slot in self._slots:
            if any(self._items_equal(slot.content, item) for item in items):
                items_to_remove.append(slot)

        for slot in items_to_remove:
            self._slots.remove(slot)

        # Create and store the chunk
        chunk = Chunk(items, label)
        return self.store(chunk)

    def _decay_all(self) -> None:
        """Apply decay to all items and remove expired ones"""
        current_time = self._time_fn()
        surviving = []

        for slot in self._slots:
            elapsed = current_time - slot.last_rehearsal

            # Check if completely expired
            if elapsed >= self.decay_time:
                continue

            # Apply activation decay
            slot.activation = max(0.0, slot.activation - self.decay_rate * elapsed)

            # Keep if still has activation
            if slot.activation > 0.01:
                surviving.append(slot)

        self._slots = surviving

    def decay_step(self, dt: float) -> None:
        """
        Advance time and apply decay.

        Args:
            dt: Time step in seconds
        """
        # Simulate time passing
        for slot in self._slots:
            slot.activation = max(0.0, slot.activation - self.decay_rate * dt)

        # Remove completely decayed items
        self._slots = [s for s in self._slots if s.activation > 0.01]

    def get_load(self) -> float:
        """
        Get current capacity usage.

        Returns:
            Ratio of used slots to capacity (0-1)
        """
        self._decay_all()
        return len(self._slots) / self.capacity

    def get_contents(self) -> List[Tuple[Any, float]]:
        """
        List all current items with their activations.

        Returns:
            List of (content, activation) tuples
        """
        self._decay_all()
        return [(slot.content, slot.activation) for slot in self._slots]

    def get_rehearsed_items(self) -> List[Any]:
        """
        Get items that have been rehearsed (activation > 0.5).

        Returns:
            List of well-maintained items
        """
        self._decay_all()
        return [slot.content for slot in self._slots if slot.activation > 0.5]

    def contains(self, item: Any) -> bool:
        """Check if item is in working memory"""
        return self.retrieve(item) is not None

    def clear(self) -> None:
        """Empty working memory"""
        self._slots.clear()

    def __len__(self) -> int:
        self._decay_all()
        return len(self._slots)

    def set_time_function(self, time_fn) -> None:
        """Set custom time function for simulation"""
        self._time_fn = time_fn
