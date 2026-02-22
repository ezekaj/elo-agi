"""
Working Memory - Dorsolateral Prefrontal Cortex Simulation

Implements limited-capacity active information maintenance.
Working memory is the "workspace" where System 2 thinking happens.

Key properties:
- LIMITED capacity (~4-7 items, Miller's law)
- Requires ACTIVE maintenance (items decay)
- Enables SERIAL processing
- Information must be refreshed to persist
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import time
from collections import OrderedDict


@dataclass
class MemorySlot:
    """A single item in working memory"""

    id: str
    content: Any
    encoding_time: float
    last_accessed: float
    access_count: int = 0
    activation: float = 1.0
    bindings: Dict[str, Any] = field(default_factory=dict)

    def refresh(self):
        """Refresh this item, resetting decay"""
        self.last_accessed = time.time()
        self.activation = 1.0
        self.access_count += 1


@dataclass
class ChunkedItem:
    """Multiple items grouped as one chunk"""

    items: List[Any]
    chunk_code: str


class WorkingMemory:
    """
    Limited-capacity active information store.

    Simulates dorsolateral prefrontal cortex:
    - Strict capacity limit (default 7, but can be 4-9)
    - Items decay without active maintenance
    - Serial processing bottleneck
    - Can chunk items to fit more in
    """

    def __init__(self, capacity: int = 7, decay_rate: float = 0.1, decay_threshold: float = 0.3):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.decay_threshold = decay_threshold

        # OrderedDict maintains insertion order
        self.slots: OrderedDict[str, MemorySlot] = OrderedDict()
        self._last_decay_time = time.time()

    def store(self, item_id: str, content: Any, priority: float = 0.5) -> MemorySlot:
        """
        Store item in working memory.

        If at capacity, lowest activation item is displaced.
        """
        current_time = time.time()

        # Check if item already exists
        if item_id in self.slots:
            slot = self.slots[item_id]
            slot.content = content
            slot.refresh()
            # Move to end (most recent)
            self.slots.move_to_end(item_id)
            return slot

        # At capacity - need to displace
        if len(self.slots) >= self.capacity:
            self._apply_decay()
            self._displace_lowest()

        # Create new slot
        slot = MemorySlot(
            id=item_id,
            content=content,
            encoding_time=current_time,
            last_accessed=current_time,
            activation=priority,
        )

        self.slots[item_id] = slot
        return slot

    def retrieve(self, item_id: str) -> Optional[Any]:
        """
        Retrieve item from working memory.

        Accessing an item refreshes it (active maintenance).
        """
        if item_id not in self.slots:
            return None

        slot = self.slots[item_id]
        slot.refresh()
        self.slots.move_to_end(item_id)

        return slot.content

    def peek(self, item_id: str) -> Optional[Any]:
        """Look at item without refreshing it"""
        if item_id not in self.slots:
            return None
        return self.slots[item_id].content

    def maintain(self, item_id: str) -> bool:
        """
        Actively maintain item, preventing decay.

        This is the "rehearsal" process that keeps items in WM.
        """
        if item_id not in self.slots:
            return False

        self.slots[item_id].refresh()
        return True

    def maintain_all(self):
        """Refresh all items (costly but possible)"""
        for slot in self.slots.values():
            slot.refresh()

    def _apply_decay(self):
        """Apply time-based decay to all items"""
        current_time = time.time()
        current_time - self._last_decay_time

        to_remove = []
        for item_id, slot in self.slots.items():
            # Decay based on time since last access
            time_since_access = current_time - slot.last_accessed
            decay = self.decay_rate * time_since_access

            slot.activation = max(0.0, slot.activation - decay)

            if slot.activation < self.decay_threshold:
                to_remove.append(item_id)

        # Remove decayed items
        for item_id in to_remove:
            del self.slots[item_id]

        self._last_decay_time = current_time

    def _displace_lowest(self):
        """Remove lowest activation item to make room"""
        if not self.slots:
            return

        # Find lowest activation
        lowest_id = min(self.slots.keys(), key=lambda k: self.slots[k].activation)
        del self.slots[lowest_id]

    def chunk(self, item_ids: List[str], chunk_id: str) -> Optional[MemorySlot]:
        """
        Combine multiple items into a single chunk.

        Chunking lets you hold more information by grouping items.
        E.g., "IBM" is one chunk, not three letters.
        """
        items = []
        for item_id in item_ids:
            if item_id in self.slots:
                items.append(self.slots[item_id].content)
                del self.slots[item_id]

        if not items:
            return None

        chunked = ChunkedItem(items=items, chunk_code=chunk_id)
        return self.store(chunk_id, chunked, priority=0.8)

    def bind(self, item_id: str, binding_name: str, binding_value: Any) -> bool:
        """
        Create binding between WM items.

        Bindings are how relational structures are maintained in WM.
        E.g., bind("X", "role", "agent") means X is the agent.
        """
        if item_id not in self.slots:
            return False

        self.slots[item_id].bindings[binding_name] = binding_value
        return True

    def get_bindings(self, item_id: str) -> Dict[str, Any]:
        """Get all bindings for an item"""
        if item_id not in self.slots:
            return {}
        return self.slots[item_id].bindings.copy()

    def query_by_binding(self, binding_name: str, binding_value: Any) -> List[str]:
        """Find items with specific binding"""
        matches = []
        for item_id, slot in self.slots.items():
            if slot.bindings.get(binding_name) == binding_value:
                matches.append(item_id)
        return matches

    @property
    def current_load(self) -> int:
        """Current number of items in WM"""
        return len(self.slots)

    @property
    def available_slots(self) -> int:
        """How many more items can be stored"""
        return self.capacity - len(self.slots)

    @property
    def is_full(self) -> bool:
        """Check if WM is at capacity"""
        return len(self.slots) >= self.capacity

    def get_all_items(self) -> List[str]:
        """Get IDs of all items currently in WM"""
        self._apply_decay()
        return list(self.slots.keys())

    def get_activations(self) -> Dict[str, float]:
        """Get activation levels of all items"""
        return {k: v.activation for k, v in self.slots.items()}

    def clear(self):
        """Clear working memory (e.g., task switch)"""
        self.slots.clear()

    def focus(self, item_ids: List[str]):
        """
        Focus attention on specific items.

        Focused items get boosted, others decay faster.
        """
        for item_id, slot in self.slots.items():
            if item_id in item_ids:
                slot.activation = min(1.0, slot.activation + 0.3)
                slot.last_accessed = time.time()
            else:
                slot.activation *= 0.8

    def serial_process(self, items: List[str]) -> List[Any]:
        """
        Process items serially (System 2 constraint).

        Returns contents in order, each accessed sequentially.
        This simulates the serial bottleneck of conscious processing.
        """
        results = []
        for item_id in items:
            content = self.retrieve(item_id)  # This refreshes each item
            if content is not None:
                results.append(content)
        return results
