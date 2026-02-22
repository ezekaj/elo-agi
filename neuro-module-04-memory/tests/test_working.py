"""Tests for working memory"""

from neuro.modules.m04_memory.working_memory.working_memory import WorkingMemory, MemorySlot


class TestWorkingMemory:
    """Tests for working memory capacity and decay"""

    def test_capacity_limit(self):
        """Test 7±2 capacity limit"""
        wm = WorkingMemory(capacity=7)

        # Store 10 items
        for i in range(10):
            wm.store(f"item_{i}")

        # Should only have 7
        assert len(wm) == 7

    def test_capacity_range(self):
        """Test capacity enforced to 5-9 range"""
        wm_low = WorkingMemory(capacity=2)
        wm_high = WorkingMemory(capacity=15)

        assert wm_low.capacity == 5
        assert wm_high.capacity == 9

    def test_store_and_retrieve(self):
        """Test basic store and retrieve"""
        wm = WorkingMemory()

        wm.store("test_item")
        result = wm.retrieve("test_item")

        assert result == "test_item"

    def test_decay_without_rehearsal(self):
        """Test items decay without rehearsal"""
        wm = WorkingMemory(capacity=7, decay_time=30.0, decay_rate=0.1)

        current_time = 0.0
        wm.set_time_function(lambda: current_time)

        wm.store("item1")

        # Advance time significantly
        current_time = 60.0  # 60 seconds
        wm.decay_step(60.0)

        # Item should have decayed
        assert len(wm) == 0

    def test_rehearsal_prevents_decay(self):
        """Test rehearsal refreshes item"""
        wm = WorkingMemory(capacity=7, decay_time=30.0, decay_rate=0.05)

        current_time = 0.0
        wm.set_time_function(lambda: current_time)

        wm.store("item1")

        # Advance time
        current_time = 10.0
        wm.rehearse("item1")  # Refresh

        # Should still be there
        assert wm.retrieve("item1") == "item1"

    def test_chunking(self):
        """Test chunking increases effective capacity"""
        wm = WorkingMemory(capacity=7)

        # Store individual items
        items = ["a", "b", "c", "d", "e"]
        for item in items:
            wm.store(item)

        # Chunk some items
        wm.chunk(["a", "b", "c"], label="abc_chunk")

        # Effective capacity increased
        # Now we have: abc_chunk, d, e = 3 slots
        # Can add more
        wm.store("f")
        wm.store("g")
        wm.store("h")
        wm.store("i")
        wm.store("j")

        assert len(wm) == 7

    def test_lowest_activation_displaced(self):
        """Test that lowest activation item is displaced when full"""
        wm = WorkingMemory(capacity=3, decay_rate=0.1)

        current_time = 0.0
        wm.set_time_function(lambda: current_time)

        wm.store("old_item")
        current_time = 5.0
        wm.decay_step(5.0)  # old_item loses activation

        wm.store("new_item1")
        wm.store("new_item2")
        wm.store("new_item3")  # Should displace old_item

        assert wm.retrieve("old_item") is None
        assert wm.retrieve("new_item3") is not None

    def test_manipulate(self):
        """Test in-place manipulation"""
        wm = WorkingMemory()

        wm.store("hello")
        wm.manipulate("hello", lambda x: x.upper())

        result = wm.retrieve("HELLO")
        assert result == "HELLO"

    def test_get_contents(self):
        """Test listing all contents"""
        wm = WorkingMemory()

        wm.store("a")
        wm.store("b")
        wm.store("c")

        contents = wm.get_contents()
        assert len(contents) == 3

        items = [c[0] for c in contents]
        assert "a" in items
        assert "b" in items
        assert "c" in items


class TestMemorySlot:
    """Tests for individual memory slots"""

    def test_slot_creation(self):
        """Test slot is created correctly"""
        slot = MemorySlot(content="test", timestamp=0.0, last_rehearsal=0.0, activation=1.0)

        assert slot.content == "test"
        assert slot.activation == 1.0
        assert slot.slot_id != ""  # Auto-generated ID
