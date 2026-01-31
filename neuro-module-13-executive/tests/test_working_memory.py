"""Tests for working memory system"""

import numpy as np
import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('/', 2)[0])

from src.working_memory import (
    CapacityLimitedStore, DLPFCNetwork, WorkingMemory, WMParams
)


class TestCapacityLimitedStore:
    """Tests for capacity-limited store"""

    def test_initialization(self):
        """Test store initialization"""
        store = CapacityLimitedStore(capacity=4, item_dim=20)

        assert store.capacity == 4
        assert store.slots.shape == (4, 20)
        assert store.get_load() == 0

    def test_encoding(self):
        """Test encoding items"""
        store = CapacityLimitedStore(capacity=4, item_dim=10)

        item = np.random.rand(10)
        slot = store.encode(item)

        assert slot >= 0
        assert store.occupied[slot]
        assert store.get_load() == 1

    def test_capacity_limit(self):
        """Test capacity is enforced"""
        store = CapacityLimitedStore(capacity=3, item_dim=10)

        for i in range(5):
            store.encode(np.random.rand(10))

        assert store.get_load() <= 3

    def test_retrieval(self):
        """Test item retrieval"""
        store = CapacityLimitedStore(capacity=4, item_dim=10)

        item = np.random.rand(10)
        store.encode(item)

        retrieved, strength = store.retrieve(item)

        assert retrieved is not None
        assert strength > 0.5

    def test_decay(self):
        """Test items decay over time"""
        store = CapacityLimitedStore(capacity=4, item_dim=10)
        store.encode(np.random.rand(10))

        initial_activation = store.activation[0]

        for _ in range(100):
            store.update(dt=1.0, decay_rate=0.1)

        assert store.activation[0] < initial_activation

    def test_clear(self):
        """Test clearing store"""
        store = CapacityLimitedStore(capacity=4, item_dim=10)
        store.encode(np.random.rand(10))
        store.encode(np.random.rand(10))

        store.clear()

        assert store.get_load() == 0

    def test_get_contents(self):
        """Test getting all contents"""
        store = CapacityLimitedStore(capacity=4, item_dim=10)
        store.encode(np.random.rand(10))
        store.encode(np.random.rand(10))

        contents = store.get_contents()

        assert len(contents) == 2


class TestDLPFCNetwork:
    """Tests for DLPFC network"""

    def test_initialization(self):
        """Test network initialization"""
        network = DLPFCNetwork(n_units=50)

        assert len(network.activity) == 50
        assert network.W_recurrent.shape == (50, 50)

    def test_encode_pattern(self):
        """Test pattern encoding"""
        network = DLPFCNetwork(n_units=50)

        pattern = np.random.rand(50)
        network.encode_pattern(pattern)

        assert np.sum(network.activity) > 0
        assert len(network.stored_patterns) == 1

    def test_maintenance(self):
        """Test active maintenance"""
        network = DLPFCNetwork(n_units=50)
        pattern = np.random.rand(50)
        network.encode_pattern(pattern)

        # Run maintenance
        for _ in range(10):
            network.maintain(dt=1.0)

        # Activity should persist
        assert np.sum(network.activity) > 0

    def test_capacity_limit(self):
        """Test pattern capacity limit"""
        params = WMParams(capacity=3)
        network = DLPFCNetwork(n_units=50, params=params)

        for _ in range(5):
            network.encode_pattern(np.random.rand(50))

        assert len(network.stored_patterns) <= 3

    def test_clear(self):
        """Test clearing network"""
        network = DLPFCNetwork(n_units=50)
        network.encode_pattern(np.random.rand(50))

        network.clear()

        assert np.sum(network.activity) == 0
        assert len(network.stored_patterns) == 0


class TestWorkingMemory:
    """Tests for integrated working memory"""

    def test_initialization(self):
        """Test WM initialization"""
        wm = WorkingMemory()

        assert wm.store is not None
        assert wm.dlpfc is not None
        assert wm.get_load() == 0

    def test_encode_and_retrieve(self):
        """Test encoding and retrieval"""
        wm = WorkingMemory()

        item = np.random.rand(100)
        wm.encode(item)

        retrieved, strength = wm.retrieve(item)

        assert retrieved is not None
        assert strength > 0

    def test_capacity(self):
        """Test capacity limits"""
        params = WMParams(capacity=3)
        wm = WorkingMemory(params)

        for _ in range(5):
            wm.encode(np.random.rand(100))

        assert wm.get_load() <= 3

    def test_is_full(self):
        """Test full detection"""
        params = WMParams(capacity=2)
        wm = WorkingMemory(params)

        assert not wm.is_full()

        wm.encode(np.random.rand(100))
        wm.encode(np.random.rand(100))

        assert wm.is_full()

    def test_focus_shift(self):
        """Test attentional focus shifting"""
        wm = WorkingMemory()
        wm.encode(np.random.rand(100))
        wm.encode(np.random.rand(100))

        wm.shift_focus(1)

        assert wm.focus == 1

    def test_n_back(self):
        """Test n-back checking"""
        wm = WorkingMemory()

        # Encode same item twice
        item = np.random.rand(100)
        wm.encode(item)
        wm.encode(np.random.rand(100))  # Different item
        wm.encode(item)  # Same as 2-back

        # Check if current matches 2-back
        assert wm.n_back_check(item, n=2)

    def test_update_state(self):
        """Test state updates"""
        wm = WorkingMemory()
        wm.encode(np.random.rand(100))

        initial_load = wm.get_load()
        wm.update_state(dt=1.0)

        # Should still have item (decay is slow)
        assert wm.get_load() <= initial_load


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
