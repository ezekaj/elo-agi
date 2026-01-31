#!/usr/bin/env python3
"""
STRESS TEST: Comprehensive testing of Module 04 Memory Systems

Tests include:
- Capacity limits under stress
- Timing precision
- Pattern completion accuracy
- Consolidation effectiveness
- Interference effects
- Edge cases
"""

import sys
import time
import numpy as np
from collections import defaultdict

# Ensure we can import from src
sys.path.insert(0, '.')

from src import (
    MemoryController,
    IconicBuffer,
    EchoicBuffer,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    ProceduralMemory,
    Engram,
)
from src.memory_processes import (
    MemoryEncoder,
    MemoryConsolidator,
    MemoryRetriever,
    Forgetter,
)


class TestResults:
    """Track test results"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, name, passed, error=None):
        if passed:
            self.passed += 1
            print(f"  ✓ {name}")
        else:
            self.failed += 1
            self.errors.append((name, error))
            print(f"  ✗ {name}: {error}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"RESULTS: {self.passed}/{total} passed ({100*self.passed/total:.1f}%)")
        if self.errors:
            print(f"\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")
        print(f"{'='*60}")
        return self.failed == 0


results = TestResults()


def test_section(name):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")


# =============================================================================
# 1. SENSORY MEMORY STRESS TESTS
# =============================================================================

def test_sensory_memory():
    test_section("SENSORY MEMORY STRESS TESTS")

    # Test 1: Iconic buffer timing precision
    try:
        iconic = IconicBuffer(decay_time=0.250)
        sim_time = 0.0
        iconic.set_time_function(lambda: sim_time)

        iconic.capture(np.random.rand(100, 100))

        # Check at precise intervals
        times_and_expected = [
            (0.0, True, 1.0),
            (0.125, True, 0.606),  # e^(-0.5) ≈ 0.606
            (0.250, True, 0.368),  # e^(-1) ≈ 0.368
            (0.500, True, 0.135),  # e^(-2) ≈ 0.135
            (1.250, False, 0.007), # e^(-5) ≈ 0.007
        ]

        all_correct = True
        for t, exp_avail, exp_strength in times_and_expected:
            sim_time = t
            _, strength = iconic.read_with_strength()
            if abs(strength - exp_strength) > 0.05:
                all_correct = False

        results.record("Iconic timing precision", all_correct,
                      "Strength decay doesn't match exponential")
    except Exception as e:
        results.record("Iconic timing precision", False, str(e))

    # Test 2: Echoic buffer rolling window
    try:
        echoic = EchoicBuffer(decay_time=3.5)
        sim_time = 0.0
        echoic.set_time_function(lambda: sim_time)

        # Add 10 chunks over 5 seconds
        for i in range(10):
            sim_time = i * 0.5
            echoic.capture(np.array([i] * 100), duration=0.5)

        # At t=5.0, chunks from t=0,0.5,1.0 should be gone (>3.5s old)
        sim_time = 5.0
        echoic.decay(sim_time)

        # Should have 7 chunks (from t=1.5 onwards)
        count = echoic.get_segment_count()
        results.record("Echoic rolling window", count == 7,
                      f"Expected 7 segments, got {count}")
    except Exception as e:
        results.record("Echoic rolling window", False, str(e))

    # Test 3: Large data handling
    try:
        iconic = IconicBuffer()
        large_data = np.random.rand(1000, 1000)  # 1M elements
        iconic.capture(large_data)
        retrieved = iconic.read()
        results.record("Large data handling (1M elements)",
                      retrieved is not None and retrieved.shape == (1000, 1000))
    except Exception as e:
        results.record("Large data handling (1M elements)", False, str(e))


# =============================================================================
# 2. WORKING MEMORY STRESS TESTS
# =============================================================================

def test_working_memory():
    test_section("WORKING MEMORY STRESS TESTS")

    # Test 1: Strict capacity enforcement
    try:
        wm = WorkingMemory(capacity=7)

        # Try to store 100 items rapidly
        for i in range(100):
            wm.store(f"item_{i}")

        # Must never exceed capacity
        results.record("Capacity never exceeded (100 stores)",
                      len(wm) == 7, f"Got {len(wm)} items")
    except Exception as e:
        results.record("Capacity never exceeded (100 stores)", False, str(e))

    # Test 2: Decay timing accuracy
    try:
        # decay_time=25.0 means item expires completely after 25s from last rehearsal
        # At 15s: within decay_time, should survive
        # At 30s: past decay_time, should be gone
        wm = WorkingMemory(capacity=7, decay_time=25.0, decay_rate=0.02)
        sim_time = [0.0]  # Use list for closure mutability
        wm.set_time_function(lambda: sim_time[0])

        wm.store("test_item")

        # Item should survive at t=15 (within decay_time=25)
        sim_time[0] = 15.0
        survived_15 = len(wm) == 1

        # Item should be gone at t=30 (past decay_time=25)
        sim_time[0] = 30.0
        gone_30 = len(wm) == 0

        results.record("Decay timing (15s survive, 30s gone)",
                      survived_15 and gone_30,
                      f"15s: {survived_15}, 30s: {gone_30}")
    except Exception as e:
        results.record("Decay timing (15s survive, 30s gone)", False, str(e))

    # Test 3: Chunking effectiveness
    try:
        wm = WorkingMemory(capacity=7)

        # Store items and chunk them before capacity forces displacement
        # First, store 3 items
        wm.store("a1")
        wm.store("a2")
        wm.store("a3")
        # Chunk them into 1 slot
        wm.chunk(["a1", "a2", "a3"], label="chunk_a")

        # Now store 3 more
        wm.store("b1")
        wm.store("b2")
        wm.store("b3")
        # Chunk them
        wm.chunk(["b1", "b2", "b3"], label="chunk_b")

        # Store 3 more
        wm.store("c1")
        wm.store("c2")
        wm.store("c3")
        # Chunk them
        wm.chunk(["c1", "c2", "c3"], label="chunk_c")

        # Should have 3 chunks, each containing 3 items = 9 effective items
        chunk_count = len(wm)
        effective_items = sum(
            len(c[0].items) if hasattr(c[0], 'items') else 1
            for c in wm.get_contents()
        )

        results.record("Chunking (9 items → 3 chunks)",
                      chunk_count == 3 and effective_items == 9,
                      f"Got {chunk_count} chunks, {effective_items} effective items")
    except Exception as e:
        results.record("Chunking (9 items → 3 chunks)", False, str(e))

    # Test 4: Activation-based displacement
    try:
        wm = WorkingMemory(capacity=3, decay_rate=0.1)
        sim_time = 0.0
        wm.set_time_function(lambda: sim_time)

        wm.store("old")
        sim_time = 5.0
        wm.decay_step(5.0)  # old item decays

        wm.store("new1")
        wm.store("new2")
        wm.store("new3")  # Should displace "old"

        old_gone = wm.retrieve("old") is None
        new3_present = wm.retrieve("new3") is not None

        results.record("Lowest activation displaced",
                      old_gone and new3_present,
                      f"old gone: {old_gone}, new3 present: {new3_present}")
    except Exception as e:
        results.record("Lowest activation displaced", False, str(e))

    # Test 5: Rehearsal prevents decay
    try:
        # decay_rate=0.05 means 0.05 per second
        # After 10s without rehearsal: 1.0 - 0.05*10 = 0.5
        # After 20s without rehearsal: 1.0 - 0.05*20 = 0.0 (gone)
        wm = WorkingMemory(capacity=7, decay_time=30.0, decay_rate=0.05)
        sim_time = 0.0
        wm.set_time_function(lambda: sim_time)

        wm.store("rehearsed")
        wm.store("forgotten")

        # Rehearse one item every 5 seconds (resets activation to 1.0)
        # Forgotten item decays: after 25s at rate 0.05 = 1.25 total decay (gone)
        for step in range(5):
            sim_time = (step + 1) * 5  # 5, 10, 15, 20, 25
            wm.decay_step(5.0)
            wm.rehearse("rehearsed")  # Reset rehearsed to 1.0

        rehearsed_present = wm.retrieve("rehearsed") is not None
        forgotten_gone = wm.retrieve("forgotten") is None

        results.record("Rehearsal prevents decay over 25s",
                      rehearsed_present and forgotten_gone,
                      f"rehearsed: {rehearsed_present}, forgotten: {forgotten_gone}")
    except Exception as e:
        results.record("Rehearsal prevents decay over 25s", False, str(e))


# =============================================================================
# 3. LONG-TERM MEMORY STRESS TESTS
# =============================================================================

def test_long_term_memory():
    test_section("LONG-TERM MEMORY STRESS TESTS")

    # Test 1: Episodic - Mass encoding
    try:
        em = EpisodicMemory()

        # Encode 1000 episodes
        for i in range(1000):
            em.encode(
                f"event_{i}",
                context={"location": f"place_{i%10}", "category": i % 5},
                emotional_valence=(i % 10) / 10 - 0.5
            )

        results.record("Episodic mass encoding (1000 episodes)",
                      len(em) == 1000, f"Got {len(em)}")
    except Exception as e:
        results.record("Episodic mass encoding (1000 episodes)", False, str(e))

    # Test 2: Episodic - Context retrieval accuracy
    try:
        em = EpisodicMemory()

        # Create episodes with specific contexts
        for i in range(100):
            em.encode(f"event_{i}", context={"location": f"place_{i%5}"})

        # Retrieve by context
        place_0 = em.retrieve_by_context({"location": "place_0"})

        # Should get exactly 20 episodes (100/5)
        results.record("Context retrieval accuracy",
                      len(place_0) == 20, f"Expected 20, got {len(place_0)}")
    except Exception as e:
        results.record("Context retrieval accuracy", False, str(e))

    # Test 3: Semantic - Network traversal
    try:
        sm = SemanticMemory()

        # Create chain: A -> B -> C -> D -> E
        sm.create_concept("A", relations=[("causes", "B", 1.0)])
        sm.create_concept("B", relations=[("causes", "C", 1.0)])
        sm.create_concept("C", relations=[("causes", "D", 1.0)])
        sm.create_concept("D", relations=[("causes", "E", 1.0)])
        sm.create_concept("E")

        path = sm.find_path("A", "E", max_depth=10)

        results.record("Semantic path finding (A→E, 4 hops)",
                      path is not None and len(path) == 4,
                      f"Path: {path}")
    except Exception as e:
        results.record("Semantic path finding (A→E, 4 hops)", False, str(e))

    # Test 4: Semantic - Spreading activation depth
    try:
        sm = SemanticMemory()

        # Create star network
        sm.create_concept("center", relations=[
            ("related-to", f"spoke_{i}", 0.8) for i in range(10)
        ])
        for i in range(10):
            sm.create_concept(f"spoke_{i}", relations=[
                ("related-to", f"outer_{i}_{j}", 0.5) for j in range(5)
            ])
            for j in range(5):
                sm.create_concept(f"outer_{i}_{j}")

        # Spread activation from center
        activations = sm.spread_activation("center", depth=3)

        # Should activate center, all spokes, and some outer
        center_active = activations.get("center", 0) > 0.9
        spokes_active = all(activations.get(f"spoke_{i}", 0) > 0.3 for i in range(10))

        results.record("Spreading activation (61 nodes)",
                      center_active and spokes_active,
                      f"Center: {activations.get('center', 0):.2f}")
    except Exception as e:
        results.record("Spreading activation (61 nodes)", False, str(e))

    # Test 5: Procedural - Competition resolution
    try:
        pm = ProceduralMemory()

        # Create competing procedures with different strengths
        for i in range(10):
            proc = pm.encode_simple(
                f"proc_{i}",
                {"trigger": "go"},
                [f"action_{i}"]
            )
            proc.strength = i / 10  # 0.0 to 0.9

        # Execute - should pick highest strength
        matches = pm.get_matching({"trigger": "go"})
        winner = pm.compete(matches)

        results.record("Procedural competition (10 competing)",
                      winner.name == "proc_9",
                      f"Winner: {winner.name}")
    except Exception as e:
        results.record("Procedural competition (10 competing)", False, str(e))


# =============================================================================
# 4. ENGRAM STRESS TESTS
# =============================================================================

def test_engram():
    test_section("ENGRAM STRESS TESTS")

    # Test 1: Pattern completion accuracy
    try:
        np.random.seed(42)
        engram = Engram(n_neurons=200, connectivity=0.2, learning_rate=0.3)

        # Create distinct pattern
        pattern = np.zeros(200)
        pattern[0:50] = 1.0  # Region A
        pattern[100:150] = 1.0  # Region B

        engram.encode(pattern)
        for _ in range(10):
            engram.consolidate()

        # Test with 25% cue
        partial = np.zeros(200)
        partial[0:25] = 1.0  # Only half of Region A

        recalled = engram.reactivate(partial, iterations=20)

        # Check if Region B activated
        region_b_activation = np.mean(recalled[100:150])

        results.record("Pattern completion (25% cue → full recall)",
                      region_b_activation > 0.3,
                      f"Region B activation: {region_b_activation:.2f}")
    except Exception as e:
        results.record("Pattern completion (25% cue → full recall)", False, str(e))

    # Test 2: Hebbian learning strengthening
    try:
        np.random.seed(42)
        engram = Engram(n_neurons=50, connectivity=0.5, learning_rate=0.5)

        # Get initial connection strength between neurons 0 and 1
        initial_weight = engram.neurons[0].connections.get(1, 0)

        # Co-activate neurons 0 and 1 repeatedly
        for _ in range(20):
            engram.neurons[0].activate(1.0)
            engram.neurons[1].activate(1.0)
            engram.hebbian_update(0.5)

        final_weight = engram.neurons[0].connections.get(1, 0)

        results.record("Hebbian strengthening (20 co-activations)",
                      final_weight > initial_weight,
                      f"Initial: {initial_weight:.3f}, Final: {final_weight:.3f}")
    except Exception as e:
        results.record("Hebbian strengthening (20 co-activations)", False, str(e))

    # Test 3: Pruning effectiveness
    try:
        np.random.seed(42)
        engram = Engram(n_neurons=100, connectivity=0.5)

        initial_connections = sum(len(n.connections) for n in engram.neurons)

        # Aggressive pruning
        engram.prune(threshold=0.3)

        final_connections = sum(len(n.connections) for n in engram.neurons)
        reduction = (initial_connections - final_connections) / initial_connections

        results.record("Pruning removes >50% of connections",
                      reduction > 0.5,
                      f"Removed {reduction*100:.1f}%")
    except Exception as e:
        results.record("Pruning removes >50% of connections", False, str(e))

    # Test 4: State transitions
    try:
        engram = Engram(n_neurons=50)
        pattern = np.random.rand(50)

        # LABILE after encode
        engram.encode(pattern)
        labile_after_encode = engram.state.value == "labile"

        # CONSOLIDATED after consolidate
        engram.consolidate()
        consolidated_after = engram.state.value == "consolidated"

        # REACTIVATED after destabilize
        engram.destabilize()
        reactivated_after = engram.state.value == "reactivated"

        # CONSOLIDATED after restabilize
        engram.restabilize()
        consolidated_final = engram.state.value == "consolidated"

        all_correct = all([labile_after_encode, consolidated_after,
                          reactivated_after, consolidated_final])

        results.record("State transitions (labile→consolidated→reactivated→consolidated)",
                      all_correct)
    except Exception as e:
        results.record("State transitions", False, str(e))

    # Test 5: Multiple pattern storage (associative memory)
    try:
        np.random.seed(42)
        engram = Engram(n_neurons=200, connectivity=0.15, learning_rate=0.2)

        # Store 5 distinct patterns
        patterns = []
        for i in range(5):
            p = np.zeros(200)
            p[i*40:(i+1)*40] = 1.0  # Non-overlapping regions
            patterns.append(p)
            engram.encode(p)
            engram.consolidate()
            engram.reset_activations()

        # Test recall of each pattern
        recalls_successful = 0
        for i, pattern in enumerate(patterns):
            cue = np.zeros(200)
            cue[i*40:i*40+20] = 1.0  # 50% of pattern as cue

            recalled = engram.reactivate(cue, iterations=15)
            similarity = np.corrcoef(pattern, recalled)[0, 1]
            if similarity > 0.5:
                recalls_successful += 1

        results.record("Multiple pattern storage (5 patterns)",
                      recalls_successful >= 3,
                      f"Recalled {recalls_successful}/5 patterns")
    except Exception as e:
        results.record("Multiple pattern storage (5 patterns)", False, str(e))


# =============================================================================
# 5. MEMORY PROCESSES STRESS TESTS
# =============================================================================

def test_memory_processes():
    test_section("MEMORY PROCESSES STRESS TESTS")

    # Test 1: Encoder modality handling
    try:
        encoder = MemoryEncoder(pattern_size=100)

        visual = encoder.encode_visual(np.random.rand(50, 50))
        auditory = encoder.encode_auditory(np.random.rand(1000))
        semantic = encoder.encode_semantic("The quick brown fox jumps over the lazy dog")
        episodic = encoder.encode_episodic("event", {"context": "test"})

        all_correct_size = all(
            len(enc.pattern) == 100
            for enc in [visual, auditory, semantic, episodic]
        )

        results.record("Encoder handles all modalities", all_correct_size)
    except Exception as e:
        results.record("Encoder handles all modalities", False, str(e))

    # Test 2: Consolidation queue processing
    try:
        consolidator = MemoryConsolidator()

        # Create mock memories
        class MockMemory:
            def __init__(self):
                self.strength = 0.3

        memories = [MockMemory() for _ in range(100)]

        for m in memories:
            consolidator.queue_for_consolidation(m)

        consolidated = consolidator.replay_cycle(iterations=3)

        # Queue should be cleared
        queue_empty = len(consolidator.replay_queue) == 0

        # All memories should be stronger
        all_stronger = all(m.strength > 0.3 for m in memories)

        results.record("Consolidation processes 100 items",
                      consolidated == 100 and queue_empty and all_stronger)
    except Exception as e:
        results.record("Consolidation processes 100 items", False, str(e))

    # Test 3: Forgetting interference
    try:
        forgetter = Forgetter()

        class MockMemory:
            def __init__(self, strength):
                self.strength = strength

        old = MockMemory(0.8)
        new = MockMemory(1.0)

        forgetter.interfere(old, new, similarity=0.7)

        # Old memory should be weakened
        results.record("Interference weakens old memory",
                      old.strength < 0.8,
                      f"Old strength: {old.strength:.2f}")
    except Exception as e:
        results.record("Interference weakens old memory", False, str(e))

    # Test 4: Retrieval-induced forgetting
    try:
        forgetter = Forgetter()

        class MockMemory:
            def __init__(self, name):
                self.name = name
                self.strength = 0.8

        retrieved = MockMemory("retrieved")
        competitors = [MockMemory(f"competitor_{i}") for i in range(10)]

        affected = forgetter.retrieve_induced_forgetting(retrieved, competitors)

        # All competitors should be weakened
        all_weakened = all(c.strength < 0.8 for c in competitors)

        results.record("Retrieval-induced forgetting (10 competitors)",
                      affected == 10 and all_weakened)
    except Exception as e:
        results.record("Retrieval-induced forgetting (10 competitors)", False, str(e))


# =============================================================================
# 6. MEMORY CONTROLLER INTEGRATION TESTS
# =============================================================================

def test_memory_controller():
    test_section("MEMORY CONTROLLER INTEGRATION TESTS")

    # Test 1: Full pipeline under load
    try:
        mc = MemoryController(wm_capacity=7)
        sim_time = 0.0
        mc.set_time_function(lambda: sim_time)

        # Process 50 visual inputs, attending to every 5th
        for i in range(50):
            mc.process_visual(np.random.rand(10, 10))
            if i % 5 == 0:
                mc.attend("visual")
            sim_time += 0.1

        # Should have ~7 items in WM (capacity limit)
        wm_count = len(mc.working)

        results.record("Pipeline handles 50 visual inputs",
                      wm_count <= 7,
                      f"WM items: {wm_count}")
    except Exception as e:
        results.record("Pipeline handles 50 visual inputs", False, str(e))

    # Test 2: Cross-store retrieval
    try:
        mc = MemoryController()

        # Store in different systems
        mc.store_in_working_memory("wm_item")
        mc.encode_experience("episodic_item", context={"type": "test"})
        mc.learn_concept("semantic_item", features={"x": 1.0})

        # Recall from each
        wm_recall = mc.recall("wm_item", memory_types=["working"])
        ep_recall = mc.recall("episodic", memory_types=["episodic"])
        sem_recall = mc.recall("semantic_item", memory_types=["semantic"])

        all_found = all([wm_recall is not None, ep_recall is not None,
                        sem_recall is not None])

        results.record("Cross-store retrieval", all_found)
    except Exception as e:
        results.record("Cross-store retrieval", False, str(e))

    # Test 3: Skill learning and execution
    try:
        mc = MemoryController()

        # Learn skill
        skill = mc.learn_skill(
            "test_skill",
            {"situation": "test", "ready": True},
            ["prepare", "execute", "finish"]
        )

        # Practice to strengthen
        for _ in range(10):
            result = mc.execute_skill({"situation": "test", "ready": True})
            mc.procedural.strengthen(skill, success=True)

        # Check automaticity
        is_automatic = skill.is_automatic(threshold=0.5)

        results.record("Skill learning and automaticity",
                      is_automatic,
                      f"Strength: {skill.strength:.2f}")
    except Exception as e:
        results.record("Skill learning and automaticity", False, str(e))

    # Test 4: Consolidation cycle
    try:
        mc = MemoryController()

        # Add items to WM
        for i in range(5):
            mc.store_in_working_memory(f"item_{i}")
            mc.working.rehearse(f"item_{i}")  # Mark as rehearsed

        # Consolidate
        consolidated = mc.consolidate()

        results.record("Consolidation cycle",
                      consolidated > 0,
                      f"Consolidated {consolidated} items")
    except Exception as e:
        results.record("Consolidation cycle", False, str(e))

    # Test 5: Time-based decay across systems
    try:
        mc = MemoryController()
        sim_time = 0.0
        mc.set_time_function(lambda: sim_time)

        # Add items
        mc.process_visual(np.array([1, 2, 3]))
        mc.store_in_working_memory("wm_test")

        initial_iconic = mc.iconic.is_available()
        initial_wm = len(mc.working)

        # Advance time significantly
        sim_time = 60.0
        mc.tick(60.0)

        final_iconic = mc.iconic.is_available()
        final_wm = len(mc.working)

        iconic_decayed = initial_iconic and not final_iconic
        wm_decayed = initial_wm > final_wm

        results.record("Time-based decay across systems",
                      iconic_decayed and wm_decayed)
    except Exception as e:
        results.record("Time-based decay across systems", False, str(e))


# =============================================================================
# 7. EDGE CASES AND STRESS TESTS
# =============================================================================

def test_edge_cases():
    test_section("EDGE CASES AND STRESS TESTS")

    # Test 1: Empty retrieval handling
    try:
        mc = MemoryController()

        result = mc.recall("nonexistent")
        ep_result = mc.episodic.retrieve_by_cue("nothing")
        sem_result = mc.semantic.retrieve("nothing")

        all_none = result is None and len(ep_result) == 0 and sem_result is None

        results.record("Empty retrieval returns None/empty", all_none)
    except Exception as e:
        results.record("Empty retrieval returns None/empty", False, str(e))

    # Test 2: Zero-sized inputs
    try:
        iconic = IconicBuffer()
        iconic.capture(np.array([]))
        result = iconic.read()

        results.record("Zero-sized input handling", result is not None)
    except Exception as e:
        results.record("Zero-sized input handling", False, str(e))

    # Test 3: Negative emotional valence
    try:
        em = EpisodicMemory()

        ep = em.encode("negative_event", emotional_valence=-0.9)
        retrieved = em.retrieve_by_emotion(-1.0, -0.5)

        results.record("Negative emotional valence handling",
                      len(retrieved) == 1)
    except Exception as e:
        results.record("Negative emotional valence handling", False, str(e))

    # Test 4: Very long semantic paths
    try:
        sm = SemanticMemory()

        # Create chain of 20 concepts
        for i in range(20):
            if i < 19:
                sm.create_concept(f"node_{i}", relations=[("next", f"node_{i+1}", 1.0)])
            else:
                sm.create_concept(f"node_{i}")

        # Find path from start to end
        path = sm.find_path("node_0", "node_19", max_depth=25)

        results.record("Long semantic path (20 hops)",
                      path is not None and len(path) == 19,
                      f"Path length: {len(path) if path else 0}")
    except Exception as e:
        results.record("Long semantic path (20 hops)", False, str(e))

    # Test 5: Concurrent-like operations
    try:
        mc = MemoryController()

        # Rapid fire operations
        for i in range(100):
            mc.process_visual(np.random.rand(10, 10))
            mc.encode_experience(f"event_{i}", context={"i": i})
            mc.store_in_working_memory(f"wm_{i}")
            if i % 10 == 0:
                mc.consolidate()

        # System should still be functional
        state = mc.get_state()

        results.record("Rapid-fire operations (100 iterations)",
                      state["episodic_count"] == 100)
    except Exception as e:
        results.record("Rapid-fire operations (100 iterations)", False, str(e))

    # Test 6: Engram with extreme parameters
    try:
        # Very sparse
        sparse = Engram(n_neurons=100, connectivity=0.01)
        sparse.encode(np.random.rand(100))

        # Very dense
        dense = Engram(n_neurons=100, connectivity=0.99)
        dense.encode(np.random.rand(100))

        results.record("Extreme engram parameters", True)
    except Exception as e:
        results.record("Extreme engram parameters", False, str(e))


# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    print("\n" + "=" * 60)
    print("MODULE 04: MEMORY SYSTEMS - COMPREHENSIVE STRESS TESTS")
    print("=" * 60)

    start_time = time.time()

    test_sensory_memory()
    test_working_memory()
    test_long_term_memory()
    test_engram()
    test_memory_processes()
    test_memory_controller()
    test_edge_cases()

    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed:.2f}s")

    return results.summary()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
