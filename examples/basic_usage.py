#!/usr/bin/env python3
"""Basic CognitiveCore usage.

Demonstrates creating a cognitive core, running the perceive-think-act
loop, and inspecting system state. This is the lowest-level API that
gives you full control over the cognitive pipeline.
"""

import numpy as np

from neuro import CognitiveCore

# Create and initialize the cognitive system
core = CognitiveCore()
status = core.initialize()
print(f"Loaded {status['loaded']} / {status['total']} modules")

# Feed sensory input
observation = np.random.randn(64)
core.perceive(observation)

# Run one cognitive cycle (global workspace competition + broadcast)
proposals = core.think()
print(f"Proposals generated: {proposals}")

# Produce motor output via active inference
output = core.act()
print(f"Action shape: {output.value.shape}")

# Inspect system state
state = core.get_state()
print(f"Cycle count : {state.cycle_count}")
print(f"Active modules: {len(state.active_modules)}")

# Run multiple cycles at once
results = core.run(steps=5)
print(f"Completed {len(results)} cycles")
