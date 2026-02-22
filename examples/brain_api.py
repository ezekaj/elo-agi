#!/usr/bin/env python3
"""Brain API demo.

Shows the simplest way to interact with NEURO: create a Brain, ask
questions, learn new information, and inspect which cognitive modules
are active. No numpy or manual loop management required.
"""

from neuro import Brain

brain = Brain()

# Ask a question -- returns a SmartResponse with .text and metadata
result = brain.think("What causes inflation?")
print(f"Answer: {result.text[:200]}")

# See which cognitive modules participated
print(f"Active modules: {brain.modules}")

# Teach the brain new context
brain.learn("The Federal Reserve sets interest rates in the United States.")

# reason() is an alias for think()
follow_up = brain.reason("How do interest rates affect inflation?")
print(f"Follow-up: {follow_up.text[:200]}")

# Reset conversation history when switching topics
brain.reset()
