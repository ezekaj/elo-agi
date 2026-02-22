#!/bin/bash
# Demo script for recording with asciinema
# Usage: asciinema rec demo.cast -c "bash scripts/record_demo.sh"

echo "# Installing ELO-AGI"
echo "pip install neuro-agi"
sleep 1

echo ""
echo "# Using the Brain API"
python3 -c "
from neuro import Brain
brain = Brain()
result = brain.think('What causes climate change?')
print(f'Response: {result.text[:200]}...')
print(f'Modules: {result.modules_used}')
print(f'Confidence: {result.confidence:.2f}')
"
sleep 2

echo ""
echo "# 38 cognitive modules ready"
python3 -c "from neuro import CognitiveCore; c = CognitiveCore(); c.initialize(); print(f'Active modules: {len(c.get_state().active_modules)}')"
