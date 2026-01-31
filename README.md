# Neuro AGI

Neuroscience-inspired AGI architecture with 38 cognitive modules.

## Installation

```bash
git clone https://github.com/ezekaj/elo-agi.git
cd elo-agi
pip install -e .
```

## Quick Start

```python
from neuro import CognitiveCore

core = CognitiveCore()
core.step()  # Run one cognitive cycle
```

## CLI Commands

```bash
neuro --version              # Check installation
neuro info                   # System information
neuro check                  # Verify all modules

neuro-demo all               # Run all 5 demos
neuro-demo medical           # Medical diagnosis reasoning
neuro-demo physics           # Physics concept learning
neuro-demo language          # Language understanding
neuro-demo arc               # ARC-style pattern reasoning
neuro-demo continual         # Continual learning demo

neuro-bench -o report.html   # Generate benchmark report
```

## Architecture

38 modules across 4 tiers:

| Tier | Modules | Purpose |
|------|---------|---------|
| Cognitive | 20 | Core cognitive functions (memory, reasoning, language, etc.) |
| Infrastructure | 6 | System core, LLM integration, knowledge management |
| Support | 5 | Benchmarks, perception, environment |
| AGI | 7 | Causal reasoning, abstraction, planning, continual learning |

### Key Modules

- **neuro-system** - Active inference loop, module coordination
- **neuro-causal** - Differentiable causal reasoning with SCMs
- **neuro-abstract** - Neuro-symbolic binding with HRR
- **neuro-continual** - Catastrophic forgetting prevention (EWC, PackNet, SI)
- **neuro-planning** - Hierarchical task planning

## Benchmarks

```
Overall Score: 0.697
Success Rate: 96.4%

Categories:
- Reasoning: 0.67
- Memory: 0.72
- Language: 0.66
- Causal: 0.70
- Abstraction: 0.71
- Continual: 0.75
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest neuro-*/tests/ -v

# Format code
black .
ruff check .
```

## License

MIT
