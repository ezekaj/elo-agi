<p align="center">
  <img src="assets/logo-full.svg" width="400" alt="NEURO">
</p>

<h3 align="center">Neuroscience-Inspired AGI Framework</h3>

<p align="center">
  38 cognitive modules implementing Free Energy Principle, Active Inference, and Global Workspace Theory.<br>
  Runs locally. Zero API cost.
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/neuro-agi?style=flat-square&color=00d4ff" alt="PyPI">
  <img src="https://img.shields.io/pypi/pyversions/neuro-agi?style=flat-square" alt="Python">
  <img src="https://img.shields.io/github/license/ezekaj/elo-agi?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/modules-38-00d4ff?style=flat-square" alt="Modules">
  <a href="https://github.com/ezekaj/elo-agi/actions/workflows/test.yml"><img src="https://img.shields.io/github/actions/workflow/status/ezekaj/elo-agi/test.yml?branch=main&style=flat-square&label=tests" alt="Tests"></a>
  <a href="https://github.com/ezekaj/elo-agi/actions/workflows/lint.yml"><img src="https://img.shields.io/github/actions/workflow/status/ezekaj/elo-agi/lint.yml?branch=main&style=flat-square&label=lint" alt="Lint"></a>
  <a href="https://github.com/ezekaj/elo-agi"><img src="https://img.shields.io/github/stars/ezekaj/elo-agi?style=flat-square" alt="Stars"></a>
</p>

---

## Why NEURO?

| Feature | NEURO | Cloud AI |
|---------|-------|----------|
| Runs locally | Yes | No |
| Learns from your code | Yes | No |
| Free forever | Yes | $20+/month |
| Private & secure | Yes | Data sent to cloud |
| Works offline | Yes | No |
| Neuroscience-based | 38 cognitive modules | Black box |

## Installation

```bash
pip install neuro-agi
```

Or install from source:

```bash
git clone https://github.com/ezekaj/elo-agi.git
cd elo-agi
pip install -e .
```

### Prerequisites

| Interface | Requirements |
|-----------|-------------|
| Python Library | Python 3.9+, numpy, scipy |
| CLI (`neuro`) | Python 3.9+ and [Ollama](https://ollama.ai) with `ministral-3:8b` model |
| Web REPL | Browser only -- [try it here](https://ezekaj.github.io/elo-agi/demo.html) |

### Setup Ollama (for CLI)

```bash
# Install from https://ollama.ai, then:
ollama serve
ollama pull ministral-3:8b
```

## Quick Start

### CLI

```bash
neuro                        # Start interactive session
neuro --version              # Check installation
neuro info                   # System information
neuro check                  # Verify all modules
```

### Python API

```python
from neuro import CognitiveCore

core = CognitiveCore()
core.initialize()
core.think()  # Run one cognitive cycle
```

```python
# Or use the simple Brain API
from neuro import Brain
brain = Brain()
result = brain.think("What causes inflation?")
print(result.text)
```

### Demos

```bash
neuro-demo all               # Run all demos
neuro-demo medical           # Medical diagnosis reasoning
neuro-demo physics           # Physics concept learning
neuro-demo language          # Language understanding
neuro-demo arc               # ARC-style pattern reasoning
neuro-demo continual         # Continual learning
```

## Features

### Cognitive Architecture
- **Global Workspace Theory** - Attention and information broadcast
- **Predictive Coding** - Free Energy Principle implementation
- **Dual Process Theory** - System 1/2 reasoning
- **Episodic Memory** - Long-term context retention
- **Metacognition** - Self-awareness and reflection

### Learning & Adaptation
- **Continual Learning** - Learns without forgetting (EWC, PackNet, SI)
- **Active Learning** - Asks questions when uncertain
- **Pattern Recognition** - Discovers coding patterns automatically
- **Knowledge Integration** - Builds understanding over time

### Developer Tools
- **Code Understanding** - Deep analysis of your codebase
- **Intelligent Suggestions** - Context-aware recommendations
- **Multi-file Reasoning** - Understands project structure
- **Tool Integration** - Git, shell, file operations

## Integrations

### Jupyter / IPython

Load the magic extension to query NEURO directly from notebook cells:

```python
%load_ext neuro.jupyter
%neuro What is the free energy principle?
```

`SmartResponse` objects render as rich HTML in Jupyter automatically.

### LangChain

Use NEURO as a drop-in LangChain LLM (requires `pip install neuro-agi[langchain]`):

```python
from neuro.integrations.langchain import NeuroCognitiveLLM

llm = NeuroCognitiveLLM()
print(llm.invoke("Explain quantum entanglement."))
```

## Architecture

38 cognitive modules across 4 tiers:

```
┌─────────────────────────────────────────────────────────────┐
│                      NEURO ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────┤
│  COGNITIVE TIER (20 modules)                                │
│  Memory, Language, Reasoning, Creativity, Executive...      │
├─────────────────────────────────────────────────────────────┤
│  INFRASTRUCTURE TIER (6 modules)                            │
│  Core System, LLM Integration, Knowledge Graph              │
├─────────────────────────────────────────────────────────────┤
│  SUPPORT TIER (5 modules)                                   │
│  Benchmarks, Perception, Environment                        │
├─────────────────────────────────────────────────────────────┤
│  AGI TIER (7 modules)                                       │
│  Causal Reasoning, Abstraction, Planning, Continual         │
└─────────────────────────────────────────────────────────────┘
```

### Key Modules

| Module | Purpose | Tests |
|--------|---------|-------|
| `neuro-system` | Active inference loop, module coordination | Core |
| `neuro-causal` | Differentiable causal reasoning with SCMs | 124 |
| `neuro-abstract` | Neuro-symbolic binding with HRR | 144 |
| `neuro-continual` | Catastrophic forgetting prevention | 88 |
| `neuro-robust` | Uncertainty quantification, OOD detection | 192 |
| `neuro-planning` | Hierarchical task planning with MCTS | 82 |

## Benchmarks

```
Overall Score: 0.697
Success Rate: 96.4%

┌─────────────┬───────┐
│ Category    │ Score │
├─────────────┼───────┤
│ Continual   │ 0.75  │
│ Memory      │ 0.72  │
│ Abstraction │ 0.71  │
│ Causal      │ 0.70  │
│ Reasoning   │ 0.67  │
│ Language    │ 0.66  │
└─────────────┴───────┘
```

Run benchmarks:
```bash
neuro-bench -o report.html
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Versioning

This project follows [Semantic Versioning](https://semver.org/). During `0.x` releases, minor versions may include breaking changes. Starting from `1.0.0`, breaking changes will only occur in major versions with deprecation warnings issued at least one minor version before removal.

See [CHANGELOG.md](CHANGELOG.md) for release history.

## Roadmap

- [x] Core cognitive architecture (38 modules)
- [x] Causal reasoning engine
- [x] Continual learning without forgetting
- [x] Package restructure and PyPI publishing
- [x] Brain API for simple one-line queries
- [x] Jupyter / IPython magic extension
- [x] LangChain integration
- [ ] External benchmark validation (ARC, GSM8K)
- [ ] VS Code extension
- [ ] Real-time streaming responses
- [ ] Multi-agent collaboration
- [ ] Voice interface

## Community

- [GitHub Discussions](https://github.com/ezekaj/elo-agi/discussions) — Ask questions, share ideas
- [Issues](https://github.com/ezekaj/elo-agi/issues) — Report bugs, request features
- [Contributing](CONTRIBUTING.md) — How to contribute

## License

MIT - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with neuroscience principles. Runs on your machine.</sub>
</p>
