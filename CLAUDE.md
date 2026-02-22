# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NEURO (elo-agi v0.9.0) — a neuroscience-inspired AGI architecture with 38 cognitive modules. Local-first, privacy-preserving AI that runs entirely on the user's machine. Python 3.9+, MIT licensed.

## Commands

```bash
# Install
pip install -e .              # Basic
pip install -e ".[dev]"       # With pytest, ruff, black, mypy
pip install -e ".[all]"       # Everything (llm, viz, dev)

# Run tests
pytest neuro-*/tests/ -v                          # All modules
cd neuro-<module> && python -m pytest tests/ -v   # Single module
pytest neuro-system/tests/ -v                     # Core system
pytest neuro-causal/tests/ -v                     # Causal (124 tests)
pytest neuro-abstract/tests/ -v                   # Abstraction (144 tests)
pytest neuro-robust/tests/ -v                     # Robustness (192 tests)

# Lint & format
ruff check .                  # Lint
ruff check . --fix            # Auto-fix
black .                       # Format (line-length=100)
mypy                          # Type check

# CLI entry points (defined in pyproject.toml [project.scripts])
neuro                         # Interactive session
neuro-demo all                # Run demos
neuro-bench -o report.html    # Run benchmarks

# API server
uvicorn api.main:app --host 0.0.0.0 --port 8080

# Docker
docker build -t elo-agi . && docker run -p 8080:8080 elo-agi
```

## Architecture

### Four-tier modular design

**Tier 1 — Cognitive Modules (20):** `neuro-module-00-integration/` through `neuro-module-19-multi-agent/`. Each follows `neuro-module-{ID}-{name}/src/` + `tests/` pattern. Module 00 (Global Workspace) coordinates attention/broadcast across all others.

**Tier 2 — Infrastructure (6):** `neuro-system` (active inference core, module loader), `neuro-llm` (LLM oracle), `neuro-knowledge` (knowledge graph), `neuro-ground` (sensors/actuators), `neuro-scale` (distributed), `neuro-transfer`.

**Tier 3 — Support (5):** `neuro-env`, `neuro-bench`, `neuro-inference`, `neuro-perception`, `neuro-integrate`.

**Tier 4 — AGI Extensions (7):** `neuro-causal`, `neuro-abstract`, `neuro-continual`, `neuro-robust`, `neuro-planning`, `neuro-credit`, `neuro-meta-reasoning`.

### Core processing pipeline

`neuro-system/src/cognitive_core.py` runs the active inference loop: **perceive → think → act**. Modules are loaded dynamically via `neuro-system/src/module_loader.py`. The `neuro/` package is the main Python package with lazy imports in `__init__.py`.

### Entry points

- **CLI:** `neuro/cli.py` (main interface, streaming responses) + `neuro/cli/` subpackage (commands, tools, agents, UI, skills)
- **API:** `api/main.py` — FastAPI with endpoints: `/api/health`, `/api/info`, `/api/modules`, `/api/chat`, `/api/repl` (sandboxed Python via `api/sandbox.py`), `/api/benchmarks`, `/api/analyze`
- **Web docs:** `docs/` — static HTML site (index, features, pricing, demo, privacy, terms)
- **Legacy:** `elo.py` (ELO v1.0), `elo_chat.py` (chat interface), `elo_cli/` (legacy CLI framework)

### Key files

- `neuro-system/src/cognitive_core.py` — CognitiveCore with active inference loop
- `neuro-system/src/active_inference.py` — Free energy minimization
- `neuro-module-00-integration/src/global_workspace.py` — Attention & broadcast
- `neuro/engine.py` — Cognitive processing engine
- `neuro/orchestrator.py` — Module coordination/scheduling
- `api/main.py` — FastAPI REST backend

## Code Conventions

- Dataclass configs with sensible defaults
- `.statistics()` method on major classes
- Clean `__all__` exports in `__init__.py`
- Simulated mode by default (no hardware required)
- Lazy imports in `neuro/__init__.py` to minimize startup overhead
- Line length: 100 (ruff and black)
- Target Python: 3.9

## Module Pattern

Every `neuro-<name>/` module follows:
```
neuro-<name>/
├── src/         # Python source
├── tests/       # pytest tests (test_*.py, test_* functions)
└── CLAUDE.md    # Module-specific guidance (optional)
```

## CI/CD

- **`.github/workflows/test.yml`** — pytest across Python 3.9-3.12, tolerates up to 8 module failures
- **`.github/workflows/lint.yml`** — ruff check on push/PR
- **`.github/workflows/publish.yml`** — PyPI publication
- **Deployment:** Fly.io (`fly.toml`), Frankfurt region, 1024MB RAM, auto-stop/start, Docker-based

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Health check + uptime |
| GET | `/api/info` | System info (modules, capabilities) |
| GET | `/api/modules` | List all cognitive modules |
| POST | `/api/chat` | Chat with cognitive AI |
| POST | `/api/repl` | Sandboxed Python REPL execution |
| POST | `/api/benchmarks` | Run benchmark suite |
| POST | `/api/analyze` | Cognitive analysis (dual-process, emotion, reasoning) |

## AGI Roadmap

1. **Causal Reasoning** — COMPLETE (`neuro-causal`, 124 tests)
2. **Compositional Abstraction** — COMPLETE (`neuro-abstract`, 144 tests)
3. **Continual Learning** — In progress (`neuro-continual`, 88 tests)
4. **Robustness** — COMPLETE (`neuro-robust`, 192 tests)
5. **Hierarchical Planning** — COMPLETE (`neuro-planning`, 82 tests)
