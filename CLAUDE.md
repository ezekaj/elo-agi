# CLAUDE.md - Neuro Cognitive Architecture

This file provides guidance to Claude Code when working with the Neuro AGI project.

## Quick Reference

```bash
# Run tests for any module
cd neuro-<module> && python -m pytest tests/ -v

# Common modules
pytest neuro-system/tests/ -v          # Core system (active inference, module loader)
pytest neuro-module-00-integration/tests/ -v  # Global Workspace
pytest neuro-llm/tests/ -v             # LLM integration
pytest neuro-bench/tests/ -v           # Benchmarks

# Setup venv (if needed)
python -m venv .venv && source .venv/bin/activate
pip install pytest numpy scipy
```

## Architecture Overview

Neuro is a neuroscience-inspired AGI architecture with 29 modules across 3 tiers:

### Tier 1: Cognitive Modules (23)
| ID | Module | Purpose |
|----|--------|---------|
| 00 | integration | Global Workspace Theory, attention, broadcast |
| 01 | predictive-coding | Free Energy Principle, hierarchical predictions |
| 02 | dual-process | System 1/2 emergence from geometry |
| 03 | reasoning-types | Dimensional, interactive, logical, perceptual |
| 04 | memory | Sensory, working, long-term memory |
| 05 | sleep-consolidation | Replay, systems consolidation, homeostasis |
| 06 | motivation | Path entropy, dopamine, curiosity |
| 07 | emotions-decisions | Emotional valuation in decisions |
| 08 | language | Language processing/generation |
| 09 | creativity | Creative thought generation |
| 10 | spatial-cognition | Spatial reasoning |
| 11 | time-perception | Embodied temporal cognition |
| 12 | learning | Adaptive learning mechanisms |
| 13 | executive | Executive control, planning |
| 14 | embodied | Embodied cognition |
| 15 | social | Social reasoning |
| 16 | consciousness | Metacognition, self-awareness |
| 17 | world-model | Internal world modeling |
| 18 | self-improvement | Self-directed improvement |
| 19 | multi-agent | Multi-agent coordination |
| 20 | causal | Counterfactual reasoning, causal DAGs, interventions |
| 21 | abstract | Neuro-symbolic binding, compositional generalization |
| 22 | robust | Uncertainty quantification, OOD detection, calibration |

### Tier 2: Infrastructure (6)
| Module | Purpose |
|--------|---------|
| neuro-system | Unified core: active inference, module loader, sensory/motor |
| neuro-llm | LLM oracle, semantic bridge, grounding, dialogue |
| neuro-knowledge | Fact store, knowledge graph, ontology, inference |
| neuro-ground | Sensors (camera, mic, proprioception), actuators (motor, speech) |
| neuro-scale | Distributed: coordinator, workers, aggregators, GPU kernels |
| neuro-transfer | Transfer learning mechanisms |

### Tier 3: Support (5)
| Module | Purpose |
|--------|---------|
| neuro-env | Environment/context management |
| neuro-bench | Benchmarks: reasoning, memory, language, planning |
| neuro-inference | Bayesian, analogical, causal reasoning engines |
| neuro-perception | Visual, auditory, multimodal integration |
| neuro-integrate | Advanced integration layer |

## Module Patterns

**File Structure:**
```
neuro-<name>/
├── src/           # Python source
├── tests/         # pytest tests
├── .gitignore     # Python standard
└── CLAUDE.md      # Module-specific guidance (if exists)
```

**Code Conventions:**
- Dataclass configs with sensible defaults
- `.statistics()` method on major classes
- Clean `__all__` exports in `__init__.py`
- Simulated mode by default (no hardware required)

**Integration Points:**
- neuro-system loads modules via ModuleLoader
- neuro-module-00 (Global Workspace) coordinates attention/broadcast
- neuro-llm provides language understanding
- neuro-ground provides sensor/actuator interfaces

## Key Files

- `neuro-system/src/core.py` - CognitiveCore with active inference loop
- `neuro-module-00-integration/src/global_workspace.py` - Attention & broadcast
- `neuro-module-01-predictive-coding/src/` - Free Energy computations
- `neuro-llm/src/oracle.py` - LLM integration
- `neuro-ground/src/sensors/` - Camera, Microphone, Proprioception
- `neuro-bench/src/benchmarks/` - AGI capability tests

## AGI Roadmap: Progress

Based on 2024-2026 research, critical gaps addressed for human-level AGI:

### Priority 1: Causal Reasoning ✅ COMPLETE
- Counterfactual reasoning, causal DAGs, intervention modeling
- Module: `neuro-causal` (124 tests)

### Priority 2: Compositional Abstraction ✅ COMPLETE
- ARC Prize core challenge: neuro-symbolic binding
- Hierarchical concept composition
- Module: `neuro-abstract` (144 tests)

### Priority 3: Continual Learning
- Hippocampal replay mechanisms
- Meta-learning for learning-to-learn
- Enhance: neuro-module-05-sleep-consolidation

### Priority 4: Robustness ✅ COMPLETE
- Adversarial training, uncertainty quantification
- OOD detection
- Module: `neuro-robust` (192 tests)

### Key Research Papers
1. Agentic AI Survey (arxiv:2510.25445)
2. Neuro-Symbolic AI 2024 (arxiv:2501.05435)
3. Embodied AI: LLMs to World Models (arxiv:2509.20021)
4. ARC Prize 2025 Technical Report (arxiv:2601.10904)
