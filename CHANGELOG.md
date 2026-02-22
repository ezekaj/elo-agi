# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Brain API (`neuro.brain.Brain`) for simple one-line cognitive queries
- Jupyter/IPython magic extension (`%load_ext neuro.jupyter`)
- LangChain integration (`neuro.integrations.langchain.NeuroCognitiveLLM`)
- Rich HTML display for SmartResponse in Jupyter notebooks
- Security scanning workflow (Bandit + CodeQL)
- Dependabot for automated dependency updates
- CODEOWNERS for PR review assignment
- Test coverage reporting in CI
- Security headers on API responses
- Getting Started and API Docs pages on website
- Public roadmap section on website

### Security
- API key verification on all sensitive endpoints
- Stable hashing with SHA-256 (replaces Python `hash()`)
- Sandbox attribute access blocking

### Changed
- CI test tolerance tightened from 8 to 2 module failures
- ruff format check is now blocking in CI
- Centralized version management via importlib.metadata
- Updated website navigation with Getting Started and API Docs links

### Fixed
- API version inconsistency (was reporting 2.0.0 instead of 0.9.0)

## [0.9.0] - 2026-01-31

### Added
- 38 cognitive modules across 4 tiers (Cognitive, Infrastructure, Support, AGI)
- Causal reasoning engine with differentiable SCMs (124 tests)
- Compositional abstraction with neuro-symbolic binding (144 tests)
- Robustness module with uncertainty quantification and OOD detection (192 tests)
- Hierarchical planning with MCTS and temporal abstraction (82 tests)
- Continual learning with EWC, PackNet, and SI (88 tests)
- Credit assignment with eligibility traces and contribution accounting
- Meta-reasoning with problem classification and strategy selection
- FastAPI REST backend with 7 endpoints
- Sandboxed Python REPL with blocked imports, memory limits, and timeouts
- Rate limiting on all API endpoints (slowapi)
- CORS whitelist for API access
- Docker deployment to Fly.io (Frankfurt region)
- PyPI package distribution (`pip install neuro-agi`)
- CLI entry point (`neuro`) with interactive sessions and streaming
- Benchmark CLI (`neuro-bench`) with HTML report generation
- Demo CLI (`neuro-demo`) with 5 demonstration scenarios
- Static documentation website (GitHub Pages)
- Privacy policy and terms of service
- SECURITY.md with vulnerability disclosure policy
- CONTRIBUTING.md with development guidelines

### Security
- Sandbox allowlist-only builtins (no eval, exec, getattr)
- Blocked dangerous imports (subprocess, socket, os, ctypes, pickle)
- Memory limits (256MB) and execution timeouts (10s) on REPL
- Non-root Docker container with health checks
- OIDC trusted publishing for PyPI releases
