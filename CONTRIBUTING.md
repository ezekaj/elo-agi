# Contributing

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ezekaj/elo-agi.git
cd elo-agi

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
pytest neuro-*/tests/ -v

# Run tests for a specific module
pytest neuro-causal/tests/ -v

# Run with coverage
pytest neuro-*/tests/ --cov=. --cov-report=term-missing
```

## Code Style

This project uses:
- **ruff** for linting
- **black** for formatting (optional)

```bash
# Check linting
ruff check .

# Fix auto-fixable issues
ruff check . --fix

# Format code (optional)
black .
```

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run tests (`pytest neuro-*/tests/ -v`)
5. Run linting (`ruff check .`)
6. Add a changelog entry to `CHANGELOG.md` under `[Unreleased]`
7. Commit your changes
8. Push to your fork
9. Open a Pull Request

## Changelog

All notable changes must be documented in `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/) format. Add entries under the `[Unreleased]` section using these categories: Added, Changed, Fixed, Removed, Security.

## Security

If you discover a security vulnerability, **do not** open a public issue. Instead, follow the process in [SECURITY.md](SECURITY.md).

## Module Structure

Each module follows this structure:

```
neuro-<name>/
├── tests/         # pytest tests
└── .gitignore

neuro/modules/<name>/
├── __init__.py    # Module exports
└── *.py           # Source files
```

Source code lives in `neuro/modules/<name>/` (the installable package). Tests remain in `neuro-<name>/tests/`.

## Adding a New Module

1. Create source files in `neuro/modules/<name>/`
2. Add `__init__.py` with exports
3. Add tests in `neuro-<name>/tests/`
4. Update `neuro/__init__.py` if adding public lazy imports
5. Add a changelog entry
