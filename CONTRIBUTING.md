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
6. Commit your changes
7. Push to your fork
8. Open a Pull Request

## Module Structure

Each module follows this structure:

```
neuro-<name>/
├── src/           # Python source files
├── tests/         # pytest tests
└── .gitignore
```

## Adding a New Module

1. Create the directory structure above
2. Add source files in `src/`
3. Add tests in `tests/`
4. Update `neuro/__init__.py` if adding public exports
