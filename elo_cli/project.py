"""Project detection and context loading."""

from pathlib import Path
from typing import Optional


PROJECT_MARKERS = {
    "python": ["pyproject.toml", "setup.py", "requirements.txt", "Pipfile", "setup.cfg"],
    "node": ["package.json", "package-lock.json", "yarn.lock"],
    "rust": ["Cargo.toml"],
    "go": ["go.mod", "go.sum"],
    "ruby": ["Gemfile", "Gemfile.lock"],
    "java": ["pom.xml", "build.gradle"],
    "dotnet": ["*.csproj", "*.sln"],
    "php": ["composer.json"],
}


def detect_project(path: str = ".") -> dict:
    """Detect project type and gather context.

    Returns:
        {
            "type": "python" | "node" | etc | "unknown",
            "marker": "pyproject.toml" | etc,
            "path": "/absolute/path",
            "context": "Project context string..."
        }
    """
    p = Path(path).resolve()

    for lang, markers in PROJECT_MARKERS.items():
        for marker in markers:
            if "*" in marker:
                # Glob pattern
                if list(p.glob(marker)):
                    return {
                        "type": lang,
                        "marker": marker,
                        "path": str(p),
                        "context": load_project_context(p, lang),
                    }
            elif (p / marker).exists():
                return {
                    "type": lang,
                    "marker": marker,
                    "path": str(p),
                    "context": load_project_context(p, lang),
                }

    return {"type": "unknown", "marker": None, "path": str(p), "context": load_generic_context(p)}


def load_project_context(path: Path, lang: str) -> str:
    """Load relevant project files for context."""
    context_parts = []

    # Always try README
    readme_content = load_readme(path)
    if readme_content:
        context_parts.append(f"README:\n{readme_content}")

    # Language-specific context
    if lang == "python":
        # pyproject.toml
        if (path / "pyproject.toml").exists():
            content = safe_read(path / "pyproject.toml", 1500)
            context_parts.append(f"pyproject.toml:\n{content}")
        # requirements.txt
        elif (path / "requirements.txt").exists():
            content = safe_read(path / "requirements.txt", 500)
            context_parts.append(f"requirements.txt:\n{content}")

    elif lang == "node":
        if (path / "package.json").exists():
            content = safe_read(path / "package.json", 1500)
            context_parts.append(f"package.json:\n{content}")

    elif lang == "rust":
        if (path / "Cargo.toml").exists():
            content = safe_read(path / "Cargo.toml", 1000)
            context_parts.append(f"Cargo.toml:\n{content}")

    elif lang == "go":
        if (path / "go.mod").exists():
            content = safe_read(path / "go.mod", 500)
            context_parts.append(f"go.mod:\n{content}")

    return "\n\n".join(context_parts)


def load_generic_context(path: Path) -> str:
    """Load generic context when project type is unknown."""
    context_parts = []

    readme_content = load_readme(path)
    if readme_content:
        context_parts.append(f"README:\n{readme_content}")

    # List top-level files
    try:
        files = [f.name for f in path.iterdir() if f.is_file()][:20]
        if files:
            context_parts.append(f"Files: {', '.join(files)}")
    except Exception:
        pass

    return "\n\n".join(context_parts)


def load_readme(path: Path) -> Optional[str]:
    """Load README file content."""
    for readme_name in ["README.md", "README.txt", "README", "readme.md"]:
        readme_path = path / readme_name
        if readme_path.exists():
            return safe_read(readme_path, 2000)
    return None


def safe_read(path: Path, max_chars: int = 2000) -> str:
    """Safely read a file with character limit."""
    try:
        content = path.read_text()
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... (truncated)"
        return content
    except Exception:
        return ""


def get_project_summary(path: str = ".") -> str:
    """Get a one-line project summary for display."""
    info = detect_project(path)

    if info["type"] == "unknown":
        return f"[dim]Unknown project at {info['path']}[/dim]"

    return f"[cyan]{info['type'].title()}[/cyan] project ({info['marker']})"
