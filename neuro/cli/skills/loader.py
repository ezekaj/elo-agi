"""
Skills Loader - Load and execute custom skills.

Skills are reusable prompts/instructions that can be invoked
via slash commands like /commit, /review, /test, etc.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
from pathlib import Path
import os
import yaml


@dataclass
class Skill:
    """A loaded skill."""

    name: str
    description: str
    prompt: str

    # Optional metadata
    tools: List[str] = field(default_factory=list)
    model: str = "inherit"
    aliases: List[str] = field(default_factory=list)

    # Argument handling
    args_description: str = ""
    required_args: List[str] = field(default_factory=list)


class SkillsLoader:
    """
    Loads and manages skills from various sources.

    Skill sources (in priority order):
    1. Project: .neuro/skills/*.md
    2. User: ~/.neuro/skills/*.md
    3. Built-in: Hardcoded defaults

    Skill file format (markdown with YAML frontmatter):
    ```
    ---
    name: commit
    description: Create a git commit with smart message
    aliases: [ci, c]
    ---

    Analyze the staged changes and create a commit...
    ```
    """

    SKILLS_DIR = ".neuro/skills"
    USER_SKILLS_DIR = "~/.neuro/skills"

    # Built-in skills
    BUILTINS = {
        "commit": Skill(
            name="commit",
            description="Create a git commit with smart message generation",
            aliases=["ci"],
            prompt="""Analyze the staged git changes and create a commit.

STEPS:
1. Run `git status` to see what's staged
2. Run `git diff --staged` to see the changes
3. Analyze the changes and generate a concise commit message
4. Ask user to confirm or edit the message
5. Create the commit

COMMIT MESSAGE FORMAT:
- First line: Type and brief description (max 50 chars)
- Types: feat, fix, docs, style, refactor, test, chore
- Example: "feat: add user authentication flow"

RULES:
- Never commit if nothing is staged
- Show the diff summary before committing
- Let user approve the message""",
        ),
        "review": Skill(
            name="review",
            description="Review code changes for issues and improvements",
            aliases=["r", "cr"],
            prompt="""Review the code changes for potential issues.

ANALYZE:
1. Check for bugs and logic errors
2. Look for security vulnerabilities
3. Identify performance issues
4. Note code style inconsistencies
5. Suggest improvements

OUTPUT FORMAT:
## Summary
Brief overview of the changes

## Issues Found
- [SEVERITY] Description of issue

## Suggestions
- Improvement recommendations

## Verdict
APPROVE / REQUEST_CHANGES / COMMENT""",
        ),
        "test": Skill(
            name="test",
            description="Run tests and analyze results",
            aliases=["t"],
            tools=["run_command"],
            prompt="""Run the project's test suite and analyze results.

STEPS:
1. Detect the testing framework (pytest, jest, go test, etc.)
2. Run the appropriate test command
3. Analyze any failures
4. Suggest fixes for failing tests

If tests fail, provide:
- Which tests failed
- Why they might be failing
- Suggested fixes""",
        ),
        "explain": Skill(
            name="explain",
            description="Explain how code works",
            aliases=["e", "what"],
            prompt="""Explain how this code works in detail.

INCLUDE:
- Purpose and functionality
- Key components and their roles
- Data flow
- Important patterns used
- Potential gotchas

Keep explanations clear and accessible.""",
        ),
        "refactor": Skill(
            name="refactor",
            description="Suggest and implement code refactoring",
            aliases=["rf"],
            prompt="""Analyze the code and suggest refactoring improvements.

CONSIDER:
- Code duplication
- Complex functions that should be split
- Better naming
- Design pattern opportunities
- Performance improvements

For each suggestion:
1. Explain the issue
2. Show the proposed change
3. Explain the benefit

Ask before making changes.""",
        ),
        "doc": Skill(
            name="doc",
            description="Generate documentation for code",
            aliases=["docs", "document"],
            prompt="""Generate documentation for the specified code.

INCLUDE:
- Function/class descriptions
- Parameter documentation
- Return value documentation
- Usage examples
- Any important notes

Match the project's documentation style if one exists.""",
        ),
    }

    def __init__(self, project_dir: str = "."):
        self.project_dir = os.path.abspath(project_dir)
        self._skills: Dict[str, Skill] = dict(self.BUILTINS)
        self._aliases: Dict[str, str] = {}

        # Build alias map for builtins
        for name, skill in self.BUILTINS.items():
            for alias in skill.aliases:
                self._aliases[alias] = name

        # Load custom skills
        self._load_custom_skills()

    def _load_custom_skills(self):
        """Load custom skills from skill directories."""
        dirs = [
            os.path.expanduser(self.USER_SKILLS_DIR),
            os.path.join(self.project_dir, self.SKILLS_DIR),
        ]

        for skills_dir in dirs:
            if os.path.exists(skills_dir):
                for file in Path(skills_dir).glob("*.md"):
                    try:
                        skill = self._load_skill_file(str(file))
                        if skill:
                            self._skills[skill.name] = skill
                            for alias in skill.aliases:
                                self._aliases[alias] = skill.name
                    except Exception as e:
                        print(f"Error loading skill {file}: {e}")

    def _load_skill_file(self, path: str) -> Optional[Skill]:
        """Load a skill from a markdown file."""
        with open(path) as f:
            content = f.read()

        # Parse frontmatter
        if not content.startswith("---"):
            # No frontmatter, use filename as name
            return Skill(
                name=Path(path).stem,
                description="",
                prompt=content.strip(),
            )

        parts = content.split("---", 2)
        if len(parts) < 3:
            return None

        try:
            frontmatter = yaml.safe_load(parts[1])
            prompt = parts[2].strip()
        except yaml.YAMLError:
            return None

        return Skill(
            name=frontmatter.get("name", Path(path).stem),
            description=frontmatter.get("description", ""),
            prompt=prompt,
            tools=frontmatter.get("tools", []),
            model=frontmatter.get("model", "inherit"),
            aliases=frontmatter.get("aliases", []),
            args_description=frontmatter.get("argsDescription", ""),
            required_args=frontmatter.get("requiredArgs", []),
        )

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a skill by name or alias."""
        # Check direct name
        if name in self._skills:
            return self._skills[name]

        # Check aliases
        if name in self._aliases:
            return self._skills.get(self._aliases[name])

        return None

    def list_skills(self) -> List[Skill]:
        """List all available skills."""
        return list(self._skills.values())

    def get_skill_names(self) -> List[str]:
        """Get all skill names and aliases."""
        names = list(self._skills.keys())
        names.extend(self._aliases.keys())
        return sorted(set(names))

    def execute_skill(
        self,
        name: str,
        args: str = "",
        context: str = "",
    ) -> str:
        """
        Build the prompt for executing a skill.

        Args:
            name: Skill name or alias
            args: Additional arguments from user
            context: Additional context (e.g., file contents)

        Returns:
            The complete prompt to send to LLM
        """
        skill = self.get_skill(name)
        if not skill:
            return f"Unknown skill: {name}"

        # Build the prompt
        parts = [skill.prompt]

        if args:
            parts.append(f"\nUSER INPUT:\n{args}")

        if context:
            parts.append(f"\nCONTEXT:\n{context}")

        return "\n".join(parts)

    def create_skill(
        self,
        name: str,
        description: str,
        prompt: str,
        aliases: List[str] = None,
        save_to: str = "project",
    ) -> bool:
        """
        Create a new custom skill.

        Args:
            name: Skill name
            description: Short description
            prompt: The skill prompt
            aliases: Optional aliases
            save_to: "project" or "user"

        Returns:
            True if saved successfully
        """
        skill = Skill(
            name=name,
            description=description,
            prompt=prompt,
            aliases=aliases or [],
        )

        # Determine save path
        if save_to == "user":
            skills_dir = os.path.expanduser(self.USER_SKILLS_DIR)
        else:
            skills_dir = os.path.join(self.project_dir, self.SKILLS_DIR)

        os.makedirs(skills_dir, exist_ok=True)

        # Build file content
        frontmatter = {
            "name": name,
            "description": description,
        }
        if aliases:
            frontmatter["aliases"] = aliases

        content = f"""---
{yaml.dump(frontmatter, default_flow_style=False).strip()}
---

{prompt}
"""

        # Save
        file_path = os.path.join(skills_dir, f"{name}.md")
        try:
            with open(file_path, "w") as f:
                f.write(content)

            # Add to loaded skills
            self._skills[name] = skill
            for alias in skill.aliases:
                self._aliases[alias] = name

            return True
        except Exception as e:
            print(f"Error saving skill: {e}")
            return False
