"""
NEURO Git Automator - Safe Git Operations

Provides:
- Safe commit workflow with validation
- Secrets detection
- Pre-commit hook support
- Branch management
- CI/CD verification
"""

import os
import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class GitStatus(Enum):
    """Status of a git operation."""
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class GitResult:
    """Result of a git operation."""
    status: GitStatus
    message: str
    output: str = ""
    error: Optional[str] = None
    files_affected: List[str] = field(default_factory=list)


@dataclass
class FileChange:
    """A changed file."""
    path: str
    status: str  # M=modified, A=added, D=deleted, ?=untracked
    staged: bool = False


class GitAutomator:
    """
    Safe git operations with validation.

    Features:
    - Secrets detection before commit
    - Pre-commit hook execution
    - Selective file staging
    - Smart commit messages
    - Branch management
    - Force-push prevention
    """

    # Patterns that might indicate secrets
    SECRET_PATTERNS = [
        r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?[\w-]{20,}',
        r'(?i)(secret|password|passwd|pwd)\s*[=:]\s*["\']?[^\s]{8,}',
        r'(?i)(token)\s*[=:]\s*["\']?[\w-]{20,}',
        r'(?i)Bearer\s+[\w-]+\.[\w-]+\.[\w-]+',  # JWT
        r'-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',
        r'(?i)AWS[_-]?(ACCESS|SECRET)[_-]?KEY',
        r'(?i)(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}',  # GitHub tokens
        r'sk-[A-Za-z0-9]{48}',  # OpenAI keys
        r'(?i)ANTHROPIC[_-]?API[_-]?KEY',
    ]

    # Files that likely contain secrets
    SENSITIVE_FILES = [
        '.env', '.env.local', '.env.production',
        'credentials.json', 'secrets.json', 'config.json',
        '.netrc', '.npmrc', '.pypirc',
        'id_rsa', 'id_ed25519', '*.pem', '*.key',
    ]

    def __init__(self, repo_path: str = ".", verbose: bool = False):
        self.repo_path = Path(repo_path).resolve()
        self.verbose = verbose

    def _run(self, cmd: List[str], check: bool = False) -> Tuple[int, str, str]:
        """Run a git command."""
        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def _log(self, msg: str):
        """Log if verbose."""
        if self.verbose:
            print(f"  [git] {msg}")

    def is_repo(self) -> bool:
        """Check if current directory is a git repo."""
        code, _, _ = self._run(["git", "rev-parse", "--git-dir"])
        return code == 0

    def get_status(self) -> List[FileChange]:
        """Get list of changed files."""
        code, stdout, _ = self._run(["git", "status", "--porcelain"])
        if code != 0:
            return []

        changes = []
        for line in stdout.strip().split('\n'):
            if not line:
                continue

            # Parse porcelain format: XY filename
            status = line[:2]
            path = line[3:].strip()

            # Handle renamed files
            if ' -> ' in path:
                path = path.split(' -> ')[1]

            staged = status[0] != ' ' and status[0] != '?'
            file_status = status[0] if staged else status[1]

            changes.append(FileChange(
                path=path,
                status=file_status,
                staged=staged
            ))

        return changes

    def get_current_branch(self) -> str:
        """Get current branch name."""
        code, stdout, _ = self._run(["git", "branch", "--show-current"])
        return stdout.strip() if code == 0 else ""

    def has_secrets(self, files: List[str]) -> List[Tuple[str, str]]:
        """
        Check files for potential secrets.

        Returns:
            List of (file, matched_pattern) tuples
        """
        found = []

        for file_path in files:
            full_path = self.repo_path / file_path

            # Check if filename matches sensitive patterns
            for pattern in self.SENSITIVE_FILES:
                if '*' in pattern:
                    import fnmatch
                    if fnmatch.fnmatch(file_path, pattern):
                        found.append((file_path, f"Sensitive filename: {pattern}"))
                        continue
                elif file_path.endswith(pattern) or file_path == pattern:
                    found.append((file_path, f"Sensitive file: {pattern}"))
                    continue

            # Check file contents
            if full_path.exists() and full_path.is_file():
                try:
                    content = full_path.read_text(errors='ignore')
                    for pattern in self.SECRET_PATTERNS:
                        if re.search(pattern, content):
                            found.append((file_path, f"Pattern match: {pattern[:30]}..."))
                            break
                except Exception:
                    pass

        return found

    def run_hooks(self) -> GitResult:
        """Run pre-commit hooks if they exist."""
        hook_path = self.repo_path / ".git" / "hooks" / "pre-commit"

        if not hook_path.exists():
            return GitResult(
                status=GitStatus.SKIPPED,
                message="No pre-commit hook found"
            )

        self._log("Running pre-commit hook...")
        code, stdout, stderr = self._run([str(hook_path)])

        if code == 0:
            return GitResult(
                status=GitStatus.SUCCESS,
                message="Pre-commit hook passed",
                output=stdout
            )
        else:
            return GitResult(
                status=GitStatus.FAILED,
                message="Pre-commit hook failed",
                output=stdout,
                error=stderr
            )

    def stage_files(self, files: List[str]) -> GitResult:
        """Stage specific files for commit."""
        if not files:
            return GitResult(
                status=GitStatus.SKIPPED,
                message="No files to stage"
            )

        cmd = ["git", "add"] + files
        code, stdout, stderr = self._run(cmd)

        if code == 0:
            return GitResult(
                status=GitStatus.SUCCESS,
                message=f"Staged {len(files)} files",
                files_affected=files
            )
        else:
            return GitResult(
                status=GitStatus.FAILED,
                message="Failed to stage files",
                error=stderr
            )

    def commit(self, message: str, allow_empty: bool = False) -> GitResult:
        """Create a commit."""
        cmd = ["git", "commit", "-m", message]
        if allow_empty:
            cmd.append("--allow-empty")

        code, stdout, stderr = self._run(cmd)

        if code == 0:
            return GitResult(
                status=GitStatus.SUCCESS,
                message="Commit created",
                output=stdout
            )
        else:
            return GitResult(
                status=GitStatus.FAILED,
                message="Commit failed",
                output=stdout,
                error=stderr
            )

    def safe_commit(
        self,
        files: List[str],
        message: str,
        check_secrets: bool = True,
        run_pre_commit: bool = True,
        author: Optional[str] = None
    ) -> GitResult:
        """
        Safely commit files with validation.

        Args:
            files: Files to commit (empty = all staged)
            message: Commit message
            check_secrets: Whether to check for secrets
            run_pre_commit: Whether to run pre-commit hooks
            author: Optional author override

        Returns:
            GitResult with status
        """
        self._log(f"Safe commit: {len(files)} files")

        # 1. Check for secrets
        if check_secrets and files:
            secrets = self.has_secrets(files)
            if secrets:
                secret_list = "\n".join([f"  - {f}: {p}" for f, p in secrets])
                return GitResult(
                    status=GitStatus.BLOCKED,
                    message=f"Potential secrets detected:\n{secret_list}",
                    files_affected=[f for f, _ in secrets]
                )

        # 2. Stage files
        if files:
            stage_result = self.stage_files(files)
            if stage_result.status == GitStatus.FAILED:
                return stage_result

        # 3. Run pre-commit hooks
        if run_pre_commit:
            hook_result = self.run_hooks()
            if hook_result.status == GitStatus.FAILED:
                return hook_result

        # 4. Create commit
        return self.commit(message)

    def push(
        self,
        remote: str = "origin",
        branch: Optional[str] = None,
        force: bool = False,
        set_upstream: bool = False
    ) -> GitResult:
        """
        Push to remote with safety checks.

        Args:
            remote: Remote name
            branch: Branch to push (default: current)
            force: Allow force push (DANGEROUS)
            set_upstream: Set upstream tracking

        Returns:
            GitResult
        """
        branch = branch or self.get_current_branch()

        # Block force push to main/master
        if force and branch in ('main', 'master'):
            return GitResult(
                status=GitStatus.BLOCKED,
                message="Force push to main/master is blocked for safety"
            )

        cmd = ["git", "push"]
        if set_upstream:
            cmd.extend(["-u", remote, branch])
        else:
            cmd.extend([remote, branch])

        if force:
            cmd.append("--force")

        self._log(f"Pushing to {remote}/{branch}")
        code, stdout, stderr = self._run(cmd)

        if code == 0:
            return GitResult(
                status=GitStatus.SUCCESS,
                message=f"Pushed to {remote}/{branch}",
                output=stdout
            )
        else:
            return GitResult(
                status=GitStatus.FAILED,
                message="Push failed",
                output=stdout,
                error=stderr
            )

    def create_branch(self, name: str, checkout: bool = True) -> GitResult:
        """Create a new branch."""
        if checkout:
            cmd = ["git", "checkout", "-b", name]
        else:
            cmd = ["git", "branch", name]

        code, stdout, stderr = self._run(cmd)

        if code == 0:
            return GitResult(
                status=GitStatus.SUCCESS,
                message=f"Created branch: {name}",
                output=stdout
            )
        else:
            return GitResult(
                status=GitStatus.FAILED,
                message=f"Failed to create branch: {name}",
                error=stderr
            )

    def get_diff(self, staged: bool = True) -> str:
        """Get diff of changes."""
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")

        code, stdout, _ = self._run(cmd)
        return stdout if code == 0 else ""

    def get_log(self, limit: int = 5, format: str = "oneline") -> List[str]:
        """Get recent commit log."""
        cmd = ["git", "log", f"-{limit}", f"--format={format}"]
        code, stdout, _ = self._run(cmd)
        return stdout.strip().split('\n') if code == 0 else []

    def generate_commit_message(self, diff: Optional[str] = None) -> str:
        """Generate a commit message based on changes."""
        if diff is None:
            diff = self.get_diff(staged=True)

        if not diff:
            return "Update files"

        # Analyze diff to generate message
        lines_added = diff.count('\n+') - diff.count('\n+++')
        lines_removed = diff.count('\n-') - diff.count('\n---')

        # Get changed files
        changes = self.get_status()
        staged = [c for c in changes if c.staged]

        if not staged:
            return "Update files"

        # Simple heuristics
        file_types = set()
        for change in staged:
            ext = Path(change.path).suffix.lower()
            file_types.add(ext or 'files')

        if len(staged) == 1:
            action = {
                'A': 'Add',
                'M': 'Update',
                'D': 'Remove',
                'R': 'Rename',
            }.get(staged[0].status, 'Update')
            return f"{action} {staged[0].path}"

        # Multiple files
        main_type = max(file_types, key=lambda t: sum(1 for c in staged if c.path.endswith(t)))
        return f"Update {len(staged)} {main_type} files (+{lines_added}/-{lines_removed})"


# Convenience functions

def safe_commit(files: List[str], message: str, repo_path: str = ".") -> GitResult:
    """Safely commit files."""
    git = GitAutomator(repo_path)
    return git.safe_commit(files, message)


def get_changes(repo_path: str = ".") -> List[FileChange]:
    """Get list of changed files."""
    git = GitAutomator(repo_path)
    return git.get_status()


def push_changes(remote: str = "origin", repo_path: str = ".") -> GitResult:
    """Push changes to remote."""
    git = GitAutomator(repo_path)
    return git.push(remote)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NEURO GIT AUTOMATOR TEST")
    print("=" * 60)

    git = GitAutomator(verbose=True)

    # Test 1: Check if repo
    print("\nTest 1: Check repository")
    print("-" * 40)
    is_repo = git.is_repo()
    print(f"  Is git repo: {is_repo}")

    if not is_repo:
        print("  Not in a git repository, skipping other tests")
    else:
        # Test 2: Get status
        print("\nTest 2: Get status")
        print("-" * 40)
        changes = git.get_status()
        print(f"  {len(changes)} changed files")
        for c in changes[:5]:
            print(f"    [{c.status}] {c.path} (staged: {c.staged})")

        # Test 3: Current branch
        print("\nTest 3: Current branch")
        print("-" * 40)
        branch = git.get_current_branch()
        print(f"  Branch: {branch}")

        # Test 4: Recent commits
        print("\nTest 4: Recent commits")
        print("-" * 40)
        commits = git.get_log(limit=3)
        for c in commits:
            print(f"  {c[:70]}...")

        # Test 5: Secrets detection
        print("\nTest 5: Secrets detection (mock)")
        print("-" * 40)
        # Create a mock file with a "secret"
        test_secrets = git.has_secrets([".env", "config.json"])
        print(f"  Detected sensitive files: {len(test_secrets)}")

        # Test 6: Generate commit message
        print("\nTest 6: Generate commit message")
        print("-" * 40)
        msg = git.generate_commit_message()
        print(f"  Suggested message: {msg}")

    print("\n" + "=" * 60)
    print("Tests completed!")
