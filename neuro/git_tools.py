"""
GitTools - Git operations for NEURO.

Provides:
- git status, diff, log
- git add, commit, push
- Branch operations
- Remote management
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class GitResult:
    """Result of a git operation."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    command: str = ""


class GitTools:
    """
    Git operations tool.
    
    Features:
    - Status and diff viewing
    - Add, commit, push operations
    - Branch management
    - Remote operations
    - Safety checks (secrets, large files)
    """

    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = Path(repo_path).expanduser() if repo_path else Path.cwd()
        self._is_repo: Optional[bool] = None

    def _run_git(self, *args: str, timeout: float = 30.0) -> GitResult:
        """Run a git command."""
        try:
            result = subprocess.run(
                ["git"] + list(args),
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return GitResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                command=" ".join(["git"] + list(args)),
            )
        except subprocess.TimeoutExpired:
            return GitResult(
                success=False,
                stderr=f"Command timed out after {timeout}s",
                command=" ".join(["git"] + list(args)),
            )
        except FileNotFoundError:
            return GitResult(
                success=False,
                stderr="git command not found. Please install git.",
                command=" ".join(["git"] + list(args)),
            )
        except Exception as e:
            return GitResult(
                success=False,
                stderr=str(e),
                command=" ".join(["git"] + list(args)),
            )

    def is_repo(self) -> bool:
        """Check if current directory is a git repository."""
        if self._is_repo is not None:
            return self._is_repo
        
        result = self._run_git("rev-parse", "--git-dir")
        self._is_repo = result.success
        return self._is_repo

    def status(self) -> Dict[str, Any]:
        """Get git status."""
        if not self.is_repo():
            return {"error": "Not a git repository"}

        # Get status
        status_result = self._run_git("status", "--porcelain")
        
        # Get current branch
        branch_result = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        
        # Get ahead/behind count
        ahead_behind_result = self._run_git(
            "rev-list", "--left-right", "--count", f"HEAD...@{{upstream}}",
            timeout=10
        )
        
        # Parse status
        changed_files = []
        if status_result.stdout:
            for line in status_result.stdout.strip().split("\n"):
                if line:
                    status_char = line[:2]
                    file_path = line[3:] if len(line) > 3 else ""
                    changed_files.append({
                        "status": status_char.strip(),
                        "file": file_path,
                    })
        
        # Parse ahead/behind
        ahead, behind = 0, 0
        if ahead_behind_result.success and ahead_behind_result.stdout:
            parts = ahead_behind_result.stdout.strip().split()
            if len(parts) == 2:
                ahead, behind = int(parts[0]), int(parts[1])
        
        return {
            "branch": branch_result.stdout.strip() if branch_result.success else "unknown",
            "changed_files": changed_files,
            "ahead": ahead,
            "behind": behind,
            "has_changes": len(changed_files) > 0,
        }

    def diff(self, path: Optional[str] = None, staged: bool = False) -> str:
        """Get git diff."""
        if not self.is_repo():
            return "Not a git repository"
        
        args = ["diff"]
        if staged:
            args.append("--staged")
        if path:
            args.append("--")
            args.append(path)
        
        result = self._run_git(*args)
        return result.stdout if result.success else result.stderr

    def diff_stat(self, staged: bool = False) -> Dict[str, int]:
        """Get diff statistics."""
        if not self.is_repo():
            return {"error": "Not a git repository"}
        
        args = ["diff", "--numstat"]
        if staged:
            args.append("--staged")
        
        result = self._run_git(*args)
        
        additions, deletions = 0, 0
        files_changed = 0
        
        if result.success and result.stdout:
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            add = int(parts[0]) if parts[0] != "-" else 0
                            del_ = int(parts[1]) if parts[1] != "-" else 0
                            additions += add
                            deletions += del_
                            files_changed += 1
                        except ValueError:
                            pass
        
        return {
            "additions": additions,
            "deletions": deletions,
            "files_changed": files_changed,
        }

    def add(self, files: List[str]) -> GitResult:
        """Stage files."""
        if not self.is_repo():
            return GitResult(success=False, stderr="Not a git repository")
        
        return self._run_git("add", *files)

    def add_all(self) -> GitResult:
        """Stage all changes."""
        return self.add(".")

    def commit(self, message: str, files: Optional[List[str]] = None) -> GitResult:
        """
        Create a commit.
        
        Args:
            message: Commit message
            files: Optional list of files to commit (if None, uses staged files)
        """
        if not self.is_repo():
            return GitResult(success=False, stderr="Not a git repository")
        
        # If specific files provided, stage them first
        if files:
            stage_result = self.add(files)
            if not stage_result.success:
                return stage_result
        
        return self._run_git("commit", "-m", message)

    def push(self, remote: str = "origin", branch: Optional[str] = None, 
             force: bool = False) -> GitResult:
        """Push to remote."""
        if not self.is_repo():
            return GitResult(success=False, stderr="Not a git repository")
        
        args = ["push", remote]
        if force:
            args.append("--force")
        if branch:
            args.append(branch)
        
        return self._run_git(*args, timeout=60.0)

    def log(self, count: int = 10) -> List[Dict[str, str]]:
        """Get commit log."""
        if not self.is_repo():
            return []
        
        format_str = "%H|%an|%ae|%ad|%s"
        result = self._run_git(
            "log", f"-{count}", f"--format={format_str}", "--date=iso"
        )
        
        commits = []
        if result.success and result.stdout:
            for line in result.stdout.strip().split("\n"):
                parts = line.split("|", 4)
                if len(parts) == 5:
                    commits.append({
                        "hash": parts[0][:7],
                        "author": parts[1],
                        "email": parts[2],
                        "date": parts[3],
                        "message": parts[4],
                    })
        
        return commits

    def create_branch(self, name: str, start_point: Optional[str] = None) -> GitResult:
        """Create a new branch."""
        if not self.is_repo():
            return GitResult(success=False, stderr="Not a git repository")
        
        args = ["checkout", "-b", name]
        if start_point:
            args.append(start_point)
        
        return self._run_git(*args)

    def switch_branch(self, name: str) -> GitResult:
        """Switch to a branch."""
        return self._run_git("checkout", name)

    def list_branches(self, remote: bool = False) -> List[str]:
        """List branches."""
        if not self.is_repo():
            return []
        
        args = ["branch"]
        if remote:
            args.append("-r")
        
        result = self._run_git(*args)
        
        branches = []
        if result.success and result.stdout:
            for line in result.stdout.strip().split("\n"):
                branch = line.strip().lstrip("* ").split(" -> ")[0]
                if branch:
                    branches.append(branch)
        
        return branches

    def check_secrets(self, content: Optional[str] = None, 
                      files: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Check for potential secrets in content or files.
        
        Patterns checked:
        - API keys (generic)
        - AWS keys
        - Private keys
        - Passwords in URLs
        - GitHub tokens
        """
        import re
        
        patterns = {
            "aws_access_key": r"AKIA[0-9A-Z]{16}",
            "aws_secret_key": r"(?i)aws[_\-]?secret[_\-]?access[_\-]?key[\s]*[=:]\s*['\"]?[A-Za-z0-9/+=]{40}",
            "github_token": r"gh[pousr]_[A-Za-z0-9_]{36,}",
            "generic_api_key": r"(?i)(api[_\-]?key|apikey)[\s]*[=:]\s*['\"]?[A-Za-z0-9]{20,}",
            "private_key": r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
            "password_in_url": r"://[^:]+:[^@]+@",
            "jwt_token": r"eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
        }
        
        findings = []
        
        if content:
            for pattern_name, pattern in patterns.items():
                matches = re.findall(pattern, content)
                if matches:
                    findings.append({
                        "type": pattern_name,
                        "count": len(matches),
                        "sample": matches[0][:20] + "..." if len(matches[0]) > 20 else matches[0],
                    })
        
        if files:
            for file_path in files:
                try:
                    file_content = Path(file_path).read_text()
                    for pattern_name, pattern in patterns.items():
                        matches = re.findall(pattern, file_content)
                        if matches:
                            findings.append({
                                "file": file_path,
                                "type": pattern_name,
                                "count": len(matches),
                            })
                except:
                    pass
        
        return findings

    def get_remotes(self) -> List[Dict[str, str]]:
        """Get configured remotes."""
        if not self.is_repo():
            return []
        
        result = self._run_git("remote", "-v")
        
        remotes = {}
        if result.success and result.stdout:
            for line in result.stdout.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    url = parts[1]
                    direction = "(fetch)" if "(fetch)" in line else "(push)"
                    if name not in remotes:
                        remotes[name] = {"name": name, "urls": []}
                    remotes[name]["urls"].append({"url": url, "direction": direction})
        
        return list(remotes.values())

    def add_remote(self, name: str, url: str) -> GitResult:
        """Add a remote."""
        return self._run_git("remote", "add", name, url)

    def fetch(self, remote: str = "origin") -> GitResult:
        """Fetch from remote."""
        return self._run_git("fetch", remote, timeout=60.0)

    def pull(self, remote: str = "origin", branch: Optional[str] = None) -> GitResult:
        """Pull from remote."""
        args = ["pull", remote]
        if branch:
            args.append(branch)
        return self._run_git(*args, timeout=120.0)


# Convenience functions for tool executor
def git_status() -> Dict[str, Any]:
    """Get git status."""
    return GitTools().status()


def git_diff(path: Optional[str] = None, staged: bool = False) -> str:
    """Get git diff."""
    return GitTools().diff(path, staged)


def git_add(files: List[str]) -> GitResult:
    """Stage files."""
    return GitTools().add(files)


def git_commit(message: str, files: Optional[List[str]] = None) -> GitResult:
    """Create commit."""
    return GitTools().commit(message, files)


def git_push(remote: str = "origin", branch: Optional[str] = None) -> GitResult:
    """Push to remote."""
    return GitTools().push(remote, branch)


def git_log(count: int = 5) -> List[Dict[str, str]]:
    """Get commit log."""
    return GitTools().log(count)


def git_check_secrets(content: Optional[str] = None) -> List[Dict[str, Any]]:
    """Check for secrets."""
    return GitTools().check_secrets(content=content)
