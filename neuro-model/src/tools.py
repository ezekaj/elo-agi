"""
Neuro AGI Tool System

Real capabilities that the AGI can use to interact with the world.
"""

import subprocess
import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


class Tools:
    """
    AGI Tool System - Real capabilities for Neuro.

    Provides actual access to:
    - Web search
    - Browser automation
    - GitHub
    - File system
    - Code execution
    - System commands
    """

    def __init__(self, work_dir: str = "."):
        self.work_dir = Path(work_dir).resolve()
        self.memory = {}  # Simple key-value memory
        self._browser = None  # Lazy-loaded browser agent

    def execute(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name."""
        tools = {
            "web_search": self.web_search,
            "web_fetch": self.web_fetch,
            "browse_web": self.browse_web,
            "github_user": self.github_user,
            "github_repos": self.github_repos,
            "github_repo_info": self.github_repo_info,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_files": self.list_files,
            "run_command": self.run_command,
            "run_python": self.run_python,
            "remember": self.remember,
            "recall": self.recall,
        }

        if tool_name not in tools:
            return ToolResult(False, "", f"Unknown tool: {tool_name}")

        try:
            return tools[tool_name](**args)
        except Exception as e:
            return ToolResult(False, "", str(e))

    # === Web Tools ===

    def web_search(self, query: str, num_results: int = 5) -> ToolResult:
        """Search the web using DuckDuckGo Instant Answer API (JSON)."""
        try:
            import urllib.request
            import urllib.parse

            # DuckDuckGo Instant Answer API - returns JSON!
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"

            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))

            results = []

            # Abstract (main answer)
            if data.get('Abstract'):
                results.append(f"**{data.get('Heading', query)}**\n{data['Abstract']}")
                if data.get('AbstractURL'):
                    results.append(f"Source: {data['AbstractURL']}")

            # Definition
            if data.get('Definition'):
                results.append(f"Definition: {data['Definition']}")

            # Answer (direct answer)
            if data.get('Answer'):
                results.append(f"Answer: {data['Answer']}")

            # Related topics
            for topic in data.get('RelatedTopics', [])[:num_results]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append(f"• {topic['Text']}")

            # Results
            for result in data.get('Results', [])[:num_results]:
                if isinstance(result, dict) and result.get('Text'):
                    results.append(f"→ {result['Text']}")

            if results:
                return ToolResult(True, "\n\n".join(results))

            # Fallback to HTML scraping if no JSON results
            return self._web_search_html(query, num_results)

        except Exception as e:
            # Fallback to HTML scraping
            return self._web_search_html(query, num_results)

    def _web_search_html(self, query: str, num_results: int = 5) -> ToolResult:
        """Fallback HTML-based search."""
        try:
            result = subprocess.run(
                ["curl", "-s", f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"],
                capture_output=True,
                text=True,
                timeout=30
            )

            html = result.stdout
            titles = re.findall(r'class="result__a"[^>]*>([^<]+)</a>', html)
            snippets = re.findall(r'class="result__snippet">([^<]+)</a>', html)

            results = []
            for i, (title, snippet) in enumerate(zip(titles[:num_results], snippets[:num_results])):
                results.append(f"{i+1}. {title}\n   {snippet}")

            if results:
                return ToolResult(True, "\n\n".join(results))
            else:
                return ToolResult(True, f"No results found for: {query}")

        except Exception as e:
            return ToolResult(False, "", f"Web search failed: {e}")

    def web_fetch(self, url: str) -> ToolResult:
        """Fetch content from a URL."""
        try:
            result = subprocess.run(
                ["curl", "-s", "-L", "--max-time", "30", url],
                capture_output=True,
                text=True,
                timeout=35
            )

            content = result.stdout
            # Strip HTML tags for readability
            text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()

            # Limit output
            if len(text) > 5000:
                text = text[:5000] + "... [truncated]"

            return ToolResult(True, text)

        except Exception as e:
            return ToolResult(False, "", f"Fetch failed: {e}")

    def browse_web(
        self,
        url: str = None,
        action: str = "read",
        selector: str = None,
        text: str = None,
        query: str = None
    ) -> ToolResult:
        """
        Full browser automation - navigate, click, fill forms, search.

        Args:
            url: URL to navigate to (for action="goto")
            action: One of "goto", "read", "click", "fill", "search", "links", "screenshot"
            selector: CSS selector (for click/fill actions)
            text: Text to type (for fill action)
            query: Search query (for search action)

        Returns:
            ToolResult with the action result
        """
        try:
            # Lazy load browser agent
            if self._browser is None:
                from browser_agent import BrowserAgent, BROWSER_AVAILABLE
                if not BROWSER_AVAILABLE:
                    return ToolResult(
                        False, "",
                        "Browser not available. Install: pip install playwright && playwright install"
                    )
                self._browser = BrowserAgent(headless=True)

            # Handle different actions
            if action == "goto" and url:
                result = self._browser.goto(url)
                if result.get('success'):
                    content = self._browser.get_content(max_length=2000)
                    return ToolResult(True, f"Opened: {result['title']}\n\n{content}")
                return ToolResult(False, "", result.get('error', 'Failed to open URL'))

            elif action == "read":
                content = self._browser.get_content()
                title = self._browser.get_title()
                url_now = self._browser.get_url()
                return ToolResult(True, f"Title: {title}\nURL: {url_now}\n\n{content}")

            elif action == "click" and selector:
                result = self._browser.click(selector)
                if result.get('success'):
                    return ToolResult(True, f"Clicked: {selector}")
                return ToolResult(False, "", result.get('error', 'Click failed'))

            elif action == "fill" and selector and text:
                result = self._browser.fill(selector, text)
                if result.get('success'):
                    return ToolResult(True, f"Filled '{selector}' with text")
                return ToolResult(False, "", result.get('error', 'Fill failed'))

            elif action == "search" and query:
                results = self._browser.search_google(query)
                if results:
                    output = f"Search results for '{query}':\n\n"
                    for i, r in enumerate(results[:10], 1):
                        output += f"{i}. {r['text']}\n   {r['href']}\n\n"
                    return ToolResult(True, output)
                return ToolResult(True, "No results found")

            elif action == "links":
                links = self._browser.get_links()
                if links:
                    output = "Links on page:\n"
                    for l in links[:15]:
                        output += f"- {l['text'][:40]}: {l['href'][:60]}\n"
                    return ToolResult(True, output)
                return ToolResult(True, "No links found")

            elif action == "screenshot":
                path = self._browser.screenshot()
                if path and not path.startswith("Error"):
                    return ToolResult(True, f"Screenshot saved: {path}")
                return ToolResult(False, "", path or "Screenshot failed")

            else:
                return ToolResult(
                    False, "",
                    "Invalid action. Use: goto, read, click, fill, search, links, screenshot"
                )

        except ImportError:
            return ToolResult(False, "", "Browser agent not available")
        except Exception as e:
            return ToolResult(False, "", f"Browser error: {e}")

    # === GitHub Tools ===

    def github_user(self, username: str) -> ToolResult:
        """Get GitHub user info using gh CLI."""
        try:
            result = subprocess.run(
                ["gh", "api", f"/users/{username}"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return ToolResult(False, "", result.stderr)

            data = json.loads(result.stdout)
            info = f"""GitHub User: {data.get('login')}
Name: {data.get('name', 'N/A')}
Bio: {data.get('bio', 'N/A')}
Location: {data.get('location', 'N/A')}
Public Repos: {data.get('public_repos', 0)}
Followers: {data.get('followers', 0)}
Following: {data.get('following', 0)}
URL: {data.get('html_url')}"""

            return ToolResult(True, info)

        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "GitHub API timed out")
        except json.JSONDecodeError:
            return ToolResult(False, "", "Failed to parse GitHub response")
        except FileNotFoundError:
            return ToolResult(False, "", "gh CLI not installed. Run: brew install gh && gh auth login")
        except Exception as e:
            return ToolResult(False, "", f"GitHub error: {e}")

    def github_repos(self, username: str, limit: int = 10) -> ToolResult:
        """List user's GitHub repositories."""
        try:
            result = subprocess.run(
                ["gh", "api", f"/users/{username}/repos?sort=updated&per_page={limit}"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return ToolResult(False, "", result.stderr)

            repos = json.loads(result.stdout)
            lines = [f"Repositories for {username}:\n"]

            for repo in repos:
                stars = repo.get('stargazers_count', 0)
                desc = repo.get('description') or 'No description'
                desc = desc[:60] if len(desc) > 60 else desc
                lines.append(f"  - {repo['name']} ({stars} stars)")
                lines.append(f"    {desc}")

            return ToolResult(True, "\n".join(lines))

        except Exception as e:
            return ToolResult(False, "", f"GitHub error: {e}")

    def github_repo_info(self, owner: str, repo: str) -> ToolResult:
        """Get detailed info about a repository."""
        try:
            result = subprocess.run(
                ["gh", "api", f"/repos/{owner}/{repo}"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                return ToolResult(False, "", result.stderr)

            data = json.loads(result.stdout)
            info = f"""Repository: {data.get('full_name')}
Description: {data.get('description', 'N/A')}
Language: {data.get('language', 'N/A')}
Stars: {data.get('stargazers_count', 0)}
Forks: {data.get('forks_count', 0)}
Open Issues: {data.get('open_issues_count', 0)}
Created: {data.get('created_at', 'N/A')[:10]}
Updated: {data.get('updated_at', 'N/A')[:10]}
URL: {data.get('html_url')}"""

            return ToolResult(True, info)

        except Exception as e:
            return ToolResult(False, "", f"GitHub error: {e}")

    # === File Tools ===

    def read_file(self, path: str) -> ToolResult:
        """Read a file's contents."""
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.work_dir / file_path

            content = file_path.read_text()

            if len(content) > 10000:
                content = content[:10000] + "\n... [truncated]"

            return ToolResult(True, content)

        except FileNotFoundError:
            return ToolResult(False, "", f"File not found: {path}")
        except Exception as e:
            return ToolResult(False, "", f"Read error: {e}")

    def write_file(self, path: str, content: str) -> ToolResult:
        """Write content to a file."""
        try:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = self.work_dir / file_path

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

            return ToolResult(True, f"Written to {file_path}")

        except Exception as e:
            return ToolResult(False, "", f"Write error: {e}")

    def list_files(self, path: str = ".", pattern: str = "*") -> ToolResult:
        """List files in a directory."""
        try:
            dir_path = Path(path)
            if not dir_path.is_absolute():
                dir_path = self.work_dir / dir_path

            files = list(dir_path.glob(pattern))
            lines = [f"Files in {dir_path}:"]

            for f in sorted(files)[:50]:
                prefix = "📁 " if f.is_dir() else "📄 "
                lines.append(f"  {prefix}{f.name}")

            if len(files) > 50:
                lines.append(f"  ... and {len(files) - 50} more")

            return ToolResult(True, "\n".join(lines))

        except Exception as e:
            return ToolResult(False, "", f"List error: {e}")

    # === Execution Tools ===

    def run_command(self, command: str, timeout: int = 60) -> ToolResult:
        """Run a shell command."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.work_dir
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"

            if len(output) > 5000:
                output = output[:5000] + "\n... [truncated]"

            return ToolResult(
                result.returncode == 0,
                output,
                None if result.returncode == 0 else f"Exit code: {result.returncode}"
            )

        except subprocess.TimeoutExpired:
            return ToolResult(False, "", f"Command timed out after {timeout}s")
        except Exception as e:
            return ToolResult(False, "", f"Command error: {e}")

    def run_python(self, code: str) -> ToolResult:
        """Execute Python code."""
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=self.work_dir
            )

            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]: {result.stderr}"

            return ToolResult(
                result.returncode == 0,
                output,
                None if result.returncode == 0 else f"Exit code: {result.returncode}"
            )

        except subprocess.TimeoutExpired:
            return ToolResult(False, "", "Python execution timed out")
        except Exception as e:
            return ToolResult(False, "", f"Python error: {e}")

    # === Memory Tools ===

    def remember(self, key: str, value: str) -> ToolResult:
        """Store something in memory."""
        self.memory[key] = value
        return ToolResult(True, f"Remembered: {key}")

    def recall(self, key: str) -> ToolResult:
        """Recall something from memory."""
        if key in self.memory:
            return ToolResult(True, self.memory[key])
        return ToolResult(False, "", f"No memory for: {key}")

    # === Tool Info ===

    @staticmethod
    def get_tool_descriptions() -> str:
        """Get descriptions of all available tools."""
        return """Available Tools:

WEB:
- web_search(query, num_results=5): Search the web
- web_fetch(url): Fetch content from a URL
- browse_web(url, action, selector, text, query): Full browser automation
  Actions: goto, read, click, fill, search, links, screenshot

GITHUB:
- github_user(username): Get user profile info
- github_repos(username, limit=10): List user's repositories
- github_repo_info(owner, repo): Get repository details

FILES:
- read_file(path): Read a file
- write_file(path, content): Write to a file
- list_files(path=".", pattern="*"): List directory contents

EXECUTION:
- run_command(command, timeout=60): Run shell command
- run_python(code): Execute Python code

MEMORY:
- remember(key, value): Store in memory
- recall(key): Retrieve from memory

To use a tool, respond with:
<tool>tool_name</tool>
<args>{"arg1": "value1", "arg2": "value2"}</args>
"""


def parse_tool_call(response: str) -> Optional[tuple]:
    """Parse a tool call from LLM response."""
    tool_match = re.search(r'<tool>(\w+)</tool>', response)
    args_match = re.search(r'<args>({.*?})</args>', response, re.DOTALL)

    if tool_match:
        tool_name = tool_match.group(1)
        args = {}

        if args_match:
            try:
                args = json.loads(args_match.group(1))
            except json.JSONDecodeError:
                pass

        return (tool_name, args)

    return None
