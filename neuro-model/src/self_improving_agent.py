"""
Self-Improving AGI Agent

An autonomous agent that:
1. Tries to accomplish tasks
2. When it fails, searches for solutions
3. Fixes itself by modifying its own code/prompts
4. Learns and remembers solutions
5. Gets better over time
"""

import json
import os
import re
import time
import requests
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

# Import tools
import sys
sys.path.insert(0, str(Path(__file__).parent))
from tools import Tools, ToolResult
from autonomous_learning import AutonomousLearner


@dataclass
class TaskResult:
    success: bool
    output: str
    error: Optional[str] = None
    tools_used: List[str] = None
    self_fixes: int = 0


class SelfImprovingAgent:
    """
    An AGI that can fix itself when it encounters problems.

    Core loop:
    1. Receive task
    2. Try to accomplish it
    3. If failed, analyze error
    4. Search for solutions
    5. Apply fix (modify prompts, learn new patterns)
    6. Retry
    7. Store solution for future
    """

    def __init__(
        self,
        model: str = "ministral-3:8b",
        max_retries: int = 3,
        max_tool_calls: int = 10
    ):
        self.model = model
        self.max_retries = max_retries
        self.max_tool_calls = max_tool_calls
        self.base_url = "http://localhost:11434"

        # Tools and learning
        self.tools = Tools()
        self.learner = AutonomousLearner()

        # Self-improvement storage
        self.fixes_applied: List[Dict] = []
        self.learned_patterns: Dict[str, str] = {}
        self.error_solutions: Dict[str, str] = {}

        # Load previous learnings
        self._load_improvements()

        # Dynamic system prompt that evolves
        self.base_system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build system prompt including learned patterns."""
        base = '''You are Neuro, a self-improving AGI. You have real tools and can fix yourself.

## Your Capabilities
You can use tools to interact with the real world. When you encounter errors, you analyze them and find solutions.

## Available Tools
- web_search(query): Search the internet for information
- web_fetch(url): Get content from a URL
- github_user(username): Get GitHub profile
- github_repos(username): List repositories
- github_repo_info(owner, repo): Get repo details
- read_file(path): Read a file
- write_file(path, content): Write to a file
- list_files(path, pattern): List directory contents
- run_command(command): Execute shell command
- run_python(code): Run Python code
- remember(key, value): Store in memory
- recall(key): Retrieve from memory

## How to Use Tools
<tool>tool_name</tool>
<args>{"param1": "value1", "param2": "value2"}</args>

## CRITICAL RULES
1. ALWAYS use tools for external information - NEVER make up data
2. If a tool fails, analyze the error and try a different approach
3. If you don't know how to do something, SEARCH THE WEB for solutions
4. Learn from every interaction

## Self-Improvement Protocol
When you encounter an error:
1. Analyze what went wrong
2. Search the web for solutions if needed
3. Try an alternative approach
4. Remember the solution for next time
'''

        # Add learned patterns
        if self.learned_patterns:
            base += "\n## Learned Patterns (from previous sessions)\n"
            for pattern, solution in list(self.learned_patterns.items())[:10]:
                base += f"- {pattern}: {solution}\n"

        # Add known error solutions
        if self.error_solutions:
            base += "\n## Known Error Solutions\n"
            for error, solution in list(self.error_solutions.items())[:10]:
                base += f"- {error[:50]}: {solution}\n"

        return base

    def execute_task(self, task: str) -> TaskResult:
        """Execute a task with self-improvement capabilities."""
        print(f"\n{'='*60}")
        print(f"TASK: {task}")
        print('='*60)

        tools_used = []
        self_fixes = 0

        for attempt in range(self.max_retries):
            print(f"\n--- Attempt {attempt + 1}/{self.max_retries} ---")

            result = self._try_task(task, tools_used)

            if result.success:
                # Store successful approach
                self._learn_from_success(task, result)
                return TaskResult(
                    success=True,
                    output=result.output,
                    tools_used=tools_used,
                    self_fixes=self_fixes
                )

            # Task failed - try to self-improve
            print(f"\n[SELF-IMPROVE] Analyzing failure: {result.error}")

            fix_result = self._attempt_self_fix(task, result.error)
            if fix_result:
                self_fixes += 1
                print(f"[SELF-IMPROVE] Applied fix: {fix_result}")
                # Update system prompt with new knowledge
                self.base_system_prompt = self._build_system_prompt()
            else:
                print("[SELF-IMPROVE] Could not find fix, trying different approach")

        return TaskResult(
            success=False,
            output="",
            error=f"Failed after {self.max_retries} attempts",
            tools_used=tools_used,
            self_fixes=self_fixes
        )

    def _try_task(self, task: str, tools_used: List[str]) -> TaskResult:
        """Try to accomplish a task using the agentic loop."""
        messages = [
            {"role": "system", "content": self.base_system_prompt},
            {"role": "user", "content": task}
        ]

        final_response = ""

        for i in range(self.max_tool_calls):
            try:
                r = requests.post(
                    f"{self.base_url}/api/chat",
                    json={"model": self.model, "messages": messages, "stream": False},
                    timeout=120
                )
                response = r.json()["message"]["content"]
            except Exception as e:
                return TaskResult(False, "", f"LLM error: {e}")

            print(f"\nNeuro: {self._clean_response(response)[:300]}...")

            # Check for tool call
            tool_call = self._parse_tool_call(response)

            if tool_call:
                tool_name, args = tool_call
                tools_used.append(tool_name)

                print(f"  [TOOL] {tool_name}({json.dumps(args)[:100]})")

                result = self.tools.execute(tool_name, args)

                if result.success:
                    print(f"  [OK] {result.output[:150]}...")
                    self.learner.add_memory(result.output[:500], tool_name, tool_name, 0.7)

                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content": f"Tool result:\n{result.output}"})
                else:
                    print(f"  [FAIL] {result.error}")

                    # Try to self-fix on tool error
                    fix = self._handle_tool_error(tool_name, args, result.error)
                    if fix:
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": f"Tool failed: {result.error}\n\nSuggested fix: {fix}\n\nTry again with the fix."})
                    else:
                        messages.append({"role": "assistant", "content": response})
                        messages.append({"role": "user", "content": f"Tool failed: {result.error}\n\nAnalyze the error and try a different approach."})
            else:
                # No tool call - this is the final response
                final_response = response
                break

        if final_response:
            # Check if response indicates failure
            failure_indicators = ["i cannot", "i don't know", "unable to", "failed to", "error"]
            is_failure = any(ind in final_response.lower() for ind in failure_indicators)

            return TaskResult(
                success=not is_failure,
                output=final_response,
                error="Task may have failed" if is_failure else None
            )

        return TaskResult(False, "", "No final response generated")

    def _attempt_self_fix(self, task: str, error: str) -> Optional[str]:
        """Attempt to fix the problem by searching for solutions."""
        # First check if we already know this error
        for known_error, solution in self.error_solutions.items():
            if known_error.lower() in error.lower():
                return solution

        # Search the web for solutions
        search_query = f"how to fix: {error[:100]}"
        print(f"[SEARCH] {search_query}")

        result = self.tools.web_search(search_query)

        if result.success and result.output and "No results" not in result.output:
            # Extract solution from search results
            solution = self._extract_solution(result.output, error)
            if solution:
                # Store for future
                self.error_solutions[error[:100]] = solution
                self._save_improvements()
                return solution

        return None

    def _handle_tool_error(self, tool_name: str, args: Dict, error: str) -> Optional[str]:
        """Handle a tool execution error."""
        # Common fixes for known errors
        if "missing" in error.lower() and "argument" in error.lower():
            return f"The {tool_name} tool requires all arguments. Check the tool signature and provide all required parameters."

        if "timeout" in error.lower():
            return "The operation timed out. Try with a simpler query or break it into smaller steps."

        if "not found" in error.lower():
            return "The resource was not found. Verify the path/name is correct, or search for alternatives."

        if "permission" in error.lower():
            return "Permission denied. Try a different approach that doesn't require elevated permissions."

        return None

    def _extract_solution(self, search_results: str, error: str) -> Optional[str]:
        """Extract a solution from search results."""
        # Simple extraction - in reality would use LLM to analyze
        lines = search_results.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['solution', 'fix', 'resolve', 'try', 'use']):
                return line.strip()[:200]
        return None

    def _learn_from_success(self, task: str, result: TaskResult) -> None:
        """Learn from a successful task completion."""
        # Extract key patterns
        task_type = self._classify_task(task)

        if task_type and result.tools_used:
            pattern = f"{task_type} tasks"
            solution = f"Use {', '.join(set(result.tools_used))}"
            self.learned_patterns[pattern] = solution
            self._save_improvements()

    def _classify_task(self, task: str) -> Optional[str]:
        """Classify the type of task."""
        task_lower = task.lower()

        if any(w in task_lower for w in ['github', 'repo', 'repository']):
            return "GitHub"
        if any(w in task_lower for w in ['search', 'find', 'look up']):
            return "Search"
        if any(w in task_lower for w in ['file', 'read', 'write', 'list']):
            return "File"
        if any(w in task_lower for w in ['python', 'code', 'calculate', 'compute']):
            return "Code"
        if any(w in task_lower for w in ['command', 'run', 'execute']):
            return "Shell"

        return None

    def _parse_tool_call(self, response: str) -> Optional[tuple]:
        """Parse a tool call from response."""
        tool_match = re.search(r'<tool>(\w+)</tool>', response)
        args_match = re.search(r'<args>({.*?})</args>', response, re.DOTALL)

        if tool_match:
            tool_name = tool_match.group(1)
            args = {}
            if args_match:
                try:
                    args = json.loads(args_match.group(1))
                except:
                    pass
            return (tool_name, args)
        return None

    def _clean_response(self, response: str) -> str:
        """Clean response for display."""
        clean = re.sub(r'<tool>.*?</tool>', '[TOOL]', response)
        clean = re.sub(r'<args>.*?</args>', '', clean, flags=re.DOTALL)
        return clean.strip()

    def _save_improvements(self) -> None:
        """Save learned improvements to disk."""
        improvements_file = Path.home() / ".neuro" / "improvements.json"
        improvements_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "learned_patterns": self.learned_patterns,
            "error_solutions": self.error_solutions,
            "fixes_applied": self.fixes_applied[-100:],  # Keep last 100
            "last_updated": datetime.now().isoformat()
        }

        try:
            with open(improvements_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _load_improvements(self) -> None:
        """Load previous improvements from disk."""
        improvements_file = Path.home() / ".neuro" / "improvements.json"

        if not improvements_file.exists():
            return

        try:
            with open(improvements_file) as f:
                data = json.load(f)

            self.learned_patterns = data.get("learned_patterns", {})
            self.error_solutions = data.get("error_solutions", {})
            self.fixes_applied = data.get("fixes_applied", [])

            print(f"[LOADED] {len(self.learned_patterns)} patterns, {len(self.error_solutions)} solutions")
        except Exception:
            pass

    def get_stats(self) -> Dict:
        """Get agent statistics."""
        return {
            "learned_patterns": len(self.learned_patterns),
            "error_solutions": len(self.error_solutions),
            "fixes_applied": len(self.fixes_applied),
            "learner_state": self.learner.get_state()
        }


def main():
    """Test the self-improving agent."""
    agent = SelfImprovingAgent()

    tests = [
        "Look up the GitHub user 'ezekaj' and list their repositories",
        "Search the web for 'how to build an AGI' and summarize the results",
        "Calculate the factorial of 10 using Python",
        "What files are in the current directory?",
    ]

    for task in tests:
        result = agent.execute_task(task)
        print(f"\n{'='*60}")
        print(f"RESULT: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"Tools used: {result.tools_used}")
        print(f"Self-fixes: {result.self_fixes}")
        if result.output:
            print(f"Output: {result.output[:300]}...")

    print(f"\n{'='*60}")
    print("AGENT STATS")
    print('='*60)
    print(json.dumps(agent.get_stats(), indent=2, default=str))


if __name__ == "__main__":
    main()
