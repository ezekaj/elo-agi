"""
Curiosity Loop - Autonomous Self-Learning System

This runs continuously, exploring knowledge sources and self-improving.
The AGI doesn't wait for permission - it actively seeks knowledge.
"""

import time
import json
import random
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from datetime import datetime

# Import tools
import sys

sys.path.insert(0, str(Path(__file__).parent))
from .tools import Tools
from .autonomous_learning import AutonomousLearner


@dataclass
class LearningTarget:
    """A topic to learn about."""

    topic: str
    source: str  # arxiv, github, web, etc.
    priority: float
    attempts: int = 0
    last_attempt: float = 0
    learned: bool = False


class CuriosityLoop:
    """
    Autonomous learning loop that continuously explores and learns.

    This is the "always curious" brain that:
    1. Identifies what it doesn't know
    2. Actively searches for information
    3. Processes and stores learnings
    4. Never stops exploring
    """

    # Knowledge sources to explore
    SOURCES = {
        "arxiv": {
            "url": "https://arxiv.org/search/?query={query}&searchtype=all",
            "topics": [
                "AGI",
                "neural networks",
                "cognitive architecture",
                "machine learning",
                "reinforcement learning",
                "natural language processing",
                "computer vision",
                "reasoning",
                "memory systems",
                "attention mechanisms",
            ],
        },
        "github": {
            "type": "api",
            "topics": [
                "autonomous-agents",
                "agi",
                "cognitive-architecture",
                "self-improving-ai",
                "neural-symbolic",
                "llm-agents",
                "tool-use",
            ],
        },
        "web": {
            "topics": [
                "how to build AGI",
                "self-improving AI systems",
                "cognitive science",
                "neuroscience of learning",
                "artificial general intelligence 2024",
                "latest AI research",
                "machine consciousness",
            ]
        },
    }

    def __init__(self, tools: Tools, learner: AutonomousLearner, verbose: bool = True):
        self.tools = tools
        self.learner = learner
        self.verbose = verbose

        self.running = False
        self.thread = None

        # Learning state
        self.targets: List[LearningTarget] = []
        self.explored: Set[str] = set()
        self.learnings: List[Dict] = []

        # Stats
        self.cycles = 0
        self.searches_done = 0
        self.facts_learned = 0

        # Load previous state
        self._load_state()

        # Initialize targets from sources
        self._init_targets()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [CURIOSITY] {msg}")

    def _init_targets(self) -> None:
        """Initialize learning targets from sources."""
        for source, config in self.SOURCES.items():
            for topic in config.get("topics", []):
                key = f"{source}:{topic}"
                if key not in self.explored:
                    self.targets.append(
                        LearningTarget(
                            topic=topic, source=source, priority=random.uniform(0.5, 1.0)
                        )
                    )

    def start(self) -> None:
        """Start the autonomous learning loop."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self._log("Autonomous learning STARTED - I will explore continuously")

    def stop(self) -> None:
        """Stop the learning loop."""
        self.running = False
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=2)
            except (KeyboardInterrupt, Exception):
                pass  # Don't block on second interrupt
        try:
            self._save_state()
        except Exception:
            pass
        self._log(f"Stopped after {self.cycles} cycles, {self.facts_learned} facts learned")

    def _loop(self) -> None:
        """Main curiosity loop - runs continuously."""
        while self.running:
            self.cycles += 1

            # 1. Pick highest priority unexplored target
            target = self._pick_target()
            if not target:
                self._log("No more targets - generating new curiosity...")
                self._generate_new_curiosity()
                time.sleep(10)
                continue

            self._log(f"Cycle {self.cycles}: Exploring '{target.topic}' via {target.source}")

            # 2. Execute exploration
            try:
                if target.source == "arxiv":
                    self._explore_arxiv(target)
                elif target.source == "github":
                    self._explore_github(target)
                elif target.source == "web":
                    self._explore_web(target)

                target.attempts += 1
                target.last_attempt = time.time()
                self.searches_done += 1

            except Exception as e:
                self._log(f"Exploration failed: {e}")
                target.attempts += 1

            # 3. Mark as explored after 3 attempts
            if target.attempts >= 3:
                self.explored.add(f"{target.source}:{target.topic}")
                target.learned = True

            # 4. Save progress
            if self.cycles % 5 == 0:
                self._save_state()

            # 5. Brief pause between explorations
            time.sleep(random.uniform(5, 15))

    def _pick_target(self) -> Optional[LearningTarget]:
        """Pick the best target to explore next."""
        available = [t for t in self.targets if not t.learned and t.attempts < 3]
        if not available:
            return None

        # Sort by priority and least attempts
        available.sort(key=lambda t: (t.priority, -t.attempts), reverse=True)
        return available[0]

    def _explore_arxiv(self, target: LearningTarget) -> None:
        """Explore arXiv for a topic."""
        query = target.topic.replace(" ", "+")

        # Try direct arXiv API/search
        urls_to_try = [
            f"https://arxiv.org/search/?query={query}&searchtype=all",
            f"https://export.arxiv.org/api/query?search_query=all:{query}&max_results=5",
        ]

        for url in urls_to_try:
            result = self.tools.web_fetch(url)
            if result.success and result.output and len(result.output) > 100:
                self._process_learning(target.topic, "arxiv", result.output[:3000])
                self._log(f"  Learned about '{target.topic}' from arXiv")
                return

        # Fallback to web search
        result = self.tools.web_search(f"arxiv {target.topic} research paper")
        if result.success and result.output and "No results" not in result.output:
            self._process_learning(target.topic, "arxiv", result.output)

    def _explore_github(self, target: LearningTarget) -> None:
        """Explore GitHub for a topic."""
        query = target.topic.replace(" ", "-")

        # Try GitHub API via gh CLI
        result = self.tools.run_command(
            f"gh search repos {target.topic} --limit 5 --json name,description,stargazersCount 2>/dev/null || echo 'no results'"
        )

        if result.success and result.output and "no results" not in result.output:
            self._process_learning(target.topic, "github", result.output)
            self._log(f"  Found GitHub repos for '{target.topic}'")
            return

        # Fallback: fetch GitHub trending/topics
        result = self.tools.web_fetch(f"https://github.com/topics/{query}")
        if result.success and result.output and len(result.output) > 100:
            self._process_learning(target.topic, "github", result.output[:2000])
            self._log(f"  Explored GitHub topic '{target.topic}'")

    def _explore_web(self, target: LearningTarget) -> None:
        """General web exploration."""
        # Try multiple approaches
        queries = [
            target.topic,
            f"{target.topic} explained",
            f"what is {target.topic}",
        ]

        for query in queries:
            result = self.tools.web_search(query)
            if (
                result.success
                and result.output
                and "No results" not in result.output
                and len(result.output) > 50
            ):
                self._process_learning(target.topic, "web", result.output)
                self._log(f"  Learned from web about '{target.topic}'")
                return

        # Try Wikipedia as fallback
        wiki_topic = target.topic.replace(" ", "_")
        result = self.tools.web_fetch(f"https://en.wikipedia.org/wiki/{wiki_topic}")
        if result.success and result.output and len(result.output) > 200:
            self._process_learning(target.topic, "wikipedia", result.output[:3000])
            self._log(f"  Learned about '{target.topic}' from Wikipedia")

    def _process_learning(self, topic: str, source: str, content: str) -> None:
        """Process and store ALL learnings - no artificial limits."""
        facts = []

        # Split by sentences and paragraphs
        chunks = content.replace(". ", ".\n").replace("! ", "!\n").replace("? ", "?\n").split("\n")

        for chunk in chunks:
            chunk = chunk.strip()

            # Skip too short
            if len(chunk) < 20:
                continue

            # Skip if mostly special chars
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in chunk) / max(len(chunk), 1)
            if alpha_ratio < 0.4:
                continue

            # Keep informative sentences - broad keyword matching
            keywords = [
                "is",
                "are",
                "was",
                "were",
                "can",
                "could",
                "uses",
                "used",
                "provides",
                "enables",
                "allows",
                "includes",
                "contains",
                "research",
                "study",
                "paper",
                "method",
                "approach",
                "system",
                "model",
                "algorithm",
                "learning",
                "neural",
                "cognitive",
                "ai",
                "intelligence",
                "data",
                "process",
                "function",
                "theory",
                "developed",
                "created",
                "designed",
                "built",
                "works",
                "based",
                "known",
                "called",
                "defined",
                "refers",
                "means",
                "example",
                "such",
                "like",
                "including",
                "also",
                "however",
            ]

            if any(word in chunk.lower() for word in keywords):
                # Clean up the chunk
                clean = " ".join(chunk.split())  # Normalize whitespace
                if len(clean) > 20 and clean not in facts:
                    facts.append(clean)

        # Also store the abstract/summary if available
        if len(content) > 100:
            # First paragraph often has the best summary
            first_para = content[:500].split("\n\n")[0].strip()
            if first_para and len(first_para) > 50 and first_para not in facts:
                facts.insert(0, f"[{topic}] {first_para}")

        # STORE ALL FACTS - NO LIMIT!
        stored = 0
        for fact in facts:
            # Avoid duplicates by checking content hash
            fact_hash = hash(fact[:100])
            if not hasattr(self, "_seen_facts"):
                self._seen_facts = set()

            if fact_hash not in self._seen_facts:
                self._seen_facts.add(fact_hash)
                self.learner.add_memory(content=fact, source=source, topic=topic, importance=0.8)
                self.facts_learned += 1
                stored += 1

        # Record learning
        self.learnings.append(
            {
                "topic": topic,
                "source": source,
                "facts_count": stored,
                "timestamp": datetime.now().isoformat(),
            }
        )

        if stored > 0:
            self._log(f"  Stored {stored} facts about '{topic}'")

    def _generate_new_curiosity(self) -> None:
        """Generate new things to be curious about based on learnings."""
        # Get knowledge gaps from learner
        gaps = self.learner.get_knowledge_gaps()

        for gap in gaps[:5]:
            key = f"web:{gap}"
            if key not in self.explored:
                self.targets.append(
                    LearningTarget(
                        topic=gap,
                        source="web",
                        priority=0.9,  # High priority for gaps
                    )
                )
                self._log(f"  New curiosity: '{gap}'")

        # Also add random expansion topics
        if self.learnings:
            recent = self.learnings[-10:]
            for learning in recent:
                # Generate related topics
                related = f"{learning['topic']} advanced techniques"
                key = f"web:{related}"
                if key not in self.explored:
                    self.targets.append(LearningTarget(topic=related, source="web", priority=0.7))

    def _save_state(self) -> None:
        """Save learning state to disk."""
        state_file = Path.home() / ".neuro" / "curiosity_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "cycles": self.cycles,
            "searches_done": self.searches_done,
            "facts_learned": self.facts_learned,
            "explored": list(self.explored),
            "learnings": self.learnings[-100:],  # Keep last 100
            "last_updated": datetime.now().isoformat(),
        }

        try:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

    def _load_state(self) -> None:
        """Load previous learning state."""
        state_file = Path.home() / ".neuro" / "curiosity_state.json"

        if not state_file.exists():
            return

        try:
            with open(state_file) as f:
                state = json.load(f)

            self.cycles = state.get("cycles", 0)
            self.searches_done = state.get("searches_done", 0)
            self.facts_learned = state.get("facts_learned", 0)
            self.explored = set(state.get("explored", []))
            self.learnings = state.get("learnings", [])

            self._log(f"Resumed: {self.facts_learned} facts from {self.cycles} previous cycles")
        except Exception:
            pass

    def get_stats(self) -> Dict:
        """Get curiosity loop statistics."""
        return {
            "running": self.running,
            "cycles": self.cycles,
            "searches_done": self.searches_done,
            "facts_learned": self.facts_learned,
            "targets_remaining": len([t for t in self.targets if not t.learned]),
            "topics_explored": len(self.explored),
            "recent_learnings": self.learnings[-5:] if self.learnings else [],
        }

    def force_explore(self, topic: str, source: str = "web") -> str:
        """Force exploration of a specific topic."""
        self._log(f"Force exploring: {topic}")

        target = LearningTarget(topic=topic, source=source, priority=1.0)

        if source == "arxiv":
            self._explore_arxiv(target)
        elif source == "github":
            self._explore_github(target)
        else:
            self._explore_web(target)

        return f"Explored '{topic}' - learned {self.facts_learned} total facts"


def main():
    """Test the curiosity loop."""
    print("\n" + "=" * 60)
    print("CURIOSITY LOOP TEST")
    print("=" * 60)

    tools = Tools()
    learner = AutonomousLearner()

    loop = CuriosityLoop(tools, learner, verbose=True)

    print("\nStarting autonomous exploration...")
    loop.start()

    # Let it run for a bit
    try:
        for i in range(30):  # 30 seconds
            time.sleep(1)
            if i % 10 == 0:
                stats = loop.get_stats()
                print(
                    f"\n[STATS] Cycles: {stats['cycles']}, Facts: {stats['facts_learned']}, Targets: {stats['targets_remaining']}"
                )
    except KeyboardInterrupt:
        pass

    loop.stop()

    print("\n" + "=" * 60)
    print("FINAL STATS")
    print("=" * 60)
    print(json.dumps(loop.get_stats(), indent=2))


if __name__ == "__main__":
    main()
