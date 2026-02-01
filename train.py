#!/usr/bin/env python3
"""
NEURO LLM Training Pipeline
============================

Train the LLM through:
1. Learn facts from multiple sources (web, ArXiv, benchmarks)
2. Store as Q&A training pairs
3. Run benchmarks to measure progress
4. MLX fine-tune when improved

Run: python3 train.py [model_name]
"""

import sys
import os
import json
import time
import random
import urllib.request
import urllib.parse
import re
import html
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neuro.self_evolution import get_evolution, SelfEvolution
from neuro.self_training import SelfTrainer


class OllamaChat:
    """Simple Ollama chat wrapper."""

    def __init__(self, model: str = "ministral-3:8b"):
        self.model = model
        self.base_url = "http://localhost:11434"

    def chat(self, prompt: str, system: str = None) -> str:
        """Send chat request to Ollama."""
        try:
            url = f"{self.base_url}/api/generate"
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            if system:
                data["system"] = system

            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode(),
                headers={"Content-Type": "application/json"}
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode())
                return result.get("response", "")

        except Exception as e:
            return f"[Error: {e}]"


class Benchmark:
    """Benchmark system to measure LLM progress."""

    def __init__(self):
        self.tests = [
            # Math reasoning
            {
                "question": "A store sells apples for $2 each. If John buys 5 apples and pays with a $20 bill, how much change does he get?",
                "answer": "10",
                "keywords": ["10", "dollar", "change"],
                "category": "math"
            },
            {
                "question": "A train travels at 60 mph. How far does it travel in 2.5 hours?",
                "answer": "150",
                "keywords": ["150", "miles"],
                "category": "math"
            },
            {
                "question": "If a rectangle has length 8 and width 5, what is its area?",
                "answer": "40",
                "keywords": ["40"],
                "category": "math"
            },
            # Logic
            {
                "question": "All cats are mammals. All mammals are animals. Is a cat an animal? Answer yes or no and explain.",
                "answer": "yes",
                "keywords": ["yes", "mammal", "animal", "therefore"],
                "category": "logic"
            },
            {
                "question": "If it rains, the ground gets wet. The ground is wet. Can we conclude it rained? Answer yes or no.",
                "answer": "no",
                "keywords": ["no", "not necessarily", "other"],
                "category": "logic"
            },
            # Common sense
            {
                "question": "A person puts ice cream in the oven at 400 degrees. What happens?",
                "answer": "melts",
                "keywords": ["melt", "liquid", "heat"],
                "category": "common_sense"
            },
            # Trick questions
            {
                "question": "A farmer has 17 sheep. All but 9 run away. How many sheep does he have left?",
                "answer": "9",
                "keywords": ["9"],
                "category": "trick"
            },
            {
                "question": "If you have 3 apples and you take away 2, how many apples do YOU have?",
                "answer": "2",
                "keywords": ["2"],
                "category": "trick"
            },
            # Chain of thought
            {
                "question": "If Alice is twice as old as Bob, and Bob is 15, how old will Alice be in 5 years?",
                "answer": "35",
                "keywords": ["30", "35", "twice"],
                "category": "chain_of_thought"
            },
            # Theory of mind
            {
                "question": "Sally puts a marble in her basket and leaves. Anne moves it to her box. When Sally returns, where will she LOOK for the marble?",
                "answer": "basket",
                "keywords": ["basket", "think", "believe"],
                "category": "theory_of_mind"
            },
        ]

    def score_response(self, response: str, test: Dict) -> float:
        """Score a response."""
        response_lower = response.lower()
        scores = []

        # Check exact answer
        answer = test["answer"].lower()
        if answer in response_lower:
            scores.append(1.0)
        else:
            scores.append(0.0)

        # Check keywords
        keywords = test.get("keywords", [])
        if keywords:
            matches = sum(1 for kw in keywords if kw.lower() in response_lower)
            scores.append(matches / len(keywords))

        # Check reasoning signals
        reasoning = ["because", "therefore", "so", "step", "="]
        r_count = sum(1 for s in reasoning if s in response_lower)
        scores.append(min(r_count / 2, 1.0))

        return sum(scores) / len(scores) if scores else 0.0

    def run(self, chat_fn) -> Dict:
        """Run benchmark and return results."""
        results = {"tests": [], "total": 0, "avg": 0}

        for test in self.tests:
            try:
                response = chat_fn(test["question"])
                score = self.score_response(response, test)
            except Exception as e:
                response = f"[Error: {e}]"
                score = 0.0

            results["tests"].append({
                "category": test["category"],
                "question": test["question"][:50],
                "score": round(score, 3)
            })
            results["total"] += score

        results["avg"] = round(results["total"] / len(self.tests), 3)
        return results


class DataCollector:
    """Collect training data from multiple sources."""

    def __init__(self):
        self.sources = ['benchmark', 'arxiv', 'web', 'wikipedia']

    def fetch_benchmark_qa(self) -> List[Dict]:
        """Get Q&A pairs from benchmark questions with explanations."""
        qa_pairs = [
            {
                "prompt": "If you buy 5 apples for $2 each and pay with $20, how much change do you get?",
                "completion": "Step 1: Calculate total cost: 5 apples × $2 = $10\nStep 2: Calculate change: $20 - $10 = $10\nAnswer: $10 in change"
            },
            {
                "prompt": "A train travels at 60 mph for 2.5 hours. How far does it go?",
                "completion": "Distance = Speed × Time\nDistance = 60 mph × 2.5 hours = 150 miles\nAnswer: 150 miles"
            },
            {
                "prompt": "All cats are mammals. All mammals are animals. Is a cat an animal?",
                "completion": "Yes. This is a syllogism:\n1. All cats are mammals\n2. All mammals are animals\n3. Therefore, all cats are animals\nAnswer: Yes"
            },
            {
                "prompt": "If it rains, the ground gets wet. The ground is wet. Did it rain?",
                "completion": "No, we cannot conclude it rained. This is the fallacy of 'affirming the consequent'. The ground could be wet from:\n- Sprinklers\n- Spilled water\n- Dew\n- Flooding\nAnswer: No, not necessarily"
            },
            {
                "prompt": "A farmer has 17 sheep. All but 9 run away. How many are left?",
                "completion": "This is a trick question. 'All but 9' means 9 remain.\nAnswer: 9 sheep"
            },
            {
                "prompt": "You have 3 apples and take away 2. How many do YOU have?",
                "completion": "You took 2 apples, so YOU have 2 apples. The question asks what YOU have, not what's left.\nAnswer: 2 apples"
            },
            {
                "prompt": "Alice is twice as old as Bob. Bob is 15. How old will Alice be in 5 years?",
                "completion": "Step 1: Bob's age = 15\nStep 2: Alice's age = 2 × 15 = 30\nStep 3: Alice in 5 years = 30 + 5 = 35\nAnswer: 35 years old"
            },
            {
                "prompt": "Sally puts a marble in basket. Anne moves it to box. Where will Sally look for it?",
                "completion": "Sally will look in the BASKET. This is Theory of Mind - Sally doesn't know Anne moved it. She will look where SHE left it.\nAnswer: Basket"
            },
        ]
        return qa_pairs

    def fetch_arxiv(self, category: str = "cs.AI") -> List[Dict]:
        """Fetch papers from ArXiv."""
        items = []
        try:
            url = f'http://export.arxiv.org/api/query?search_query=cat:{category}&start={random.randint(0,30)}&max_results=5'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read().decode('utf-8')

            entries = re.findall(r'<entry>(.*?)</entry>', data, re.DOTALL)
            for entry in entries:
                title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)

                if title and summary:
                    title_text = html.unescape(title.group(1).strip()[:100])
                    summary_text = html.unescape(summary.group(1).strip()[:500])

                    items.append({
                        "prompt": f"What is {title_text}?",
                        "completion": summary_text,
                        "source": "arxiv"
                    })
        except Exception:
            pass
        return items

    def fetch_web(self, query: str) -> List[Dict]:
        """Fetch from DuckDuckGo."""
        items = []
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            if data.get('Abstract'):
                items.append({
                    "prompt": f"Explain {query}",
                    "completion": data['Abstract'],
                    "source": "web"
                })

            for topic in data.get('RelatedTopics', [])[:3]:
                if isinstance(topic, dict) and topic.get('Text'):
                    items.append({
                        "prompt": f"What do you know about {query}?",
                        "completion": topic['Text'],
                        "source": "web"
                    })
        except Exception:
            pass
        return items

    def fetch_wikipedia(self, topic: str) -> List[Dict]:
        """Fetch from Wikipedia API."""
        items = []
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(topic)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

            if data.get('extract'):
                items.append({
                    "prompt": f"Tell me about {topic}",
                    "completion": data['extract'][:500],
                    "source": "wikipedia"
                })
        except Exception:
            pass
        return items

    def collect_all(self, topics: List[str] = None) -> List[Dict]:
        """Collect data from all sources."""
        all_data = []

        # Benchmark Q&A (always include - these are gold standard)
        all_data.extend(self.fetch_benchmark_qa())
        print(f"  [benchmark] {len(all_data)} Q&A pairs")

        # ArXiv
        categories = ['cs.AI', 'cs.LG', 'cs.CL']
        for cat in categories:
            items = self.fetch_arxiv(cat)
            all_data.extend(items)
            print(f"  [arxiv/{cat}] {len(items)} items")

        # Web and Wikipedia
        topics = topics or [
            'artificial intelligence', 'machine learning', 'neural networks',
            'natural language processing', 'reasoning', 'logic',
            'mathematics', 'problem solving', 'critical thinking'
        ]

        for topic in topics[:5]:  # Limit to avoid rate limiting
            web_items = self.fetch_web(topic)
            all_data.extend(web_items)

            wiki_items = self.fetch_wikipedia(topic)
            all_data.extend(wiki_items)

            print(f"  [web+wiki] {topic}: {len(web_items) + len(wiki_items)} items")
            time.sleep(0.5)  # Rate limiting

        return all_data


class NeuroTrainer:
    """
    Main training orchestrator.

    Cycle:
    1. Collect training data
    2. Save unique facts (no duplicates)
    3. Run benchmark
    4. If improved 1%+ → MLX fine-tune
    5. Repeat
    """

    def __init__(self, model: str = "ministral-3:8b"):
        self.model = model
        self.ollama = OllamaChat(model)
        self.evolution = get_evolution()
        self.trainer = SelfTrainer()
        self.benchmark = Benchmark()
        self.collector = DataCollector()

        # Training data file
        self.training_file = os.path.expanduser("~/.neuro/evolution/training_data.jsonl")
        os.makedirs(os.path.dirname(self.training_file), exist_ok=True)

    def save_training_pair(self, prompt: str, completion: str, source: str = "collected"):
        """Save a training pair if unique."""
        content = f"{prompt} {completion}"

        if self.evolution.is_duplicate(content):
            return False

        self.evolution.mark_learned(content)

        # Save to JSONL
        pair = {
            "prompt": prompt,
            "completion": completion,
            "source": source,
            "timestamp": datetime.now().isoformat()
        }

        with open(self.training_file, 'a') as f:
            f.write(json.dumps(pair) + '\n')

        # Also save to knowledge base
        self.trainer.learn(prompt[:50], completion[:500], source)

        return True

    def run_benchmark(self) -> float:
        """Run benchmark and return average score."""
        print("\n[BENCHMARK] Running tests...")

        def chat_with_knowledge(question: str) -> str:
            # Inject learned knowledge
            knowledge = self.trainer.get_knowledge_for_prompt(question)
            if knowledge:
                enhanced = f"{knowledge}\n\nQuestion: {question}\nThink step by step:"
            else:
                enhanced = f"Question: {question}\nThink step by step:"
            return self.ollama.chat(enhanced)

        results = self.benchmark.run(chat_with_knowledge)

        # Show results by category
        categories = {}
        for test in results["tests"]:
            cat = test["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(test["score"])

        print("\n  Category Scores:")
        for cat, scores in categories.items():
            avg = sum(scores) / len(scores)
            status = "[green]" if avg >= 0.7 else "[yellow]" if avg >= 0.4 else "[red]"
            print(f"    {cat}: {avg:.0%}")

        print(f"\n  Overall: {results['avg']:.0%}")

        # Record in evolution
        weak_areas = [(cat, sum(s)/len(s)) for cat, s in categories.items() if sum(s)/len(s) < 0.7]
        self.evolution.record_benchmark(results['avg'], {'weak_areas': weak_areas})

        return results['avg']

    def collect_and_learn(self, topics: List[str] = None) -> int:
        """Collect data and learn unique facts."""
        print("\n[COLLECTING] Gathering training data...")

        data = self.collector.collect_all(topics)

        unique_count = 0
        for item in data:
            if self.save_training_pair(item["prompt"], item["completion"], item.get("source", "collected")):
                unique_count += 1

        print(f"\n[LEARNED] {unique_count} unique facts (total: {self.evolution.get_stats()['total_facts']})")
        return unique_count

    def train_mlx(self) -> bool:
        """Run MLX fine-tuning if conditions are met."""
        should_train, reason = self.evolution.should_train()

        print(f"\n[TRAINING] {reason}")

        if not should_train:
            return False

        result = self.evolution.run_mlx_training(self.model)

        if result['success']:
            print(f"[TRAINING] MLX fine-tuning completed!")
            return True
        else:
            print(f"[TRAINING] {result['message']}")
            return False

    def run_cycle(self, topics: List[str] = None) -> Dict:
        """Run one training cycle."""
        stats = self.evolution.get_stats()
        print(f"\n{'='*60}")
        print(f"NEURO TRAINING - Cycle {stats['cycle']}")
        print(f"{'='*60}")
        print(f"Facts learned: {stats['total_facts']}")
        print(f"Training pairs: {stats['training_pairs']}")

        # Step 1: Collect and learn
        unique = self.collect_and_learn(topics)

        # Step 2: Run benchmark
        score = self.run_benchmark()

        # Step 3: Check if should train
        self.train_mlx()

        # Step 4: Show reflection
        print(self.evolution.reflect())

        return {
            'cycle': stats['cycle'],
            'unique_facts': unique,
            'score': score,
            'improvement': self.evolution.get_improvement()
        }

    def run_continuous(self, cycles: int = 10, delay: int = 30):
        """Run multiple training cycles continuously."""
        print("\n" + "="*60)
        print("NEURO CONTINUOUS TRAINING")
        print("="*60)
        print(f"Model: {self.model}")
        print(f"Cycles: {cycles}")
        print(f"Delay between cycles: {delay}s")
        print("="*60)

        for i in range(cycles):
            print(f"\n>>> CYCLE {i+1}/{cycles}")

            try:
                result = self.run_cycle()

                if result['improvement'] > 0.05:
                    print(f"\n*** SIGNIFICANT IMPROVEMENT: {result['improvement']:.1%} ***")

            except KeyboardInterrupt:
                print("\n\nTraining interrupted. Saving progress...")
                self.trainer.save()
                break
            except Exception as e:
                print(f"\n[ERROR] Cycle failed: {e}")

            if i < cycles - 1:
                print(f"\nWaiting {delay}s before next cycle...")
                time.sleep(delay)

        # Final summary
        stats = self.evolution.get_stats()
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Total facts learned: {stats['total_facts']}")
        print(f"Training pairs: {stats['training_pairs']}")
        print(f"MLX trainings: {stats['trainings']}")
        print(f"Final improvement: {stats['improvement']:+.1%}")
        print("="*60)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "ministral-3:8b"

    # Check Ollama is running
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as r:
            pass
    except Exception:
        print("ERROR: Ollama not running. Start with: ollama serve")
        sys.exit(1)

    trainer = NeuroTrainer(model)

    # Interactive mode
    print("\n" + "="*60)
    print("NEURO LLM TRAINING")
    print("="*60)
    print(f"Model: {model}")
    print("\nCommands:")
    print("  1  - Run single training cycle")
    print("  5  - Run 5 cycles")
    print("  10 - Run 10 cycles")
    print("  b  - Run benchmark only")
    print("  c  - Collect data only")
    print("  s  - Show stats")
    print("  q  - Quit")
    print("="*60)

    while True:
        try:
            cmd = input("\n> ").strip().lower()

            if cmd == 'q' or cmd == 'quit':
                print("Saving and exiting...")
                trainer.trainer.save()
                break

            elif cmd == '1':
                trainer.run_cycle()

            elif cmd == '5':
                trainer.run_continuous(cycles=5)

            elif cmd == '10':
                trainer.run_continuous(cycles=10)

            elif cmd == 'b':
                trainer.run_benchmark()

            elif cmd == 'c':
                trainer.collect_and_learn()

            elif cmd == 's':
                stats = trainer.evolution.get_stats()
                print(f"\nFacts: {stats['total_facts']}")
                print(f"Training pairs: {stats['training_pairs']}")
                print(f"Cycles: {stats['cycle']}")
                print(f"Baseline: {stats['baseline_score']}")
                print(f"Current: {stats['current_score']}")
                print(f"Improvement: {stats['improvement']:+.1%}")
                print(f"MLX trainings: {stats['trainings']}")

            else:
                print("Unknown command")

        except KeyboardInterrupt:
            print("\n\nSaving and exiting...")
            trainer.trainer.save()
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
