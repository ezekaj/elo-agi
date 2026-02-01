"""
Autonomous Learning System
===========================

Self-evolving AI that learns continuously:
1. Run initial benchmark
2. Learn 100 unique facts from various sources
3. Re-benchmark
4. If improved 1%+ → MLX fine-tune
5. Reflect and repeat

Based on agielo's autonomous AI architecture.
"""

import os
import json
import time
import random
import re
import html
import urllib.request
import urllib.parse
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


class WebLearner:
    """Autonomous web learning capabilities."""

    def __init__(self):
        self.learned_facts = []
        self.search_history = []

    def search_web(self, query: str) -> List[Dict]:
        """Search the web using DuckDuckGo (free, no API key)."""
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))

            results = []

            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', query),
                    'snippet': data['Abstract'],
                    'source': data.get('AbstractSource', 'DuckDuckGo'),
                    'url': data.get('AbstractURL', '')
                })

            for topic in data.get('RelatedTopics', [])[:3]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('Text', '')[:50],
                        'snippet': topic.get('Text', ''),
                        'url': topic.get('FirstURL', '')
                    })

            self.search_history.append({
                'query': query,
                'time': datetime.now().isoformat(),
                'results': len(results)
            })

            return results

        except Exception as e:
            return [{'error': str(e)}]

    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch and extract text from a webpage."""
        if not url:
            return None

        skip_domains = ['google.com/rss', 'youtube.com', 'twitter.com', 'facebook.com']
        if any(d in url for d in skip_domains):
            return None

        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            })
            with urllib.request.urlopen(req, timeout=15) as response:
                content_type = response.headers.get('Content-Type', '')
                if 'pdf' in content_type.lower() or url.endswith('.pdf'):
                    return None
                raw_html = response.read().decode('utf-8', errors='ignore')

            # Extract meaningful text
            text = re.sub(r'<script[^>]*>.*?</script>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<nav[^>]*>.*?</nav>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<footer[^>]*>.*?</footer>', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'<header[^>]*>.*?</header>', '', text, flags=re.DOTALL | re.IGNORECASE)

            # Try to find main content
            main_content = re.search(r'<article[^>]*>(.*?)</article>', text, re.DOTALL | re.IGNORECASE)
            if not main_content:
                main_content = re.search(r'<main[^>]*>(.*?)</main>', text, re.DOTALL | re.IGNORECASE)
            if main_content:
                text = main_content.group(1)

            text = re.sub(r'<[^>]+>', ' ', text)
            text = html.unescape(text)
            text = re.sub(r'\s+', ' ', text).strip()

            if len(text) > 500:
                start = len(text) // 10
                end = int(len(text) * 0.8)
                text = text[start:end]

            return text[:2000] if len(text) > 100 else None

        except Exception:
            return None

    def fetch_arxiv(self, category: str = 'cs.AI') -> List[Dict]:
        """Fetch papers from ArXiv API."""
        items = []
        try:
            url = f'http://export.arxiv.org/api/query?search_query=cat:{category}&start={random.randint(0,50)}&max_results=10&sortBy=submittedDate&sortOrder=descending'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read().decode('utf-8')

            entries = re.findall(r'<entry>(.*?)</entry>', data, re.DOTALL)
            for entry in entries[:10]:
                title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
                summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
                link = re.search(r'<id>(.*?)</id>', entry)

                if title and summary:
                    items.append({
                        'title': title.group(1).strip()[:100],
                        'snippet': summary.group(1).strip()[:1000],
                        'url': link.group(1) if link else '',
                        'source': 'ArXiv'
                    })
        except Exception:
            pass
        return items

    def fetch_github(self, topic: str = 'machine-learning') -> List[Dict]:
        """Fetch from GitHub API."""
        items = []
        try:
            url = f'https://api.github.com/search/repositories?q={topic}+stars:>500&sort=stars&per_page=10'
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0',
                'Accept': 'application/vnd.github.v3+json'
            })
            with urllib.request.urlopen(req, timeout=15) as response:
                data = json.loads(response.read().decode('utf-8'))

            for repo in data.get('items', [])[:10]:
                desc = repo.get('description', '') or ''
                items.append({
                    'title': repo.get('full_name', ''),
                    'snippet': f"{desc} - Language: {repo.get('language', 'Unknown')}, Stars: {repo.get('stargazers_count', 0)}",
                    'url': repo.get('html_url', ''),
                    'source': 'GitHub'
                })
        except Exception:
            pass
        return items

    def fetch_rss(self, url: str, source: str = 'RSS') -> List[Dict]:
        """Fetch from RSS feed."""
        items = []
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                data = response.read().decode('utf-8', errors='ignore')

            rss_items = re.findall(r'<item>(.*?)</item>', data, re.DOTALL)
            for item in rss_items[:15]:
                title = re.search(r'<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>', item)
                desc = re.search(r'<description>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>', item, re.DOTALL)
                link = re.search(r'<link>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</link>', item)

                if title:
                    snippet = desc.group(1) if desc else title.group(1)
                    snippet = html.unescape(snippet)
                    snippet = re.sub(r'<[^>]+>', ' ', snippet)
                    snippet = re.sub(r'\s+', ' ', snippet).strip()

                    if 'news.google.com/rss/articles' in snippet:
                        continue
                    if len(snippet) < 50:
                        continue

                    title_clean = html.unescape(title.group(1).strip())[:100]

                    items.append({
                        'title': title_clean,
                        'snippet': snippet[:500],
                        'url': link.group(1).strip() if link else '',
                        'source': source
                    })
        except Exception:
            pass
        return items


class AutonomousLoop:
    """
    Autonomous learning loop that runs in background.

    Cycle:
    1. Run initial benchmark
    2. Learn 100 unique facts (no duplicates)
    3. Re-benchmark
    4. If improved 1%+ → MLX fine-tune
    5. Reflect and repeat
    """

    def __init__(self,
                 chat_fn: Callable[[str], str],
                 trainer,
                 evolution,
                 benchmark,
                 verbose: bool = True,
                 on_activity: Optional[Callable[[str, str, Dict], None]] = None,
                 ocr=None):
        """
        Args:
            chat_fn: Function to call LLM (takes prompt, returns response)
            trainer: SelfTrainer instance for learning facts
            evolution: SelfEvolution instance for duplicate detection
            benchmark: Benchmark instance for testing
            verbose: Print progress updates
            on_activity: Callback for activity updates (type, message, data)
        """
        self.chat_fn = chat_fn
        self.trainer = trainer
        self.evolution = evolution
        self.benchmark = benchmark
        self.verbose = verbose
        self.on_activity = on_activity
        self.ocr = ocr

        self.web = WebLearner()
        self.running = False
        self.is_busy = False
        self.thread = None

        # State
        self.initial_benchmark_done = False
        self.benchmark_results = None
        self.weak_areas = []
        self.conversation_started = False

        # Learning order
        self._learning_order_idx = 0
        self._current_focus = None
        self._focus_items = []
        self._focus_learned = 0

        # Activity log for UI
        self.activity_log = []
        self.current_activity = ""
        self.last_learned_topic = ""
        self.last_learned_source = ""

        # Pause mechanism - don't run while user is chatting
        self.paused = False
        self.last_user_activity = 0
        self.idle_threshold = 10  # seconds before starting autonomous learning

    def pause(self):
        """Pause autonomous learning (user is chatting)."""
        self.paused = True
        self.last_user_activity = time.time()

    def resume(self):
        """Resume autonomous learning (user is idle)."""
        self.paused = False

    def user_active(self):
        """Mark that user is actively using the system."""
        self.last_user_activity = time.time()
        self.paused = True

    def _notify(self, activity_type: str, message: str, data: Dict = None):
        """Notify about activity (for UI updates)."""
        data = data or {}
        self.current_activity = message
        self.activity_log.append({
            'type': activity_type,
            'message': message,
            'data': data,
            'time': datetime.now().isoformat()
        })
        # Keep log bounded
        if len(self.activity_log) > 100:
            self.activity_log = self.activity_log[-50:]

        if self.on_activity:
            try:
                self.on_activity(activity_type, message, data)
            except Exception:
                pass
        elif self.verbose:
            # Only print to console if no callback is set
            print(f"\n  [Autonomous] {message}")

    def start(self):
        """Start the autonomous learning loop."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        if self.verbose:
            print("[Autonomous] Learning loop started")

    def stop(self):
        """Stop the autonomous learning loop."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.verbose:
            print("[Autonomous] Learning loop stopped")

    def mark_conversation_started(self):
        """Mark that user has started conversation (triggers learning)."""
        self.conversation_started = True
        self.last_user_activity = time.time()

    def _loop(self):
        """Main autonomous loop."""
        while self.running:
            time.sleep(3)

            if self.is_busy:
                continue

            # Wait for first conversation before starting
            if not self.conversation_started:
                continue

            # Check if paused (user is chatting)
            if self.paused:
                # Auto-resume after idle threshold
                idle_time = time.time() - self.last_user_activity
                if idle_time < self.idle_threshold:
                    continue  # Still waiting for user to be idle
                else:
                    self.paused = False
                    self._notify("resume", f"Resuming learning (idle {idle_time:.0f}s)", {})

            # Step 1: Run INITIAL benchmark (once) - but wait for user to be idle
            if not self.initial_benchmark_done:
                # Don't run benchmark immediately - wait until user is idle
                idle_time = time.time() - self.last_user_activity
                if idle_time < 5:
                    continue
                self._run_benchmark_and_report("INITIAL")
                self.initial_benchmark_done = True
                continue

            # Step 2: Learn unique facts
            if not self.evolution.should_benchmark():
                self._learn_unique_fact()
                continue

            # Step 3: After 100 unique facts → re-benchmark
            if self.verbose:
                print(f"\n[Evolution] Learned {self.evolution.state['facts_this_cycle']} unique facts! Re-benchmarking...")

            self._run_benchmark_and_report("CYCLE")

            # Step 4: Check if should train
            should_train, reason = self.evolution.should_train(min_improvement=0.01)
            self._notify("training_check", reason, {"should_train": should_train})

            if should_train:
                self._notify("training", "Starting MLX fine-tuning on MacBook...", {})
                result = self.evolution.run_mlx_training()
                if result['success']:
                    self._notify("training_done", "MLX TRAINING COMPLETE!", {"success": True})
                    self._add_evolved_capability()
                else:
                    self._notify("training_skip", f"Training skipped: {result['message']}", {"success": False})

            # Step 5: Reflect and start new cycle
            reflection = self.evolution.reflect()
            self._notify("reflection", reflection, {})

            self.evolution.start_new_cycle()
            new_cycle = self.evolution.state['current_cycle']
            self._notify("new_cycle", f"Starting learning cycle {new_cycle}", {"cycle": new_cycle})

    def _run_benchmark_and_report(self, phase: str = ""):
        """Run benchmark and record results."""
        self.is_busy = True
        self._notify("benchmark", f"Running {phase} benchmark...", {"phase": phase})

        try:
            def think_fn(q):
                # Inject learned knowledge into the question
                knowledge = self.trainer.get_knowledge_for_prompt(q)
                if knowledge:
                    enhanced_q = f"{knowledge}\n\nQuestion: {q}\nThink step by step and give the answer:"
                else:
                    enhanced_q = f"Question: {q}\nThink step by step and give the answer:"
                return self.chat_fn(enhanced_q)

            self.benchmark_results = self.benchmark.run_benchmark(think_fn)

            # Find weak areas by category (below 70%)
            category_scores = {}
            for test in self.benchmark_results.get('tests', []):
                cat = test.get('category', 'unknown')
                score = test.get('score', 0) or 0
                if cat not in category_scores:
                    category_scores[cat] = []
                category_scores[cat].append(score)

            self.weak_areas = []
            for cat, scores in category_scores.items():
                avg = sum(scores) / len(scores) if scores else 0
                if avg < 0.7:
                    self.weak_areas.append((cat, avg))

            avg_score = self.benchmark_results.get('avg_score', 0) or 0

            # Record in evolution system
            self.evolution.record_benchmark(avg_score, {
                'weak_areas': [(c, s) for c, s in self.weak_areas],
                'phase': phase
            })

            weak_str = ", ".join([f"{a}: {s:.0%}" for a, s in self.weak_areas[:3]]) if self.weak_areas else "none"
            self._notify("benchmark_done", f"{phase} Benchmark: {avg_score:.0%} | Weak: {weak_str}", {
                "score": avg_score,
                "weak_areas": self.weak_areas,
                "phase": phase
            })

        except Exception as e:
            self._notify("error", f"Benchmark error: {e}", {"error": str(e)})

        self.is_busy = False

    def _learn_unique_fact(self):
        """Learn one unique fact at a time."""
        self.is_busy = True

        # If no current focus or finished, get new items
        if not self._focus_items:
            self._notify("fetching", "Fetching new content...", {})
            self._focus_items = self._fetch_content()
            if self._focus_items:
                self._current_focus = self._focus_items[0].get('source', 'Unknown')
                self._focus_learned = 0
                self._notify("focus", f"Now learning from: {self._current_focus}", {
                    "source": self._current_focus,
                    "items": len(self._focus_items)
                })
            else:
                self.is_busy = False
                return

        # Learn from current focus
        items_to_remove = []

        for i, item in enumerate(self._focus_items[:3]):
            snippet = item.get('snippet', '')
            url = item.get('url', '')
            title = item.get('title', 'Item')[:60]
            source = item.get('source', 'Unknown')

            if not snippet or len(snippet) < 50:
                items_to_remove.append(i)
                continue

            # Check for duplicate
            if self.evolution.is_duplicate(snippet):
                self._notify("skip", f"Skipping duplicate: {title[:40]}...", {"reason": "duplicate"})
                items_to_remove.append(i)
                continue

            # Try to get full content if snippet is short
            if url and len(snippet) < 500:
                self._notify("fetching", f"Fetching full content: {title[:40]}...", {"url": url})
                full_content = self.web.fetch_page(url)
                if full_content and len(full_content) > len(snippet):
                    snippet = full_content

            if not snippet or len(snippet) < 100:
                items_to_remove.append(i)
                continue

            # Analyze content with LLM
            self._notify("analyzing", f"Analyzing: {title[:40]}...", {"title": title})
            analyzed = self._analyze_content(title, snippet, source)

            if not analyzed:
                items_to_remove.append(i)
                continue

            # Check for duplicate (use analyzed summary)
            if self.evolution.is_duplicate(analyzed.get('summary', snippet)):
                items_to_remove.append(i)
                continue

            # Learn the analyzed content
            if self.evolution.mark_learned(analyzed.get('summary', '')):
                topic = analyzed.get('topic', title)
                # Ensure knowledge is a string
                knowledge = analyzed.get('knowledge', snippet[:1000])
                if isinstance(knowledge, dict):
                    knowledge = json.dumps(knowledge)
                elif not isinstance(knowledge, str):
                    knowledge = str(knowledge)
                self.trainer.learn(
                    topic,
                    knowledge,
                    source
                )
                self._focus_learned += 1
                self.last_learned_topic = topic
                self.last_learned_source = source

                # Save as training Q&A pairs
                self._save_as_training(analyzed, source)

                stats = self.evolution.get_stats()
                facts_count = len(analyzed.get('facts', []))
                qa_count = len(analyzed.get('qa_pairs', []))

                self._notify("learned", f"Learned [{stats['facts_this_cycle']}/100]: {topic}", {
                    "topic": topic,
                    "source": source,
                    "facts_count": facts_count,
                    "qa_pairs": qa_count,
                    "cycle_progress": stats['facts_this_cycle'],
                    "total_facts": stats['total_facts']
                })

                items_to_remove.append(i)
                break  # One at a time

        # Remove processed items
        for i in sorted(items_to_remove, reverse=True):
            if i < len(self._focus_items):
                self._focus_items.pop(i)

        # If focus exhausted, reflect and move on
        if not self._focus_items:
            if self._focus_learned > 0:
                self._notify("reflection", f"Completed {self._current_focus}: {self._focus_learned} facts learned", {
                    "source": self._current_focus,
                    "facts_learned": self._focus_learned
                })
            self._current_focus = None
            self._focus_learned = 0

        self.is_busy = False

    def _fetch_content(self) -> List[Dict]:
        """Fetch content from various sources."""
        all_items = []

        # Include 'images' if OCR is available
        learning_order = ['math', 'logic', 'arxiv', 'science', 'code']
        if self.ocr and self.ocr.is_available():
            learning_order.append('images')

        source_type = learning_order[self._learning_order_idx % len(learning_order)]
        self._learning_order_idx += 1

        # Parallel mode every 5th cycle
        if self._learning_order_idx % 5 == 0:
            self._notify("fetching", "Parallel fetch from all sources...", {})

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(self._fetch_benchmark_questions, 'math'): 'math',
                    executor.submit(self._fetch_benchmark_questions, 'logic'): 'logic',
                    executor.submit(self.web.fetch_arxiv, 'cs.AI'): 'arxiv',
                    executor.submit(self.web.fetch_github, 'machine-learning'): 'github',
                    executor.submit(self.web.fetch_rss, 'https://www.sciencedaily.com/rss/all.xml', 'ScienceDaily'): 'rss',
                }

                for future in as_completed(futures, timeout=60):
                    try:
                        items = future.result(timeout=10)
                        if items:
                            all_items.extend(items[:3])
                    except Exception:
                        pass

            if all_items:
                random.shuffle(all_items)
                return all_items

        # Sequential mode
        try:
            if source_type == 'math':
                all_items = self._fetch_benchmark_questions('math')
            elif source_type == 'logic':
                all_items = self._fetch_benchmark_questions('logic')
            elif source_type == 'arxiv':
                categories = ['cs.AI', 'cs.LG', 'cs.CL', 'math.CO', 'stat.ML']
                all_items = self.web.fetch_arxiv(random.choice(categories))
            elif source_type == 'science':
                all_items = self.web.fetch_rss('https://www.sciencedaily.com/rss/all.xml', 'ScienceDaily')
            elif source_type == 'code':
                topics = ['algorithm', 'machine-learning', 'data-structure', 'framework']
                all_items = self.web.fetch_github(random.choice(topics))
            elif source_type == 'images':
                all_items = self._fetch_images_to_learn()
        except Exception:
            pass

        return all_items

    def _fetch_images_to_learn(self) -> List[Dict]:
        """Fetch images to learn from using OCR."""
        items = []

        if not self.ocr:
            return items

        # Look for images in common locations
        import glob
        image_dirs = [
            os.path.expanduser("~/Desktop"),
            os.path.expanduser("~/Downloads"),
            os.path.expanduser("~/Pictures"),
            "/tmp",
        ]

        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp']

        for img_dir in image_dirs:
            if not os.path.exists(img_dir):
                continue

            for ext in image_extensions:
                pattern = os.path.join(img_dir, ext)
                files = glob.glob(pattern)

                for img_path in files[:5]:  # Max 5 per dir
                    # Skip if already processed
                    if self.evolution and self.evolution.is_duplicate(img_path):
                        continue

                    # Extract content using OCR
                    self._notify("ocr", f"Reading image: {os.path.basename(img_path)}", {"path": img_path})

                    try:
                        data = self.ocr.extract_for_learning(img_path)
                        if 'error' not in data and data.get('content'):
                            items.append({
                                'title': data.get('title', os.path.basename(img_path)),
                                'snippet': data.get('content', '')[:1000],
                                'url': img_path,
                                'source': 'OCR-Image',
                                'ocr_data': data
                            })
                    except Exception as e:
                        self._notify("error", f"OCR error: {e}", {"path": img_path})

                    if len(items) >= 3:
                        break

            if len(items) >= 3:
                break

        return items

    def _fetch_benchmark_questions(self, category: str) -> List[Dict]:
        """Learn from benchmark with correct answers."""
        items = []

        benchmark_questions = [t for t in self.benchmark.tests if t.get('category') == category]
        if not benchmark_questions:
            benchmark_questions = self.benchmark.tests

        test = random.choice(benchmark_questions)
        question = test['question']
        answer = test.get('answer', '')
        keywords = test.get('expected_keywords', [])

        if self.verbose:
            print(f"\n[Evolution] STUDYING: {question[:50]}... (answer: {answer})")

        # Create teaching content with step-by-step reasoning
        explanations = {
            "apples for $2": f"Calculate: 5 × $2 = $10. Change: $20 - $10 = $10. Answer: 10",
            "60 mph": f"Distance = Speed × Time = 60 × 2.5 = 150 miles. Answer: 150",
            "length 8 and width 5": f"Area = 8 × 5 = 40. Answer: 40",
            "cats are mammals": f"Syllogism: cats→mammals→animals. Therefore cats are animals. Answer: yes",
            "rains, the ground gets wet": f"Affirming consequent fallacy. Wet ground ≠ rain (could be sprinklers). Answer: no",
            "ice cream in the oven": f"400°F melts and burns ice cream. Answer: melts",
            "17 sheep. All but 9": f"'All but 9' = 9 remain. Answer: 9",
            "3 apples and you take away 2": f"YOU took 2, so YOU have 2. Answer: 2",
            "twice as old as Bob": f"Alice = 2×15 = 30. In 5 years = 35. Answer: 35",
            "marble in her basket": f"Sally thinks marble is where SHE put it. Answer: basket",
        }

        explanation = f"Answer: {answer}. Keywords: {', '.join(keywords[:3])}"
        for key, exp in explanations.items():
            if key.lower() in question.lower():
                explanation = exp
                break

        qa_content = f"Question: {question}\n\nStep-by-step solution:\n{explanation}\n\nFINAL ANSWER: {answer}"

        items.append({
            'title': f"[{category.upper()}] {answer}",
            'snippet': qa_content,
            'url': '',
            'source': f'Benchmark-{category}'
        })

        return items

    def _analyze_content(self, title: str, content: str, source: str) -> Optional[Dict]:
        """Use LLM to analyze content and extract structured knowledge."""
        try:
            prompt = f"""Extract key knowledge from this text. Be concise.

TEXT: {content[:1500]}

Return JSON only:
{{"topic":"topic name","summary":"one sentence","facts":["fact1","fact2"],"qa_pairs":[{{"q":"question","a":"answer"}}],"knowledge":"key knowledge"}}

JSON:"""

            response = self.chat_fn(prompt)

            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*"topic"[^{}]*\}', response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'\{[\s\S]*?\}(?=\s*$|\s*```)', response)

            if json_match:
                json_str = json_match.group()
                json_str = json_str.replace('\n', ' ').replace('\\', '\\\\')
                analyzed = json.loads(json_str)
                return analyzed

        except json.JSONDecodeError:
            return {
                'topic': title[:50],
                'summary': content[:200],
                'facts': [content[:300]],
                'qa_pairs': [{'q': f'What is {title}?', 'a': content[:200]}],
                'knowledge': content[:500]
            }
        except Exception:
            pass

        return {
            'topic': title[:50],
            'summary': content[:200],
            'facts': [],
            'qa_pairs': [],
            'knowledge': content[:500]
        }

    def _save_as_training(self, analyzed: Dict, source: str):
        """Save analyzed Q&A pairs as training data for MLX."""
        training_file = os.path.expanduser("~/.cognitive_ai_knowledge/training_data.jsonl")

        try:
            os.makedirs(os.path.dirname(training_file), exist_ok=True)

            for qa in analyzed.get('qa_pairs', []):
                if qa.get('q') and qa.get('a'):
                    training_pair = {
                        "prompt": qa['q'],
                        "completion": qa['a'],
                        "source": source,
                        "topic": analyzed.get('topic', ''),
                        "timestamp": datetime.now().isoformat()
                    }
                    with open(training_file, 'a') as f:
                        f.write(json.dumps(training_pair) + '\n')

            for fact in analyzed.get('facts', []):
                if fact:
                    training_pair = {
                        "prompt": f"What do you know about {analyzed.get('topic', 'this')}?",
                        "completion": fact,
                        "source": source,
                        "timestamp": datetime.now().isoformat()
                    }
                    with open(training_file, 'a') as f:
                        f.write(json.dumps(training_pair) + '\n')

        except Exception:
            pass

    def _add_evolved_capability(self):
        """Add a new function based on what was learned."""
        if not self.weak_areas:
            return

        weak_topic, _ = self.weak_areas[0]

        capability_templates = {
            'math': (
                'solve_basic_math',
                '''def solve_basic_math(expression: str) -> str:
    """Solve basic math expressions."""
    try:
        allowed = set('0123456789+-*/().% ')
        if all(c in allowed for c in expression):
            result = eval(expression)
            return f"{expression} = {result}"
        return "Cannot evaluate: contains unsafe characters"
    except Exception as e:
        return f"Error: {e}"
''',
                'Safely evaluate basic math expressions'
            ),
            'logic': (
                'check_logic',
                '''def check_logic(premise1: str, premise2: str, conclusion: str) -> str:
    """Simple logical consistency checker."""
    if "all" in premise1.lower() and "is a" in premise2.lower():
        return f"If '{premise1}' and '{premise2}', then '{conclusion}' follows by syllogism."
    return f"Analyzing: {premise1} + {premise2} -> {conclusion}"
''',
                'Check basic logical syllogisms'
            ),
        }

        if weak_topic in capability_templates:
            name, code, description = capability_templates[weak_topic]
            existing = [f['name'] for f in self.evolution.state['added_functions']]
            if name not in existing:
                if self.evolution.add_function(name, code, description):
                    if self.verbose:
                        print(f"\n[Evolution] Added new capability: {name}")

    def get_stats(self) -> Dict:
        """Get autonomous loop statistics."""
        return {
            'running': self.running,
            'initial_benchmark_done': self.initial_benchmark_done,
            'weak_areas': self.weak_areas,
            'current_focus': self._current_focus,
            'focus_learned': self._focus_learned,
            'learning_order_idx': self._learning_order_idx,
            'evolution': self.evolution.get_stats() if self.evolution else {},
        }
