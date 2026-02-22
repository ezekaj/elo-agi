"""
Smart Wrapper: Routes LLM queries through cognitive modules.

Makes any LLM smarter by adding:
- Memory retrieval (episodic + semantic)
- Reasoning analysis (selects appropriate reasoning type)
- Cognitive processing (perceive → think → act via Global Workspace)
- Context enrichment (builds enriched prompts from module outputs)

Usage:
    from neuro.wrapper import SmartWrapper

    wrapper = SmartWrapper()  # Auto-detects LLM (Ollama → API keys → mock)
    result = wrapper.query("If we raise prices 20%, what happens to revenue?")
    print(result.text)
    print(result.modules_used)
    print(result.processing_steps)
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

from neuro.modules.llm.llm_interface import (
    LLMOracle,
    LLMConfig,
    LLMResponse,
    MockLLM,
    AnthropicLLM,
    OpenAILLM,
    create_llm,
    HAS_ANTHROPIC,
    HAS_OPENAI,
)
from neuro.modules.llm.semantic_bridge import SemanticBridge


@dataclass
class SmartResponse:
    """Response from the Smart Wrapper."""

    text: str
    modules_used: List[str]
    cognitive_context: Dict[str, Any]
    confidence: float
    processing_steps: List[str]
    latency: float = 0.0
    provider: str = "mock"

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        import html as html_mod

        conf_pct = int(self.confidence * 100)
        conf_color = "#4caf50" if conf_pct >= 70 else "#ff9800" if conf_pct >= 40 else "#f44336"

        modules_html = " ".join(
            f'<span style="display:inline-block;background:#e3f2fd;color:#1565c0;'
            f'padding:2px 8px;border-radius:12px;font-size:12px;margin:2px;">'
            f"{html_mod.escape(m)}</span>"
            for m in self.modules_used
        )

        steps_html = "\n".join(
            f"<li style='margin:2px 0;font-size:13px;color:#555;'>{html_mod.escape(s)}</li>"
            for s in self.processing_steps
        )

        escaped_text = html_mod.escape(self.text).replace("\n", "<br>")

        return (
            f'<div style="font-family:system-ui,sans-serif;border:1px solid #e0e0e0;'
            f'border-radius:8px;padding:16px;margin:8px 0;max-width:800px;">'
            f'  <div style="font-size:15px;line-height:1.6;margin-bottom:12px;">'
            f"{escaped_text}</div>"
            f'  <div style="margin-bottom:8px;">'
            f'    <span style="font-size:12px;color:#888;margin-right:8px;">Modules:</span>'
            f"{modules_html}</div>"
            f'  <div style="margin-bottom:8px;">'
            f'    <span style="font-size:12px;color:#888;margin-right:8px;">'
            f"Confidence: {conf_pct}%</span>"
            f'    <div style="display:inline-block;width:120px;height:8px;'
            f'background:#eee;border-radius:4px;vertical-align:middle;">'
            f'      <div style="width:{conf_pct}%;height:100%;background:{conf_color};'
            f'border-radius:4px;"></div>'
            f"    </div>"
            f"  </div>"
            f'  <div style="font-size:12px;color:#999;margin-bottom:4px;">'
            f"{self.provider} | {self.latency:.2f}s</div>"
            f'  <details style="margin-top:8px;">'
            f'    <summary style="cursor:pointer;font-size:12px;color:#888;">'
            f"Processing steps ({len(self.processing_steps)})</summary>"
            f'    <ol style="margin:4px 0;padding-left:20px;">{steps_html}</ol>'
            f"  </details>"
            f"</div>"
        )


class OllamaLLM(LLMOracle):
    """LLM Oracle using Ollama's local API."""

    def __init__(self, config: Optional[LLMConfig] = None):
        config = config or LLMConfig(provider="ollama", model="llama3.2")
        super().__init__(config)
        self._base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def query(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        self._query_count += 1
        start_time = time.time()

        cache_key = self._get_cache_key(prompt, system)
        cached = self._check_cache(cache_key)
        if cached:
            return cached

        import urllib.request
        import json

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps(
            {
                "model": self.config.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            }
        ).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{self._base_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            text = data.get("message", {}).get("content", "")
            tokens = data.get("eval_count", len(text.split()))
        except Exception as e:
            text = f"Error communicating with Ollama: {e}"
            tokens = 0

        response = LLMResponse(
            text=text,
            tokens_used=tokens,
            latency=time.time() - start_time,
            model=self.config.model,
            cached=False,
        )
        self._store_cache(cache_key, response)
        return response

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding via Ollama."""
        import urllib.request
        import json

        payload = json.dumps(
            {
                "model": self.config.model,
                "input": text,
            }
        ).encode("utf-8")

        try:
            req = urllib.request.Request(
                f"{self._base_url}/api/embed",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            embeddings = data.get("embeddings", [[]])
            full = np.array(embeddings[0], dtype=np.float32)

            if len(full) > self.config.embedding_dim:
                return full[: self.config.embedding_dim]
            elif len(full) < self.config.embedding_dim:
                return np.pad(full, (0, self.config.embedding_dim - len(full)))
            return full
        except Exception:
            embedding = np.zeros(self.config.embedding_dim, dtype=np.float32)
            for i, char in enumerate(text):
                idx = (ord(char) + i) % self.config.embedding_dim
                embedding[idx] += ord(char) / 256.0
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding


def _check_ollama_available() -> bool:
    """Check if Ollama is running locally."""
    import urllib.request

    try:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


def auto_detect_llm(provider: str = "auto", model: Optional[str] = None) -> tuple:
    """
    Auto-detect the best available LLM provider.

    Priority: Ollama (free, local) → OpenAI → Anthropic → Mock

    Returns:
        (LLMOracle instance, provider name)
    """
    if provider != "auto":
        if provider == "ollama":
            config = LLMConfig(provider="ollama", model=model or "llama3.2:3b")
            return OllamaLLM(config), "ollama"
        config = LLMConfig(provider=provider, model=model or "gpt-4o-mini")
        if provider == "anthropic" and model is None:
            config.model = "claude-3-haiku-20240307"
        return create_llm(config), provider

    # Auto-detect
    if _check_ollama_available():
        config = LLMConfig(provider="ollama", model=model or "llama3.2:3b")
        return OllamaLLM(config), "ollama"

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key and HAS_OPENAI:
        config = LLMConfig(
            provider="openai",
            model=model or "gpt-4o-mini",
            api_key=openai_key,
        )
        return OpenAILLM(config), "openai"

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key and HAS_ANTHROPIC:
        config = LLMConfig(
            provider="anthropic",
            model=model or "claude-3-haiku-20240307",
            api_key=anthropic_key,
        )
        return AnthropicLLM(config), "anthropic"

    return MockLLM(), "mock"


class SmartWrapper:
    """
    Routes LLM queries through cognitive modules for enhanced responses.

    The wrapper adds cognitive capabilities on top of any LLM:
    1. Encodes input via SemanticBridge (text → cognitive vector)
    2. Retrieves relevant memories
    3. Analyzes reasoning requirements
    4. Runs cognitive cycle (perceive → think → act)
    5. Builds enriched prompt with cognitive context
    6. Queries LLM with enriched context
    7. Learns from the interaction
    """

    def __init__(self, provider: str = "auto", model: Optional[str] = None):
        self.llm, self.provider_name = auto_detect_llm(provider, model)
        self.bridge = SemanticBridge(self.llm)

        # Lazy-load heavy modules
        self._core = None
        self._memory = None
        self._reasoning = None
        self._history: List[Dict[str, str]] = []

    def _get_core(self):
        if self._core is None:
            from neuro.modules.system.cognitive_core import CognitiveCore

            self._core = CognitiveCore()
            self._core.initialize()
        return self._core

    def _get_memory(self):
        if self._memory is None:
            from neuro.modules.m04_memory.memory_controller import MemoryController

            self._memory = MemoryController()
        return self._memory

    def _get_reasoning(self):
        if self._reasoning is None:
            from neuro.modules.m03_reasoning_types.reasoning_orchestrator import (
                ReasoningOrchestrator,
            )

            self._reasoning = ReasoningOrchestrator()
        return self._reasoning

    def query(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> SmartResponse:
        """
        Enhanced LLM query routed through cognitive modules.

        Args:
            message: User message
            history: Optional conversation history [{"role": "user/assistant", "content": "..."}]

        Returns:
            SmartResponse with text, modules used, cognitive context, and processing trace
        """
        start_time = time.time()
        steps = []
        modules_used = []
        context = {}

        # Use provided history or internal
        conv_history = history or self._history

        # Step 1: Encode input
        steps.append("Encoding input via SemanticBridge")
        input_vector = self.bridge.encode(message)
        modules_used.append("semantic_bridge")

        # Step 2: Analyze reasoning requirements
        steps.append("Analyzing reasoning requirements")
        try:
            reasoning = self._get_reasoning()
            task_analysis = reasoning.analyze_task(message)
            context["reasoning"] = {
                "primary_type": task_analysis.primary_type.value,
                "required_types": [t.value for t in task_analysis.required_types],
                "complexity": task_analysis.complexity,
                "confidence": task_analysis.confidence,
            }
            modules_used.append("reasoning_orchestrator")
            steps.append(
                f"  Primary reasoning: {task_analysis.primary_type.value} "
                f"(complexity: {task_analysis.complexity:.2f})"
            )
        except Exception as e:
            context["reasoning"] = {"error": str(e)}
            steps.append(f"  Reasoning analysis skipped: {e}")

        # Step 3: Memory retrieval
        steps.append("Retrieving relevant memories")
        memory_context = []
        try:
            memory = self._get_memory()
            # Store input in sensory memory
            memory.process_visual(input_vector[: min(len(input_vector), 64)])
            attended = memory.attend("visual")
            if attended is not None:
                memory_context.append("Recent sensory input attended")
            modules_used.append("memory_controller")
            steps.append(f"  Memory items retrieved: {len(memory_context)}")
        except Exception as e:
            steps.append(f"  Memory retrieval skipped: {e}")

        # Step 4: Cognitive cycle
        steps.append("Running cognitive cycle (perceive → think → act)")
        workspace_proposals = 0
        try:
            core = self._get_core()
            core.perceive(input_vector)
            workspace_proposals = core.think()
            state = core.get_state()
            context["cognitive"] = {
                "cycle_count": state.cycle_count,
                "active_modules": state.active_modules[:10],
                "workspace_proposals": workspace_proposals,
                "has_workspace_content": len(state.workspace_contents) > 0,
            }
            modules_used.extend(["cognitive_core", "global_workspace", "active_inference"])
            steps.append(f"  Workspace proposals: {workspace_proposals}")
            steps.append(f"  Active modules: {len(state.active_modules)}")
        except Exception as e:
            context["cognitive"] = {"error": str(e)}
            steps.append(f"  Cognitive cycle skipped: {e}")

        # Step 5: Build enriched prompt
        steps.append("Building enriched LLM prompt")
        enriched_prompt = self._build_prompt(message, conv_history, context)

        # Step 6: Query LLM
        steps.append(f"Querying LLM ({self.provider_name})")
        try:
            system_prompt = self._build_system_prompt(context)
            response = self.llm.query(enriched_prompt, system=system_prompt)
            text = response.text
            steps.append(
                f"  Response generated ({response.tokens_used} tokens, {response.latency:.2f}s)"
            )
        except Exception as e:
            text = f"I encountered an error processing your request: {e}"
            steps.append(f"  LLM error: {e}")

        # Step 7: Learn from interaction
        steps.append("Learning from interaction")
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": text})
        if len(self._history) > 40:
            self._history = self._history[-20:]

        # Calculate confidence
        confidence = self._calculate_confidence(context, workspace_proposals)

        latency = time.time() - start_time
        steps.append(f"Total processing: {latency:.2f}s")

        return SmartResponse(
            text=text,
            modules_used=list(set(modules_used)),
            cognitive_context=context,
            confidence=confidence,
            processing_steps=steps,
            latency=latency,
            provider=self.provider_name,
        )

    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt with cognitive context."""
        parts = [
            "You are NEURO, an AI enhanced by a neuroscience-inspired cognitive architecture.",
            "Your responses are enriched by 38 cognitive modules including memory, reasoning,",
            "causal analysis, planning, and emotional processing.",
            "",
        ]

        # Add reasoning guidance
        reasoning = context.get("reasoning", {})
        if "primary_type" in reasoning:
            parts.append(
                f"Reasoning analysis suggests using {reasoning['primary_type']} reasoning."
            )
            if reasoning.get("required_types"):
                parts.append(
                    f"Additional reasoning types: {', '.join(reasoning['required_types'])}"
                )
            parts.append("")

        # Add cognitive state
        cognitive = context.get("cognitive", {})
        if "active_modules" in cognitive:
            n_modules = len(cognitive["active_modules"])
            parts.append(
                f"Cognitive state: {n_modules} modules active, "
                f"{cognitive.get('workspace_proposals', 0)} workspace proposals."
            )
            parts.append("")

        parts.extend(
            [
                "Respond thoughtfully and clearly. When appropriate, show your reasoning process.",
                "If the query involves causal relationships, explain the chain of causation.",
                "If it involves planning, break down the steps.",
                "If it involves memory or past context, reference relevant information.",
            ]
        )

        return "\n".join(parts)

    def _build_prompt(
        self,
        message: str,
        history: List[Dict[str, str]],
        context: Dict[str, Any],
    ) -> str:
        """Build enriched prompt with conversation history and cognitive context."""
        parts = []

        # Add recent history for context
        if history:
            recent = history[-6:]
            if recent:
                parts.append("Recent conversation:")
                for msg in recent:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if len(content) > 200:
                        content = content[:200] + "..."
                    parts.append(f"  {role}: {content}")
                parts.append("")

        # Add cognitive insights
        reasoning = context.get("reasoning", {})
        if reasoning.get("primary_type"):
            parts.append(
                f"[Cognitive insight: This requires {reasoning['primary_type']} reasoning]"
            )

        parts.append(message)
        return "\n".join(parts)

    def _calculate_confidence(self, context: Dict[str, Any], proposals: int) -> float:
        """Calculate overall confidence from cognitive processing."""
        scores = []

        reasoning = context.get("reasoning", {})
        if "confidence" in reasoning:
            scores.append(reasoning["confidence"])

        cognitive = context.get("cognitive", {})
        if cognitive.get("has_workspace_content"):
            scores.append(0.8)
        if proposals > 0:
            scores.append(min(1.0, proposals / 10.0))

        if not scores:
            return 0.5

        return float(np.mean(scores))

    def reset(self) -> None:
        """Reset conversation history and module state."""
        self._history = []
        if self._core is not None:
            self._core.reset()
        if self._memory is not None:
            self._memory = None
        self.bridge.reset()

    def get_statistics(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        stats = {
            "provider": self.provider_name,
            "history_length": len(self._history),
            "llm_stats": self.llm.get_statistics(),
            "bridge_stats": self.bridge.get_statistics(),
        }
        if self._core is not None:
            stats["core_stats"] = self._core.get_statistics()
        return stats


def smart_query(message: str, provider: str = "auto", model: Optional[str] = None) -> SmartResponse:
    """
    Quick one-shot smart query.

    Args:
        message: The query to process
        provider: LLM provider ("auto", "ollama", "openai", "anthropic", "mock")
        model: Optional model name

    Returns:
        SmartResponse with enhanced LLM output

    Example:
        >>> from neuro.wrapper import smart_query
        >>> result = smart_query("What causes inflation?")
        >>> print(result.text)
        >>> print(result.modules_used)
    """
    wrapper = SmartWrapper(provider=provider, model=model)
    return wrapper.query(message)
