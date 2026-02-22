"""IPython magic extension for NEURO cognitive architecture.

Provides %neuro line magic and %%neuro cell magic for interactive
cognitive processing inside Jupyter notebooks.

Usage:
    %load_ext neuro.jupyter
    %neuro think What causes inflation?
    %neuro modules
    %neuro info
    %neuro benchmark
"""

try:
    from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
    from IPython.display import display, HTML
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

from typing import Optional


MODULE_CATALOG = [
    ("00", "Global Workspace", "Attention & broadcast coordination"),
    ("01", "Attention", "Selective attention & salience"),
    ("02", "Perception", "Sensory pattern recognition"),
    ("03", "Reasoning Types", "Deductive, inductive, abductive reasoning"),
    ("04", "Memory", "Episodic, semantic, procedural memory"),
    ("05", "Learning", "Online & batch learning"),
    ("06", "Language", "Natural language understanding"),
    ("07", "Emotion", "Appraisal & affective processing"),
    ("08", "Motivation", "Goal prioritization & drives"),
    ("09", "Planning", "Action planning & sequencing"),
    ("10", "Metacognition", "Self-monitoring & confidence"),
    ("11", "Creativity", "Divergent thinking & novelty"),
    ("12", "Social Cognition", "Theory of Mind & social reasoning"),
    ("13", "Decision Making", "Utility evaluation & choice"),
    ("14", "Consciousness", "Phenomenal awareness model"),
    ("15", "Self Model", "Identity & autobiographical self"),
    ("16", "Predictive Coding", "Hierarchical prediction error"),
    ("17", "World Model", "Internal environment simulation"),
    ("18", "Self Improvement", "Darwin-Godel machine"),
    ("19", "Multi-Agent", "Swarm intelligence & coordination"),
    ("--", "System Core", "Active inference loop"),
    ("--", "LLM Oracle", "Language model interface"),
    ("--", "Knowledge Graph", "Structured knowledge storage"),
    ("--", "Grounding", "Sensor & actuator bridge"),
    ("--", "Scaling", "Distributed processing"),
    ("--", "Transfer", "Cross-domain transfer learning"),
    ("--", "Environment", "Environment abstraction"),
    ("--", "Benchmarks", "Performance evaluation suite"),
    ("--", "Inference", "Probabilistic inference"),
    ("--", "Perception Layer", "Multi-modal perception"),
    ("--", "Integration", "Cross-module integration"),
    ("--", "Causal", "Causal reasoning & SCMs"),
    ("--", "Abstract", "Compositional abstraction"),
    ("--", "Continual", "Continual learning"),
    ("--", "Robust", "Robustness & OOD detection"),
    ("--", "Hierarchical Planning", "MCTS & goal decomposition"),
    ("--", "Credit Assignment", "Eligibility traces & contribution"),
    ("--", "Meta-Reasoning", "Problem classification & style selection"),
]


def _modules_html() -> str:
    """Build an HTML table of all cognitive modules."""
    rows = ""
    for mid, name, desc in MODULE_CATALOG:
        badge_bg = "#e8f5e9" if mid.isdigit() else "#fff3e0"
        badge_color = "#2e7d32" if mid.isdigit() else "#e65100"
        rows += (
            f"<tr>"
            f"<td style='padding:4px 8px;'>"
            f"<span style='display:inline-block;background:{badge_bg};color:{badge_color};"
            f"padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600;'>"
            f"{mid}</span></td>"
            f"<td style='padding:4px 8px;font-weight:500;'>{name}</td>"
            f"<td style='padding:4px 8px;color:#666;font-size:13px;'>{desc}</td>"
            f"</tr>"
        )
    return (
        f"<div style='font-family:system-ui,sans-serif;'>"
        f"<h3 style='margin:0 0 8px 0;'>NEURO Cognitive Modules "
        f"({len(MODULE_CATALOG)})</h3>"
        f"<table style='border-collapse:collapse;width:100%;max-width:700px;'>"
        f"<thead><tr style='border-bottom:2px solid #ddd;'>"
        f"<th style='padding:4px 8px;text-align:left;font-size:12px;color:#999;'>ID</th>"
        f"<th style='padding:4px 8px;text-align:left;font-size:12px;color:#999;'>Module</th>"
        f"<th style='padding:4px 8px;text-align:left;font-size:12px;color:#999;'>"
        f"Description</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _info_html() -> str:
    """Build an HTML info card for the framework."""
    try:
        from neuro import __version__
    except Exception:
        __version__ = "0.9.0"

    return (
        f"<div style='font-family:system-ui,sans-serif;border:1px solid #e0e0e0;"
        f"border-radius:8px;padding:16px;max-width:400px;'>"
        f"<h3 style='margin:0 0 12px 0;'>NEURO AGI</h3>"
        f"<table style='border-collapse:collapse;'>"
        f"<tr><td style='padding:2px 12px 2px 0;color:#888;'>Version</td>"
        f"<td><b>{__version__}</b></td></tr>"
        f"<tr><td style='padding:2px 12px 2px 0;color:#888;'>Modules</td>"
        f"<td><b>{len(MODULE_CATALOG)}</b></td></tr>"
        f"<tr><td style='padding:2px 12px 2px 0;color:#888;'>Tier 1 (Cognitive)</td>"
        f"<td>20</td></tr>"
        f"<tr><td style='padding:2px 12px 2px 0;color:#888;'>Tier 2 (Infra)</td>"
        f"<td>6</td></tr>"
        f"<tr><td style='padding:2px 12px 2px 0;color:#888;'>Tier 3 (Support)</td>"
        f"<td>5</td></tr>"
        f"<tr><td style='padding:2px 12px 2px 0;color:#888;'>Tier 4 (AGI)</td>"
        f"<td>7</td></tr>"
        f"<tr><td style='padding:2px 12px 2px 0;color:#888;'>Python</td>"
        f"<td>3.9+</td></tr>"
        f"<tr><td style='padding:2px 12px 2px 0;color:#888;'>License</td>"
        f"<td>MIT</td></tr>"
        f"</table></div>"
    )


if HAS_IPYTHON:
    @magics_class
    class NeuroMagic(Magics):
        """IPython magics for NEURO cognitive architecture."""

        def __init__(self, shell):
            super().__init__(shell)
            self._brain = None

        def _get_brain(self):
            if self._brain is None:
                from neuro.brain import Brain
                self._brain = Brain()
            return self._brain

        @line_magic
        def neuro(self, line: str) -> None:
            """NEURO line magic.

            Subcommands:
                think <query>   -- cognitive query
                modules         -- list all 38 modules
                info            -- framework info
                benchmark       -- run benchmarks
            """
            line = line.strip()
            if not line:
                display(HTML(
                    "<div style='font-family:system-ui,sans-serif;color:#888;'>"
                    "Usage: <code>%neuro think &lt;query&gt;</code> | "
                    "<code>%neuro modules</code> | "
                    "<code>%neuro info</code> | "
                    "<code>%neuro benchmark</code></div>"
                ))
                return

            parts = line.split(None, 1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""

            if cmd == "think":
                if not args:
                    display(HTML(
                        "<div style='color:#c62828;'>Provide a query: "
                        "<code>%neuro think What is consciousness?</code></div>"
                    ))
                    return
                brain = self._get_brain()
                result = brain.think(args)
                display(HTML(result._repr_html_()))

            elif cmd == "modules":
                display(HTML(_modules_html()))

            elif cmd == "info":
                display(HTML(_info_html()))

            elif cmd == "benchmark":
                display(HTML(
                    "<div style='font-family:system-ui,sans-serif;color:#888;'>"
                    "Running benchmarks...</div>"
                ))
                try:
                    from neuro.wrapper import smart_query
                    result = smart_query(
                        "Run a quick self-assessment: rate your reasoning, "
                        "memory, and planning capabilities on a 0-10 scale."
                    )
                    display(HTML(result._repr_html_()))
                except Exception as e:
                    display(HTML(
                        f"<div style='color:#c62828;'>Benchmark error: {e}</div>"
                    ))

            else:
                display(HTML(
                    f"<div style='color:#c62828;'>Unknown subcommand: "
                    f"<code>{cmd}</code>. "
                    f"Try <code>think</code>, <code>modules</code>, "
                    f"<code>info</code>, or <code>benchmark</code>.</div>"
                ))

        @cell_magic
        def neuro(self, line: str, cell: str) -> None:
            """NEURO cell magic -- sends the entire cell body as a think query.

            Usage:
                %%neuro
                Explain the relationship between inflation
                and interest rates in detail.
            """
            query = cell.strip()
            if not query:
                display(HTML(
                    "<div style='color:#c62828;'>Cell body is empty. "
                    "Write your query in the cell.</div>"
                ))
                return
            brain = self._get_brain()
            result = brain.think(query)
            display(HTML(result._repr_html_()))


def load_ipython_extension(ipython) -> None:
    """Register NEURO magics with IPython.

    Called automatically by ``%load_ext neuro.jupyter``.
    """
    if not HAS_IPYTHON:
        print("IPython is required for NEURO magics.")
        return
    ipython.register_magics(NeuroMagic)


def unload_ipython_extension(ipython) -> None:
    """Called when the extension is unloaded."""
    pass
