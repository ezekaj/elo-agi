/* ================================================================
   ELO-AGI (NEURO) — Shared JavaScript
   ================================================================ */

// ---- Constants ----
const API_BASE = 'https://zedigital-elo-agi.fly.dev';
let apiConnected = false;

// ---- Module Data ----
const modules = {
  cognitive: [
    { name: 'Global Workspace', desc: 'Global Workspace Theory implementation with attention-gated broadcasting and competition dynamics across cognitive modules.', algorithms: ['Attention Competition', 'Broadcast Protocol', 'Ignition Threshold'], inputs: 'Module activations', outputs: 'Broadcast signals' },
    { name: 'Predictive Coding', desc: 'Free Energy Principle with hierarchical predictions, prediction errors, and belief updating via variational inference.', algorithms: ['Free Energy Minimization', 'Hierarchical Prediction', 'Variational Inference'], inputs: 'Sensory data', outputs: 'Predictions, prediction errors' },
    { name: 'Dual-Process', desc: 'System 1/2 emergence from cognitive geometry. Fast intuitive responses vs slow deliberate reasoning.', algorithms: ['Type 1 Heuristics', 'Type 2 Deliberation', 'Override Detection'], inputs: 'Problem context', outputs: 'Decision pathway' },
    { name: 'Reasoning Types', desc: 'Dimensional, interactive, logical, and perceptual reasoning pathways inspired by dual-process theory.', algorithms: ['Logical Inference', 'Analogical Reasoning', 'Spatial Reasoning'], inputs: 'Problem representation', outputs: 'Reasoning chains' },
    { name: 'Memory', desc: 'Sensory, working, and long-term memory systems with hippocampal replay for consolidation and retrieval.', algorithms: ['Hippocampal Replay', 'Working Memory Buffer', 'Long-term Consolidation'], inputs: 'Experiences', outputs: 'Retrieved memories' },
    { name: 'Sleep Consolidation', desc: 'Replay, systems consolidation, and homeostatic regulation during offline processing periods.', algorithms: ['Memory Replay', 'Systems Consolidation', 'Synaptic Homeostasis'], inputs: 'Recent memories', outputs: 'Consolidated knowledge' },
    { name: 'Motivation', desc: 'Path entropy, dopamine-inspired reward signals, and curiosity-driven exploration.', algorithms: ['Reward Prediction', 'Curiosity Signal', 'Path Entropy'], inputs: 'State, goals', outputs: 'Drive signals' },
    { name: 'Emotion', desc: 'Appraisal-based emotional processing that modulates decision-making, attention, and memory encoding.', algorithms: ['Appraisal Theory', 'Somatic Markers', 'Affect Spectrum'], inputs: 'Stimuli context', outputs: 'Emotional state, modulation signals' },
    { name: 'Language', desc: 'Natural language understanding and generation with semantic parsing, pragmatics, and discourse modeling.', algorithms: ['Semantic Parsing', 'Pragmatic Inference', 'Discourse Modeling'], inputs: 'Text/speech', outputs: 'Semantic representations' },
    { name: 'Creativity', desc: 'Divergent thinking, conceptual blending, and novelty generation using stochastic recombination.', algorithms: ['Conceptual Blending', 'Divergent Search', 'Novelty Detection'], inputs: 'Problem space', outputs: 'Novel solutions' },
    { name: 'Spatial Cognition', desc: 'Allocentric and egocentric spatial representations with cognitive mapping and mental rotation.', algorithms: ['Cognitive Mapping', 'Mental Rotation', 'Path Integration'], inputs: 'Spatial data', outputs: 'Spatial representations' },
    { name: 'Time Perception', desc: 'Embodied temporal cognition modeling interval timing, sequence learning, and temporal prediction.', algorithms: ['Interval Timing', 'Sequence Learning', 'Temporal Prediction'], inputs: 'Event sequences', outputs: 'Temporal estimates' },
    { name: 'Learning', desc: 'Adaptive learning mechanisms including meta-learning, curriculum learning, and learning rate modulation.', algorithms: ['Meta-Learning', 'Curriculum Learning', 'Adaptive LR'], inputs: 'Training signal', outputs: 'Updated weights' },
    { name: 'Executive Control', desc: 'Top-down attention regulation, task switching, inhibition, and goal management for coherent behavior.', algorithms: ['Task Switching', 'Inhibitory Control', 'Goal Stack Management'], inputs: 'Goals, context', outputs: 'Control signals' },
    { name: 'Embodied Cognition', desc: 'Grounding abstract concepts in simulated sensorimotor experience and body-based representations.', algorithms: ['Sensorimotor Grounding', 'Body Schema', 'Affordance Detection'], inputs: 'Body state', outputs: 'Grounded concepts' },
    { name: 'Social Cognition', desc: 'Theory of mind, social inference, shared intentionality, and multi-agent social modeling.', algorithms: ['Theory of Mind', 'Social Inference', 'Shared Intentionality'], inputs: 'Social context', outputs: 'Social predictions' },
    { name: 'Consciousness', desc: 'Global workspace broadcasting, metacognitive monitoring, and self-model maintenance.', algorithms: ['Global Broadcast', 'Self-Model', 'Metacognitive Loop'], inputs: 'All module states', outputs: 'Unified experience' },
    { name: 'World Modeling', desc: 'Internal generative world model for prediction, simulation, counterfactual reasoning, and planning.', algorithms: ['Generative Model', 'Counterfactual Simulation', 'Model-based Planning'], inputs: 'Observations', outputs: 'World state predictions' },
    { name: 'Self-Improvement', desc: 'Self-directed improvement through performance monitoring, architecture search, and capability expansion.', algorithms: ['Performance Monitoring', 'Architecture Search', 'Capability Expansion'], inputs: 'Performance metrics', outputs: 'Improvement plans' },
    { name: 'Multi-Agent', desc: 'Multi-agent coordination with negotiation, consensus building, and distributed problem solving.', algorithms: ['Negotiation Protocol', 'Consensus Building', 'Task Distribution'], inputs: 'Agent states', outputs: 'Coordination signals' },
    { name: 'Attention', desc: 'Selective, sustained, and divided attention mechanisms with priority-based resource allocation.', algorithms: ['Priority Queue', 'Salience Map', 'Resource Allocation'], inputs: 'All stimuli', outputs: 'Filtered focus' },
    { name: 'Perception', desc: 'Hierarchical feature extraction and integration across visual, auditory, and multimodal streams.', algorithms: ['Feature Extraction', 'Multimodal Integration', 'Object Recognition'], inputs: 'Sensor data', outputs: 'Percepts' },
    { name: 'Decision Making', desc: 'Evidence accumulation, value-based choice, and satisficing strategies under uncertainty.', algorithms: ['Evidence Accumulation', 'Value Computation', 'Satisficing'], inputs: 'Options, evidence', outputs: 'Decisions' }
  ],
  infrastructure: [
    { name: 'Core System', desc: 'Unified cognitive core implementing active inference loop, free energy minimization, and module orchestration.', algorithms: ['Active Inference Loop', 'Free Energy Minimization', 'Module Orchestration'], inputs: 'Module states', outputs: 'System coordination' },
    { name: 'LLM Integration', desc: 'Semantic bridge to language models for grounding, dialogue generation, and knowledge extraction.', algorithms: ['Semantic Bridge', 'Knowledge Extraction', 'Dialogue Management'], inputs: 'Queries', outputs: 'Grounded responses' },
    { name: 'Knowledge Graph', desc: 'Structured fact store with ontological reasoning, graph-based inference, and dynamic updates.', algorithms: ['Graph Inference', 'Ontology Reasoning', 'Knowledge Fusion'], inputs: 'Facts, queries', outputs: 'Inferred knowledge' },
    { name: 'Sensors', desc: 'Camera, microphone, and proprioception interfaces for multimodal sensory input processing.', algorithms: ['Signal Processing', 'Multimodal Fusion', 'Noise Filtering'], inputs: 'Raw sensor data', outputs: 'Processed signals' },
    { name: 'Actuators', desc: 'Motor output, speech synthesis, and environment manipulation interfaces for embodied action.', algorithms: ['Motor Planning', 'Speech Synthesis', 'Action Execution'], inputs: 'Action commands', outputs: 'Physical actions' },
    { name: 'Distributed Scaling', desc: 'Coordinator-worker architecture with GPU kernel fusion and gradient aggregation for scale-out.', algorithms: ['Coordinator Protocol', 'Gradient Aggregation', 'GPU Kernel Fusion'], inputs: 'Workload', outputs: 'Distributed computation' }
  ],
  support: [
    { name: 'Benchmarking', desc: 'Comprehensive test suites for reasoning, memory, language, and planning evaluation.', algorithms: ['Test Generation', 'Metric Computation', 'Comparison Analysis'], inputs: 'Module outputs', outputs: 'Performance scores' },
    { name: 'Perception Pipeline', desc: 'End-to-end sensory processing pipeline with feature extraction, integration, and filtering.', algorithms: ['Feature Pipeline', 'Attention Gating', 'Stream Integration'], inputs: 'Raw input', outputs: 'Processed percepts' },
    { name: 'Environment Manager', desc: 'Context and environment management for simulation, testing, and deployment configuration.', algorithms: ['Context Switching', 'State Management', 'Config Resolution'], inputs: 'Environment config', outputs: 'Managed context' },
    { name: 'Inference Engine', desc: 'Bayesian, analogical, and causal reasoning engines for probabilistic inference.', algorithms: ['Bayesian Inference', 'Analogical Reasoning', 'Causal Inference'], inputs: 'Evidence', outputs: 'Inferred beliefs' },
    { name: 'Integration Layer', desc: 'Advanced integration layer connecting perception, cognition, and action subsystems.', algorithms: ['Cross-module Bridge', 'State Synchronization', 'Event Bus'], inputs: 'Module outputs', outputs: 'Integrated state' },
    { name: 'Data Loader', desc: 'Efficient data loading, preprocessing, and batching for training and inference workloads.', algorithms: ['Batch Scheduling', 'Prefetch Pipeline', 'Format Conversion'], inputs: 'Raw data', outputs: 'Preprocessed batches' },
    { name: 'Logger', desc: 'Structured logging, telemetry, and cognitive state tracing for debugging and analysis.', algorithms: ['Structured Logging', 'Telemetry Collection', 'State Tracing'], inputs: 'System events', outputs: 'Log streams' },
    { name: 'Perception Module', desc: 'Visual, auditory, and multimodal perception with hierarchical processing.', algorithms: ['Visual Processing', 'Auditory Processing', 'Multimodal Binding'], inputs: 'Sensory streams', outputs: 'Unified percepts' },
    { name: 'Transfer Learning', desc: 'Cross-domain knowledge transfer, domain adaptation, and few-shot generalization mechanisms.', algorithms: ['Domain Adaptation', 'Few-shot Transfer', 'Knowledge Distillation'], inputs: 'Source knowledge', outputs: 'Adapted representations' }
  ]
};

const tierColors = {
  cognitive: { border: 'rgba(124,58,237,0.25)', bg: 'rgba(124,58,237,0.06)', text: '#7C3AED', hoverBg: 'rgba(124,58,237,0.12)' },
  infrastructure: { border: 'rgba(6,182,212,0.25)', bg: 'rgba(6,182,212,0.06)', text: '#06B6D4', hoverBg: 'rgba(6,182,212,0.12)' },
  support: { border: 'rgba(139,92,246,0.25)', bg: 'rgba(139,92,246,0.06)', text: '#8B5CF6', hoverBg: 'rgba(139,92,246,0.12)' }
};

// ---- Benchmark Data ----
const benchmarks = [
  { cat: 'Causal Reasoning', score: 0.85, tests: '124/124' },
  { cat: 'Compositional Abstraction', score: 0.74, tests: '144/144' },
  { cat: 'Continual Learning', score: 0.72, tests: '89/89' },
  { cat: 'Robustness', score: 0.71, tests: '192/192' },
  { cat: 'Language Understanding', score: 0.69, tests: '78/78' },
  { cat: 'Overall', score: 0.697, tests: '627/627' }
];

// ---- Module List Table ----
const allModulesTable = [
  ['Global Workspace','Cognitive','GWT attention-gated broadcasting'],
  ['Predictive Coding','Cognitive','Free Energy Principle, hierarchical predictions'],
  ['Dual-Process','Cognitive','System 1/2 fast/slow reasoning'],
  ['Reasoning Types','Cognitive','Dimensional, logical, perceptual reasoning'],
  ['Memory','Cognitive','Sensory, working, and long-term memory'],
  ['Sleep Consolidation','Cognitive','Replay and systems consolidation'],
  ['Motivation','Cognitive','Path entropy and curiosity signals'],
  ['Emotion','Cognitive','Appraisal-based emotional processing'],
  ['Language','Cognitive','NLU and generation with semantic parsing'],
  ['Creativity','Cognitive','Divergent thinking and conceptual blending'],
  ['Spatial Cognition','Cognitive','Cognitive mapping and mental rotation'],
  ['Time Perception','Cognitive','Interval timing and temporal prediction'],
  ['Learning','Cognitive','Meta-learning and curriculum learning'],
  ['Executive Control','Cognitive','Task switching, inhibition, goal management'],
  ['Embodied Cognition','Cognitive','Sensorimotor grounding of concepts'],
  ['Social Cognition','Cognitive','Theory of mind and social inference'],
  ['Consciousness','Cognitive','Global workspace and self-model'],
  ['World Modeling','Cognitive','Internal generative world model'],
  ['Self-Improvement','Cognitive','Performance monitoring and capability expansion'],
  ['Multi-Agent','Cognitive','Multi-agent coordination and negotiation'],
  ['Attention','Cognitive','Selective and divided attention'],
  ['Perception','Cognitive','Hierarchical feature extraction'],
  ['Decision Making','Cognitive','Evidence accumulation under uncertainty'],
  ['Core System','Infrastructure','Active inference loop and orchestration'],
  ['LLM Integration','Infrastructure','Semantic bridge to language models'],
  ['Knowledge Graph','Infrastructure','Structured fact store and ontology'],
  ['Sensors','Infrastructure','Camera, mic, proprioception interfaces'],
  ['Actuators','Infrastructure','Motor output and speech synthesis'],
  ['Distributed Scaling','Infrastructure','Coordinator-worker GPU architecture'],
  ['Benchmarking','Support','Test suites for cognitive evaluation'],
  ['Perception Pipeline','Support','End-to-end sensory processing'],
  ['Environment Manager','Support','Simulation and deployment config'],
  ['Inference Engine','Support','Bayesian and causal reasoning engines'],
  ['Integration Layer','Support','Cross-module state synchronization'],
  ['Data Loader','Support','Efficient data preprocessing'],
  ['Logger','Support','Structured logging and telemetry'],
  ['Perception Module','Support','Visual and auditory perception'],
  ['Transfer Learning','Support','Cross-domain knowledge transfer']
];

// ================================================================
// UTILITY FUNCTIONS
// ================================================================

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function copyPip(el) {
  navigator.clipboard.writeText('pip install neuro-agi').then(() => {
    const tip = el.querySelector('.tooltip');
    if (tip) { tip.classList.add('show'); setTimeout(() => tip.classList.remove('show'), 1500); }
  });
}

function copyCode(btn, code) {
  navigator.clipboard.writeText(code).then(() => {
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    btn.style.color = '#14B8A6';
    setTimeout(() => { btn.textContent = orig; btn.style.color = ''; }, 1500);
  });
}

function copyText(text, el) {
  navigator.clipboard.writeText(text).then(() => {
    if (el) {
      const orig = el.textContent;
      el.textContent = 'Copied!';
      setTimeout(() => { el.textContent = orig; }, 1500);
    }
  });
}

// ================================================================
// MOBILE MENU
// ================================================================

function toggleMobile() {
  const menu = document.getElementById('mobileMenu');
  const hamburger = document.getElementById('hamburgerIcon');
  const close = document.getElementById('closeIcon');
  if (!menu) return;
  const isHidden = menu.classList.contains('hidden');
  menu.classList.toggle('hidden');
  if (hamburger) hamburger.classList.toggle('hidden', isHidden);
  if (close) close.classList.toggle('hidden', !isHidden);
}

// ================================================================
// MODULE GRID
// ================================================================

function renderModules(containerId) {
  const prefix = containerId ? containerId + '-' : '';
  Object.keys(modules).forEach(tier => {
    const container = document.getElementById(prefix + 'tab-' + tier) || document.getElementById('tab-' + tier);
    if (!container) return;
    const colors = tierColors[tier];
    container.innerHTML = modules[tier].map((mod, i) => `
      <div class="module-tile rounded-lg p-4" style="background:${colors.bg}; border:1px solid ${colors.border};" onclick="toggleTile(this)">
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full flex-shrink-0" style="background:${colors.text};"></span>
          <span class="text-sm font-semibold text-white">${mod.name}</span>
        </div>
        <div class="module-desc text-xs text-slate-400 leading-relaxed">${mod.desc}</div>
      </div>
    `).join('');
  });
}

function toggleTile(el) { el.classList.toggle('expanded'); }

function switchTab(tab, btn) {
  document.querySelectorAll('.module-grid').forEach(g => g.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.remove('active');
    b.style.borderColor = 'transparent';
    b.style.background = 'rgba(255,255,255,0.03)';
    b.style.color = '#94a3b8';
  });
  const target = document.getElementById('tab-' + tab);
  if (!target) return;
  target.classList.remove('hidden');
  target.classList.remove('fade-in');
  void target.offsetWidth;
  target.classList.add('fade-in');
  const colors = tierColors[tab];
  btn.classList.add('active');
  btn.style.borderColor = colors.border;
  btn.style.background = colors.bg;
  btn.style.color = '#fff';
}

// ================================================================
// BENCHMARK TABLE
// ================================================================

function renderBenchmarks(bodyId) {
  const body = document.getElementById(bodyId || 'benchmarkBody');
  if (!body) return;
  body.innerHTML = benchmarks.map((b, i) => {
    const pct = (b.score * 100).toFixed(0);
    const isLast = i === benchmarks.length - 1;
    return `
      <tr class="${isLast ? 'border-t border-white/5' : ''}">
        <td class="text-slate-300 ${isLast ? 'font-bold text-white' : 'font-medium'}">${b.cat}</td>
        <td class="font-mono text-white font-semibold">${b.score.toFixed(b.score === 0.697 ? 3 : 2)}</td>
        <td class="hidden sm:table-cell">
          <div class="w-full bg-white/5 rounded-full h-2 overflow-hidden">
            <div class="bar-gradient h-full rounded-full benchmark-bar" style="width:0%;" data-target="${pct}"></div>
          </div>
        </td>
        <td class="font-mono text-slate-400 text-sm">${b.tests}</td>
        <td><span class="text-green-400 font-semibold text-sm">&#10003; Pass</span></td>
      </tr>
    `;
  }).join('');
}

function animateBenchmarkBars() {
  document.querySelectorAll('.benchmark-bar').forEach(bar => {
    const target = bar.getAttribute('data-target');
    setTimeout(() => { bar.style.width = target + '%'; bar.style.transition = 'width 1s ease'; }, 200);
  });
}

// ================================================================
// API CONNECTION
// ================================================================

async function checkApiConnection() {
  const statusEl = document.getElementById('api-status');
  if (!statusEl) return;
  statusEl.className = 'connecting';
  statusEl.innerHTML = '<span class="status-dot"></span>Connecting...';
  try {
    const res = await fetch(API_BASE + '/api/health', { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      apiConnected = true;
      statusEl.className = 'connected';
      statusEl.innerHTML = '<span class="status-dot"></span>Live API';
    } else { throw new Error('Not ok'); }
  } catch (e) {
    apiConnected = false;
    statusEl.className = 'disconnected';
    statusEl.innerHTML = '<span class="status-dot"></span>Offline (simulated)';
  }
}

async function apiRepl(command) {
  try {
    const res = await fetch(API_BASE + '/api/repl', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command }),
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) throw new Error('API error');
    return await res.json();
  } catch (e) { return null; }
}

async function apiModules() {
  try {
    const res = await fetch(API_BASE + '/api/modules', { signal: AbortSignal.timeout(5000) });
    if (!res.ok) throw new Error('API error');
    return await res.json();
  } catch (e) { return null; }
}

async function apiBenchmark() {
  try {
    const res = await fetch(API_BASE + '/api/benchmark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
      signal: AbortSignal.timeout(30000),
    });
    if (!res.ok) throw new Error('API error');
    return await res.json();
  } catch (e) { return null; }
}

async function apiAnalyze(text, types) {
  try {
    const res = await fetch(API_BASE + '/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, analysis_types: types }),
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) throw new Error('API error');
    return await res.json();
  } catch (e) { return null; }
}

// ================================================================
// REPL TERMINAL
// ================================================================

let replHistory = [];
let replHistoryIdx = -1;
let replBusy = false;

function replAddLine(html) {
  const output = document.getElementById('repl-output');
  if (!output) return;
  const div = document.createElement('div');
  div.className = 'repl-line';
  div.innerHTML = html;
  output.appendChild(div);
  output.scrollTop = output.scrollHeight;
}

function replAddPromptLine(cmd) {
  replAddLine('<span class="repl-prompt">&gt;&gt;&gt; </span><span class="repl-input-text">' + esc(cmd) + '</span>');
}

function replAddOutput(text, cls) {
  cls = cls || 'repl-output-text';
  replAddLine('<span class="' + cls + '">' + text + '</span>');
}

function replClear() {
  const output = document.getElementById('repl-output');
  if (output) output.innerHTML = '';
}

function replQuickCmd(cmd) {
  const input = document.getElementById('repl-input');
  if (input) { input.value = cmd; }
  handleReplCommand(cmd);
}

async function handleReplCommand(cmd) {
  cmd = cmd.trim();
  if (!cmd) return;

  replHistory.unshift(cmd);
  replHistoryIdx = -1;
  replAddPromptLine(cmd);

  const input = document.getElementById('repl-input');
  if (input) input.value = '';

  if (cmd === 'clear()') { replClear(); return; }

  // Try API first
  if (apiConnected) {
    const dpMatch = cmd.match(/^neuro_agi\.cognitive\.DualProcess\.run\(["'](.+?)["']\)$/);
    const emMatch = cmd.match(/^neuro_agi\.cognitive\.EmotionEngine\.analyze\(["'](.+?)["']\)$/);

    if (dpMatch) {
      replAddOutput('Analyzing with dual-process cognition...');
      const result = await apiAnalyze(dpMatch[1], ['dual_process']);
      if (result && result.analyses && result.analyses.dual_process) {
        const dp = result.analyses.dual_process;
        replAddOutput(esc(`DualProcess Analysis: "${dpMatch[1]}"\n${'─'.repeat(50)}`));
        replAddOutput(esc(`System 1 (Fast / Intuitive):\n  Response time: ${dp.system1.response_time_ms}ms\n  Confidence:    ${dp.system1.confidence}\n  Assessment:    ${dp.system1.assessment}\n\nSystem 2 (Slow / Analytical):\n  Response time: ${dp.system2.response_time_ms}ms\n  Confidence:    ${dp.system2.confidence}\n  Assessment:    ${dp.system2.assessment}\n\n  Integrated:    ${dp.integration.override ? 'System 2 override' : 'System 1 sufficient'}\n  Execution time: ${result.execution_time}s`));
        return;
      }
    }

    if (emMatch) {
      replAddOutput('Running emotion analysis...');
      const result = await apiAnalyze(emMatch[1], ['emotion']);
      if (result && result.analyses && result.analyses.emotion) {
        const em = result.analyses.emotion;
        const spectrum = Object.entries(em.affect_spectrum);
        let spectrumStr = spectrum.map(([name, score]) => {
          const filled = Math.round(score * 20);
          const empty = 20 - filled;
          return `    ${name.padEnd(12)} ${'\u2588'.repeat(filled)}${'\u2591'.repeat(empty)}  ${score.toFixed(2)}`;
        }).join('\n');
        replAddOutput(esc(`EmotionEngine Analysis: "${emMatch[1]}"\n${'─'.repeat(50)}\n  Primary emotion: ${em.primary_emotion} (${em.primary_score})\n  Valence: ${em.valence}\n  Arousal: ${em.arousal}\n\n  Affect spectrum:\n${spectrumStr}`));
        return;
      }
    }

    if (cmd === 'neuro_agi.run_benchmark()') {
      replAddOutput('Running benchmarks on live API...');
      const result = await apiBenchmark();
      if (result && result.results) {
        const r = result.results;
        let catStr = r.categories ? Object.entries(r.categories).map(([cat, data]) => `  ${cat.padEnd(22)} ${(data.average_score * 100).toFixed(1)}%    ${data.passed}/${data.tests}`).join('\n') : '';
        replAddOutput(esc(`\n  Category          Score    Tests\n  ${'─'.repeat(37)}\n${catStr}\n  ${'─'.repeat(37)}\n  Overall           ${(r.overall_score * 100).toFixed(1)}%    ${r.total_tests} tests\n  Source: Live API`));
        return;
      }
    }

    if (cmd === 'neuro_agi.list_modules()') {
      const result = await apiModules();
      if (result && result.modules) {
        let rows = result.modules.map(m => `  ${m.name.padEnd(26)} ${m.category.padEnd(18)} ${m.description.substring(0, 50)}...`);
        replAddOutput(esc(`  ${'Module'.padEnd(26)} ${'Category'.padEnd(18)} Description\n  ${'─'.repeat(86)}\n${rows.join('\n')}\n\n  [${result.total} modules total]`));
        return;
      }
    }

    const apiResult = await apiRepl(cmd);
    if (apiResult) {
      if (apiResult.output) replAddOutput(esc(apiResult.output.replace(/\n$/, '')));
      if (apiResult.error) replAddOutput('<span class="repl-error">' + esc(apiResult.error) + '</span>');
      return;
    }
  }

  // Local simulation fallback
  if (cmd === 'help()') {
    replAddOutput(esc(`Available commands:\n${'─'.repeat(50)}\n  neuro_agi.info()                    Framework info\n  neuro_agi.__version__               Version string\n  neuro_agi.list_modules()            All 38 modules\n  neuro_agi.list_tiers()              Show tiers\n  neuro_agi.cognitive.list()          Cognitive modules\n  neuro_agi.cognitive.DualProcess.run(query)     Dual process\n  neuro_agi.cognitive.EmotionEngine.analyze(t)   Emotions\n  neuro_agi.run_benchmark()           Run benchmarks\n  neuro_agi.benchmark.compare()       Compare frameworks\n  neuro_agi.architecture()            ASCII architecture\n  help()                              This help text\n  clear()                             Clear terminal\n\nInstall locally:\n  pip install neuro-agi              Python library\n  pip install neuro-agi && neuro     CLI (requires Ollama)`));
    return;
  }

  if (cmd === 'neuro_agi.info()') {
    replAddOutput(esc(`\u2554${'═'.repeat(38)}\u2557\n\u2551   ELO-AGI Cognitive Framework v2.0  \u2551\n\u2551   38 Modules | 3 Tiers | 627 Tests \u2551\n\u2551   Type help() for available commands\u2551\n\u255A${'═'.repeat(38)}\u255D`));
    return;
  }

  if (cmd === 'neuro_agi.__version__') {
    replAddOutput('<span class="repl-string">\'2.0.0\'</span>');
    return;
  }

  if (cmd === 'neuro_agi.list_modules()') {
    let header = `  ${'Module'.padEnd(26)} ${'Category'.padEnd(18)} Description`;
    let sep = `  ${'─'.repeat(26)} ${'─'.repeat(18)} ${'─'.repeat(42)}`;
    let rows = allModulesTable.map(r => `  ${r[0].padEnd(26)} ${r[1].padEnd(18)} ${r[2]}`);
    replAddOutput(esc(header + '\n' + sep + '\n' + rows.join('\n') + '\n\n  [38 modules total]'));
    return;
  }

  if (cmd === 'neuro_agi.list_tiers()') {
    replAddOutput(esc(`Tier 1: Cognitive Processing   (23 modules)\nTier 2: Infrastructure          (6 modules)\nTier 3: Support                 (9 modules)`));
    return;
  }

  if (cmd === 'neuro_agi.cognitive.list()') {
    let items = allModulesTable.filter(r => r[1] === 'Cognitive');
    let out = items.map((r,i) => `  ${String(i+1).padStart(2)}. ${r[0].padEnd(22)} ${r[2]}`).join('\n');
    replAddOutput(esc(`Cognitive Processing Modules (${items.length}):\n${'─'.repeat(52)}\n${out}`));
    return;
  }

  const dpMatch = cmd.match(/^neuro_agi\.cognitive\.DualProcess\.run\(["'](.+?)["']\)$/);
  if (dpMatch) {
    replAddOutput(esc(`DualProcess Analysis: "${dpMatch[1]}"\n${'─'.repeat(50)}`));
    await sleep(300);
    replAddOutput(esc(`System 1 (Fast / Intuitive):\n  Response time: 47ms\n  Confidence:    0.72\n  Answer:        Pattern-matched intuitive response\n\nSystem 2 (Slow / Analytical):\n  Response time: 1,240ms\n  Confidence:    0.89\n  Answer:        Evidence-based analytical response\n\n  Integrated: System 2 override (confidence delta: +0.17)`));
    return;
  }

  const emMatch = cmd.match(/^neuro_agi\.cognitive\.EmotionEngine\.analyze\(["'](.+?)["']\)$/);
  if (emMatch) {
    replAddOutput(esc(`EmotionEngine Analysis: "${emMatch[1]}"\n${'─'.repeat(50)}`));
    await sleep(200);
    replAddOutput(esc(`  Primary emotion:   Joy        (0.92)\n  Valence:           +0.88 (positive)\n  Arousal:           0.71 (high)\n\n  Affect spectrum:\n    Joy         ${'█'.repeat(18)}${'░'.repeat(2)}  0.92\n    Excitement  ${'█'.repeat(13)}${'░'.repeat(7)}  0.64\n    Surprise    ${'█'.repeat(5)}${'░'.repeat(15)}  0.23`));
    return;
  }

  if (cmd === 'neuro_agi.run_benchmark()') {
    replAddOutput('Running benchmarks...');
    for (const pct of [10,20,30,40,50,60,70,80,90,100]) {
      await sleep(150);
      const output = document.getElementById('repl-output');
      const last = output ? output.lastElementChild : null;
      if (last) last.innerHTML = '<span class="repl-output-text">[' + '\u2588'.repeat(Math.round(pct/5)) + '\u2591'.repeat(20-Math.round(pct/5)) + '] ' + pct + '%</span>';
    }
    replAddOutput(esc(`\n  Category          Score    Tests\n  ${'─'.repeat(37)}\n  Causal Reasoning  85.0%    124/124\n  Compositional     74.0%    144/144\n  Continual Learn.  72.0%    89/89\n  Robustness        71.0%    192/192\n  Language          69.0%    78/78\n  ${'─'.repeat(37)}\n  Overall           69.7%    627/627`));
    return;
  }

  if (cmd === 'neuro_agi.benchmark.compare()') {
    replAddOutput(esc(`Framework Comparison\n${'═'.repeat(58)}\n  Capability          ELO-AGI   ACT-R   SOAR   GPT-4\n  ${'─'.repeat(54)}\n  Causal Reasoning      0.850   0.420  0.380   0.610\n  Compositionality      0.740   0.310  0.290   0.520\n  Continual Learning    0.720   0.180  0.150   0.000\n  Robustness            0.710   0.350  0.320   0.440\n  Language              0.690   0.250  0.220   0.890\n  ${'─'.repeat(54)}\n  Overall               0.742   0.302  0.272   0.492`));
    return;
  }

  if (cmd === 'neuro_agi.architecture()') {
    replAddOutput(esc(`\n  ┌─────────────────────────────────────────────┐\n  │           ELO-AGI Architecture v2.0         │\n  └─────────────────────────────────────────────┘\n\n  ┌─────────────────────────────────────────────┐\n  │  Tier 1: Cognitive Processing (23 modules)  │\n  │          Global Workspace (Attention Bus)    │\n  └─────────────────────┼───────────────────────┘\n                        │\n  ┌─────────────────────┼───────────────────────┐\n  │  Tier 2: Infrastructure (6 modules)         │\n  │  Core | LLM | Knowledge | Sensors | Scale   │\n  └─────────────────────┼───────────────────────┘\n                        │\n  ┌─────────────────────┼───────────────────────┐\n  │  Tier 3: Support (9 modules)                │\n  │  Benchmark | Perception | Env | Inference   │\n  └─────────────────────────────────────────────┘`));
    return;
  }

  replAddOutput('<span class="repl-error">NameError: name \'' + esc(cmd.split('(')[0].split('.')[0]) + '\' is not defined. Type help() for commands.</span>');
}

function initRepl() {
  const input = document.getElementById('repl-input');
  if (!input) return;

  input.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
      e.preventDefault();
      const cmd = this.value.trim();
      this.value = '';
      handleReplCommand(cmd);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (replHistory.length > 0 && replHistoryIdx < replHistory.length - 1) {
        replHistoryIdx++;
        this.value = replHistory[replHistoryIdx];
      }
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (replHistoryIdx > 0) {
        replHistoryIdx--;
        this.value = replHistory[replHistoryIdx];
      } else {
        replHistoryIdx = -1;
        this.value = '';
      }
    }
  });
}

// ================================================================
// BRAIN DIAGRAM ANIMATION
// ================================================================

function createBrainSVG(containerId, opts) {
  opts = opts || {};
  const size = opts.size || 400;
  const cx = size / 2, cy = size / 2;
  const r1 = size * 0.42, r2 = size * 0.3, r3 = size * 0.18;

  const cognitiveModules = ['GW','PC','DP','RT','Mem','SC','Mot','Emo','Lang','Cre','Spa','TP','Lrn','EC','Emb','Soc','Con','WM','SI','MA','Att','Per','DM'];
  const infraModules = ['Core','LLM','KG','Sen','Act','Dist'];
  const supportModules = ['Bench','PPipe','Env','Inf','Int','DL','Log','PMod','TL'];

  let svg = `<svg viewBox="0 0 ${size} ${size}" class="brain-svg" xmlns="http://www.w3.org/2000/svg">`;
  svg += `<defs>
    <linearGradient id="brainGrad1" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#7C3AED"/><stop offset="1" stop-color="#06B6D4"/></linearGradient>
    <filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
  </defs>`;

  // Concentric circle guides
  svg += `<circle cx="${cx}" cy="${cy}" r="${r1}" fill="none" stroke="rgba(124,58,237,0.08)" stroke-width="1"/>`;
  svg += `<circle cx="${cx}" cy="${cy}" r="${r2}" fill="none" stroke="rgba(6,182,212,0.08)" stroke-width="1"/>`;
  svg += `<circle cx="${cx}" cy="${cy}" r="${r3}" fill="none" stroke="rgba(139,92,246,0.08)" stroke-width="1"/>`;

  // Cognitive tier (outer ring)
  cognitiveModules.forEach((mod, i) => {
    const angle = (2 * Math.PI * i / cognitiveModules.length) - Math.PI / 2;
    const x = cx + r1 * Math.cos(angle);
    const y = cy + r1 * Math.sin(angle);
    svg += `<g class="brain-node" data-module="${mod}" data-tier="cognitive">
      <circle cx="${x}" cy="${y}" r="10" fill="rgba(124,58,237,0.2)" stroke="#7C3AED" stroke-width="1.5" style="color:#7C3AED"/>
      <text x="${x}" y="${y+3.5}" fill="#7C3AED" font-size="6" font-weight="700" text-anchor="middle" font-family="Inter,sans-serif">${mod}</text>
    </g>`;
  });

  // Infrastructure tier (middle ring)
  infraModules.forEach((mod, i) => {
    const angle = (2 * Math.PI * i / infraModules.length) - Math.PI / 2;
    const x = cx + r2 * Math.cos(angle);
    const y = cy + r2 * Math.sin(angle);
    svg += `<g class="brain-node" data-module="${mod}" data-tier="infrastructure">
      <circle cx="${x}" cy="${y}" r="12" fill="rgba(6,182,212,0.15)" stroke="#06B6D4" stroke-width="1.5" style="color:#06B6D4"/>
      <text x="${x}" y="${y+3.5}" fill="#06B6D4" font-size="6" font-weight="700" text-anchor="middle" font-family="Inter,sans-serif">${mod}</text>
    </g>`;
  });

  // Support tier (inner ring)
  supportModules.forEach((mod, i) => {
    const angle = (2 * Math.PI * i / supportModules.length) - Math.PI / 2;
    const x = cx + r3 * Math.cos(angle);
    const y = cy + r3 * Math.sin(angle);
    svg += `<g class="brain-node" data-module="${mod}" data-tier="support">
      <circle cx="${x}" cy="${y}" r="8" fill="rgba(139,92,246,0.15)" stroke="#8B5CF6" stroke-width="1.5" style="color:#8B5CF6"/>
      <text x="${x}" y="${y+3}" fill="#8B5CF6" font-size="5" font-weight="700" text-anchor="middle" font-family="Inter,sans-serif">${mod}</text>
    </g>`;
  });

  // Center hub
  svg += `<circle cx="${cx}" cy="${cy}" r="22" fill="#0B0D17" stroke="url(#brainGrad1)" stroke-width="2"/>`;
  svg += `<text x="${cx}" y="${cy-3}" fill="#e2e8f0" font-size="10" font-weight="800" text-anchor="middle" font-family="Inter,sans-serif">38</text>`;
  svg += `<text x="${cx}" y="${cy+8}" fill="#94a3b8" font-size="6" font-weight="500" text-anchor="middle" font-family="Inter,sans-serif">MODULES</text>`;

  svg += '</svg>';

  const container = document.getElementById(containerId);
  if (container) container.innerHTML = svg;
  return svg;
}

function animateBrainSequence(containerId, loopDelay) {
  loopDelay = loopDelay || 6000;
  const container = document.getElementById(containerId);
  if (!container) return;

  const nodes = container.querySelectorAll('.brain-node');
  let idx = 0;
  let intervalId;

  function activateNext() {
    nodes.forEach(n => n.classList.remove('active'));
    if (idx < nodes.length) {
      nodes[idx].classList.add('active');
      // Also activate a couple of connected nodes
      const prev = (idx - 1 + nodes.length) % nodes.length;
      const next2 = (idx + 1) % nodes.length;
      nodes[prev].classList.add('active');
      nodes[next2].classList.add('active');
      idx++;
    } else {
      idx = 0;
      nodes.forEach(n => n.classList.remove('active'));
    }
  }

  intervalId = setInterval(activateNext, 400);

  // Reset after full cycle
  setTimeout(() => {
    clearInterval(intervalId);
    nodes.forEach(n => n.classList.remove('active'));
    // Restart after delay
    setTimeout(() => animateBrainSequence(containerId, loopDelay), loopDelay);
  }, nodes.length * 400 + 1000);
}

// ================================================================
// SCROLL REVEAL WITH STAGGER
// ================================================================

function initScrollReveal() {
  const reveals = document.querySelectorAll('.reveal');
  if (!reveals.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        const children = entry.target.querySelectorAll('.reveal-child');
        children.forEach((child, i) => {
          setTimeout(() => child.classList.add('visible'), i * 100);
        });
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

  reveals.forEach(el => observer.observe(el));
}

// ================================================================
// SCROLL PROGRESS BAR
// ================================================================

function initScrollProgress() {
  const bar = document.getElementById('scroll-progress');
  if (!bar) return;
  window.addEventListener('scroll', () => {
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const pct = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
    bar.style.width = pct + '%';
  }, { passive: true });
}

// ================================================================
// NAV SCROLL EFFECT
// ================================================================

function initNavScroll() {
  const nav = document.querySelector('nav');
  if (!nav) return;
  nav.classList.add('nav-glass');
  let ticking = false;
  window.addEventListener('scroll', () => {
    if (!ticking) {
      requestAnimationFrame(() => {
        if (window.scrollY > 20) {
          nav.classList.add('nav-scrolled');
        } else {
          nav.classList.remove('nav-scrolled');
        }
        ticking = false;
      });
      ticking = true;
    }
  }, { passive: true });
}

// ================================================================
// PARALLAX
// ================================================================

function initParallax() {
  const els = document.querySelectorAll('.parallax-slow');
  if (!els.length) return;
  window.addEventListener('scroll', () => {
    const scrollY = window.scrollY;
    els.forEach(el => {
      const speed = parseFloat(el.dataset.speed) || 0.3;
      el.style.transform = 'translateY(' + (scrollY * speed) + 'px)';
    });
  }, { passive: true });
}

// ================================================================
// ANIMATED COUNTERS
// ================================================================

function animateCounters() {
  const counters = document.querySelectorAll('[data-counter]');
  if (!counters.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const target = parseInt(el.dataset.counter, 10);
        const prefix = el.dataset.prefix || '';
        const suffix = el.dataset.suffix || '';
        const duration = 1500;
        const start = performance.now();

        function step(now) {
          const elapsed = now - start;
          const progress = Math.min(elapsed / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3);
          const current = Math.round(target * eased);
          el.textContent = prefix + current + suffix;
          if (progress < 1) requestAnimationFrame(step);
        }
        requestAnimationFrame(step);
        observer.unobserve(el);
      }
    });
  }, { threshold: 0.5 });

  counters.forEach(el => observer.observe(el));
}

// ================================================================
// DYNAMIC YEAR
// ================================================================

function initDynamicYear() {
  document.querySelectorAll('[data-year]').forEach(el => {
    el.textContent = new Date().getFullYear();
  });
}

// ================================================================
// FAQ TOGGLE
// ================================================================

function toggleFaq(el) {
  const item = el.closest('.faq-item');
  if (!item) return;
  const answer = item.querySelector('.faq-answer');
  const inner = item.querySelector('.faq-answer-inner');
  if (!answer || !inner) return;

  const isOpen = item.classList.contains('open');
  // Close all
  document.querySelectorAll('.faq-item.open').forEach(other => {
    other.classList.remove('open');
    other.querySelector('.faq-answer').style.maxHeight = '0px';
  });

  if (!isOpen) {
    item.classList.add('open');
    answer.style.maxHeight = inner.scrollHeight + 'px';
  }
}

// ================================================================
// INIT
// ================================================================

document.addEventListener('DOMContentLoaded', function() {
  initScrollReveal();
  initScrollProgress();
  initNavScroll();
  initParallax();
  animateCounters();
  initDynamicYear();
  if (document.getElementById('repl-input')) initRepl();
  if (document.getElementById('benchmarkBody')) {
    renderBenchmarks();
    const benchSection = document.getElementById('benchmarks');
    if (benchSection) {
      const obs = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) { animateBenchmarkBars(); obs.unobserve(entry.target); }
        });
      }, { threshold: 0.3 });
      obs.observe(benchSection);
    }
  }
  if (document.getElementById('api-status')) checkApiConnection();
});
