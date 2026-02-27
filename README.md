<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GenAI Engineering Roadmap</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@300;400;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #07080f;
    --surface: #0e1020;
    --border: #1e2240;
    --accent1: #7c3aff;
    --accent2: #00d4ff;
    --accent3: #ff6b35;
    --accent4: #39ff14;
    --text: #e8eaf6;
    --muted: #6b7280;
    --glow1: rgba(124,58,255,0.25);
    --glow2: rgba(0,212,255,0.2);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated grid background */
  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(124,58,255,0.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(124,58,255,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .container {
    position: relative;
    z-index: 1;
    max-width: 1100px;
    margin: 0 auto;
    padding: 60px 24px 100px;
  }

  /* Header */
  header {
    text-align: center;
    margin-bottom: 72px;
    animation: fadeDown 0.8s ease both;
  }

  .tag {
    display: inline-block;
    font-size: 11px;
    letter-spacing: 0.2em;
    color: var(--accent2);
    border: 1px solid rgba(0,212,255,0.3);
    padding: 4px 14px;
    margin-bottom: 20px;
    text-transform: uppercase;
  }

  h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(36px, 6vw, 72px);
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #fff 30%, var(--accent1) 60%, var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 16px;
  }

  header p {
    color: var(--muted);
    font-size: 13px;
    letter-spacing: 0.05em;
  }

  /* Phase container */
  .phase {
    position: relative;
    margin-bottom: 48px;
    animation: fadeUp 0.6s ease both;
  }

  .phase:nth-child(1) { animation-delay: 0.1s; }
  .phase:nth-child(2) { animation-delay: 0.2s; }
  .phase:nth-child(3) { animation-delay: 0.3s; }
  .phase:nth-child(4) { animation-delay: 0.4s; }
  .phase:nth-child(5) { animation-delay: 0.5s; }

  /* Vertical connector */
  .phase:not(:last-child)::after {
    content: '';
    position: absolute;
    left: 32px;
    bottom: -48px;
    width: 2px;
    height: 48px;
    background: linear-gradient(to bottom, var(--accent1), transparent);
  }

  .phase-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 16px;
  }

  .phase-num {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 800;
    flex-shrink: 0;
    position: relative;
  }

  .phase-num::before {
    content: '';
    position: absolute;
    inset: -2px;
    border-radius: 50%;
    padding: 2px;
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
  }

  .p1 .phase-num { background: rgba(124,58,255,0.15); color: var(--accent1); border: 2px solid var(--accent1); box-shadow: 0 0 20px var(--glow1); }
  .p2 .phase-num { background: rgba(0,212,255,0.1); color: var(--accent2); border: 2px solid var(--accent2); box-shadow: 0 0 20px var(--glow2); }
  .p3 .phase-num { background: rgba(255,107,53,0.1); color: var(--accent3); border: 2px solid var(--accent3); box-shadow: 0 0 20px rgba(255,107,53,0.2); }
  .p4 .phase-num { background: rgba(57,255,20,0.08); color: var(--accent4); border: 2px solid var(--accent4); box-shadow: 0 0 20px rgba(57,255,20,0.15); }
  .p5 .phase-num { background: rgba(255,215,0,0.08); color: #ffd700; border: 2px solid #ffd700; box-shadow: 0 0 20px rgba(255,215,0,0.15); }

  .phase-title {
    font-family: 'Syne', sans-serif;
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.01em;
  }

  .phase-subtitle {
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 2px;
  }

  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px;
    padding-left: 80px;
  }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 16px;
    position: relative;
    transition: all 0.25s ease;
    cursor: default;
  }

  .card:hover {
    border-color: var(--accent1);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(124,58,255,0.15);
  }

  .p2 .card:hover { border-color: var(--accent2); box-shadow: 0 8px 32px rgba(0,212,255,0.15); }
  .p3 .card:hover { border-color: var(--accent3); box-shadow: 0 8px 32px rgba(255,107,53,0.15); }
  .p4 .card:hover { border-color: var(--accent4); box-shadow: 0 8px 32px rgba(57,255,20,0.1); }
  .p5 .card:hover { border-color: #ffd700; box-shadow: 0 8px 32px rgba(255,215,0,0.12); }

  .card-icon {
    font-size: 20px;
    margin-bottom: 8px;
    display: block;
  }

  .card-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 6px;
    color: #fff;
  }

  .card-desc {
    font-size: 11px;
    color: var(--muted);
    line-height: 1.6;
  }

  .badge {
    display: inline-block;
    font-size: 9px;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    margin-top: 10px;
    text-transform: uppercase;
  }

  .badge-core { background: rgba(124,58,255,0.2); color: var(--accent1); }
  .badge-tool { background: rgba(0,212,255,0.1); color: var(--accent2); }
  .badge-arch { background: rgba(255,107,53,0.15); color: var(--accent3); }
  .badge-prod { background: rgba(57,255,20,0.1); color: var(--accent4); }
  .badge-adv  { background: rgba(255,215,0,0.1); color: #ffd700; }

  /* Timeline bar */
  .timeline {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 24px 0 32px 80px;
    font-size: 10px;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }

  .tl-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent1);
    box-shadow: 0 0 8px var(--accent1);
  }
  .tl-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, var(--accent1), var(--accent2), var(--accent3), var(--accent4), #ffd700);
  }

  @keyframes fadeDown {
    from { opacity: 0; transform: translateY(-24px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* Footer note */
  .footer-note {
    text-align: center;
    margin-top: 60px;
    font-size: 11px;
    color: var(--muted);
    letter-spacing: 0.05em;
    border-top: 1px solid var(--border);
    padding-top: 32px;
  }

  @media (max-width: 600px) {
    .cards { padding-left: 0; grid-template-columns: 1fr 1fr; }
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="tag">Engineering Roadmap · 2024–2026</div>
    <h1>Generative AI</h1>
    <p>A structured path from foundations to frontier systems</p>
  </header>

  <div class="timeline">
    <div class="tl-dot"></div>
    <span>Foundations</span>
    <div class="tl-line"></div>
    <span>Build</span>
    <div class="tl-line" style="flex:0.5"></div>
    <span>Scale</span>
    <div class="tl-line" style="flex:0.5"></div>
    <span>Deploy</span>
    <div class="tl-line" style="flex:0.5"></div>
    <span>Frontier</span>
    <div class="tl-dot" style="background:#ffd700;box-shadow:0 0 8px #ffd700"></div>
  </div>

  <!-- Phase 1 -->
  <div class="phase p1">
    <div class="phase-header">
      <div class="phase-num">01</div>
      <div>
        <div class="phase-title">Core Foundations</div>
        <div class="phase-subtitle">What every GenAI engineer must know</div>
      </div>
    </div>
    <div class="cards">
      <div class="card">
        <span class="card-icon">🧮</span>
        <div class="card-title">Math & Statistics</div>
        <div class="card-desc">Linear algebra, probability, calculus, information theory</div>
        <span class="badge badge-core">Core</span>
      </div>
      <div class="card">
        <span class="card-icon">🧠</span>
        <div class="card-title">Deep Learning</div>
        <div class="card-desc">Backprop, optimizers, loss functions, regularization</div>
        <span class="badge badge-core">Core</span>
      </div>
      <div class="card">
        <span class="card-icon">🔤</span>
        <div class="card-title">Tokenization & Embeddings</div>
        <div class="card-desc">BPE, SentencePiece, word2vec, semantic spaces</div>
        <span class="badge badge-core">Core</span>
      </div>
      <div class="card">
        <span class="card-icon">⚡</span>
        <div class="card-title">Transformers</div>
        <div class="card-desc">Attention, positional encoding, encoder/decoder, KV-cache</div>
        <span class="badge badge-core">Core</span>
      </div>
    </div>
  </div>

  <!-- Phase 2 -->
  <div class="phase p2">
    <div class="phase-header">
      <div class="phase-num">02</div>
      <div>
        <div class="phase-title">Tooling & Frameworks</div>
        <div class="phase-subtitle">The modern GenAI stack</div>
      </div>
    </div>
    <div class="cards">
      <div class="card">
        <span class="card-icon">🔥</span>
        <div class="card-title">PyTorch / JAX</div>
        <div class="card-desc">Tensor ops, autograd, distributed training primitives</div>
        <span class="badge badge-tool">Tool</span>
      </div>
      <div class="card">
        <span class="card-icon">🤗</span>
        <div class="card-title">Hugging Face</div>
        <div class="card-desc">Transformers, PEFT, Datasets, Inference API</div>
        <span class="badge badge-tool">Tool</span>
      </div>
      <div class="card">
        <span class="card-icon">🔗</span>
        <div class="card-title">LangChain / LlamaIndex</div>
        <div class="card-desc">Chains, agents, document loaders, RAG pipelines</div>
        <span class="badge badge-tool">Tool</span>
      </div>
      <div class="card">
        <span class="card-icon">🗄️</span>
        <div class="card-title">Vector DBs</div>
        <div class="card-desc">Pinecone, Weaviate, Qdrant, pgvector, FAISS</div>
        <span class="badge badge-tool">Tool</span>
      </div>
      <div class="card">
        <span class="card-icon">📊</span>
        <div class="card-title">Experiment Tracking</div>
        <div class="card-desc">W&B, MLflow, LangSmith, Comet</div>
        <span class="badge badge-tool">Tool</span>
      </div>
    </div>
  </div>

  <!-- Phase 3 -->
  <div class="phase p3">
    <div class="phase-header">
      <div class="phase-num">03</div>
      <div>
        <div class="phase-title">Architecture & Patterns</div>
        <div class="phase-subtitle">Building production GenAI systems</div>
      </div>
    </div>
    <div class="cards">
      <div class="card">
        <span class="card-icon">📚</span>
        <div class="card-title">RAG Systems</div>
        <div class="card-desc">Retrieval-Augmented Generation, chunking strategies, hybrid search</div>
        <span class="badge badge-arch">Architecture</span>
      </div>
      <div class="card">
        <span class="card-icon">🎯</span>
        <div class="card-title">Fine-Tuning</div>
        <div class="card-desc">LoRA, QLoRA, instruction tuning, RLHF, DPO</div>
        <span class="badge badge-arch">Architecture</span>
      </div>
      <div class="card">
        <span class="card-icon">🤖</span>
        <div class="card-title">Agentic Systems</div>
        <div class="card-desc">Tool use, ReAct, planning loops, multi-agent orchestration</div>
        <span class="badge badge-arch">Architecture</span>
      </div>
      <div class="card">
        <span class="card-icon">🖼️</span>
        <div class="card-title">Multimodal</div>
        <div class="card-desc">Vision-language, image gen, audio-LLM, cross-modal fusion</div>
        <span class="badge badge-arch">Architecture</span>
      </div>
      <div class="card">
        <span class="card-icon">💬</span>
        <div class="card-title">Prompt Engineering</div>
        <div class="card-desc">CoT, few-shot, system prompts, structured output, DSPy</div>
        <span class="badge badge-arch">Architecture</span>
      </div>
    </div>
  </div>

  <!-- Phase 4 -->
  <div class="phase p4">
    <div class="phase-header">
      <div class="phase-num">04</div>
      <div>
        <div class="phase-title">Production & Scale</div>
        <div class="phase-subtitle">Reliability, latency, cost at scale</div>
      </div>
    </div>
    <div class="cards">
      <div class="card">
        <span class="card-icon">🚀</span>
        <div class="card-title">Inference Optimization</div>
        <div class="card-desc">Quantization, vLLM, TensorRT-LLM, speculative decoding</div>
        <span class="badge badge-prod">Production</span>
      </div>
      <div class="card">
        <span class="card-icon">📈</span>
        <div class="card-title">LLMOps</div>
        <div class="card-desc">Prompt versioning, A/B testing, drift detection, evals</div>
        <span class="badge badge-prod">Production</span>
      </div>
      <div class="card">
        <span class="card-icon">🛡️</span>
        <div class="card-title">Safety & Guardrails</div>
        <div class="card-desc">Jailbreak defense, PII redaction, output filtering, red-teaming</div>
        <span class="badge badge-prod">Production</span>
      </div>
      <div class="card">
        <span class="card-icon">💰</span>
        <div class="card-title">Cost Engineering</div>
        <div class="card-desc">Token budgets, caching, batching, model routing, fallbacks</div>
        <span class="badge badge-prod">Production</span>
      </div>
      <div class="card">
        <span class="card-icon">🔍</span>
        <div class="card-title">Evaluation</div>
        <div class="card-desc">RAGAS, MT-Bench, custom evals, human-in-the-loop scoring</div>
        <span class="badge badge-prod">Production</span>
      </div>
    </div>
  </div>

  <!-- Phase 5 -->
  <div class="phase p5">
    <div class="phase-header">
      <div class="phase-num">05</div>
      <div>
        <div class="phase-title">Frontier Systems</div>
        <div class="phase-subtitle">Cutting edge — where research meets engineering</div>
      </div>
    </div>
    <div class="cards">
      <div class="card">
        <span class="card-icon">🏋️</span>
        <div class="card-title">Pre-training at Scale</div>
        <div class="card-desc">Data pipelines, distributed training, MoE, FSDP/Megatron</div>
        <span class="badge badge-adv">Advanced</span>
      </div>
      <div class="card">
        <span class="card-icon">🌐</span>
        <div class="card-title">Reasoning Models</div>
        <div class="card-desc">Chain-of-thought distillation, o1-style test-time compute, MCTS</div>
        <span class="badge badge-adv">Advanced</span>
      </div>
      <div class="card">
        <span class="card-icon">🔬</span>
        <div class="card-title">Interpretability</div>
        <div class="card-desc">Mechanistic interp, activation steering, sparse autoencoders</div>
        <span class="badge badge-adv">Advanced</span>
      </div>
      <div class="card">
        <span class="card-icon">🌍</span>
        <div class="card-title">Long-context & Memory</div>
        <div class="card-desc">1M+ tokens, RoPE scaling, external memory, retrieval augmentation</div>
        <span class="badge badge-adv">Advanced</span>
      </div>
    </div>
  </div>

  <div class="footer-note">
    GenAI Roadmap · Built for engineers who ship · Updated 2025 &nbsp;·&nbsp; Master one phase before moving to the next
  </div>

</div>
</body>
</html>
