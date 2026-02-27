# 🌌 Generative AI Roadmap
**Engineering Roadmap · 2024–2026** *A structured path from foundations to frontier systems*

---

### 🟢 Foundations ━━━ 🛠️ Build ━━━ ⚖️ Scale ━━━ 🚀 Deploy ━━━ 🌌 Frontier

---

## 01. Core Foundations
> **What every GenAI engineer must know**

| Topic | Description | Label |
| :--- | :--- | :--- |
| 🧮 **Math & Statistics** | Linear algebra, probability, calculus, information theory | `CORE` |
| 🧠 **Deep Learning** | Backprop, optimizers, loss functions, regularization | `CORE` |
| 🔤 **Tokenization & Embeddings** | BPE, SentencePiece, word2vec, semantic spaces | `CORE` |
| ⚡ **Transformers** | Attention, positional encoding, encoder/decoder, KV-cache | `CORE` |

---

## 02. Tooling & Frameworks
> **The modern GenAI stack**

| Tool | Description | Label |
| :--- | :--- | :--- |
| 🔥 **PyTorch / JAX** | Tensor ops, autograd, distributed training primitives | `TOOL` |
| 🤗 **Hugging Face** | Transformers, PEFT, Datasets, Inference API | `TOOL` |
| 🔗 **LangChain / LlamaIndex** | Chains, agents, document loaders, RAG pipelines | `TOOL` |
| 🗄️ **Vector DBs** | Pinecone, Weaviate, Qdrant, pgvector, FAISS | `TOOL` |
| 📊 **Experiment Tracking** | W&B, MLflow, LangSmith, Comet | `TOOL` |

---

## 03. Architecture & Patterns
> **Building production GenAI systems**

| Architecture | Description | Label |
| :--- | :--- | :--- |
| 📚 **RAG Systems** | Retrieval-Augmented Generation, chunking strategies, hybrid search | `ARCH` |
| 🎯 **Fine-Tuning** | LoRA, QLoRA, instruction tuning, RLHF, DPO | `ARCH` |
| 🤖 **Agentic Systems** | Tool use, ReAct, planning loops, multi-agent orchestration | `ARCH` |
| 🖼️ **Multimodal** | Vision-language, image gen, audio-LLM, cross-modal fusion | `ARCH` |
| 💬 **Prompt Engineering** | CoT, few-shot, system prompts, structured output, DSPy | `ARCH` |

---

## 04. Production & Scale
> **Reliability, latency, cost at scale**

| Production | Description | Label |
| :--- | :--- | :--- |
| 🚀 **Inference Optimization** | Quantization, vLLM, TensorRT-LLM, speculative decoding | `PROD` |
| 📈 **LLMOps** | Prompt versioning, A/B testing, drift detection, evals | `PROD` |
| 🛡️ **Safety & Guardrails** | Jailbreak defense, PII redaction, output filtering, red-teaming | `PROD` |
| 💰 **Cost Engineering** | Token budgets, caching, batching, model routing, fallbacks | `PROD` |
| 🔍 **Evaluation** | RAGAS, MT-Bench, custom evals, human-in-the-loop scoring | `PROD` |

---

## 05. Frontier Systems
> **Cutting edge — where research meets engineering**

| Advanced | Description | Label |
| :--- | :--- | :--- |
| 🏋️ **Pre-training at Scale** | Data pipelines, distributed training, MoE, FSDP/Megatron | `ADV` |
| 🌐 **Reasoning Models** | Chain-of-thought distillation, o1-style test-time compute, MCTS | `ADV` |
| 🔬 **Interpretability** | Mechanistic interp, activation steering, sparse autoencoders | `ADV` |
| 🌍 **Long-context & Memory** | 1M+ tokens, RoPE scaling, external memory, retrieval augmentation | `ADV` |



# 🌌 Build *Five* GenAI Projects
**GenAI Project Series ·*Curated projects from zero-to-hero · Each one harder than the last · Ship all five to be production-ready*

---

### 🟡 Starter ━━━ 🟢 Beginner ━━━ 🔵 Intermediate ━━━ 🟠 Advanced ━━━ 🟣 Expert

---

## 01. AI-Powered FAQ Chatbot
> **"Hello, World" of GenAI — but actually useful**

| Detail | Context |
| :--- | :--- |
| **Level** | `STARTER` |
| **Effort** | ~1–2 days |
| **Complexity** | ▓░░░░░░░░░ (18%) |

### 📝 Description
Build a simple conversational chatbot that answers questions from a predefined knowledge base (FAQ document or JSON). The user types a question, the LLM reads the context and responds naturally. No databases, no vector search — just prompt engineering and an API call.

* **What you'll learn:** OpenAI/Anthropic API, System prompts, Context injection, Prompt templates, Streamlit UI.
* **Tech Stack:** Python, OpenAI SDK, Streamlit, JSON/TXT.

> **📦 WHAT YOU SHIP:** A working Streamlit web app where users can ask questions and get instant, context-aware answers — deployed locally or on Streamlit Cloud.

---

## 02. Document Q&A with RAG
> **Upload any PDF — interrogate it like a senior analyst**

| Detail | Context |
| :--- | :--- |
| **Level** | `BEGINNER` |
| **Effort** | ~1 week |
| **Complexity** | ▓▓▓░░░░░░░ (36%) |



### 📝 Description
Upgrade the chatbot with Retrieval-Augmented Generation. Users upload a PDF, the system chunks and embeds it into a vector store, and retrieves the most relevant chunks at query time. Adds meaningful engineering: chunking strategy, embedding models, similarity search, and citation sourcing.

* **What you'll learn:** Chunking strategies, Embeddings, Vector similarity search, RAG pipeline, Source citations.
* **Tech Stack:** LangChain, FAISS / Chroma, OpenAI Embeddings, PyMuPDF, FastAPI.

> **📦 WHAT YOU SHIP:** A full-stack app where users drag-and-drop any PDF, ask questions, and get answers with exact page citations.

---

## 03. AI Research Agent with Tool Use
> **Give the LLM hands — let it search, read, and reason autonomously**

| Detail | Context |
| :--- | :--- |
| **Level** | `INTERMEDIATE` |
| **Effort** | ~2–3 weeks |
| **Complexity** | ▓▓▓▓▓░░░░░ (55%) |

### 📝 Description
Build an autonomous research agent that can browse the web, summarize articles, extract data, and produce structured reports. Implement the **ReAct** loop (Reason → Act → Observe) and handle multi-step tool chaining, retries, and final synthesis.

* **What you'll learn:** Tool/function calling, ReAct pattern, Agent loops, Multi-step planning, Error recovery, Structured output.
* **Tech Stack:** LangChain Agents, Tavily / SerpAPI, Pydantic, Redis (memory), Next.js frontend.

> **📦 WHAT YOU SHIP:** An agent you can prompt with "Research the top 5 AI startups in 2025" and it returns a polished structured report autonomously.

---

## 04. Domain-Specific Fine-Tuned Model + API
> **Stop prompting. Start training. Own the model.**

| Detail | Context |
| :--- | :--- |
| **Level** | `ADVANCED` |
| **Effort** | ~4–6 weeks |
| **Complexity** | ▓▓▓▓▓▓▓░░░ (74%) |

### 📝 Description
Fine-tune an open-source model (Mistral 7B or Llama 3) on a domain-specific dataset (medical, legal, or code). Use LoRA/QLoRA for efficient training, evaluate with quantitative benchmarks, and wrap it in a production FastAPI service with auth and rate limiting.

* **What you'll learn:** LoRA / QLoRA, Instruction tuning, Dataset curation, PEFT, Model evaluation, HF Hub deployment, API productionization.
* **Tech Stack:** Hugging Face PEFT, bitsandbytes, TRL / SFTTrainer, W&B, FastAPI, vLLM, Docker.

> **📦 WHAT YOU SHIP:** A domain-expert LLM outperforming GPT-3.5 on your chosen task, served via a production API with auth + rate limiting.

---

## 05. Multi-Agent Autonomous Coding Assistant
> **Build the system that builds systems. Ship a junior dev in a box.**

| Detail | Context |
| :--- | :--- |
| **Level** | `EXPERT` |
| **Effort** | ~2–3 months |
| **Complexity** | ▓▓▓▓▓▓▓▓▓▓ (95%) |



### 📝 Description
Design a production multi-agent system where specialized agents (Planner, Coder, Reviewer, Tester) collaborate to solve complex software engineering tasks. Includes human-in-the-loop approval, persistent memory, and a full observability stack.

* **What you'll learn:** Multi-agent orchestration, Agent communication protocols, Persistent memory, Human-in-the-loop, Streaming SSE, LLM observability, Cost optimization, Sandboxed code execution.
* **Tech Stack:** LangGraph, AutoGen / CrewAI, LangSmith, PostgreSQL + pgvector, WebSockets, E2B (sandboxes), Kubernetes, OpenTelemetry.

> **📦 WHAT YOU SHIP:** A working AI coding assistant where describing a feature in English leads to a swarm of agents planning, coding, and submitting a pull request.

<h3 align="left">Connect with me:</h3>
<p align="left">
<a href="https://linkedin.com/in/harshkumar-8h/" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="harshkumar-8h/" height="30" width="40" /></a>
</p>


