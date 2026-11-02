# LLM Debate Framework

A multi-model debate engine powered by **LangGraph** where Claude, ChatGPT, Gemini, and DeepSeek debate each other across 3 structured stages to produce a unified, consensus-driven best answer — running on their free web interfaces via Chrome DevTools Protocol (CDP) today, with paid API support ready for the future.

```
       [Topic]
          │
          ▼
   ┌──────────────┐
   │   Stage 1    │ ──▶ (Parallel dispatch: Claude, ChatGPT, Gemini, DeepSeek formulate independent views)
   └──────────────┘
          │
          ▼
   ┌──────────────┐
   │   Stage 2    │ ──▶ (Cross-examination & critique: Each model evaluates the others' arguments)
   └──────────────┘
          │
          ▼
   ┌──────────────┐
   │   Stage 3    │ ──▶ (Judge Synthesis: Designated judge produces authoritative consensus report)
   └──────────────┘
```

---

## Prerequisites

- **Python 3.13+** with [uv](https://docs.astral.sh/uv/)
- **Google Chrome** or **Chromium** browser

---

## Quick Start

### 1. Launch Chrome with Remote Debugging
Start your existing Google Chrome profile with remote debugging enabled on port 9222. Because this attaches to your local browser profile, your active logins to Claude, ChatGPT, Gemini, and DeepSeek are used directly without bot challenges:

```bash
./scripts/run_chrome.sh
```

Check CDP connection and open tabs anytime:
```bash
uv run python -m debate check-cdp
```

### 2. Run via Web Dashboard
Start the local FastAPI server:
```bash
uv run uvicorn debate.server.app:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser:
* Enter any debate question or problem statement.
* Toggle which models participate (Claude, ChatGPT, Gemini, DeepSeek).
* Select the Stage 3 Judge model.
* Watch real-time streaming updates across each stage with side-by-side color-coded model cards.
* Copy or export the synthesized final consensus report.

### 3. Run via CLI
Run debates directly from your terminal:
```bash
uv run python -m debate run "Should startups choose PostgreSQL or MongoDB for their primary datastore?" --models claude,chatgpt,gemini,deepseek --judge claude --export
```

Transcripts are automatically formatted and saved to `debates/YYYY-MM-DD-<slug>.md`.

---

## Running the Test Suite

Run all unit, graph transition, API, and simulation tests:
```bash
uv run pytest
```