# LLM Debate Framework: Architecture & Design Specification

**Date**: 2026-11-02  
**Status**: Approved  
**Target Repository**: `langgraph-debate`

---

## 1. Executive Summary & Problem Statement

Users often receive divergent, inconsistent, or hallucinated answers when asking complex technical and analytical questions across different LLMs (Claude, ChatGPT, Gemini, DeepSeek). Existing workflows require manually cross-referencing tabs or paying for multiple commercial API keys.

This framework orchestrates a multi-model debate powered by **LangGraph** where Claude, ChatGPT, Gemini, and DeepSeek debate each other across 3 structured stages to produce a synthesized, consensus-driven best answer. The framework is designed to run **entirely on free logged-in web sessions** using Chrome DevTools Protocol (CDP) automation via Playwright today, with a polymorphic driver layer enabling seamless upgrade to official paid APIs later.

---

## 2. System Architecture

The system consists of four isolated layers:

```
┌────────────────────────────────────────────────────────┐
│                   User Interfaces                      │
│   ┌──────────────────────────┐  ┌──────────────────┐   │
│   │   Web Dashboard (UI)     │  │  Rich CLI Runner │   │
│   └────────────┬─────────────┘  └────────┬─────────┘   │
└────────────────┼─────────────────────────┼─────────────┘
                 │ HTTP / SSE              │ Direct Call
┌────────────────▼─────────────────────────▼─────────────┐
│             FastAPI Application Layer                  │
│       /api/debate/start, /api/debate/{id}/events       │
└────────────────────────┬───────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────┐
│              LangGraph Debate Engine                   │
│   Stage 1: Init  ──▶  Stage 2: Critique  ──▶  Stage 3: Synthesis
│             (Typed State & Resilient Graph)            │
└────────────────────────┬───────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────┐
│            Polymorphic Model Drivers                   │
│   ┌────────────────────────────────────────────────┐   │
│   │             BaseModelDriver (ABC)              │   │
│   └───────────────┬────────────────┬───────────────┘   │
│                   │                │                   │
│                   ▼                ▼                   │
│         BrowserCDPDriver        APIDriver              │
│       (Local Chrome CDP)     (Future Paid SDKs)        │
└───────────────────┬────────────────────────────────────┘
                    │
                    ▼
         Local Chrome (Port 9222)
  [Claude.ai] [ChatGPT.com] [Gemini] [DeepSeek]
```

---

## 3. LangGraph State Machine Specification

The debate flow is modeled as a StateGraph with strict typing and runtime error tolerance.

### 3.1 State Schema (`DebateState`)

```python
from typing import TypedDict, List, Dict, Optional

class DebateState(TypedDict):
    topic: str
    active_models: List[str]            # ["claude", "chatgpt", "gemini", "deepseek"]
    judge_model: str                    # e.g. "claude"
    current_stage: str                  # "init" | "critique" | "synthesis" | "completed"
    stage_1_answers: Dict[str, str]     # model_id -> answer text
    stage_2_critiques: Dict[str, str]   # model_id -> critique & rebuttal text
    stage_3_synthesis: Optional[str]    # synthesized final report
    status_events: List[Dict]           # list of timestamped lifecycle events
    errors: Dict[str, str]              # model_id -> error message
```

### 3.2 Graph Nodes & Transitions

1. **`stage_1_init_node`**:
   - Concurrently dispatches the raw topic prompt to all models in `active_models`.
   - Collects answers in `stage_1_answers`.
   - Drops models that timed out or failed into `errors`.
   - **Gate**: Requires at least 2 successful models. If fewer than 2 models respond, aborts with an actionable error.
2. **`stage_2_critique_node`**:
   - For each active model, formats the other models' Stage 1 answers and dispatches the cross-critique prompt concurrently.
   - Collects responses into `stage_2_critiques`.
   - Drops any failing model from `active_models`.
3. **`stage_3_synthesis_node`**:
   - Evaluates whether the configured `judge_model` is still healthy in `active_models`. If not, promotes the first healthy model in `active_models`.
   - Packages the topic, all Stage 1 answers, and all Stage 2 critiques into the synthesis prompt.
   - Dispatches to the judge model and stores the final result in `stage_3_synthesis`.
   - Emits the completion event.

---

## 4. Model Driver Abstraction & Web Session Automation

### 4.1 Driver Abstraction (`BaseModelDriver`)

```python
from abc import ABC, abstractmethod

class BaseModelDriver(ABC):
    model_name: str

    @abstractmethod
    async def send_prompt(self, prompt: str, new_chat: bool = True) -> str:
        """Sends prompt, waits for generation to finish, returns text."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Checks if session is connected and ready to accept prompts."""
        pass
```

### 4.2 Browser CDP Connection Architecture (`BrowserCDPDriver`)

* **Host Connection**: Playwright attaches to an existing running Google Chrome instance via `connect_over_cdp("http://localhost:9222")`.
* **Chrome Launcher Helper (`run_chrome.sh`)**:
  ```bash
  #!/usr/bin/env bash
  google-chrome --remote-debugging-port=9222 --user-data-dir="${HOME}/.config/google-chrome" &
  ```
  This launches Chrome using the user's primary profile so all active logins, cookies, and Cloudflare clearances are already present.
* **Tab Matching & Lifecycle**:
  - The driver queries the open browser contexts and pages for matching URLs:
    - Claude: `*claude.ai*`
    - ChatGPT: `*chatgpt.com*`
    - Gemini: `*gemini.google.com*`
    - DeepSeek: `*chat.deepseek.com*`
  - If a tab is found, the driver brings it to front and clicks "New Chat" (or opens the base domain). If no tab is open, the driver opens a new tab navigating to the respective service.

### 4.3 Web Service Adapters (DOM Selectors & Completion)

| Service | Target URL | Prompt Input Selector | Submit Method | Generation Completion Indicator | Content Extraction Selector |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Claude** | `https://claude.ai/new` | `div[contenteditable="true"]` | Keyboard `Enter` | `button[aria-label="Stop Response"]` disappears | Latest `div[data-is-streaming="false"]` or `div.font-claude-message` |
| **ChatGPT** | `https://chatgpt.com/` | `#prompt-textarea` | Click `button[data-testid="send-button"]` | Stop streaming button disappears; send button re-appears | Latest `[data-message-author-role="assistant"]` |
| **Gemini** | `https://gemini.google.com/app` | `div[contenteditable="true"]` | Click `.send-button` or `Enter` | `.sparkle-loading` / typing animation ends | Latest `model-response` container text |
| **DeepSeek** | `https://chat.deepseek.com/` | `textarea` | Click send icon or `Enter` | Stop generating button toggles off | Latest assistant message element |

---

## 5. 3-Stage Debate Protocol & Prompting

### 5.1 Stage 1: Independent Exploration Prompt

```text
You are participating in an expert panel debate on the following topic:
"{topic}"

Please provide your comprehensive, definitive answer.
- State your core argument and conclusion clearly.
- Provide rigorous supporting reasoning, evidence, or technical justification.
- Explicitly state any critical assumptions, edge cases, or trade-offs in your position.
```

### 5.2 Stage 2: Cross-Critique & Rebuttal Prompt

```text
You are an expert panelist evaluating other AI models' answers to the topic:
"{topic}"

Here are the initial answers submitted by the panel:
---
{formatted_other_answers}
---

Perform a rigorous cross-examination:
1. Critical Analysis: Identify where other models made logical fallacies, overlooked edge cases, or made unsupported claims.
2. Strengths: Highlight which points or insights from other models are genuinely superior or complementary.
3. Rebuttal & Refinement: Defend or adjust your original thesis in response to the panel, providing your refined stance.
```

### 5.3 Stage 3: Judge Synthesis Prompt

```text
You are the designated Judge and Synthesizer for this expert panel debate.

TOPIC:
"{topic}"

INITIAL PROPOSALS:
{formatted_stage_1_answers}

CROSS-EXAMINATIONS & CRITIQUES:
{formatted_stage_2_critiques}

Synthesize the debate into a single, authoritative final report formatted as follows:

# Final Debate Synthesis: {topic}

## 1. Executive Summary & Core Consensus
Directly answer the topic. What is the unified, evidence-based conclusion?

## 2. Comprehensive Synthesized Solution
Present the comprehensive solution combining the best arguments from all panelists while filtering out refuted claims.

## 3. Key Trade-offs & Nuances
Detail the legitimate points of divergence or contextual trade-offs identified during the debate.

## 4. Panel Scorecard
- Claude: Core contribution and any blindspots identified.
- ChatGPT: Core contribution and any blindspots identified.
- Gemini: Core contribution and any blindspots identified.
- DeepSeek: Core contribution and any blindspots identified.
```

---

## 6. Web Dashboard, Server-Sent Events & CLI

### 6.1 Backend API Specification (`FastAPI`)

* `POST /api/debate/start`
  - Body:
    ```json
    {
      "topic": "Should Python 3.13 free-threaded GIL be adopted immediately for web apps?",
      "models": ["claude", "chatgpt", "gemini", "deepseek"],
      "judge": "claude"
    }
    ```
  - Response: `{"debate_id": "uuid-v4", "status": "started"}`
* `GET /api/debate/{debate_id}/events`
  - Protocol: `text/event-stream` (SSE)
  - Event payload format:
    ```json
    {
      "event": "stage_change" | "model_start" | "model_done" | "model_error" | "synthesis_done",
      "stage": "init" | "critique" | "synthesis" | "completed",
      "model": "claude",
      "content": "...",
      "timestamp": "2026-11-02T16:40:00Z"
    }
    ```
* `GET /api/health`
  - Returns CDP connection status (`connected: true/false`), listening port, and detected browser tabs for each supported model.

### 6.2 Web Dashboard UI

* **Technology**: Vanilla HTML5, modern CSS with custom theme variables, and Vanilla JavaScript with ES modules (lightweight, zero build-step overhead, fast local hosting).
* **Components**:
  1. **Control Header**: Debate topic input, model selection chips, judge selector, and live Chrome CDP status badge.
  2. **Stage Progression Tracker**: Animated 3-step timeline showing current state, active processing spinners, and completion checkmarks.
  3. **Panel Columns (4-Up Grid)**: Side-by-side cards for Claude, ChatGPT, Gemini, and DeepSeek with custom brand color accents, real-time typing indicators, and markdown rendering.
  4. **Synthesized Hero Section**: Prominently displayed card with consensus view, scorecard tabs, and a "Copy Markdown / Save Report" button.

### 6.3 Command Line Interface (CLI)

```bash
python -m debate run "Is SQLite suitable for production?" --models claude,chatgpt,gemini,deepseek --judge claude --export
```
* Uses Python `rich` for multi-column live progress spinners and formatted terminal output.
* Automatically exports debate transcripts to `debates/YYYY-MM-DD-<slug>.md`.

---

## 7. Directory Structure

```
langgraph-debate/
├── docs/
│   └── superpowers/specs/
│       └── 2026-11-02-llm-debate-framework-design.md
├── debate/
│   ├── __init__.py
│   ├── __main__.py             # CLI entrypoint
│   ├── config.py               # Settings & environment variables
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py            # DebateState TypedDict
│   │   ├── nodes.py            # Stage 1, 2, 3 node logic
│   │   ├── prompts.py          # Prompt templates
│   │   └── graph.py            # LangGraph StateGraph builder
│   ├── drivers/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseModelDriver ABC
│   │   ├── browser.py          # BrowserCDPDriver (Playwright CDP)
│   │   ├── adapters/
│   │   │   ├── __init__.py
│   │   │   ├── claude.py       # Claude DOM adapter
│   │   │   ├── chatgpt.py      # ChatGPT DOM adapter
│   │   │   ├── gemini.py       # Gemini DOM adapter
│   │   │   └── deepseek.py     # DeepSeek DOM adapter
│   │   └── api.py              # APIDriver (future drop-in)
│   ├── server/
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI application
│   │   ├── sse.py              # Event streaming helper
│   │   └── static/             # Web UI Dashboard (HTML, CSS, JS)
│   │       ├── index.html
│   │       ├── styles.css
│   │       └── app.js
│   └── utils/
│       ├── cdp_check.py        # Chrome debugging health checker
│       └── exporter.py         # Markdown transcript generator
├── scripts/
│   └── run_chrome.sh           # Helper script to launch Chrome with CDP
├── tests/
│   ├── test_state.py           # Unit tests for state & graph transitions
│   ├── test_prompts.py         # Unit tests for prompt formatting
│   └── test_mock_driver.py     # End-to-end debate simulation with mock drivers
├── pyproject.toml              # Dependencies & package metadata
└── README.md
```

---

## 8. Verification & Testing Strategy

1. **Unit Testing (`pytest`)**:
   - Verify `DebateState` transitions with a `MockModelDriver` simulating Stage 1, 2, and 3 responses without needing browser access.
   - Test graceful degradation: simulate 1 model failure in Stage 1 and verify the graph successfully completes with the remaining 3 models.
   - Test judge fallback: simulate the judge model failing in Stage 1 and verify automatic reassignment.
2. **Browser Driver Health Testing**:
   - `python -m debate check-cdp`: Verifies connection to `localhost:9222` and reports tab detection status.
3. **End-to-End Test**:
   - Execute a complete 3-stage debate across all 4 models and assert that the Stage 3 synthesis report is fully formed and saved.
