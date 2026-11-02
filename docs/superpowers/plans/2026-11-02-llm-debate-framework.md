# LLM Debate Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-model debate framework in Python using LangGraph where Claude, ChatGPT, Gemini, and DeepSeek debate each other across 3 stages (Initial Answers $\to$ Cross-Critique $\to$ Judge Consensus) using their free web interfaces via Chrome DevTools Protocol (CDP) and Playwright, accessible via a modern Web Dashboard and Rich CLI.

**Architecture:** A LangGraph state machine orchestrates the 3-stage debate with parallel dispatch and graceful degradation. A polymorphic `BaseModelDriver` cleanly separates the debate engine from model communication, implemented initially by `BrowserCDPDriver` attaching to Chrome remote debugging port 9222. A FastAPI backend serves an SSE live event stream to an interactive glassmorphic dark-mode dashboard alongside a terminal Rich CLI runner.

**Tech Stack:** Python 3.13, `uv`, LangGraph, Playwright, FastAPI, Uvicorn, Rich, Pytest, Pytest-Asyncio, Vanilla HTML/CSS/JS.

**Spec:** `docs/superpowers/specs/2026-11-02-llm-debate-framework-design.md`

## Global Constraints

- **Python Version**: `>=3.13` managed exclusively with `uv`.
- **Package Management**: All dependencies installed via `uv add` / `uv add --dev`. Commands executed with `uv run`.
- **Browser Automation**: Must connect to local Chrome on `--remote-debugging-port=9222` via Playwright CDP without requiring paid API keys.
- **Graceful Degradation**: Debate must proceed as long as $\ge 2$ models succeed. Judge must automatically fall back to a healthy model if the designated judge drops out.
- **UI Aesthetics**: Sleek dark-mode dashboard with side-by-side color-coded model cards (Claude: amber, ChatGPT: emerald, Gemini: royal blue, DeepSeek: cyan).

---

### Task 1: Project Setup, Dependencies with `uv`, and Base Driver Interface

**Files:**
- Create: `pyproject.toml`
- Create: `debate/__init__.py`
- Create: `debate/drivers/__init__.py`
- Create: `debate/drivers/base.py`
- Create: `tests/__init__.py`
- Create: `tests/test_base_driver.py`

**Interfaces:**
- Consumes: Nothing
- Produces: `BaseModelDriver(ABC)` with `send_prompt(prompt: str, new_chat: bool = True) -> str` and `health_check() -> bool`.

- [ ] **Step 1: Install core dependencies with `uv`**

Run:
```bash
uv add langgraph playwright fastapi uvicorn rich python-dotenv
uv add --dev pytest pytest-asyncio httpx
uv run playwright install chromium
```

- [ ] **Step 2: Write failing test for `BaseModelDriver`**

Create `tests/test_base_driver.py`:
```python
import pytest
from debate.drivers.base import BaseModelDriver

class DummyDriver(BaseModelDriver):
    def __init__(self, name: str = "dummy"):
        self.model_name = name

    async def send_prompt(self, prompt: str, new_chat: bool = True) -> str:
        return f"Response to: {prompt}"

    async def health_check(self) -> bool:
        return True

@pytest.mark.asyncio
async def test_dummy_driver_implements_interface():
    driver = DummyDriver()
    assert driver.model_name == "dummy"
    assert await driver.health_check() is True
    res = await driver.send_prompt("Hello")
    assert res == "Response to: Hello"

def test_cannot_instantiate_abstract_driver():
    with pytest.raises(TypeError):
        BaseModelDriver()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_base_driver.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'debate'`

- [ ] **Step 4: Implement `BaseModelDriver`**

Create `debate/__init__.py` (empty).  
Create `debate/drivers/__init__.py`:
```python
from debate.drivers.base import BaseModelDriver

__all__ = ["BaseModelDriver"]
```

Create `debate/drivers/base.py`:
```python
from abc import ABC, abstractmethod

class BaseModelDriver(ABC):
    model_name: str

    @abstractmethod
    async def send_prompt(self, prompt: str, new_chat: bool = True) -> str:
        """Sends a prompt to the model and returns its full response."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verifies session is active and responsive."""
        pass
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_base_driver.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock debate/ tests/
git commit -m "feat: setup project dependencies with uv and create BaseModelDriver interface"
```

---

### Task 2: LangGraph State Schema and Prompt Templates

**Files:**
- Create: `debate/graph/__init__.py`
- Create: `debate/graph/state.py`
- Create: `debate/graph/prompts.py`
- Create: `tests/test_prompts.py`

**Interfaces:**
- Consumes: Nothing
- Produces:
  - `DebateState` TypedDict
  - `build_stage_1_prompt(topic: str) -> str`
  - `build_stage_2_prompt(topic: str, other_answers: dict[str, str]) -> str`
  - `build_stage_3_prompt(topic: str, stage_1_answers: dict[str, str], stage_2_critiques: dict[str, str]) -> str`

- [ ] **Step 1: Write failing test for prompts and state**

Create `tests/test_prompts.py`:
```python
from debate.graph.state import DebateState
from debate.graph.prompts import (
    build_stage_1_prompt,
    build_stage_2_prompt,
    build_stage_3_prompt,
)

def test_build_stage_1_prompt():
    prompt = build_stage_1_prompt("Should we use Rust or Go?")
    assert "Should we use Rust or Go?" in prompt
    assert "expert panel debate" in prompt

def test_build_stage_2_prompt():
    other_answers = {
        "claude": "Use Rust for memory safety.",
        "chatgpt": "Use Go for simple concurrency.",
    }
    prompt = build_stage_2_prompt("Should we use Rust or Go?", other_answers)
    assert "Should we use Rust or Go?" in prompt
    assert "claude" in prompt
    assert "Use Rust for memory safety." in prompt
    assert "Cross-examination" in prompt or "cross-examination" in prompt

def test_build_stage_3_prompt():
    answers = {"claude": "Rust", "chatgpt": "Go"}
    critiques = {"claude": "Go lacks borrow checker", "chatgpt": "Rust has steep curve"}
    prompt = build_stage_3_prompt("Should we use Rust or Go?", answers, critiques)
    assert "Judge and Synthesizer" in prompt
    assert "Executive Summary" in prompt
    assert "Panel Scorecard" in prompt

def test_state_structure():
    state: DebateState = {
        "topic": "Testing",
        "active_models": ["claude", "gemini"],
        "judge_model": "claude",
        "current_stage": "init",
        "stage_1_answers": {},
        "stage_2_critiques": {},
        "stage_3_synthesis": None,
        "status_events": [],
        "errors": {},
    }
    assert state["topic"] == "Testing"
    assert state["judge_model"] == "claude"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompts.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'debate.graph'`

- [ ] **Step 3: Implement `state.py` and `prompts.py`**

Create `debate/graph/__init__.py`:
```python
from debate.graph.state import DebateState

__all__ = ["DebateState"]
```

Create `debate/graph/state.py`:
```python
from typing import TypedDict, List, Dict, Optional, Any

class DebateState(TypedDict):
    topic: str
    active_models: List[str]
    judge_model: str
    current_stage: str
    stage_1_answers: Dict[str, str]
    stage_2_critiques: Dict[str, str]
    stage_3_synthesis: Optional[str]
    status_events: List[Dict[str, Any]]
    errors: Dict[str, str]
```

Create `debate/graph/prompts.py`:
```python
from typing import Dict

def build_stage_1_prompt(topic: str) -> str:
    return (
        f"You are participating in an expert panel debate on the following topic:\n"
        f"\"{topic}\"\n\n"
        f"Please provide your comprehensive, definitive answer.\n"
        f"- State your core argument and conclusion clearly.\n"
        f"- Provide rigorous supporting reasoning, evidence, or technical justification.\n"
        f"- Explicitly state any critical assumptions, edge cases, or trade-offs in your position."
    )

def build_stage_2_prompt(topic: str, other_answers: Dict[str, str]) -> str:
    formatted_answers = "\n\n".join(
        f"### {model_name.upper()}:\n{answer}"
        for model_name, answer in other_answers.items()
    )
    return (
        f"You are an expert panelist evaluating other AI models' answers to the topic:\n"
        f"\"{topic}\"\n\n"
        f"Here are the initial answers submitted by the panel:\n"
        f"---\n"
        f"{formatted_answers}\n"
        f"---\n\n"
        f"Perform a rigorous cross-examination:\n"
        f"1. Critical Analysis: Identify where other models made logical fallacies, overlooked edge cases, or made unsupported claims.\n"
        f"2. Strengths: Highlight which points or insights from other models are genuinely superior or complementary.\n"
        f"3. Rebuttal & Refinement: Defend or adjust your original thesis in response to the panel, providing your refined stance."
    )

def build_stage_3_prompt(topic: str, stage_1_answers: Dict[str, str], stage_2_critiques: Dict[str, str]) -> str:
    formatted_answers = "\n\n".join(
        f"### {model_name.upper()} (Initial):\n{answer}"
        for model_name, answer in stage_1_answers.items()
    )
    formatted_critiques = "\n\n".join(
        f"### {model_name.upper()} (Critique & Rebuttal):\n{critique}"
        for model_name, critique in stage_2_critiques.items()
    )
    return (
        f"You are the designated Judge and Synthesizer for this expert panel debate.\n\n"
        f"TOPIC:\n\"{topic}\"\n\n"
        f"INITIAL PROPOSALS:\n{formatted_answers}\n\n"
        f"CROSS-EXAMINATIONS & CRITIQUES:\n{formatted_critiques}\n\n"
        f"Synthesize the debate into a single, authoritative final report formatted as follows:\n\n"
        f"# Final Debate Synthesis: {topic}\n\n"
        f"## 1. Executive Summary & Core Consensus\n"
        f"Directly answer the topic. What is the unified, evidence-based conclusion?\n\n"
        f"## 2. Comprehensive Synthesized Solution\n"
        f"Present the comprehensive solution combining the best arguments from all panelists while filtering out refuted claims.\n\n"
        f"## 3. Key Trade-offs & Nuances\n"
        f"Detail the legitimate points of divergence or contextual trade-offs identified during the debate.\n\n"
        f"## 4. Panel Scorecard\n"
        f"Provide a brief evaluation of each model's contribution and corrected blindspots."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompts.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debate/graph/ tests/test_prompts.py
git commit -m "feat: implement DebateState schema and prompt builders"
```

---

### Task 3: LangGraph Debate Engine Nodes and StateGraph

**Files:**
- Create: `debate/graph/nodes.py`
- Create: `debate/graph/graph.py`
- Create: `tests/test_debate_graph.py`

**Interfaces:**
- Consumes: `DebateState`, `BaseModelDriver`, prompt builders
- Produces:
  - `build_debate_graph(drivers: dict[str, BaseModelDriver]) -> CompiledStateGraph`
  - Handles parallel dispatch, graceful degradation, and judge fallback.

- [ ] **Step 1: Write failing test with mock drivers for LangGraph execution**

Create `tests/test_debate_graph.py`:
```python
import pytest
from debate.drivers.base import BaseModelDriver
from debate.graph.graph import build_debate_graph
from debate.graph.state import DebateState

class MockModelDriver(BaseModelDriver):
    def __init__(self, name: str, should_fail: bool = False):
        self.model_name = name
        self.should_fail = should_fail
        self.prompts_received = []

    async def send_prompt(self, prompt: str, new_chat: bool = True) -> str:
        self.prompts_received.append(prompt)
        if self.should_fail:
            raise RuntimeError(f"{self.model_name} session timed out")
        return f"Response from {self.model_name} for prompt of length {len(prompt)}"

    async def health_check(self) -> bool:
        return not self.should_fail

@pytest.mark.asyncio
async def test_full_debate_graph_success():
    drivers = {
        "claude": MockModelDriver("claude"),
        "chatgpt": MockModelDriver("chatgpt"),
        "gemini": MockModelDriver("gemini"),
    }
    graph = build_debate_graph(drivers)
    initial_state: DebateState = {
        "topic": "Should Python use static typing?",
        "active_models": ["claude", "chatgpt", "gemini"],
        "judge_model": "claude",
        "current_stage": "init",
        "stage_1_answers": {},
        "stage_2_critiques": {},
        "stage_3_synthesis": None,
        "status_events": [],
        "errors": {},
    }
    final_state = await graph.ainvoke(initial_state)

    assert len(final_state["stage_1_answers"]) == 3
    assert len(final_state["stage_2_critiques"]) == 3
    assert final_state["stage_3_synthesis"] is not None
    assert "Response from claude" in final_state["stage_3_synthesis"]
    assert final_state["current_stage"] == "completed"

@pytest.mark.asyncio
async def test_graceful_degradation_when_one_model_fails():
    drivers = {
        "claude": MockModelDriver("claude", should_fail=True),
        "chatgpt": MockModelDriver("chatgpt"),
        "gemini": MockModelDriver("gemini"),
    }
    graph = build_debate_graph(drivers)
    initial_state: DebateState = {
        "topic": "Graceful degradation test",
        "active_models": ["claude", "chatgpt", "gemini"],
        "judge_model": "claude",
        "current_stage": "init",
        "stage_1_answers": {},
        "stage_2_critiques": {},
        "stage_3_synthesis": None,
        "status_events": [],
        "errors": {},
    }
    final_state = await graph.ainvoke(initial_state)

    assert "claude" in final_state["errors"]
    assert "chatgpt" in final_state["stage_1_answers"]
    assert "gemini" in final_state["stage_1_answers"]
    # Judge fallback should promote chatgpt or gemini
    assert final_state["judge_model"] in ["chatgpt", "gemini"]
    assert final_state["stage_3_synthesis"] is not None
    assert final_state["current_stage"] == "completed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_debate_graph.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'debate.graph.graph'`

- [ ] **Step 3: Implement `nodes.py` and `graph.py`**

Create `debate/graph/nodes.py`:
```python
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any
from debate.drivers.base import BaseModelDriver
from debate.graph.state import DebateState
from debate.graph.prompts import (
    build_stage_1_prompt,
    build_stage_2_prompt,
    build_stage_3_prompt,
)

def make_event(event_type: str, stage: str, model: str = "", content: str = "") -> Dict[str, Any]:
    return {
        "event": event_type,
        "stage": stage,
        "model": model,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

def create_stage_1_node(drivers: Dict[str, BaseModelDriver]):
    async def stage_1_node(state: DebateState) -> Dict[str, Any]:
        topic = state["topic"]
        active = list(state["active_models"])
        answers = dict(state.get("stage_1_answers", {}))
        errors = dict(state.get("errors", {}))
        events = list(state.get("status_events", []))

        events.append(make_event("stage_change", "init", content="Stage 1: Collecting initial answers"))

        async def query_model(name: str):
            driver = drivers.get(name)
            if not driver:
                return name, False, f"Driver for {name} not found"
            prompt = build_stage_1_prompt(topic)
            try:
                events.append(make_event("model_start", "init", model=name))
                res = await driver.send_prompt(prompt, new_chat=True)
                events.append(make_event("model_done", "init", model=name, content=res))
                return name, True, res
            except Exception as e:
                events.append(make_event("model_error", "init", model=name, content=str(e)))
                return name, False, str(e)

        results = await asyncio.gather(*(query_model(m) for m in active))

        new_active = []
        for name, success, content in results:
            if success:
                answers[name] = content
                new_active.append(name)
            else:
                errors[name] = content

        if len(new_active) < 2:
            raise RuntimeError(f"Debate cannot continue: only {len(new_active)} model(s) responded: {new_active}")

        return {
            "stage_1_answers": answers,
            "active_models": new_active,
            "errors": errors,
            "status_events": events,
            "current_stage": "critique",
        }
    return stage_1_node

def create_stage_2_node(drivers: Dict[str, BaseModelDriver]):
    async def stage_2_node(state: DebateState) -> Dict[str, Any]:
        topic = state["topic"]
        active = list(state["active_models"])
        stage_1_answers = state["stage_1_answers"]
        critiques = dict(state.get("stage_2_critiques", {}))
        errors = dict(state.get("errors", {}))
        events = list(state.get("status_events", []))

        events.append(make_event("stage_change", "critique", content="Stage 2: Cross-critique & rebuttal"))

        async def query_critique(name: str):
            driver = drivers.get(name)
            if not driver:
                return name, False, f"Driver for {name} not found"
            others = {m: ans for m, ans in stage_1_answers.items() if m != name}
            prompt = build_stage_2_prompt(topic, others)
            try:
                events.append(make_event("model_start", "critique", model=name))
                res = await driver.send_prompt(prompt, new_chat=False)
                events.append(make_event("model_done", "critique", model=name, content=res))
                return name, True, res
            except Exception as e:
                events.append(make_event("model_error", "critique", model=name, content=str(e)))
                return name, False, str(e)

        results = await asyncio.gather(*(query_critique(m) for m in active))

        new_active = []
        for name, success, content in results:
            if success:
                critiques[name] = content
                new_active.append(name)
            else:
                errors[name] = content

        if len(new_active) < 2:
            raise RuntimeError(f"Debate aborted during Stage 2: fewer than 2 active models remaining ({len(new_active)})")

        return {
            "stage_2_critiques": critiques,
            "active_models": new_active,
            "errors": errors,
            "status_events": events,
            "current_stage": "synthesis",
        }
    return stage_2_node

def create_stage_3_node(drivers: Dict[str, BaseModelDriver]):
    async def stage_3_node(state: DebateState) -> Dict[str, Any]:
        topic = state["topic"]
        active = list(state["active_models"])
        judge = state.get("judge_model", "")
        events = list(state.get("status_events", []))
        errors = dict(state.get("errors", {}))

        events.append(make_event("stage_change", "synthesis", content="Stage 3: Synthesizing final consensus"))

        if judge not in active:
            fallback = active[0]
            events.append(make_event("stage_change", "synthesis", content=f"Judge {judge} unavailable, falling back to {fallback}"))
            judge = fallback

        driver = drivers.get(judge)
        if not driver:
            raise RuntimeError(f"Driver for judge {judge} not found")

        prompt = build_stage_3_prompt(topic, state["stage_1_answers"], state["stage_2_critiques"])
        try:
            events.append(make_event("model_start", "synthesis", model=judge))
            synthesis = await driver.send_prompt(prompt, new_chat=False)
            events.append(make_event("synthesis_done", "completed", model=judge, content=synthesis))
        except Exception as e:
            errors[judge] = str(e)
            raise RuntimeError(f"Stage 3 judge synthesis failed: {e}")

        return {
            "judge_model": judge,
            "stage_3_synthesis": synthesis,
            "status_events": events,
            "errors": errors,
            "current_stage": "completed",
        }
    return stage_3_node
```

Create `debate/graph/graph.py`:
```python
from typing import Dict
from langgraph.graph import StateGraph, END
from debate.drivers.base import BaseModelDriver
from debate.graph.state import DebateState
from debate.graph.nodes import (
    create_stage_1_node,
    create_stage_2_node,
    create_stage_3_node,
)

def build_debate_graph(drivers: Dict[str, BaseModelDriver]):
    workflow = StateGraph(DebateState)

    workflow.add_node("stage_1_init", create_stage_1_node(drivers))
    workflow.add_node("stage_2_critique", create_stage_2_node(drivers))
    workflow.add_node("stage_3_synthesis", create_stage_3_node(drivers))

    workflow.set_entry_point("stage_1_init")
    workflow.add_edge("stage_1_init", "stage_2_critique")
    workflow.add_edge("stage_2_critique", "stage_3_synthesis")
    workflow.add_edge("stage_3_synthesis", END)

    return workflow.compile()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_debate_graph.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debate/graph/ tests/test_debate_graph.py
git commit -m "feat: implement LangGraph debate nodes and state graph with error handling"
```

---

### Task 4: Chrome Remote Debugging Helper & Health Checker

**Files:**
- Create: `scripts/run_chrome.sh`
- Create: `debate/utils/__init__.py`
- Create: `debate/utils/cdp_check.py`
- Create: `tests/test_cdp_check.py`

**Interfaces:**
- Consumes: Nothing
- Produces:
  - `scripts/run_chrome.sh`: executable script launching Chrome with `--remote-debugging-port=9222`
  - `check_cdp_status(host: str = "http://localhost:9222") -> dict`

- [ ] **Step 1: Write failing test for CDP checker**

Create `tests/test_cdp_check.py`:
```python
import pytest
from unittest.mock import patch, MagicMock
from debate.utils.cdp_check import check_cdp_status

def test_check_cdp_status_offline():
    status = check_cdp_status("http://localhost:99999")
    assert status["connected"] is False
    assert "error" in status

@patch("urllib.request.urlopen")
def test_check_cdp_status_online(mock_urlopen):
    mock_resp = MagicMock()
    mock_resp.read.return_value = b'[{"url": "https://claude.ai/chat/1"}, {"url": "https://chatgpt.com/"}]'
    mock_urlopen.return_value.__enter__.return_value = mock_resp

    status = check_cdp_status("http://localhost:9222")
    assert status["connected"] is True
    assert status["tabs"]["claude"] is True
    assert status["tabs"]["chatgpt"] is True
    assert status["tabs"]["gemini"] is False
    assert status["tabs"]["deepseek"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cdp_check.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'debate.utils'`

- [ ] **Step 3: Implement `scripts/run_chrome.sh` and `debate/utils/cdp_check.py`**

Create `scripts/run_chrome.sh`:
```bash
#!/usr/bin/env bash
set -e

PORT=${1:-9222}
USER_DATA_DIR="${HOME}/.config/google-chrome"

echo "======================================================="
echo "Launching Google Chrome with Remote Debugging on :$PORT"
echo "Profile directory: $USER_DATA_DIR"
echo "======================================================="

# Detect Chrome binary
if command -v google-chrome &> /dev/null; then
    CHROME_BIN="google-chrome"
elif command -v google-chrome-stable &> /dev/null; then
    CHROME_BIN="google-chrome-stable"
elif command -v chromium-browser &> /dev/null; then
    CHROME_BIN="chromium-browser"
elif command -v chromium &> /dev/null; then
    CHROME_BIN="chromium"
else
    echo "Error: Google Chrome or Chromium not found in PATH."
    exit 1
fi

"$CHROME_BIN" --remote-debugging-port="$PORT" --user-data-dir="$USER_DATA_DIR" > /dev/null 2>&1 &
echo "Chrome started in background (PID: $!)."
echo "You can now run 'python -m debate check-cdp' to verify connection."
```
Make executable: `chmod +x scripts/run_chrome.sh`

Create `debate/utils/__init__.py`:
```python
from debate.utils.cdp_check import check_cdp_status

__all__ = ["check_cdp_status"]
```

Create `debate/utils/cdp_check.py`:
```python
import json
import urllib.request
from typing import Dict, Any

def check_cdp_status(endpoint: str = "http://localhost:9222") -> Dict[str, Any]:
    json_url = f"{endpoint.rstrip('/')}/json/list"
    result: Dict[str, Any] = {
        "connected": False,
        "endpoint": endpoint,
        "tabs": {
            "claude": False,
            "chatgpt": False,
            "gemini": False,
            "deepseek": False,
        },
        "raw_tab_count": 0,
    }
    try:
        req = urllib.request.Request(json_url, headers={"User-Agent": "LLM-Debate/1.0"})
        with urllib.request.urlopen(req, timeout=1.5) as response:
            data = json.loads(response.read().decode("utf-8"))
            result["connected"] = True
            result["raw_tab_count"] = len(data)
            for item in data:
                url = item.get("url", "").lower()
                if "claude.ai" in url:
                    result["tabs"]["claude"] = True
                if "chatgpt.com" in url or "openai.com" in url:
                    result["tabs"]["chatgpt"] = True
                if "gemini.google.com" in url:
                    result["tabs"]["gemini"] = True
                if "deepseek.com" in url:
                    result["tabs"]["deepseek"] = True
    except Exception as e:
        result["connected"] = False
        result["error"] = str(e)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cdp_check.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_chrome.sh debate/utils/ tests/test_cdp_check.py
git commit -m "feat: add Chrome CDP launch script and CDP health check utility"
```

---

### Task 5: Browser CDP Model Driver & Web Service Adapters

**Files:**
- Create: `debate/drivers/adapters/__init__.py`
- Create: `debate/drivers/adapters/claude.py`
- Create: `debate/drivers/adapters/chatgpt.py`
- Create: `debate/drivers/adapters/gemini.py`
- Create: `debate/drivers/adapters/deepseek.py`
- Create: `debate/drivers/browser.py`
- Create: `tests/test_browser_driver.py`

**Interfaces:**
- Consumes: Playwright CDP, `BaseModelDriver`
- Produces:
  - `BrowserCDPDriver(model_name: str, cdp_url: str = "http://localhost:9222")`
  - Specialized adapters for Claude, ChatGPT, Gemini, DeepSeek handling DOM typing, response streaming wait, and markdown extraction.

- [ ] **Step 1: Write unit tests for adapter configurations and driver instantiation**

Create `tests/test_browser_driver.py`:
```python
import pytest
from debate.drivers.adapters import get_adapter
from debate.drivers.browser import BrowserCDPDriver

def test_get_adapter_exists_for_all_models():
    for name in ["claude", "chatgpt", "gemini", "deepseek"]:
        adapter = get_adapter(name)
        assert adapter is not None
        assert adapter.name == name
        assert adapter.base_url.startswith("https://")

def test_get_adapter_unknown_raises():
    with pytest.raises(ValueError):
        get_adapter("unknown_model")

def test_browser_driver_init():
    driver = BrowserCDPDriver("claude", cdp_url="http://localhost:9222")
    assert driver.model_name == "claude"
    assert driver.cdp_url == "http://localhost:9222"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_browser_driver.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'debate.drivers.adapters'`

- [ ] **Step 3: Implement web adapters for Claude, ChatGPT, Gemini, DeepSeek**

Create `debate/drivers/adapters/__init__.py`:
```python
from typing import Dict, Type
from debate.drivers.adapters.claude import ClaudeAdapter
from debate.drivers.adapters.chatgpt import ChatGPTAdapter
from debate.drivers.adapters.gemini import GeminiAdapter
from debate.drivers.adapters.deepseek import DeepSeekAdapter

ADAPTERS: Dict[str, Any] = {
    "claude": ClaudeAdapter(),
    "chatgpt": ChatGPTAdapter(),
    "gemini": GeminiAdapter(),
    "deepseek": DeepSeekAdapter(),
}

def get_adapter(name: str):
    name_clean = name.lower().strip()
    if name_clean not in ADAPTERS:
        raise ValueError(f"Unknown model adapter: '{name}'. Available: {list(ADAPTERS.keys())}")
    return ADAPTERS[name_clean]

__all__ = ["get_adapter", "ClaudeAdapter", "ChatGPTAdapter", "GeminiAdapter", "DeepSeekAdapter"]
```

Create `debate/drivers/adapters/claude.py`:
```python
from playwright.async_api import Page

class ClaudeAdapter:
    name = "claude"
    base_url = "https://claude.ai/new"
    match_url_pattern = "claude.ai"

    async def submit_prompt(self, page: Page, prompt: str):
        # Claude uses contenteditable div
        input_elem = await page.wait_for_selector('div[contenteditable="true"]', timeout=15000)
        await input_elem.click()
        await page.keyboard.insert_text(prompt)
        await page.wait_for_timeout(300)
        await page.keyboard.press("Enter")

    async def wait_for_completion(self, page: Page, timeout_seconds: int = 120) -> str:
        # Wait for stop generating button to disappear
        await page.wait_for_timeout(1000)
        try:
            stop_btn = await page.query_selector('button[aria-label*="Stop"]')
            if stop_btn:
                await page.wait_for_selector('button[aria-label*="Stop"]', state="detached", timeout=timeout_seconds * 1000)
        except Exception:
            pass
        # Extract last assistant message
        msgs = await page.query_selector_all('div.font-claude-message, div[data-is-streaming="false"]')
        if msgs:
            return (await msgs[-1].inner_text()).strip()
        # Fallback to general markdown container
        all_blocks = await page.query_selector_all('div.prose, div[class*="standard-markdown"]')
        if all_blocks:
            return (await all_blocks[-1].inner_text()).strip()
        return "Claude finished responding."
```

Create `debate/drivers/adapters/chatgpt.py`:
```python
from playwright.async_api import Page

class ChatGPTAdapter:
    name = "chatgpt"
    base_url = "https://chatgpt.com/"
    match_url_pattern = "chatgpt.com"

    async def submit_prompt(self, page: Page, prompt: str):
        input_elem = await page.wait_for_selector('#prompt-textarea, div[contenteditable="true"]', timeout=15000)
        await input_elem.click()
        await page.keyboard.insert_text(prompt)
        await page.wait_for_timeout(300)
        await page.keyboard.press("Enter")

    async def wait_for_completion(self, page: Page, timeout_seconds: int = 120) -> str:
        await page.wait_for_timeout(1000)
        try:
            # Wait for stop generation button to detach
            await page.wait_for_selector('button[data-testid="stop-button"]', state="detached", timeout=timeout_seconds * 1000)
        except Exception:
            pass
        # Get latest assistant message
        msgs = await page.query_selector_all('[data-message-author-role="assistant"]')
        if msgs:
            return (await msgs[-1].inner_text()).strip()
        return "ChatGPT finished responding."
```

Create `debate/drivers/adapters/gemini.py`:
```python
from playwright.async_api import Page

class GeminiAdapter:
    name = "gemini"
    base_url = "https://gemini.google.com/app"
    match_url_pattern = "gemini.google.com"

    async def submit_prompt(self, page: Page, prompt: str):
        input_elem = await page.wait_for_selector('rich-textarea div[contenteditable="true"], div.input-area', timeout=15000)
        await input_elem.click()
        await page.keyboard.insert_text(prompt)
        await page.wait_for_timeout(300)
        await page.keyboard.press("Enter")

    async def wait_for_completion(self, page: Page, timeout_seconds: int = 120) -> str:
        await page.wait_for_timeout(1500)
        try:
            await page.wait_for_selector('.sparkle-loading, .loading-indicator', state="detached", timeout=timeout_seconds * 1000)
        except Exception:
            pass
        msgs = await page.query_selector_all('model-response, message-content, .model-response-text')
        if msgs:
            return (await msgs[-1].inner_text()).strip()
        return "Gemini finished responding."
```

Create `debate/drivers/adapters/deepseek.py`:
```python
from playwright.async_api import Page

class DeepSeekAdapter:
    name = "deepseek"
    base_url = "https://chat.deepseek.com/"
    match_url_pattern = "deepseek.com"

    async def submit_prompt(self, page: Page, prompt: str):
        input_elem = await page.wait_for_selector('textarea', timeout=15000)
        await input_elem.click()
        await page.keyboard.insert_text(prompt)
        await page.wait_for_timeout(300)
        await page.keyboard.press("Enter")

    async def wait_for_completion(self, page: Page, timeout_seconds: int = 120) -> str:
        await page.wait_for_timeout(1000)
        try:
            await page.wait_for_selector('.ds-icon-stop, button[aria-label*="Stop"]', state="detached", timeout=timeout_seconds * 1000)
        except Exception:
            pass
        msgs = await page.query_selector_all('.ds-markdown, div[class*="message-assistant"]')
        if msgs:
            return (await msgs[-1].inner_text()).strip()
        return "DeepSeek finished responding."
```

Create `debate/drivers/browser.py`:
```python
import asyncio
from playwright.async_api import async_playwright
from debate.drivers.base import BaseModelDriver
from debate.drivers.adapters import get_adapter

class BrowserCDPDriver(BaseModelDriver):
    def __init__(self, model_name: str, cdp_url: str = "http://localhost:9222"):
        self.model_name = model_name.lower().strip()
        self.cdp_url = cdp_url
        self.adapter = get_adapter(self.model_name)
        self._lock = asyncio.Lock()

    async def _get_or_create_page(self, browser_context):
        for page in browser_context.pages:
            if self.adapter.match_url_pattern in page.url.lower():
                await page.bring_to_front()
                return page
        page = await browser_context.new_page()
        await page.goto(self.adapter.base_url, timeout=30000, wait_until="domcontentloaded")
        return page

    async def send_prompt(self, prompt: str, new_chat: bool = True) -> str:
        async with self._lock:
            async with async_playwright() as p:
                try:
                    browser = await p.chromium.connect_over_cdp(self.cdp_url)
                except Exception as e:
                    raise ConnectionError(
                        f"Cannot connect to Chrome at {self.cdp_url}. "
                        f"Make sure Chrome was started with './scripts/run_chrome.sh'. Error: {e}"
                    )
                context = browser.contexts[0] if browser.contexts else await browser.new_context()
                page = await self._get_or_create_page(context)

                if new_chat and page.url.lower() != self.adapter.base_url.lower():
                    await page.goto(self.adapter.base_url, timeout=30000, wait_until="domcontentloaded")

                await self.adapter.submit_prompt(page, prompt)
                response = await self.adapter.wait_for_completion(page)
                return response

    async def health_check(self) -> bool:
        async with async_playwright() as p:
            try:
                browser = await p.chromium.connect_over_cdp(self.cdp_url)
                return True
            except Exception:
                return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_browser_driver.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debate/drivers/ tests/test_browser_driver.py
git commit -m "feat: implement BrowserCDPDriver and service adapters for Claude, ChatGPT, Gemini, DeepSeek"
```

---

### Task 6: Markdown Transcript Exporter & Rich CLI Runner

**Files:**
- Create: `debate/utils/exporter.py`
- Create: `debate/__main__.py`
- Create: `tests/test_exporter.py`

**Interfaces:**
- Consumes: `DebateState`
- Produces:
  - `export_debate_to_markdown(state: DebateState, output_dir: str = "debates") -> str`
  - CLI commands: `python -m debate run ...`, `python -m debate check-cdp`

- [ ] **Step 1: Write failing test for markdown exporter**

Create `tests/test_exporter.py`:
```python
import os
import tempfile
from debate.graph.state import DebateState
from debate.utils.exporter import export_debate_to_markdown

def test_export_debate_to_markdown():
    state: DebateState = {
        "topic": "Should we use Next.js?",
        "active_models": ["claude", "chatgpt"],
        "judge_model": "claude",
        "current_stage": "completed",
        "stage_1_answers": {"claude": "Good for SSR.", "chatgpt": "Good for fullstack."},
        "stage_2_critiques": {"claude": "Complex routing.", "chatgpt": "Vercel lock-in."},
        "stage_3_synthesis": "# Final Consensus\nUse Next.js when SSR is needed.",
        "status_events": [],
        "errors": {},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = export_debate_to_markdown(state, output_dir=tmpdir)
        assert os.path.exists(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        assert "Should we use Next.js?" in content
        assert "Stage 1: Initial Proposals" in content
        assert "Good for SSR." in content
        assert "Final Consensus" in content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_exporter.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'debate.utils.exporter'`

- [ ] **Step 3: Implement `exporter.py` and `__main__.py`**

Create `debate/utils/exporter.py`:
```python
import os
import re
from datetime import datetime, timezone
from debate.graph.state import DebateState

def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)[:50]

def export_debate_to_markdown(state: DebateState, output_dir: str = "debates") -> str:
    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    slug = slugify(state["topic"])
    filename = f"{date_str}-{slug}.md"
    filepath = os.path.join(output_dir, filename)

    lines = [
        f"# LLM Debate: {state['topic']}",
        f"",
        f"**Date**: {date_str}  ",
        f"**Participants**: {', '.join(state.get('active_models', []))}  ",
        f"**Judge**: {state.get('judge_model', 'None')}  ",
        f"",
        f"---",
        f"",
        f"## Stage 1: Initial Proposals",
        f"",
    ]
    for model, answer in state.get("stage_1_answers", {}).items():
        lines.append(f"### {model.upper()}")
        lines.append(answer)
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Stage 2: Cross-Examinations & Critiques")
    lines.append("")
    for model, critique in state.get("stage_2_critiques", {}).items():
        lines.append(f"### {model.upper()}")
        lines.append(critique)
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Stage 3: Judge Synthesis & Consensus")
    lines.append("")
    lines.append(state.get("stage_3_synthesis") or "*(No synthesis produced)*")
    lines.append("")

    if state.get("errors"):
        lines.append("---")
        lines.append("## Errors Encountered")
        for m, err in state["errors"].items():
            lines.append(f"- **{m}**: {err}")
        lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return filepath
```

Create `debate/__main__.py`:
```python
import argparse
import asyncio
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from debate.utils.cdp_check import check_cdp_status
from debate.drivers.browser import BrowserCDPDriver
from debate.graph.graph import build_debate_graph
from debate.graph.state import DebateState
from debate.utils.exporter import export_debate_to_markdown

console = Console()

def run_check_cdp(args):
    status = check_cdp_status(args.endpoint)
    table = Table(title="Chrome DevTools Protocol (CDP) Status")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Connected", "[green]Yes[/green]" if status["connected"] else "[red]No[/red]")
    table.add_row("Endpoint", status["endpoint"])
    table.add_row("Open Tabs", str(status.get("raw_tab_count", 0)))

    for model, open_state in status.get("tabs", {}).items():
        val = "[green]Open[/green]" if open_state else "[yellow]Not detected[/yellow]"
        table.add_row(f"Tab: {model.capitalize()}", val)

    console.print(table)
    if not status["connected"]:
        console.print("\n[yellow]Tip:[/yellow] Run [bold]./scripts/run_chrome.sh[/bold] to launch Chrome with remote debugging.")

def run_debate_cli(args):
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    judge = args.judge.strip().lower()

    console.print(Panel(f"[bold cyan]Starting Debate[/bold cyan]\nTopic: {args.topic}\nModels: {', '.join(models)}\nJudge: {judge}"))

    drivers = {m: BrowserCDPDriver(m, cdp_url=args.endpoint) for m in models}
    graph = build_debate_graph(drivers)

    initial_state: DebateState = {
        "topic": args.topic,
        "active_models": models,
        "judge_model": judge,
        "current_stage": "init",
        "stage_1_answers": {},
        "stage_2_critiques": {},
        "stage_3_synthesis": None,
        "status_events": [],
        "errors": {},
    }

    async def execute():
        with console.status("[bold green]Running debate through LangGraph..."):
            return await graph.ainvoke(initial_state)

    result = asyncio.run(execute())

    if result.get("stage_3_synthesis"):
        console.print(Panel(Markdown(result["stage_3_synthesis"]), title=f"[bold green]Final Synthesis (Judge: {result.get('judge_model')})[/bold green]"))

    if args.export:
        filepath = export_debate_to_markdown(result)
        console.print(f"[bold green]Saved debate transcript to:[/bold green] {filepath}")

def main():
    parser = argparse.ArgumentParser(description="LLM Debate Framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # cdp check command
    check_p = subparsers.add_parser("check-cdp", help="Check Chrome CDP connection and tabs")
    check_p.add_argument("--endpoint", default="http://localhost:9222", help="CDP endpoint")
    check_p.set_defaults(func=run_check_cdp)

    # run command
    run_p = subparsers.add_parser("run", help="Run a multi-model debate")
    run_p.add_argument("topic", help="Topic to debate")
    run_p.add_argument("--models", default="claude,chatgpt,gemini,deepseek", help="Comma-separated models")
    run_p.add_argument("--judge", default="claude", help="Model to synthesize final consensus")
    run_p.add_argument("--endpoint", default="http://localhost:9222", help="CDP endpoint")
    run_p.add_argument("--export", action="store_true", help="Export report to markdown")
    run_p.set_defaults(func=run_debate_cli)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_exporter.py -v`  
Expected: PASS

- [ ] **Step 5: Test CLI check-cdp command**

Run: `uv run python -m debate check-cdp`  
Expected: Prints CDP status table without errors.

- [ ] **Step 6: Commit**

```bash
git add debate/utils/exporter.py debate/__main__.py tests/test_exporter.py
git commit -m "feat: implement markdown exporter and Rich CLI command runner"
```

---

### Task 7: FastAPI Backend & Server-Sent Events (SSE) Streaming

**Files:**
- Create: `debate/server/__init__.py`
- Create: `debate/server/sse.py`
- Create: `debate/server/app.py`
- Create: `tests/test_api_server.py`

**Interfaces:**
- Consumes: LangGraph engine, CDP checker
- Produces:
  - `POST /api/debate/start` -> `{debate_id}`
  - `GET /api/debate/{debate_id}/events` -> SSE stream
  - `GET /api/health` -> CDP & tab status

- [ ] **Step 1: Write failing test for FastAPI routes**

Create `tests/test_api_server.py`:
```python
import pytest
from httpx import AsyncClient, ASGITransport
from debate.server.app import app

@pytest.mark.asyncio
async def test_health_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "connected" in data
    assert "tabs" in data

@pytest.mark.asyncio
async def test_start_debate_validation():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/api/debate/start", json={"topic": "", "models": []})
    assert response.status_code == 422
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api_server.py -v`  
Expected: FAIL with `ModuleNotFoundError: No module named 'debate.server'`

- [ ] **Step 3: Implement `sse.py` and `app.py`**

Create `debate/server/__init__.py`:
```python
from debate.server.app import app

__all__ = ["app"]
```

Create `debate/server/sse.py`:
```python
import asyncio
import json
from typing import Dict, AsyncGenerator

class SSEManager:
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}

    def get_or_create_queue(self, debate_id: str) -> asyncio.Queue:
        if debate_id not in self._queues:
            self._queues[debate_id] = asyncio.Queue()
        return self._queues[debate_id]

    async def broadcast_event(self, debate_id: str, event_data: dict):
        if debate_id in self._queues:
            await self._queues[debate_id].put(event_data)

    async def event_generator(self, debate_id: str) -> AsyncGenerator[str, None]:
        q = self.get_or_create_queue(debate_id)
        while True:
            event_data = await q.get()
            yield f"data: {json.dumps(event_data)}\n\n"
            if event_data.get("event") == "synthesis_done" or event_data.get("event") == "fatal_error":
                break

sse_manager = SSEManager()
```

Create `debate/server/app.py`:
```python
import uuid
import asyncio
from typing import List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from debate.utils.cdp_check import check_cdp_status
from debate.server.sse import sse_manager
from debate.drivers.browser import BrowserCDPDriver
from debate.graph.graph import build_debate_graph
from debate.graph.state import DebateState
from debate.utils.exporter import export_debate_to_markdown

app = FastAPI(title="LLM Debate Framework API")

class DebateRequest(BaseModel):
    topic: str = Field(..., min_length=3)
    models: List[str] = Field(default=["claude", "chatgpt", "gemini", "deepseek"], min_length=2)
    judge: str = Field(default="claude")

debates_db: dict = {}

@app.get("/api/health")
async def health_check():
    return check_cdp_status()

@app.post("/api/debate/start")
async def start_debate(req: DebateRequest):
    debate_id = str(uuid.uuid4())
    models = [m.lower().strip() for m in req.models]
    judge = req.judge.lower().strip()

    debates_db[debate_id] = {
        "status": "running",
        "topic": req.topic,
        "models": models,
        "judge": judge,
    }

    async def run_in_background():
        drivers = {m: BrowserCDPDriver(m) for m in models}
        graph = build_debate_graph(drivers)
        initial_state: DebateState = {
            "topic": req.topic,
            "active_models": models,
            "judge_model": judge,
            "current_stage": "init",
            "stage_1_answers": {},
            "stage_2_critiques": {},
            "stage_3_synthesis": None,
            "status_events": [],
            "errors": {},
        }
        try:
            # Stream status events as graph executes
            final_state = await graph.ainvoke(initial_state)
            for event in final_state.get("status_events", []):
                await sse_manager.broadcast_event(debate_id, event)
            debates_db[debate_id]["status"] = "completed"
            debates_db[debate_id]["state"] = final_state
            export_debate_to_markdown(final_state)
        except Exception as e:
            await sse_manager.broadcast_event(debate_id, {"event": "fatal_error", "content": str(e)})
            debates_db[debate_id]["status"] = "failed"
            debates_db[debate_id]["error"] = str(e)

    asyncio.create_task(run_in_background())
    return {"debate_id": debate_id, "status": "started"}

@app.get("/api/debate/{debate_id}/events")
async def stream_events(debate_id: str):
    if debate_id not in debates_db:
        raise HTTPException(status_code=404, detail="Debate not found")
    return StreamingResponse(
        sse_manager.event_generator(debate_id),
        media_type="text/event-stream",
    )

# Mount static files for Dashboard UI
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api_server.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debate/server/ tests/test_api_server.py
git commit -m "feat: implement FastAPI server with SSE streaming and health endpoints"
```

---

### Task 8: Interactive Web Dashboard UI

**Files:**
- Create: `debate/server/static/index.html`
- Create: `debate/server/static/styles.css`
- Create: `debate/server/static/app.js`

**Interfaces:**
- Consumes: `/api/health`, `/api/debate/start`, `/api/debate/{id}/events`
- Produces: Responsive glassmorphic dark-mode web app with live stage timeline, 4-column side-by-side comparison, and final synthesis card with Markdown rendering.

- [ ] **Step 1: Create `debate/server/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LLM Debate Arena</title>
  <link rel="stylesheet" href="styles.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <!-- Marked for markdown parsing -->
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
  <div class="app-container">
    <header class="header">
      <div class="header-left">
        <div class="logo">⚡ LLM Debate Arena</div>
        <div id="cdp-status" class="status-pill status-offline">● Chrome CDP Offline</div>
      </div>
      <div class="header-right">
        <button id="refresh-cdp-btn" class="btn btn-secondary">Refresh CDP</button>
      </div>
    </header>

    <section class="control-panel">
      <div class="form-group">
        <label for="topic-input">Debate Topic</label>
        <textarea id="topic-input" rows="2" placeholder="e.g. Should startups choose PostgreSQL or MongoDB for their primary datastore?"></textarea>
      </div>

      <div class="panel-options">
        <div class="model-selection">
          <label>Participating Models:</label>
          <div class="chips">
            <label class="chip chip-claude"><input type="checkbox" value="claude" checked> Claude</label>
            <label class="chip chip-chatgpt"><input type="checkbox" value="chatgpt" checked> ChatGPT</label>
            <label class="chip chip-gemini"><input type="checkbox" value="gemini" checked> Gemini</label>
            <label class="chip chip-deepseek"><input type="checkbox" value="deepseek" checked> DeepSeek</label>
          </div>
        </div>

        <div class="judge-selection">
          <label for="judge-select">Stage 3 Judge:</label>
          <select id="judge-select">
            <option value="claude">Claude</option>
            <option value="chatgpt">ChatGPT</option>
            <option value="gemini">Gemini</option>
            <option value="deepseek">DeepSeek</option>
          </select>
        </div>

        <button id="start-debate-btn" class="btn btn-primary">Start Debate</button>
      </div>
    </section>

    <!-- Stage Progress Stepper -->
    <section class="stepper-section">
      <div class="stepper">
        <div id="step-1" class="step">
          <div class="step-num">1</div>
          <div class="step-label">Initial Arguments</div>
        </div>
        <div class="step-line"></div>
        <div id="step-2" class="step">
          <div class="step-num">2</div>
          <div class="step-label">Cross-Critique</div>
        </div>
        <div class="step-line"></div>
        <div id="step-3" class="step">
          <div class="step-num">3</div>
          <div class="step-label">Judge Synthesis</div>
        </div>
      </div>
    </section>

    <!-- 4-Model Side-by-side Grid -->
    <main class="models-grid">
      <div class="model-card card-claude" id="card-claude">
        <div class="card-header">
          <span class="model-name">🟠 Claude</span>
          <span class="card-status" id="status-claude">Idle</span>
        </div>
        <div class="card-body" id="content-claude">Waiting for debate to begin...</div>
      </div>

      <div class="model-card card-chatgpt" id="card-chatgpt">
        <div class="card-header">
          <span class="model-name">🟢 ChatGPT</span>
          <span class="card-status" id="status-chatgpt">Idle</span>
        </div>
        <div class="card-body" id="content-chatgpt">Waiting for debate to begin...</div>
      </div>

      <div class="model-card card-gemini" id="card-gemini">
        <div class="card-header">
          <span class="model-name">🔵 Gemini</span>
          <span class="card-status" id="status-gemini">Idle</span>
        </div>
        <div class="card-body" id="content-gemini">Waiting for debate to begin...</div>
      </div>

      <div class="model-card card-deepseek" id="card-deepseek">
        <div class="card-header">
          <span class="model-name">🟣 DeepSeek</span>
          <span class="card-status" id="status-deepseek">Idle</span>
        </div>
        <div class="card-body" id="content-deepseek">Waiting for debate to begin...</div>
      </div>
    </main>

    <!-- Final Synthesis Hero Section -->
    <section class="synthesis-section" id="synthesis-section" style="display: none;">
      <div class="synthesis-card">
        <div class="synthesis-header">
          <h2>🏆 Final Synthesized Consensus</h2>
          <button id="copy-report-btn" class="btn btn-secondary">Copy Report</button>
        </div>
        <div class="synthesis-body" id="synthesis-content"></div>
      </div>
    </section>
  </div>
  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Create `debate/server/static/styles.css`**

```css
:root {
  --bg-primary: #0a0c10;
  --bg-secondary: #131720;
  --bg-card: rgba(26, 32, 46, 0.7);
  --border-color: rgba(255, 255, 255, 0.08);
  --text-main: #f0f3f6;
  --text-muted: #8b949e;
  
  --claude-accent: #d97706;
  --chatgpt-accent: #10b981;
  --gemini-accent: #3b82f6;
  --deepseek-accent: #8b5cf6;
  --primary-accent: #6366f1;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Inter', sans-serif;
  background-color: var(--bg-primary);
  color: var(--text-main);
  line-height: 1.5;
  padding: 24px;
}

.app-container { max-width: 1400px; margin: 0 auto; display: flex; flex-direction: column; gap: 24px; }

.header {
  display: flex; justify-content: space-between; align-items: center;
  padding-bottom: 16px; border-bottom: 1px solid var(--border-color);
}
.header-left { display: flex; align-items: center; gap: 16px; }
.logo { font-size: 1.5rem; font-weight: 700; color: #fff; }

.status-pill {
  font-size: 0.8rem; font-weight: 600; padding: 4px 12px; border-radius: 9999px;
}
.status-online { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid #10b981; }
.status-offline { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid #ef4444; }

.control-panel {
  background: var(--bg-card); backdrop-filter: blur(12px);
  border: 1px solid var(--border-color); border-radius: 12px; padding: 20px;
  display: flex; flex-direction: column; gap: 16px;
}
.form-group label { display: block; font-weight: 600; margin-bottom: 8px; font-size: 0.9rem; }
textarea {
  width: 100%; background: #0d1117; border: 1px solid var(--border-color);
  border-radius: 8px; color: #fff; padding: 12px; font-family: inherit; font-size: 1rem;
}

.panel-options { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 16px; }
.chips { display: flex; gap: 8px; }
.chip {
  padding: 6px 12px; border-radius: 8px; background: rgba(255,255,255,0.05);
  font-size: 0.85rem; font-weight: 500; cursor: pointer; display: flex; align-items: center; gap: 6px;
  border: 1px solid var(--border-color);
}
.btn {
  padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer;
  border: none; transition: 0.2s ease;
}
.btn-primary { background: var(--primary-accent); color: #fff; }
.btn-primary:hover { opacity: 0.9; }
.btn-secondary { background: rgba(255,255,255,0.08); color: #fff; }

/* Stepper */
.stepper { display: flex; justify-content: center; align-items: center; gap: 16px; }
.step { display: flex; align-items: center; gap: 8px; opacity: 0.5; }
.step.active { opacity: 1; font-weight: 700; color: var(--primary-accent); }
.step.completed { opacity: 1; color: #10b981; }
.step-num { width: 28px; height: 28px; border-radius: 50%; background: var(--bg-card); display: flex; align-items: center; justify-content: center; border: 1px solid var(--border-color); }
.step-line { width: 60px; height: 2px; background: var(--border-color); }

/* Models Grid */
.models-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px;
}
.model-card {
  background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border-color);
  display: flex; flex-direction: column; min-height: 400px; max-height: 600px;
}
.card-claude { border-top: 3px solid var(--claude-accent); }
.card-chatgpt { border-top: 3px solid var(--chatgpt-accent); }
.card-gemini { border-top: 3px solid var(--gemini-accent); }
.card-deepseek { border-top: 3px solid var(--deepseek-accent); }

.card-header {
  padding: 12px 16px; border-bottom: 1px solid var(--border-color);
  display: flex; justify-content: space-between; align-items: center; font-weight: 600;
}
.card-status { font-size: 0.75rem; color: var(--text-muted); }
.card-body {
  padding: 16px; overflow-y: auto; font-size: 0.9rem; flex: 1;
}

/* Synthesis */
.synthesis-card {
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(16, 185, 129, 0.05));
  border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 24px;
}
.synthesis-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.synthesis-body { line-height: 1.6; }
```

- [ ] **Step 3: Create `debate/server/static/app.js`**

```javascript
document.addEventListener("DOMContentLoaded", () => {
  const cdpStatusEl = document.getElementById("cdp-status");
  const refreshCdpBtn = document.getElementById("refresh-cdp-btn");
  const startBtn = document.getElementById("start-debate-btn");
  const topicInput = document.getElementById("topic-input");
  const judgeSelect = document.getElementById("judge-select");
  const synthesisSection = document.getElementById("synthesis-section");
  const synthesisContent = document.getElementById("synthesis-content");
  const copyBtn = document.getElementById("copy-report-btn");

  let rawSynthesisMarkdown = "";

  async function checkCdp() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.connected) {
        cdpStatusEl.className = "status-pill status-online";
        cdpStatusEl.textContent = `● Chrome CDP Online (${data.raw_tab_count} tabs)`;
      } else {
        cdpStatusEl.className = "status-pill status-offline";
        cdpStatusEl.textContent = "● Chrome CDP Offline";
      }
    } catch {
      cdpStatusEl.className = "status-pill status-offline";
      cdpStatusEl.textContent = "● API Server Offline";
    }
  }

  refreshCdpBtn.addEventListener("click", checkCdp);
  checkCdp();

  function updateStep(stage) {
    document.querySelectorAll(".step").forEach(el => el.classList.remove("active", "completed"));
    if (stage === "init") {
      document.getElementById("step-1").classList.add("active");
    } else if (stage === "critique") {
      document.getElementById("step-1").classList.add("completed");
      document.getElementById("step-2").classList.add("active");
    } else if (stage === "synthesis") {
      document.getElementById("step-1").classList.add("completed");
      document.getElementById("step-2").classList.add("completed");
      document.getElementById("step-3").classList.add("active");
    } else if (stage === "completed") {
      document.getElementById("step-1").classList.add("completed");
      document.getElementById("step-2").classList.add("completed");
      document.getElementById("step-3").classList.add("completed");
    }
  }

  startBtn.addEventListener("click", async () => {
    const topic = topicInput.value.trim();
    if (!topic) {
      alert("Please enter a debate topic.");
      return;
    }

    const selectedModels = Array.from(document.querySelectorAll(".chips input:checked")).map(cb => cb.value);
    if (selectedModels.length < 2) {
      alert("Please select at least 2 models.");
      return;
    }

    startBtn.disabled = true;
    startBtn.textContent = "Debating...";
    synthesisSection.style.display = "none";

    selectedModels.forEach(m => {
      const body = document.getElementById(`content-${m}`);
      const status = document.getElementById(`status-${m}`);
      if (body) body.innerHTML = "<em>Waiting for turn...</em>";
      if (status) status.textContent = "Queued";
    });

    try {
      const res = await fetch("/api/debate/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic,
          models: selectedModels,
          judge: judgeSelect.value,
        }),
      });
      const data = await res.json();
      const debateId = data.debate_id;

      const eventSource = new EventSource(`/api/debate/${debateId}/events`);
      eventSource.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        handleDebateEvent(payload);
        if (payload.event === "synthesis_done" || payload.event === "fatal_error") {
          eventSource.close();
          startBtn.disabled = false;
          startBtn.textContent = "Start Debate";
        }
      };
    } catch (err) {
      alert(`Error starting debate: ${err.message}`);
      startBtn.disabled = false;
      startBtn.textContent = "Start Debate";
    }
  });

  function handleDebateEvent(data) {
    if (data.event === "stage_change") {
      updateStep(data.stage);
    } else if (data.event === "model_start") {
      const status = document.getElementById(`status-${data.model}`);
      if (status) status.textContent = "Thinking...";
    } else if (data.event === "model_done") {
      const status = document.getElementById(`status-${data.model}`);
      const body = document.getElementById(`content-${data.model}`);
      if (status) status.textContent = "Submitted";
      if (body) body.innerHTML = marked.parse(data.content);
    } else if (data.event === "model_error") {
      const status = document.getElementById(`status-${data.model}`);
      const body = document.getElementById(`content-${data.model}`);
      if (status) status.textContent = "Failed";
      if (body) body.innerHTML = `<span style="color:#ef4444">Error: ${data.content}</span>`;
    } else if (data.event === "synthesis_done") {
      updateStep("completed");
      rawSynthesisMarkdown = data.content;
      synthesisContent.innerHTML = marked.parse(data.content);
      synthesisSection.style.display = "block";
      synthesisSection.scrollIntoView({ behavior: "smooth" });
    }
  }

  copyBtn.addEventListener("click", () => {
    if (rawSynthesisMarkdown) {
      navigator.clipboard.writeText(rawSynthesisMarkdown);
      copyBtn.textContent = "Copied!";
      setTimeout(() => copyBtn.textContent = "Copy Report", 2000);
    }
  });
});
```

- [ ] **Step 4: Verify static mounting with test**

Add to `tests/test_api_server.py`:
```python
@pytest.mark.asyncio
async def test_static_index_serves():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/")
    assert response.status_code == 200
    assert "LLM Debate Arena" in response.text
```

Run: `uv run pytest tests/test_api_server.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add debate/server/static/ tests/test_api_server.py
git commit -m "feat: implement glassmorphic web dashboard UI with real-time SSE updates"
```

---

### Task 9: End-to-End Verification & Integration Test

**Files:**
- Create: `tests/test_e2e_simulation.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: All modules
- Produces: Complete verified end-to-end simulation asserting all 3 stages, markdown report export, and documentation in README.md.

- [ ] **Step 1: Write end-to-end integration test**

Create `tests/test_e2e_simulation.py`:
```python
import os
import pytest
from debate.drivers.base import BaseModelDriver
from debate.graph.graph import build_debate_graph
from debate.graph.state import DebateState
from debate.utils.exporter import export_debate_to_markdown

class FullSimulationDriver(BaseModelDriver):
    def __init__(self, name: str):
        self.model_name = name

    async def send_prompt(self, prompt: str, new_chat: bool = True) -> str:
        if "expert panel debate" in prompt:
            return f"[{self.model_name}] Initial thesis: We should favor clarity over cleverness."
        elif "cross-examination" in prompt:
            return f"[{self.model_name}] Critique: Other models had good points, but overlooked latency."
        else:
            return (
                f"# Final Debate Synthesis\n\n"
                f"## 1. Executive Summary\nUnified agreement reached.\n\n"
                f"## 2. Comprehensive Solution\nImplement balanced approach.\n\n"
                f"## 3. Scorecard\nAll models contributed."
            )

    async def health_check(self) -> bool:
        return True

@pytest.mark.asyncio
async def test_complete_e2e_debate_lifecycle(tmp_path):
    models = ["claude", "chatgpt", "gemini", "deepseek"]
    drivers = {m: FullSimulationDriver(m) for m in models}
    graph = build_debate_graph(drivers)

    state: DebateState = {
        "topic": "Microservices vs Monolith for Seed Startups",
        "active_models": models,
        "judge_model": "claude",
        "current_stage": "init",
        "stage_1_answers": {},
        "stage_2_critiques": {},
        "stage_3_synthesis": None,
        "status_events": [],
        "errors": {},
    }

    result = await graph.ainvoke(state)
    assert result["current_stage"] == "completed"
    assert len(result["stage_1_answers"]) == 4
    assert len(result["stage_2_critiques"]) == 4
    assert result["stage_3_synthesis"] is not None

    export_path = export_debate_to_markdown(result, output_dir=str(tmp_path))
    assert os.path.exists(export_path)
    with open(export_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Microservices vs Monolith" in content
    assert "Final Debate Synthesis" in content
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_e2e_simulation.py -v`  
Expected: PASS

- [ ] **Step 3: Update `README.md` with complete usage instructions**

Update `README.md`:
```markdown
# LLM Debate Framework

A multi-model debate engine powered by **LangGraph** where Claude, ChatGPT, Gemini, and DeepSeek debate each other across 3 structured stages to produce a unified, consensus-driven best answer — running on their free web interfaces via Chrome DevTools Protocol (CDP) today, with paid API support ready for the future.

## Prerequisites

- Python 3.13+ with [uv](https://docs.astral.sh/uv/)
- Google Chrome or Chromium browser

## Quick Start

### 1. Launch Chrome with Remote Debugging
Ensure Chrome is started with remote debugging enabled on port 9222 using your normal user profile:
```bash
./scripts/run_chrome.sh
```
Verify the connection and open tabs:
```bash
uv run python -m debate check-cdp
```

### 2. Run via Web Dashboard
Start the local server:
```bash
uv run uvicorn debate.server.app:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser. Enter your topic, select participating models and a judge, and watch the live debate unfold.

### 3. Run via CLI
```bash
uv run python -m debate run "Should startups adopt microservices early?" --models claude,chatgpt,gemini,deepseek --judge claude --export
```

## Running Tests

```bash
uv run pytest
```
```

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest`  
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_simulation.py README.md
git commit -m "docs: complete README and add end-to-end debate simulation tests"
```
