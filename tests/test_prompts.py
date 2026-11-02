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
    assert "claude" in prompt or "CLAUDE" in prompt
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
