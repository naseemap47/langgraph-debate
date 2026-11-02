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
