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
        if "Judge and Synthesizer" in prompt:
            return (
                f"# Final Debate Synthesis\n\n"
                f"## 1. Executive Summary\nUnified agreement reached.\n\n"
                f"## 2. Comprehensive Solution\nImplement balanced approach.\n\n"
                f"## 3. Scorecard\nAll models contributed."
            )
        elif "cross-examination" in prompt or "Cross-examination" in prompt:
            return f"[{self.model_name}] Critique: Other models had good points, but overlooked latency."
        else:
            return f"[{self.model_name}] Initial thesis: We should favor clarity over cleverness."


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
