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
