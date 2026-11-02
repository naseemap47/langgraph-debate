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
