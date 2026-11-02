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
