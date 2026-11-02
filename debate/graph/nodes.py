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
