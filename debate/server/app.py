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

static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")
