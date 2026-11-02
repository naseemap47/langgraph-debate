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
