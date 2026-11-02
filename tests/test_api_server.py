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
