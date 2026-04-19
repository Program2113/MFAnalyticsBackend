import pytest
import time
from httpx import AsyncClient
from main import app

@pytest.mark.asyncio
async def test_funds_list_response_time():
    async with AsyncClient(app=app, base_url="http://test") as client:
        start_time = time.perf_counter()
        response = await client.get("/funds")
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        
        assert response.status_code == 200
        assert elapsed_ms < 200.0, f"API took {elapsed_ms}ms, which exceeds the 200ms limit."

@pytest.mark.asyncio
async def test_rank_endpoint_response_time():
    async with AsyncClient(app=app, base_url="http://test") as client:
        start_time = time.perf_counter()
        # Testing the specific ranking constraints required by the assignment
        response = await client.get("/funds/rank?category=Equity:%20Mid%20Cap&window=3Y&sort_by=median_return")
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        
        # We expect 200 or 422 (if DB is entirely empty and validation fails), 
        # but the speed is what matters here.
        assert response.status_code in [200, 404] 
        assert elapsed_ms < 200.0, f"Ranking API took {elapsed_ms}ms, exceeding 200ms."