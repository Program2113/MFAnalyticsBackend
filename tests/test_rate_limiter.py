import time
import asyncio
import pytest
import pytest_asyncio
import redis.asyncio as aioredis
from main import REDIS_URL, RATE_LIMIT_LUA

@pytest_asyncio.fixture
async def test_redis():
    # 1. Create a fresh client strictly bound to the current test's event loop
    client = aioredis.from_url(REDIS_URL, decode_responses=True)
    await client.flushdb()
    
    yield client  # 2. Hand this fresh client to the test
    
    # 3. Clean up and explicitly sever the connection pool after the test finishes
    await client.flushdb()
    await client.connection_pool.disconnect()

@pytest.mark.asyncio
async def test_per_second_limit(test_redis):
    now_ms = int(time.time() * 1000)
    
    # 2 requests should be allowed immediately
    res1 = await test_redis.eval(RATE_LIMIT_LUA, 3, "rl:sec", "rl:min", "rl:hr", str(now_ms))
    res2 = await test_redis.eval(RATE_LIMIT_LUA, 3, "rl:sec", "rl:min", "rl:hr", str(now_ms))
    
    # 3rd request in the same millisecond should be blocked
    res3 = await test_redis.eval(RATE_LIMIT_LUA, 3, "rl:sec", "rl:min", "rl:hr", str(now_ms))
    
    assert res1 == 1
    assert res2 == 1
    assert res3 == 0

@pytest.mark.asyncio
async def test_concurrent_access_race_conditions(test_redis):
    now_ms = int(time.time() * 1000)
    
    # Fire 10 simultaneous requests
    tasks = [
        test_redis.eval(RATE_LIMIT_LUA, 3, "rl:sec", "rl:min", "rl:hr", str(now_ms))
        for _ in range(10)
    ]
    results = await asyncio.gather(*tasks)
    
    # Exactly 2 should succeed, 8 should fail (2/sec limit)
    assert results.count(1) == 2
    assert results.count(0) == 8

@pytest.mark.asyncio
async def test_state_persistence_and_ttl(test_redis):
    now_ms = int(time.time() * 1000)
    await test_redis.eval(RATE_LIMIT_LUA, 3, "rl:sec", "rl:min", "rl:hr", str(now_ms))
    
    # Verify the global sequence key exists and has a 24-hour TTL (86400s)
    ttl = await test_redis.ttl("rl:seq")
    assert ttl > 0
    assert ttl <= 86400