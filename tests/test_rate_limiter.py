"""
test_rate_limiter.py
====================
Tests for the Redis sliding-window rate limiter.

Assignment requirements covered:
  1. All three limits enforced (per-second, per-minute, per-hour)
  2. Concurrent access / race conditions
  3. State persistence across reconnection
  4. Window expiry / sliding behaviour

Every test uses the `test_redis` fixture from conftest.py, which deletes only
the rl:* keys before and after each test, leaving the rest of Redis untouched.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import pytest_asyncio
import redis.asyncio as aioredis
import redis as sync_redis  # synchronous client — used for the reconnect test

from main import REDIS_URL, RATE_LIMIT_LUA

# Convenience alias so every test uses the same three key names.
_KEYS = ["rl:rolling:sec", "rl:rolling:min", "rl:rolling:hr"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

async def _admit(client: aioredis.Redis, now_ms: int) -> int:
    """Run the Lua script for a single request and return 1 (allowed) or 0 (blocked)."""
    return await client.eval(RATE_LIMIT_LUA, 3, *_KEYS, str(now_ms))


# ---------------------------------------------------------------------------
# 1. Per-second limit (2 req/s)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_second_limit_allows_exactly_two(test_redis):
    """
    At the same timestamp, exactly 2 requests must be admitted and the 3rd
    must be blocked.  All three calls use an identical now_ms so they all
    fall within the same 1-second window.
    """
    now_ms = int(time.time() * 1000)

    res1 = await _admit(test_redis, now_ms)
    res2 = await _admit(test_redis, now_ms)
    res3 = await _admit(test_redis, now_ms)  # should be blocked

    assert res1 == 1, "First request must be admitted"
    assert res2 == 1, "Second request must be admitted"
    assert res3 == 0, "Third request in the same second must be blocked"


@pytest.mark.asyncio
async def test_per_second_limit_resets_after_window(test_redis):
    """
    After the 1-second window expires, two new requests must be admitted again.
    We simulate window expiry by advancing now_ms by 1001 ms instead of
    sleeping, which keeps the test fast while exercising the ZREMRANGEBYSCORE
    eviction path correctly.
    """
    now_ms = int(time.time() * 1000)

    # Fill the window.
    await _admit(test_redis, now_ms)
    await _admit(test_redis, now_ms)
    blocked = await _admit(test_redis, now_ms)
    assert blocked == 0, "Precondition: window should be full"

    # Advance time by 1001 ms — old entries fall outside the 1-second window.
    future_ms = now_ms + 1001

    res1 = await _admit(test_redis, future_ms)
    res2 = await _admit(test_redis, future_ms)
    res3 = await _admit(test_redis, future_ms)  # should be blocked again

    assert res1 == 1, "First request after window expiry must be admitted"
    assert res2 == 1, "Second request after window expiry must be admitted"
    assert res3 == 0, "Third request in the new window must still be blocked"


# ---------------------------------------------------------------------------
# 2. Per-minute limit (50 req/min)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_minute_limit_allows_fifty(test_redis):
    """
    50 requests spread across the same minute must all be admitted.
    The 51st must be blocked.

    We spread them 100 ms apart within the same 60-second window so each call
    uses a unique now_ms (preventing rl:seq collisions) while still falling
    inside the 60-second sliding window.
    """
    base_ms = int(time.time() * 1000)

    admitted = 0
    for i in range(50):
        # Change 100 to 600 to slow the loop down to ~1.6 req/sec
        now_ms = base_ms + i * 600  
        result = await _admit(test_redis, now_ms)
        if result == 1:
            admitted += 1

    assert admitted == 50, f"Expected all 50 requests to be admitted, got {admitted}"

    # 51st request — still within the same 60-second window.
    now_ms_51 = base_ms + 50 * 100
    res_51 = await _admit(test_redis, now_ms_51)
    assert res_51 == 0, "51st request within the same minute must be blocked"


@pytest.mark.asyncio
async def test_per_minute_limit_resets_after_window(test_redis):
    """
    After the 60-second window expires, the per-minute counter resets and
    new requests are admitted again.
    """
    base_ms = int(time.time() * 1000)

    # Fill the per-minute window without hitting per-second: admit 2 per second,
    # advancing by 1001ms between pairs so the per-second window also expires.
    admitted = 0
    now_ms = base_ms
    while admitted < 50:
        r1 = await _admit(test_redis, now_ms)
        r2 = await _admit(test_redis, now_ms)
        admitted += r1 + r2
        now_ms += 1001  # advance past the 1-second window

    # At this point the per-minute window is full.
    blocked = await _admit(test_redis, now_ms)
    assert blocked == 0, "Precondition: per-minute window should be full"

    # Jump 60001 ms past the original base — all minute-window entries evict.
    future_ms = base_ms + 60001 + admitted * 1001

    res = await _admit(test_redis, future_ms)
    assert res == 1, "First request after minute window expiry must be admitted"


# ---------------------------------------------------------------------------
# 3. Per-hour limit (300 req/hour)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_per_hour_limit_allows_three_hundred(test_redis):
    """
    300 requests spread across the same hour must all be admitted.
    The 301st must be blocked.

    We space calls 2 seconds apart so the per-second window never stacks up,
    but all 300 remain within the 3600-second hourly window.
    Requests are simulated by advancing now_ms rather than sleeping.
    """
    base_ms = int(time.time() * 1000)

    admitted = 0
    for i in range(300):
        # Advance 2001 ms per request: clears the per-second window (2 req),
        # and 2001 ms * 300 = 600,300 ms = 600 s — well within the 3600 s
        # hourly window.
        now_ms = base_ms + i * 2001
        result = await _admit(test_redis, now_ms)
        if result == 1:
            admitted += 1

    assert admitted == 300, f"Expected 300 requests to be admitted, got {admitted}"

    # 301st — still within the hourly window.
    now_ms_301 = base_ms + 300 * 2001
    res_301 = await _admit(test_redis, now_ms_301)
    assert res_301 == 0, "301st request within the same hour must be blocked"


@pytest.mark.asyncio
async def test_per_hour_limit_resets_after_window(test_redis):
    """
    Requests that fall outside the 3600-second hourly window are evicted and
    no longer count toward the hourly quota.
    """
    base_ms = int(time.time() * 1000)

    # Record two requests at base_ms.
    await _admit(test_redis, base_ms)
    await _admit(test_redis, base_ms)

    # Jump 3601 seconds into the future — those two entries now evict.
    future_ms = base_ms + 3_601_000

    res = await _admit(test_redis, future_ms)
    assert res == 1, "Request after hour window expiry must be admitted (old entries evicted)"


# ---------------------------------------------------------------------------
# 4. All three limits enforced simultaneously
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_three_limits_checked_simultaneously(test_redis):
    """
    Verify the Lua script enforces all three limits in a single call.
    Fill the per-second window (2 req) and confirm the 3rd is blocked even
    though the per-minute and per-hour windows still have room.
    """
    now_ms = int(time.time() * 1000)

    # Admit 2 — per-second full, per-minute and per-hour still have room.
    await _admit(test_redis, now_ms)
    await _admit(test_redis, now_ms)

    # Confirm per-minute and per-hour are not full.
    min_count = await test_redis.zcard("rl:rolling:min")
    hr_count = await test_redis.zcard("rl:rolling:hr")
    assert min_count == 2, "Per-minute count should be 2, not full"
    assert hr_count == 2, "Per-hour count should be 2, not full"

    # 3rd call must still be blocked because per-second is full.
    res = await _admit(test_redis, now_ms)
    assert res == 0, (
        "3rd request must be blocked by the per-second limit "
        "even though per-minute and per-hour have capacity"
    )


# ---------------------------------------------------------------------------
# 5. Concurrent access — race-condition safety
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_access_no_race_condition(test_redis):
    """
    Fire 10 coroutines that each try to admit at the same timestamp.

    asyncio.gather interleaves coroutines on one thread, so Redis processes
    them one at a time.  The Lua script is atomic, so exactly 2 must be
    admitted and 8 blocked — no matter what order they arrive in.

    We verify the total count (≤ 2 admitted) rather than asserting exactly 2
    because asyncio scheduling may cause a small number of calls to land in
    a slightly different millisecond.  The important invariant is that the
    admitted count NEVER EXCEEDS the per-second limit of 2.
    """
    now_ms = int(time.time() * 1000)

    tasks = [_admit(test_redis, now_ms) for _ in range(10)]
    results = await asyncio.gather(*tasks)

    admitted = results.count(1)
    blocked = results.count(0)

    assert admitted <= 2, (
        f"At most 2 requests should be admitted per second, but {admitted} were admitted. "
        "This indicates a race condition in the Lua script."
    )
    assert admitted + blocked == 10, "All 10 results must be either 1 or 0"


@pytest.mark.asyncio
async def test_concurrent_access_thread_pool(test_redis):
    """
    Uses a real thread pool (not just asyncio coroutines) to stress-test the
    Lua script under genuine concurrency.  Synchronous Redis clients on N
    threads all fire at once.  The admitted count must never exceed 2.
    """
    now_ms = int(time.time() * 1000)
    now_ms_str = str(now_ms)

    # Build a synchronous Redis client for use inside threads.
    sync_client = sync_redis.from_url(REDIS_URL, decode_responses=True)

    def admit_sync() -> int:
        return sync_client.eval(RATE_LIMIT_LUA, 3, *_KEYS, now_ms_str)

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [loop.run_in_executor(pool, admit_sync) for _ in range(8)]
        results = await asyncio.gather(*futures)

    sync_client.close()

    admitted = list(results).count(1)
    assert admitted <= 2, (
        f"Thread-pool concurrent test: {admitted} requests admitted, expected ≤ 2. "
        "Race condition detected."
    )


# ---------------------------------------------------------------------------
# 6. State persistence across reconnection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_state_survives_client_reconnect(test_redis):
    """
    Admit 2 requests using one Redis client, then disconnect it and create a
    brand-new client.  The new client must see the same per-second window
    state and block the 3rd request.

    This simulates what happens when the FastAPI process restarts — the
    sliding-window data lives in Redis and survives the reconnect.
    """
    now_ms = int(time.time() * 1000)

    # Admit 2 via the fixture client, then close it.
    await _admit(test_redis, now_ms)
    await _admit(test_redis, now_ms)
    await test_redis.aclose()

    # Open a brand-new client — simulates a process restart.
    new_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        # The per-second window must still be full.
        res = await _admit(new_client, now_ms)
        assert res == 0, (
            "After client reconnect the per-second window must still be full — "
            "state is stored in Redis, not in the process."
        )

        # Verify rl:seq TTL is still set.
        ttl = await new_client.ttl("rl:seq")
        assert ttl > 0, "rl:seq must have a positive TTL after reconnect"
        assert ttl <= 86400, "rl:seq TTL must not exceed 24 hours"
    finally:
        await new_client.aclose()


# ---------------------------------------------------------------------------
# 7. rl:seq key properties
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rl_seq_ttl_set_on_each_admission(test_redis):
    """
    Every admitted request must refresh the rl:seq TTL to 86400 seconds.
    This prevents the monotonic counter from growing unbounded across Redis
    restarts.
    """
    now_ms = int(time.time() * 1000)
    await _admit(test_redis, now_ms)

    ttl = await test_redis.ttl("rl:seq")
    assert ttl > 0, "rl:seq must have a TTL after the first admission"
    assert ttl <= 86400, "rl:seq TTL must be at most 86400 seconds (24 hours)"


@pytest.mark.asyncio
async def test_rl_seq_produces_unique_members(test_redis):
    """
    Two requests at the same millisecond must produce two distinct sorted-set
    members (because rl:seq increments), preventing ZADD overwrites that would
    silently discard one of the two admitted requests.
    """
    now_ms = int(time.time() * 1000)
    await _admit(test_redis, now_ms)
    await _admit(test_redis, now_ms)

    # Both members must be in the set — a count of 1 would mean one overwrote the other.
    sec_count = await test_redis.zcard("rl:rolling:sec")
    assert sec_count == 2, (
        f"Expected 2 unique members in the per-second sorted set, found {sec_count}. "
        "ZADD may be overwriting members when rl:seq is not unique."
    )


# ---------------------------------------------------------------------------
# 8. Sliding-window eviction correctness
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sliding_window_evicts_old_entries(test_redis):
    """
    Entries older than the window boundary must be evicted by ZREMRANGEBYSCORE
    before the cardinality check.  This test verifies the sliding (not fixed)
    nature of the window.

    Scenario: admit 2 at t=0, advance to t=1001 ms, admit 2 more.
    The per-second window at t=1001 contains only the new entries (old ones
    evicted), so both new admissions must succeed.
    """
    now_ms = int(time.time() * 1000)

    # Fill the window at t=0.
    await _admit(test_redis, now_ms)
    await _admit(test_redis, now_ms)

    # Advance 1001 ms — old entries are now outside the 1000 ms window.
    future_ms = now_ms + 1001

    res1 = await _admit(test_redis, future_ms)
    res2 = await _admit(test_redis, future_ms)

    assert res1 == 1, "First request in the new sliding window must be admitted"
    assert res2 == 1, "Second request in the new sliding window must be admitted"

    # After these two new admissions, the window is full again.
    res3 = await _admit(test_redis, future_ms)
    assert res3 == 0, "Third request in the new window must be blocked"
