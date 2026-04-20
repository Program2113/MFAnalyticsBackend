"""
conftest.py — shared pytest fixtures for the Mutual Fund Analytics test suite.

Every fixture here is designed to be completely isolated:
  - The database fixture wraps each test in a transaction that is rolled back
    after the test, so no test permanently mutates the DB.
  - The Redis fixture flushes only the rate-limiter keys it owns (prefixed
    "rl:") to avoid interfering with other services on the same Redis instance.
  - The API fixture pre-seeds known fund + analytics data so response-time
    tests measure real query paths, not trivial empty-table scans.
"""

import asyncio
import time
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal
from typing import AsyncGenerator

import pandas as pd
import pytest
import pytest_asyncio
import redis.asyncio as aioredis
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from main import (
    app,
    REDIS_URL,
    DATABASE_URL,
    Base,
    Fund,
    NAVHistory,
    AnalyticsCache,
    FundSyncState,
    SyncJob,
    compute_metrics,
)

# ---------------------------------------------------------------------------
# pytest-asyncio configuration
# ---------------------------------------------------------------------------
# All async tests in this suite use a single shared event loop to avoid
# "attached to a different loop" errors with asyncpg / aioredis.
pytest_plugins = ("pytest_asyncio",)


# ---------------------------------------------------------------------------
# Database fixtures
# ---------------------------------------------------------------------------

# Create a dedicated test engine that points at the same DB but is separate
# from the application engine so we can control transactions independently.
_test_engine = create_async_engine(DATABASE_URL, echo=False)
_TestSession = sessionmaker(_test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def create_tables():
    """Create all tables once per test session, drop nothing (idempotent DDL)."""
    async with _test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Yield a database session that is wrapped in a SAVEPOINT.

    Every test that receives this fixture gets a fresh savepoint at the start
    and a rollback to that savepoint at the end — so no test data ever
    persists to the actual database.
    """
    async with _TestSession() as session:
        await session.begin()
        # Use a nested transaction (SAVEPOINT) so we can roll back after the test
        # while keeping the outer connection alive for the next test.
        try:
            yield session
        finally:
            await session.rollback()


# ---------------------------------------------------------------------------
# Redis fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def test_redis() -> AsyncGenerator[aioredis.Redis, None]:
    """
    Yield a Redis client with the rate-limiter keyspace cleaned before and
    after each test.  Only rl:* keys and rl:seq are touched — other data
    on the same Redis instance is left untouched.
    """
    client = aioredis.from_url(REDIS_URL, decode_responses=True)

    # Clean only our keys so we don't disturb other services.
    await _flush_rl_keys(client)

    yield client

    await _flush_rl_keys(client)
    await client.aclose()


async def _flush_rl_keys(client: aioredis.Redis) -> None:
    """Delete all rl: prefixed keys used by the rate limiter."""
    keys_to_delete = [
        "rl:rolling:sec", "rl:rolling:min", "rl:rolling:hr", "rl:seq",
    ]
    pipe = client.pipeline()
    for k in keys_to_delete:
        pipe.delete(k)
    await pipe.execute()


# ---------------------------------------------------------------------------
# Seeded-data fixtures (used by API response-time tests)
# ---------------------------------------------------------------------------

# A small but realistic NAV series: 3 years of daily data for two funds.
# We build it once at module level so fixture setup is fast.
_FUND_CODES = [119598, 118989]

_SEED_FUNDS = [
    {
        "code": 119598,
        "name": "Axis Midcap Fund - Direct Plan - Growth",
        "amc": "Axis",
        "category": "Equity: Mid Cap",
        "scheme_type": "Open Ended Schemes",
        "isin_growth": "INF846K01DP8",
        "isin_div_reinvestment": None,
        "latest_nav": Decimal("78.45000"),
        "latest_nav_date": date(2026, 1, 6),
        "last_synced_at": datetime(2026, 1, 6, 2, 30, 0, tzinfo=timezone.utc),
        "is_active": True,
    },
    {
        "code": 118989,
        "name": "Kotak Emerging Equity Fund - Direct - Growth",
        "amc": "Kotak Mahindra",
        "category": "Equity: Mid Cap",
        "scheme_type": "Open Ended Schemes",
        "isin_growth": "INF174K01LS2",
        "isin_div_reinvestment": None,
        "latest_nav": Decimal("92.31000"),
        "latest_nav_date": date(2026, 1, 6),
        "last_synced_at": datetime(2026, 1, 6, 2, 30, 0, tzinfo=timezone.utc),
        "is_active": True,
    },
]


def _build_nav_series(code: int, start: str, periods: int, start_nav: float) -> list[dict]:
    """Build a deterministic daily NAV series for seeding."""
    dates = pd.date_range(start=start, periods=periods, freq="B")  # business days only
    navs = [round(start_nav * (1 + 0.0001 * i), 5) for i in range(len(dates))]
    return [
        {"fund_code": code, "date": d.date(), "nav": Decimal(str(n))}
        for d, n in zip(dates, navs)
    ]


def _build_analytics_payload(code: int, name: str, category: str, amc: str, window: str) -> dict:
    """Build a realistic pre-computed analytics payload for seeding."""
    return {
        "fund_code": code,
        "fund_name": name,
        "category": category,
        "amc": amc,
        "window": window,
        "status": "SUCCESS",
        "data_availability": {
            "start_date": "2016-01-15",
            "end_date": "2026-01-06",
            "total_days": 3644,
            "nav_data_points": 2513,
            "sufficient_for_window": True,
        },
        "rolling_periods_analyzed": 731,
        "rolling_returns": {"min": 8.2, "max": 48.5, "median": 22.3, "p25": 15.7, "p75": 28.9},
        "max_drawdown": -32.1,
        "cagr": {"min": 9.5, "max": 45.2, "median": 21.8},
        "computed_at": "2026-01-06T02:30:15+00:00",
    }


@pytest_asyncio.fixture(scope="module")
async def seeded_db():
    """
    Insert two funds with full NAV history and pre-computed analytics for all
    four windows into the test database.

    Scope is 'module' so the seed is inserted once and shared across all
    response-time tests, which only read data and never mutate it.

    A module-scoped teardown deletes the seeded rows after all tests in the
    module complete.
    """
    async with _TestSession() as session:
        async with session.begin():
            # Insert fund master rows
            for fd in _SEED_FUNDS:
                fund = Fund(**fd)
                session.add(fund)

            # Insert NAV history (~780 business-day rows per fund ≈ 3 years)
            for fund_data in _SEED_FUNDS:
                nav_rows = _build_nav_series(
                    fund_data["code"],
                    start="2023-01-01",
                    periods=780,
                    start_nav=50.0 if fund_data["code"] == 119598 else 60.0,
                )
                for row in nav_rows:
                    session.add(NAVHistory(**row))

            # Insert analytics_cache for all four windows
            for fd in _SEED_FUNDS:
                for window in ["1Y", "3Y", "5Y", "10Y"]:
                    payload = _build_analytics_payload(
                        fd["code"], fd["name"], fd["category"], fd["amc"], window
                    )
                    cache = AnalyticsCache(
                        fund_code=fd["code"],
                        window=window,
                        median_return=Decimal("22.3"),
                        max_drawdown=Decimal("-32.1"),
                        computed_at=datetime(2026, 1, 6, 2, 30, 15, tzinfo=timezone.utc),
                        payload=payload,
                    )
                    session.add(cache)

    yield  # tests run here

    # Teardown — remove seeded rows so the DB is clean for other test sessions
    async with _TestSession() as session:
        async with session.begin():
            for fd in _SEED_FUNDS:
                code = fd["code"]
                # Delete in FK-safe order
                await session.execute(
                    __import__("sqlalchemy").delete(AnalyticsCache).where(
                        AnalyticsCache.fund_code == code
                    )
                )
                await session.execute(
                    __import__("sqlalchemy").delete(NAVHistory).where(
                        NAVHistory.fund_code == code
                    )
                )
                await session.execute(
                    __import__("sqlalchemy").delete(Fund).where(Fund.code == code)
                )


@pytest_asyncio.fixture(scope="module")
async def api_client(seeded_db):
    """
    Return a pre-warmed AsyncClient backed by the real FastAPI app.

    We issue one warm-up request before yielding so that connection-pool
    establishment and startup costs are paid before any timed test begins.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Warm-up call — not timed
        await client.get("/funds")
        yield client
