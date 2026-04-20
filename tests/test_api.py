"""
test_api.py
===========
Tests for API contract correctness and the <200 ms response-time requirement.

Assignment requirements covered:
  1. GET /funds                    — response time < 200 ms
  2. GET /funds/rank               — response time < 200 ms
  3. GET /funds/{code}             — response time < 200 ms
  4. GET /funds/{code}/analytics   — response time < 200 ms (the key endpoint)
  5. Correct HTTP status codes and response shape for each endpoint

Design notes:
  - All timed tests use the `api_client` fixture from conftest.py, which is
    pre-warmed (one warm-up request already issued) and backed by a seeded DB.
    This means we measure real handler + query latency, not startup costs.
  - Timing measurements use time.perf_counter() — the highest-resolution
    timer available in Python — and are taken as tightly around client.get()
    as possible.
  - We run each timed call 3 times and assert on the median to smooth out
    occasional OS scheduling jitter.
  - Status-code tests use a separate lightweight client so they don't
    depend on the seeded DB being in a particular state.
"""

import statistics
import time

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from main import app
from conftest import _FUND_CODES  # the two seeded scheme codes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _timed_get(client: AsyncClient, url: str, runs: int = 3) -> tuple[int, float]:
    """
    Issue `runs` GET requests to `url`, return (status_code, median_ms).
    We always use the median to reduce sensitivity to scheduling jitter.
    """
    latencies = []
    status_code = None
    for _ in range(runs):
        t0 = time.perf_counter()
        response = await client.get(url)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        status_code = response.status_code
    return status_code, statistics.median(latencies)


LIMIT_MS = 200.0  # assignment requirement


# ---------------------------------------------------------------------------
# 1. GET /funds — list endpoint
# ---------------------------------------------------------------------------

class TestGetFunds:

    @pytest.mark.asyncio
    async def test_funds_list_returns_200(self, api_client):
        response = await api_client.get("/funds")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_funds_list_response_shape(self, api_client):
        response = await api_client.get("/funds")
        body = response.json()
        assert "total" in body, "Response must contain 'total' field"
        assert "funds" in body, "Response must contain 'funds' field"
        assert isinstance(body["funds"], list)

    @pytest.mark.asyncio
    async def test_funds_list_contains_seeded_funds(self, api_client):
        response = await api_client.get("/funds")
        body = response.json()
        codes = {f["fund_code"] for f in body["funds"]}
        for code in _FUND_CODES:
            assert code in codes, f"Seeded fund {code} must appear in /funds response"

    @pytest.mark.asyncio
    async def test_funds_list_category_filter(self, api_client):
        response = await api_client.get("/funds?category=Mid+Cap")
        assert response.status_code == 200
        body = response.json()
        for fund in body["funds"]:
            assert "Mid Cap" in fund["category"], (
                f"Fund {fund['fund_code']} does not match category filter"
            )

    @pytest.mark.asyncio
    async def test_funds_list_amc_filter(self, api_client):
        response = await api_client.get("/funds?amc=Axis")
        assert response.status_code == 200
        body = response.json()
        for fund in body["funds"]:
            assert "Axis" in fund["amc"], (
                f"Fund {fund['fund_code']} does not match AMC filter"
            )

    @pytest.mark.asyncio
    async def test_funds_list_response_time(self, api_client):
        """GET /funds must respond in < 200 ms (median of 3 runs)."""
        status, median_ms = await _timed_get(api_client, "/funds")
        assert status == 200
        assert median_ms < LIMIT_MS, (
            f"GET /funds took {median_ms:.1f} ms (median), exceeds {LIMIT_MS} ms limit"
        )


# ---------------------------------------------------------------------------
# 2. GET /funds/{code} — fund detail endpoint
# ---------------------------------------------------------------------------

class TestGetFundDetail:

    @pytest.mark.asyncio
    async def test_known_fund_returns_200(self, api_client):
        response = await api_client.get(f"/funds/{_FUND_CODES[0]}")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_unknown_fund_returns_404(self, api_client):
        response = await api_client.get("/funds/999999999")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_fund_detail_response_shape(self, api_client):
        response = await api_client.get(f"/funds/{_FUND_CODES[0]}")
        body = response.json()
        required_fields = ["fund_code", "fund_name", "amc", "category"]
        for field in required_fields:
            assert field in body, f"Field '{field}' missing from /funds/{{code}} response"

    @pytest.mark.asyncio
    async def test_fund_detail_code_matches_request(self, api_client):
        code = _FUND_CODES[0]
        response = await api_client.get(f"/funds/{code}")
        body = response.json()
        assert body["fund_code"] == code

    @pytest.mark.asyncio
    async def test_fund_detail_response_time(self, api_client):
        """GET /funds/{code} must respond in < 200 ms (median of 3 runs)."""
        url = f"/funds/{_FUND_CODES[0]}"
        status, median_ms = await _timed_get(api_client, url)
        assert status == 200
        assert median_ms < LIMIT_MS, (
            f"GET /funds/{{code}} took {median_ms:.1f} ms (median), exceeds {LIMIT_MS} ms limit"
        )


# ---------------------------------------------------------------------------
# 3. GET /funds/{code}/analytics — analytics endpoint
# ---------------------------------------------------------------------------

class TestGetFundAnalytics:

    @pytest.mark.asyncio
    async def test_analytics_returns_200(self, api_client):
        url = f"/funds/{_FUND_CODES[0]}/analytics?window=3Y"
        response = await api_client.get(url)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_analytics_unknown_fund_returns_404(self, api_client):
        response = await api_client.get("/funds/999999999/analytics?window=1Y")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_analytics_missing_window_returns_422(self, api_client):
        """window query param is required; omitting it must return 422."""
        response = await api_client.get(f"/funds/{_FUND_CODES[0]}/analytics")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analytics_invalid_window_returns_422(self, api_client):
        """Invalid window values must be rejected with 422."""
        response = await api_client.get(f"/funds/{_FUND_CODES[0]}/analytics?window=2Y")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_analytics_response_shape(self, api_client):
        response = await api_client.get(f"/funds/{_FUND_CODES[0]}/analytics?window=3Y")
        body = response.json()
        required_fields = [
            "fund_code", "fund_name", "category", "amc", "window",
            "status", "data_availability", "rolling_periods_analyzed",
            "computed_at",
        ]
        for field in required_fields:
            assert field in body, f"Field '{field}' missing from analytics response"

    @pytest.mark.asyncio
    async def test_analytics_rolling_returns_has_quartiles(self, api_client):
        response = await api_client.get(f"/funds/{_FUND_CODES[0]}/analytics?window=3Y")
        body = response.json()
        if body["status"] == "SUCCESS":
            rr = body["rolling_returns"]
            assert "p25" in rr, "rolling_returns must include p25"
            assert "p75" in rr, "rolling_returns must include p75"

    @pytest.mark.asyncio
    async def test_analytics_all_windows(self, api_client):
        """All four window values must return 200 for a seeded fund."""
        for window in ["1Y", "3Y", "5Y", "10Y"]:
            response = await api_client.get(
                f"/funds/{_FUND_CODES[0]}/analytics?window={window}"
            )
            assert response.status_code == 200, (
                f"Window {window} returned {response.status_code}, expected 200"
            )

    @pytest.mark.asyncio
    async def test_analytics_response_time_1y(self, api_client):
        """GET /funds/{code}/analytics?window=1Y must respond in < 200 ms."""
        url = f"/funds/{_FUND_CODES[0]}/analytics?window=1Y"
        status, median_ms = await _timed_get(api_client, url)
        assert status == 200
        assert median_ms < LIMIT_MS, (
            f"analytics 1Y took {median_ms:.1f} ms (median), exceeds {LIMIT_MS} ms limit"
        )

    @pytest.mark.asyncio
    async def test_analytics_response_time_3y(self, api_client):
        """GET /funds/{code}/analytics?window=3Y must respond in < 200 ms."""
        url = f"/funds/{_FUND_CODES[0]}/analytics?window=3Y"
        status, median_ms = await _timed_get(api_client, url)
        assert status == 200
        assert median_ms < LIMIT_MS, (
            f"analytics 3Y took {median_ms:.1f} ms (median), exceeds {LIMIT_MS} ms limit"
        )

    @pytest.mark.asyncio
    async def test_analytics_response_time_10y(self, api_client):
        """GET /funds/{code}/analytics?window=10Y must respond in < 200 ms."""
        url = f"/funds/{_FUND_CODES[0]}/analytics?window=10Y"
        status, median_ms = await _timed_get(api_client, url)
        assert status == 200
        assert median_ms < LIMIT_MS, (
            f"analytics 10Y took {median_ms:.1f} ms (median), exceeds {LIMIT_MS} ms limit"
        )


# ---------------------------------------------------------------------------
# 4. GET /funds/rank — ranking endpoint
# ---------------------------------------------------------------------------

class TestGetFundsRank:

    @pytest.mark.asyncio
    async def test_rank_returns_200(self, api_client):
        response = await api_client.get(
            "/funds/rank?category=Mid+Cap&window=3Y&sort_by=median_return"
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_rank_missing_category_returns_422(self, api_client):
        response = await api_client.get("/funds/rank?window=3Y")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_rank_missing_window_returns_422(self, api_client):
        response = await api_client.get("/funds/rank?category=Mid+Cap")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_rank_invalid_window_returns_422(self, api_client):
        response = await api_client.get("/funds/rank?category=Mid+Cap&window=2Y")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_rank_invalid_sort_by_returns_422(self, api_client):
        response = await api_client.get(
            "/funds/rank?category=Mid+Cap&window=3Y&sort_by=invalid"
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_rank_response_shape(self, api_client):
        response = await api_client.get(
            "/funds/rank?category=Mid+Cap&window=3Y&sort_by=median_return"
        )
        body = response.json()
        required_fields = ["category", "window", "sorted_by", "total_funds", "showing", "funds"]
        for field in required_fields:
            assert field in body, f"Field '{field}' missing from /funds/rank response"

    @pytest.mark.asyncio
    async def test_rank_sorted_by_median_return_descending(self, api_client):
        """Results must be sorted by median_return descending."""
        response = await api_client.get(
            "/funds/rank?category=Mid+Cap&window=3Y&sort_by=median_return&limit=50"
        )
        body = response.json()
        funds = body["funds"]
        if len(funds) < 2:
            pytest.skip("Need at least 2 funds to verify sort order")
        returns = [f["metrics"].get(f"median_return_3y", 0) for f in funds]
        assert returns == sorted(returns, reverse=True), (
            "Funds must be sorted by median_return descending"
        )

    @pytest.mark.asyncio
    async def test_rank_sorted_by_max_drawdown_ascending(self, api_client):
        """
        When sort_by=max_drawdown, results are sorted ascending (least negative
        drawdown first = best performing fund).
        """
        response = await api_client.get(
            "/funds/rank?category=Mid+Cap&window=3Y&sort_by=max_drawdown&limit=50"
        )
        body = response.json()
        funds = body["funds"]
        if len(funds) < 2:
            pytest.skip("Need at least 2 funds to verify sort order")
        drawdowns = [f["metrics"].get("max_drawdown_3y", 0) for f in funds]
        assert drawdowns == sorted(drawdowns), (
            "Funds must be sorted by max_drawdown ascending"
        )

    @pytest.mark.asyncio
    async def test_rank_limit_respected(self, api_client):
        response = await api_client.get(
            "/funds/rank?category=Mid+Cap&window=3Y&sort_by=median_return&limit=1"
        )
        body = response.json()
        assert len(body["funds"]) <= 1, "limit=1 must return at most 1 fund"

    @pytest.mark.asyncio
    async def test_rank_response_time_median_return(self, api_client):
        """GET /funds/rank (median_return sort) must respond in < 200 ms."""
        url = "/funds/rank?category=Mid+Cap&window=3Y&sort_by=median_return"
        status, median_ms = await _timed_get(api_client, url)
        assert status == 200
        assert median_ms < LIMIT_MS, (
            f"GET /funds/rank took {median_ms:.1f} ms (median), exceeds {LIMIT_MS} ms limit"
        )

    @pytest.mark.asyncio
    async def test_rank_response_time_max_drawdown(self, api_client):
        """GET /funds/rank (max_drawdown sort) must respond in < 200 ms."""
        url = "/funds/rank?category=Mid+Cap&window=3Y&sort_by=max_drawdown"
        status, median_ms = await _timed_get(api_client, url)
        assert status == 200
        assert median_ms < LIMIT_MS, (
            f"GET /funds/rank (drawdown sort) took {median_ms:.1f} ms, exceeds {LIMIT_MS} ms limit"
        )

    @pytest.mark.asyncio
    async def test_rank_route_not_swallowed_by_fund_detail_route(self, api_client):
        """
        /funds/rank must resolve as the rank endpoint, not be mistaken for
        /funds/{code} with code='rank' (which would be a 422 or 404).
        """
        response = await api_client.get(
            "/funds/rank?category=Mid+Cap&window=3Y"
        )
        # If routing is broken, FastAPI tries to parse 'rank' as an integer
        # and returns 422. A 200 proves the route is wired correctly.
        assert response.status_code == 200, (
            f"/funds/rank returned {response.status_code}; "
            "route may have been swallowed by /funds/{{code}}"
        )


# ---------------------------------------------------------------------------
# 5. GET /sync/status — pipeline status endpoint
# ---------------------------------------------------------------------------

class TestSyncStatus:

    @pytest.mark.asyncio
    async def test_sync_status_returns_200(self, api_client):
        response = await api_client.get("/sync/status")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_sync_status_response_shape(self, api_client):
        response = await api_client.get("/sync/status")
        body = response.json()
        assert "status" in body, "Response must contain 'status' field"
