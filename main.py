"""
Mutual Fund Analytics Platform
================================
FastAPI backend that ingests NAV data from mfapi.in, computes rolling
performance analytics, and serves ranking/analytics queries.

Rate limits enforced (all three simultaneously via Redis Lua):
  - 2 requests / second
  - 50 requests / minute
  - 300 requests / hour
"""

import asyncio
import logging
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional, Tuple
from contextlib import asynccontextmanager

import httpx
import numpy as np
import pandas as pd
import redis.asyncio as aioredis
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATABASE_URL = "postgresql+asyncpg://mf_user:mf_password@localhost:5432/mf_analytics"
REDIS_URL = "redis://localhost:6379"
MFAPI_BASE_URL = "https://api.mfapi.in/mf"

# Analytics windows: label -> years
WINDOWS: Dict[str, int] = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}

TARGET_AMCS = ["ICICI Prudential", "HDFC", "Axis", "SBI", "Kotak Mahindra"]
TARGET_CATEGORIES = {
    "Equity: Mid Cap": ["MIDCAP", "MID CAP"],
    "Equity: Small Cap": ["SMALLCAP", "SMALL CAP"],
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DATABASE SETUP
# ---------------------------------------------------------------------------
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM MODELS
# ---------------------------------------------------------------------------
class Fund(Base):
    __tablename__ = "funds"

    code = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    amc = Column(String, nullable=False)
    category = Column(String, nullable=False)
    scheme_type = Column(String, nullable=True)
    isin_growth = Column(String, nullable=True)
    isin_div_reinvestment = Column(String, nullable=True)
    latest_nav = Column(Numeric(15, 5), nullable=True)
    latest_nav_date = Column(Date, nullable=True)
    last_synced_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    __table_args__ = (
        # Speeds up /funds?category=... and /funds/rank category filter
        Index("ix_fund_category_active", "category", "is_active"),
    )


class NAVHistory(Base):
    __tablename__ = "nav_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fund_code = Column(Integer, index=True, nullable=False)
    date = Column(Date, index=True, nullable=False)
    nav = Column(Numeric(15, 5), nullable=False)

    __table_args__ = (UniqueConstraint("fund_code", "date", name="uq_nav_fund_date"),)


class AnalyticsCache(Base):
    __tablename__ = "analytics_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fund_code = Column(Integer, index=True, nullable=False)
    window = Column(String, nullable=False)
    median_return = Column(Numeric, nullable=True)
    max_drawdown = Column(Numeric, nullable=True)
    computed_at = Column(DateTime(timezone=True), nullable=False)
    payload = Column(JSON, nullable=False)

    __table_args__ = (
        UniqueConstraint("fund_code", "window", name="uq_analytics_fund_window"),
        # Speeds up /funds/rank sort_by=median_return
        Index("ix_analytics_window_median", "window", "median_return"),
        # Speeds up /funds/rank sort_by=max_drawdown
        Index("ix_analytics_window_drawdown", "window", "max_drawdown"),
    )


class FundSyncState(Base):
    __tablename__ = "fund_sync_state"

    fund_code = Column(Integer, primary_key=True)
    sync_state = Column(String, nullable=False, default="PENDING")
    last_nav_date = Column(Date, nullable=True)
    last_backfill_at = Column(DateTime(timezone=True), nullable=True)
    last_analytics_at = Column(DateTime(timezone=True), nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    last_error = Column(String, nullable=True)
    last_job_id = Column(Integer, nullable=True, index=True)


class SyncJob(Base):
    __tablename__ = "sync_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    status = Column(String, nullable=False, default="PENDING")
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    current_fund_code = Column(Integer, nullable=True)
    processed_funds = Column(Integer, nullable=False, default=0)
    failed_funds = Column(Integer, nullable=False, default=0)
    discovered_codes = Column(JSON, nullable=False, default=list)
    last_error = Column(String, nullable=True)


# ---------------------------------------------------------------------------
# PYDANTIC — EXTERNAL API MODELS (mfapi.in)
# ---------------------------------------------------------------------------

class MfapiSchemeListItem(BaseModel):
    """
    Represents one entry from GET /mf (the full scheme list).

    The actual API returns camelCase keys. We capture isinGrowth and
    isinDivReinvestment here so that discovery can pre-populate ISIN data
    without an extra per-scheme API call.
    """

    schemeCode: int
    schemeName: str
    isinGrowth: Optional[str] = None
    isinDivReinvestment: Optional[str] = None


class MfapiNAVData(BaseModel):
    """Single NAV data point returned inside NAVHistoryResponse.data[]."""

    date: str   # DD-MM-YYYY
    nav: str    # "52.34500"


class MfapiSchemeMeta(BaseModel):
    """
    Metadata block embedded in GET /mf/{code} (NAV history) responses.
    Field names use snake_case as returned by the API.
    """

    fund_house: Optional[str] = None
    scheme_type: Optional[str] = None
    scheme_category: Optional[str] = None
    scheme_code: int
    scheme_name: str
    isin_growth: Optional[str] = None
    isin_div_reinvestment: Optional[str] = None


class MfapiNAVHistoryResponse(BaseModel):
    """Response shape for GET /mf/{scheme_code}."""

    meta: MfapiSchemeMeta
    data: List[MfapiNAVData]
    status: Literal["SUCCESS", "ERROR"]


class MfapiLatestNAVItem(BaseModel):
    """
    Response shape for GET /mf/{scheme_code}/latest.

    The /latest endpoint returns a *flat* JSON object (NOT the
    {meta, data[], status} envelope used by the full history endpoint).
    Field names are camelCase here, unlike the snake_case meta block in
    NAVHistoryResponse.

    Property accessors expose a consistent snake_case interface so that
    verified_target_key() and fund-population code can treat both
    MfapiSchemeMeta and MfapiLatestNAVItem uniformly.
    """

    schemeCode: int
    schemeName: str
    fundHouse: Optional[str] = None
    schemeType: Optional[str] = None
    schemeCategory: Optional[str] = None
    isinGrowth: Optional[str] = None
    isinDivReinvestment: Optional[str] = None
    nav: str    # "52.34500"
    date: str   # "18-04-2026" (DD-MM-YYYY)

    # --- snake_case compatibility properties ---

    @property
    def scheme_code(self) -> int:
        return self.schemeCode

    @property
    def scheme_name(self) -> str:
        return self.schemeName

    @property
    def fund_house(self) -> Optional[str]:
        return self.fundHouse

    @property
    def scheme_type(self) -> Optional[str]:
        return self.schemeType

    @property
    def scheme_category(self) -> Optional[str]:
        return self.schemeCategory

    @property
    def isin_growth(self) -> Optional[str]:
        return self.isinGrowth

    @property
    def isin_div_reinvestment(self) -> Optional[str]:
        return self.isinDivReinvestment


# ---------------------------------------------------------------------------
# PYDANTIC — INTERNAL API RESPONSE MODELS
# ---------------------------------------------------------------------------

class LatestNAVOut(BaseModel):
    nav: str
    date: str   # ISO 8601: YYYY-MM-DD


class FundListItemOut(BaseModel):
    fund_code: int
    fund_name: str
    amc: str
    category: str
    latest_nav: Optional[LatestNAVOut] = None


class FundDetailsOut(BaseModel):
    fund_code: int
    fund_name: str
    amc: str
    category: str
    scheme_type: Optional[str] = None
    isin_growth: Optional[str] = None
    isin_div_reinvestment: Optional[str] = None
    latest_nav: Optional[LatestNAVOut] = None
    last_synced_at: Optional[str] = None


class DataAvailabilityOut(BaseModel):
    start_date: str         # ISO 8601: YYYY-MM-DD
    end_date: str           # ISO 8601: YYYY-MM-DD
    total_days: int
    nav_data_points: int
    sufficient_for_window: bool


class DistributionOut(BaseModel):
    min: float
    max: float
    median: float
    p25: Optional[float] = None
    p75: Optional[float] = None


class AnalyticsResponseOut(BaseModel):
    fund_code: int
    fund_name: str
    category: str
    amc: str
    window: str
    status: Literal["SUCCESS", "INSUFFICIENT_HISTORY"]
    reason: Optional[str] = None
    data_availability: DataAvailabilityOut
    rolling_periods_analyzed: int = 0
    rolling_returns: Optional[DistributionOut] = None
    max_drawdown: Optional[float] = None
    cagr: Optional[DistributionOut] = None
    computed_at: str


class RankedFundOut(BaseModel):
    rank: int
    fund_code: int
    fund_name: str
    amc: str
    current_nav: Optional[float] = None
    last_updated: Optional[str] = None   # ISO 8601: YYYY-MM-DD
    metrics: Dict[str, float]


class RankResponseOut(BaseModel):
    category: str
    window: str
    sorted_by: str
    total_funds: int
    showing: int
    funds: List[RankedFundOut]


# ---------------------------------------------------------------------------
# REDIS RATE LIMITER
# ---------------------------------------------------------------------------
redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

# Atomic Lua script: checks all three sliding-window counters and, only if
# all limits have room, records the new request across all three windows in
# a single round-trip.  Returns 1 (allowed) or 0 (throttled).
#
# rl:seq is a global monotonic counter used to make ZADD members unique
# when multiple requests land in the same millisecond.  It is given a
# 24-hour TTL so it doesn't grow unbounded across Redis restarts.
RATE_LIMIT_LUA = """
local now_ms   = tonumber(ARGV[1])
local sec_key  = KEYS[1]
local min_key  = KEYS[2]
local hr_key   = KEYS[3]

redis.call('ZREMRANGEBYSCORE', sec_key, '-inf', now_ms - 1000)
redis.call('ZREMRANGEBYSCORE', min_key, '-inf', now_ms - 60000)
redis.call('ZREMRANGEBYSCORE', hr_key,  '-inf', now_ms - 3600000)

local sec_count = redis.call('ZCARD', sec_key)
local min_count = redis.call('ZCARD', min_key)
local hr_count  = redis.call('ZCARD', hr_key)

if sec_count >= 2 or min_count >= 50 or hr_count >= 300 then
    return 0
end

local seq    = redis.call('INCR', 'rl:seq')
local member = tostring(now_ms) .. '-' .. tostring(seq)

redis.call('ZADD',    sec_key, now_ms, member)
redis.call('PEXPIRE', sec_key, 2000)
redis.call('ZADD',    min_key, now_ms, member)
redis.call('PEXPIRE', min_key, 65000)
redis.call('ZADD',    hr_key,  now_ms, member)
redis.call('PEXPIRE', hr_key,  3605000)

-- Refresh seq TTL daily so it never grows unbounded
redis.call('EXPIRE', 'rl:seq', 86400)

return 1
"""


async def wait_for_rate_limit() -> None:
    """Block (with 200 ms polling) until the rate limiter grants a slot."""
    while True:
        now_ms = int(time.time() * 1000)
        allowed = await redis_client.eval(
            RATE_LIMIT_LUA,
            3,
            "rl:rolling:sec",
            "rl:rolling:min",
            "rl:rolling:hr",
            str(now_ms),
        )
        if allowed == 1:
            return
        await asyncio.sleep(0.2)


async def mfapi_get_json(client: httpx.AsyncClient, url: str) -> Any:
    """
    Fetch a URL, honouring the three-tier rate limit.
    Retries up to 5 times on HTTP 429 with exponential back-off.
    """
    attempts = 0
    while True:
        attempts += 1
        await wait_for_rate_limit()
        response = await client.get(url)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429 and attempts < 6:
            await asyncio.sleep(min(2 ** attempts, 10) + 0.1)
            continue
        response.raise_for_status()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def round2(v: float) -> float:
    return round(float(v), 2)


def compute_distribution(values: List[float], include_quartiles: bool) -> Dict[str, float]:
    s = pd.Series(values, dtype="float64")
    out: Dict[str, float] = {
        "min": round2(float(s.min())),
        "max": round2(float(s.max())),
        "median": round2(float(s.median())),
    }
    if include_quartiles:
        out["p25"] = round2(float(s.quantile(0.25)))
        out["p75"] = round2(float(s.quantile(0.75)))
    return out


def compute_max_drawdown(nav_series: pd.Series) -> float:
    """
    Returns the worst peak-to-trough decline as a negative percentage.
    E.g. -32.1 means the fund fell 32.1% from its previous peak.
    """
    cumulative_max = nav_series.cummax()
    drawdown = ((nav_series - cumulative_max) / cumulative_max) * 100.0
    return round2(float(drawdown.min()))


def build_data_availability(df: pd.DataFrame, sufficient_for_window: bool) -> Dict[str, Any]:
    start_dt = df.index[0].date()
    end_dt = df.index[-1].date()
    return {
        "start_date": start_dt.isoformat(),   # YYYY-MM-DD (ISO 8601)
        "end_date": end_dt.isoformat(),
        "total_days": (end_dt - start_dt).days,
        "nav_data_points": int(len(df)),
        "sufficient_for_window": sufficient_for_window,
    }


def compute_metrics(df: pd.DataFrame, window_years: int) -> Dict[str, Any]:
    """
    Vectorised computation of rolling returns, CAGR, and drawdown.

    For each trading day t we find the NAV exactly `window_years` years
    prior using binary search (O(log n) per lookup via numpy searchsorted),
    then compute total return and CAGR in bulk.  Max drawdown per window is
    computed via a vectorised rolling cummax approach.

    Time complexity: O(n log n) vs the previous O(n²) row-iteration approach.
    """
    if df.empty:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "reason": "No NAV data",
            "data_availability": {
                "start_date": "",
                "end_date": "",
                "total_days": 0,
                "nav_data_points": 0,
                "sufficient_for_window": False,
            },
            "rolling_periods_analyzed": 0,
        }

    df = df.sort_index().copy()
    nav_values: np.ndarray = df["nav"].to_numpy(dtype=np.float64)
    timestamps: np.ndarray = df.index.to_numpy()  # numpy datetime64

    # window_td = np.timedelta64(int(365.25 * window_years * 24 * 3600), "s")
    target_seconds = int((365.25 * window_years * 24 * 3600) - (2 * 24 * 3600))
    window_td = np.timedelta64(target_seconds, "s")

    rolling_returns: List[float] = []
    rolling_cagrs: List[float] = []
    rolling_drawdowns: List[float] = []

    for i in range(len(df)):
        target_ts = timestamps[i] - window_td
        # searchsorted returns the insertion point; subtract 1 to get the
        # last index whose timestamp is <= target_ts.
        j = int(np.searchsorted(timestamps, target_ts, side="right")) - 1
        if j < 0:
            # No data point far enough back for this window
            continue

        base_nav = nav_values[j]
        current_nav = nav_values[i]
        if base_nav <= 0:
            continue

        elapsed_days = max((timestamps[i] - timestamps[j]) / np.timedelta64(1, "D"), 1.0)
        years = round(elapsed_days / 365.25, 1)
        total_return = ((current_nav / base_nav) - 1.0) * 100.0
        cagr = (((current_nav / base_nav) ** (1.0 / years)) - 1.0) * 100.0

        # Drawdown over the window [j+1 .. i] (the same slice used for the
        # return calculation).  We include the base point so the series
        # starts at a known NAV.
        window_slice = pd.Series(nav_values[j : i + 1], dtype="float64")
        rolling_returns.append(total_return)
        rolling_cagrs.append(cagr)
        rolling_drawdowns.append(compute_max_drawdown(window_slice))

    if not rolling_returns:
        return {
            "status": "INSUFFICIENT_HISTORY",
            "reason": f"Not enough data for {window_years}Y window",
            "data_availability": build_data_availability(df, False),
            "rolling_periods_analyzed": 0,
        }

    return {
        "status": "SUCCESS",
        "data_availability": build_data_availability(df, True),
        "rolling_periods_analyzed": len(rolling_returns),
        "rolling_returns": compute_distribution(rolling_returns, include_quartiles=True),
        "max_drawdown": round2(min(rolling_drawdowns)),
        "cagr": compute_distribution(rolling_cagrs, include_quartiles=False),
    }


def match_target_amc(name_upper: str) -> Optional[str]:
    """
    Maps a fund house string (already uppercased) to one of the five
    canonical TARGET_AMCS names.

    Kotak Mahindra only requires "KOTAK" — "MAHINDRA" alone would
    incorrectly match Mahindra Manulife funds.
    """
    checks: Dict[str, List[str]] = {
        "ICICI Prudential": ["ICICI", "PRUDENTIAL"],
        "HDFC": ["HDFC"],
        "Axis": ["AXIS"],
        "SBI": ["SBI"],
        "Kotak Mahindra": ["KOTAK"],
    }
    for amc, tokens in checks.items():
        if all(token in name_upper for token in tokens):
            return amc
    return None


def normalize_assignment_category(scheme_category: Optional[str], scheme_name: str) -> Optional[str]:
    cat = (scheme_category or "").upper()
    name = scheme_name.upper()
    if any(token in cat or token in name for token in TARGET_CATEGORIES["Equity: Mid Cap"]):
        return "Equity: Mid Cap"
    if any(token in cat or token in name for token in TARGET_CATEGORIES["Equity: Small Cap"]):
        return "Equity: Small Cap"
    return None


def is_direct_growth_scheme(name: str) -> bool:
    name_upper = name.upper()
    return "DIRECT" in name_upper and "GROWTH" in name_upper


# Union type accepted by verified_target_key so it can handle both the
# full-history meta block (MfapiSchemeMeta) and the flat /latest response
# (MfapiLatestNAVItem) without duplicating logic.
_SchemeMetaLike = Any  # MfapiSchemeMeta | MfapiLatestNAVItem


def verified_target_key(meta: _SchemeMetaLike) -> Optional[Tuple[str, str]]:
    """
    Returns (amc, category) if the scheme matches our target universe
    (correct AMC, correct category, Direct Growth plan), else None.

    Accepts both MfapiSchemeMeta (from full history) and MfapiLatestNAVItem
    (from /latest) since both expose .scheme_name, .fund_house,
    .scheme_category via a uniform interface.
    """
    if not is_direct_growth_scheme(meta.scheme_name):
        return None
    amc = match_target_amc((meta.fund_house or meta.scheme_name).upper())
    if not amc:
        amc = match_target_amc(meta.scheme_name.upper())
    if not amc:
        return None
    category = normalize_assignment_category(meta.scheme_category, meta.scheme_name)
    if not category:
        return None
    return amc, category


# ---------------------------------------------------------------------------
# MFAPI FETCH HELPERS
# ---------------------------------------------------------------------------

async def fetch_scheme_history(client: httpx.AsyncClient, code: int) -> MfapiNAVHistoryResponse:
    """Fetch full NAV history from GET /mf/{code}."""
    data_raw = await mfapi_get_json(client, f"{MFAPI_BASE_URL}/{code}")
    data = MfapiNAVHistoryResponse.model_validate(data_raw)
    if data.status != "SUCCESS":
        raise RuntimeError(f"mfapi returned {data.status} for scheme {code}")
    return data


async def fetch_scheme_latest(client: httpx.AsyncClient, code: int) -> MfapiLatestNAVItem:
    """
    Fetch only the most recent NAV from GET /mf/{code}/latest.

    IMPORTANT: this endpoint returns a flat LatestNAVItem object, NOT the
    {meta, data[], status} envelope returned by the full history endpoint.
    """
    data_raw = await mfapi_get_json(client, f"{MFAPI_BASE_URL}/{code}/latest")
    return MfapiLatestNAVItem.model_validate(data_raw)


async def fetch_all_latest_navs(client: httpx.AsyncClient) -> Dict[int, MfapiLatestNAVItem]:
    """
    Fetch the latest NAV for *all* schemes in a single call to GET /mf/latest.

    Used during daily incremental sync: one API request instead of one per
    tracked fund, dramatically reducing quota consumption.

    Returns a dict keyed by schemeCode for O(1) lookup.
    """
    data_raw = await mfapi_get_json(client, f"{MFAPI_BASE_URL}/latest")
    result: Dict[int, MfapiLatestNAVItem] = {}
    for item_raw in data_raw:
        try:
            item = MfapiLatestNAVItem.model_validate(item_raw)
            result[item.schemeCode] = item
        except Exception:
            # Malformed entries in the bulk response are skipped silently
            continue
    return result


def extract_latest_nav_from_history(nav_items: List[MfapiNAVData]) -> Optional[Tuple[date, Decimal]]:
    """
    Extract the most recent (date, nav) pair from a NAVHistoryResponse.data
    list.  The list is newest-first from the API, but we use max() to be
    robust against any ordering variation.
    """
    if not nav_items:
        return None
    latest_item = max(nav_items, key=lambda x: datetime.strptime(x.date, "%d-%m-%Y"))
    return datetime.strptime(latest_item.date, "%d-%m-%Y").date(), Decimal(latest_item.nav)


def extract_latest_nav_from_item(item: MfapiLatestNAVItem) -> Tuple[date, Decimal]:
    """Extract (date, nav) from a flat MfapiLatestNAVItem response."""
    nav_date = datetime.strptime(item.date, "%d-%m-%Y").date()
    return nav_date, Decimal(item.nav)


# ---------------------------------------------------------------------------
# SCHEME DISCOVERY
# ---------------------------------------------------------------------------

async def discover_schemes(client: httpx.AsyncClient) -> List[int]:
    """
    Identify exactly one scheme code per (AMC, category) combination.

    Strategy:
      1. Download the full scheme list from GET /mf — one API call.
         MfapiSchemeListItem captures ISIN codes from this list so we
         don't need an extra call per fund during discovery.
      2. Filter candidates in-process: Direct Growth plans only, matching
         AMC and category tokens from the scheme name alone.
      3. For each (AMC, category) bucket, validate the top candidate via
         GET /mf/{code} to confirm the server-side meta matches our filters.
         Stop at the first confirmed match per bucket.

    Raises RuntimeError if any of the 10 target (AMC, category) combinations
    cannot be verified — callers should treat this as a hard failure.
    """
    logger.info("Starting validated scheme discovery")
    master_list_raw = await mfapi_get_json(client, MFAPI_BASE_URL)
    master_list = [MfapiSchemeListItem.model_validate(item) for item in master_list_raw]

    candidates: Dict[Tuple[str, str], List[MfapiSchemeListItem]] = {
        (amc, category): []
        for amc in TARGET_AMCS
        for category in TARGET_CATEGORIES
    }

    for item in master_list:
        if not is_direct_growth_scheme(item.schemeName):
            continue
        amc = match_target_amc(item.schemeName.upper())
        category = normalize_assignment_category(None, item.schemeName)
        if amc and category:
            candidates[(amc, category)].append(item)

    discovered: Dict[Tuple[str, str], int] = {}
    for key, bucket in candidates.items():
        # Deterministic ordering: prefer shorter names (less likely to be
        # a variant/plan), then alphabetical, then by code for tie-breaking.
        bucket = sorted(bucket, key=lambda x: (len(x.schemeName), x.schemeName, x.schemeCode))
        for item in bucket:
            try:
                history = await fetch_scheme_history(client, item.schemeCode)
            except Exception:
                logger.exception("Discovery validation failed for scheme %s", item.schemeCode)
                continue
            if verified_target_key(history.meta) == key:
                discovered[key] = item.schemeCode
                logger.info(
                    "Verified scheme %s (%s) for %s / %s",
                    item.schemeCode,
                    item.schemeName,
                    key[0],
                    key[1],
                )
                break

    missing = [key for key in candidates if key not in discovered]
    if missing:
        raise RuntimeError(f"Unable to verify all target schemes. Missing: {missing}")

    # Return codes in a stable order: AMCs x categories
    return [discovered[(amc, category)] for amc in TARGET_AMCS for category in TARGET_CATEGORIES]


# ---------------------------------------------------------------------------
# DATABASE HELPERS
# ---------------------------------------------------------------------------

async def upsert_sync_state(session: AsyncSession, fund_code: int, **kwargs: Any) -> None:
    stmt = insert(FundSyncState).values(fund_code=fund_code, **kwargs)
    stmt = stmt.on_conflict_do_update(
        index_elements=[FundSyncState.fund_code], set_=kwargs
    )
    await session.execute(stmt)


async def update_sync_job(session: AsyncSession, job_id: int, **kwargs: Any) -> None:
    job = await session.get(SyncJob, job_id)
    if not job:
        raise RuntimeError(f"Sync job {job_id} not found")
    for key, value in kwargs.items():
        setattr(job, key, value)
    await session.flush()


async def load_nav_dataframe(session: AsyncSession, fund_code: int) -> pd.DataFrame:
    """
    Load the full NAV history for a fund from the database into a DataFrame
    indexed by date (ascending).

    Analytics are computed from the DB — not from raw API response data —
    so that partial previous runs accumulate correctly and a short API
    response never overwrites analytics computed from a longer history.
    """
    rows = (
        await session.execute(
            select(NAVHistory.date, NAVHistory.nav)
            .where(NAVHistory.fund_code == fund_code)
            .order_by(NAVHistory.date.asc())
        )
    ).all()

    if not rows:
        return pd.DataFrame(columns=["nav"]).rename_axis("date")

    df = pd.DataFrame(rows, columns=["date", "nav"])
    df["nav"] = df["nav"].astype(float)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


# ---------------------------------------------------------------------------
# CORE SYNC: PROCESS A SINGLE SCHEME
# ---------------------------------------------------------------------------

async def process_scheme(
    session: AsyncSession,
    client: httpx.AsyncClient,
    code: int,
    job_id: int,
    prefetched_latest: Optional[MfapiLatestNAVItem] = None,
) -> None:
    """
    Sync one scheme:
      1. Try the /latest shortcut (uses prefetched bulk data if available,
         otherwise fetches individually).  If the fund is already up-to-date
         and analytics are fresh, return early — consuming no extra quota.
      2. On shortcut failure, fall back to full history from GET /mf/{code}.
      3. Upsert new NAV rows into nav_history.
      4. Reload the full NAV history from the DB (not from the API response)
         and recompute analytics for all four windows.
      5. Persist analytics to analytics_cache and update sync state.

    The `prefetched_latest` parameter accepts a MfapiLatestNAVItem obtained
    from the bulk GET /mf/latest call so that daily incremental syncs for
    all funds consume only one API request total for the latest-NAV check.
    """
    await upsert_sync_state(session, code, sync_state="RUNNING", last_error=None, last_job_id=job_id)
    await update_sync_job(session, job_id, current_fund_code=code)
    await session.commit()

    now_utc = datetime.now(timezone.utc)
    state = await session.get(FundSyncState, code)
    fund = await session.get(Fund, code)

    # ------------------------------------------------------------------
    # Step 1: /latest shortcut — skip full history if already up-to-date
    # ------------------------------------------------------------------
    try:
        latest_item: MfapiLatestNAVItem = (
            prefetched_latest
            if prefetched_latest is not None
            else await fetch_scheme_latest(client, code)
        )

        latest_target_key = verified_target_key(latest_item)
        if latest_target_key is None:
            raise RuntimeError(
                f"Scheme {code} /latest response does not match "
                "target AMC/category/direct-growth filters"
            )

        latest_nav_date, latest_nav_value = extract_latest_nav_from_item(latest_item)

        if not fund:
            fund = Fund(code=code)
            session.add(fund)

        fund.name = latest_item.scheme_name
        fund.amc = latest_item.fund_house or latest_target_key[0]
        fund.category = latest_target_key[1]
        fund.scheme_type = latest_item.scheme_type
        fund.isin_growth = latest_item.isin_growth
        fund.isin_div_reinvestment = latest_item.isin_div_reinvestment
        fund.latest_nav = latest_nav_value
        fund.latest_nav_date = latest_nav_date
        fund.last_synced_at = now_utc
        fund.is_active = True

        if (
            state
            and state.last_nav_date
            and latest_nav_date <= state.last_nav_date
            and state.last_analytics_at is not None
        ):
            # NAV hasn't changed since last sync and analytics are fresh —
            # commit the fund metadata update and return early.
            await upsert_sync_state(
                session,
                code,
                sync_state="SUCCESS",
                last_nav_date=state.last_nav_date,
                last_backfill_at=now_utc,
                last_analytics_at=state.last_analytics_at,
                retry_count=0,
                last_error=None,
                last_job_id=job_id,
            )
            await session.commit()
            return

    except Exception:
        logger.exception(
            "Latest NAV shortcut failed for scheme %s; falling back to full history", code
        )
        await session.rollback()
        # Re-fetch state and fund after rollback
        state = await session.get(FundSyncState, code)
        fund = await session.get(Fund, code)
        now_utc = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Step 2: Full history fetch (backfill or shortcut fallback)
    # ------------------------------------------------------------------
    data = await fetch_scheme_history(client, code)
    target_key = verified_target_key(data.meta)
    if target_key is None:
        raise RuntimeError(
            f"Scheme {code} does not match target AMC/category/direct-growth filters"
        )
    _, category = target_key

    latest_point = extract_latest_nav_from_history(data.data)
    latest_nav_date = latest_point[0] if latest_point else None
    latest_nav_value = latest_point[1] if latest_point else None

    if not fund:
        fund = Fund(code=code)
        session.add(fund)

    fund.name = data.meta.scheme_name
    fund.amc = data.meta.fund_house or target_key[0]
    fund.category = category
    fund.scheme_type = data.meta.scheme_type
    fund.isin_growth = data.meta.isin_growth
    fund.isin_div_reinvestment = data.meta.isin_div_reinvestment
    fund.latest_nav = latest_nav_value
    fund.latest_nav_date = latest_nav_date
    fund.last_synced_at = now_utc
    fund.is_active = True

    # ------------------------------------------------------------------
    # Step 3: Upsert new NAV rows
    # ------------------------------------------------------------------
    last_stored_date = state.last_nav_date if state else None
    to_insert: List[Dict[str, Any]] = []

    for item in data.data:
        nav_date = datetime.strptime(item.date, "%d-%m-%Y").date()
        nav_val = Decimal(item.nav)
        if last_stored_date is None or nav_date > last_stored_date:
            to_insert.append({"fund_code": code, "date": nav_date, "nav": nav_val})

    inserted_new_rows = bool(to_insert)
    if to_insert:
        stmt = insert(NAVHistory).values(to_insert)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=[NAVHistory.fund_code, NAVHistory.date]
        )
        await session.execute(stmt)

    if not inserted_new_rows and state and state.last_analytics_at and latest_nav_date == state.last_nav_date:
        # Nothing new to process; preserve existing analytics.
        await upsert_sync_state(
            session,
            code,
            sync_state="SUCCESS",
            last_nav_date=state.last_nav_date,
            last_backfill_at=now_utc,
            last_analytics_at=state.last_analytics_at,
            retry_count=0,
            last_error=None,
            last_job_id=job_id,
        )
        await session.commit()
        return

    # ------------------------------------------------------------------
    # Step 4: Reload full history from DB and recompute analytics
    #
    # We load from the DB rather than from `data.data` so that analytics
    # always reflect the complete accumulated history, not just whatever
    # the API returned in this single response.
    # ------------------------------------------------------------------
    df = await load_nav_dataframe(session, code)

    for win_label, win_years in WINDOWS.items():
        metrics = compute_metrics(df, win_years)
        computed_at = datetime.now(timezone.utc)
        payload: Dict[str, Any] = {
            "fund_code": code,
            "fund_name": fund.name,
            "category": fund.category,
            "amc": fund.amc,
            "window": win_label,
            "computed_at": computed_at.isoformat(),
            **metrics,
        }

        median_return: Optional[Decimal] = None
        max_drawdown: Optional[Decimal] = None
        if metrics.get("status") == "SUCCESS":
            median_return = Decimal(str(metrics["rolling_returns"]["median"]))
            max_drawdown = Decimal(str(metrics["max_drawdown"]))

        stmt = insert(AnalyticsCache).values(
            fund_code=code,
            window=win_label,
            median_return=median_return,
            max_drawdown=max_drawdown,
            computed_at=computed_at,
            payload=payload,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[AnalyticsCache.fund_code, AnalyticsCache.window],
            set_={
                "median_return": median_return,
                "max_drawdown": max_drawdown,
                "computed_at": computed_at,
                "payload": payload,
            },
        )
        await session.execute(stmt)

    # ------------------------------------------------------------------
    # Step 5: Mark fund as successfully synced
    # ------------------------------------------------------------------
    await upsert_sync_state(
        session,
        code,
        sync_state="SUCCESS",
        last_nav_date=latest_nav_date,
        last_backfill_at=now_utc,
        last_analytics_at=now_utc,
        retry_count=0,
        last_error=None,
        last_job_id=job_id,
    )
    await session.commit()


# ---------------------------------------------------------------------------
# PIPELINE WORKER
# ---------------------------------------------------------------------------

async def get_or_create_resumable_job(session: AsyncSession) -> SyncJob:
    stmt = (
        select(SyncJob)
        .where(SyncJob.status.in_(["RUNNING", "FAILED", "PENDING"]))
        .order_by(SyncJob.id.desc())
    )
    existing = (await session.execute(stmt)).scalars().first()
    if existing:
        return existing
    job = SyncJob(status="PENDING", discovered_codes=[])
    session.add(job)
    await session.flush()
    return job


async def pending_codes_for_job(session: AsyncSession, job: SyncJob) -> List[int]:
    """Return scheme codes that have not yet completed successfully in this job."""
    if not job.discovered_codes:
        return []
    rows = (
        await session.execute(
            select(FundSyncState).where(
                FundSyncState.last_job_id == job.id,
                FundSyncState.sync_state == "SUCCESS",
            )
        )
    ).scalars().all()
    successful_codes = {row.fund_code for row in rows}
    return [int(c) for c in job.discovered_codes if int(c) not in successful_codes]


async def backfill_pipeline(job_id: Optional[int] = None) -> None:
    """
    Main pipeline entry point.

    For daily incremental syncs, fetches GET /mf/latest once (1 API call)
    and distributes the prefetched data to each process_scheme() call so
    that the /latest shortcut path costs zero additional quota per fund.

    Job status lifecycle:
      PENDING -> RUNNING -> SUCCESS   (all funds processed)
      PENDING -> RUNNING -> FAILED    (at least one fund failed; pipeline
                                       still processes remaining funds so
                                       partial results are available)

    Resumability: if a job already has discovered_codes persisted from a
    previous run, scheme discovery is skipped and only pending (not already
    SUCCESS) codes are retried.
    """
    async with AsyncSessionLocal() as session:
        if job_id is not None:
            job = await session.get(SyncJob, job_id)
            if not job:
                raise RuntimeError(f"Sync job {job_id} not found")
        else:
            job = await get_or_create_resumable_job(session)

        started_at = job.started_at or datetime.now(timezone.utc)
        await update_sync_job(
            session, job.id,
            status="RUNNING",
            started_at=started_at,
            completed_at=None,
            last_error=None,
        )
        await session.commit()
        await redis_client.set("sync_status", "running")
        await redis_client.set("sync_started_at", started_at.isoformat())
        await redis_client.set("sync_job_id", str(job.id))

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # --- Discovery (skipped if resuming) ---
                if not job.discovered_codes:
                    codes = await discover_schemes(client)
                    await update_sync_job(session, job.id, discovered_codes=codes)
                    await session.commit()
                else:
                    codes = [int(c) for c in job.discovered_codes]

                pending_codes = await pending_codes_for_job(session, job)
                if not pending_codes:
                    pending_codes = codes

                # --- Bulk /latest prefetch (1 API call for all funds) ---
                prefetched: Dict[int, MfapiLatestNAVItem] = {}
                try:
                    prefetched = await fetch_all_latest_navs(client)
                    logger.info("Prefetched latest NAVs for %d schemes", len(prefetched))
                except Exception:
                    logger.warning(
                        "Bulk /mf/latest prefetch failed; will fetch individually per fund",
                        exc_info=True,
                    )

                # --- Per-fund processing ---
                had_failure = False
                for code in pending_codes:
                    await redis_client.set("sync_current_scheme", str(code))
                    try:
                        await process_scheme(
                            session,
                            client,
                            code,
                            job.id,
                            prefetched_latest=prefetched.get(code),
                        )
                        await update_sync_job(
                            session,
                            job.id,
                            processed_funds=job.processed_funds + 1,
                            current_fund_code=code,
                        )
                        await session.commit()
                        await session.refresh(job)
                    except Exception as exc:
                        had_failure = True
                        logger.exception("Failed processing scheme %s", code)
                        existing = await session.get(FundSyncState, code)
                        retries = (existing.retry_count if existing else 0) + 1
                        await upsert_sync_state(
                            session,
                            code,
                            sync_state="FAILED",
                            retry_count=retries,
                            last_error=str(exc),
                            last_job_id=job.id,
                        )
                        await update_sync_job(
                            session,
                            job.id,
                            failed_funds=job.failed_funds + 1,
                            current_fund_code=code,
                            last_error=str(exc),
                        )
                        await session.commit()
                        await session.refresh(job)

                # Final status reflects whether any fund failed
                final_status = "FAILED" if had_failure else "SUCCESS"
                await update_sync_job(
                    session,
                    job.id,
                    status=final_status,
                    completed_at=datetime.now(timezone.utc),
                    current_fund_code=None,
                )
                await session.commit()

        except Exception as exc:
            logger.exception("Pipeline crashed during discovery or setup")
            await update_sync_job(
                session,
                job.id,
                status="FAILED",
                completed_at=datetime.now(timezone.utc),
                last_error=str(exc),
            )
            await session.commit()
        finally:
            await redis_client.set("sync_status", "idle")
            await redis_client.set("sync_last_run", datetime.now(timezone.utc).isoformat())
            await redis_client.delete("sync_current_scheme")
            await redis_client.delete("sync_job_id")


# ---------------------------------------------------------------------------
# RESPONSE BUILDERS
# ---------------------------------------------------------------------------

def latest_nav_out(fund: Fund) -> Optional[LatestNAVOut]:
    if fund.latest_nav is None or fund.latest_nav_date is None:
        return None
    return LatestNAVOut(
        nav=f"{Decimal(str(fund.latest_nav)):.5f}",
        date=fund.latest_nav_date.isoformat(),   # YYYY-MM-DD
    )


def fund_list_item_out(fund: Fund) -> FundListItemOut:
    return FundListItemOut(
        fund_code=fund.code,
        fund_name=fund.name,
        amc=fund.amc,
        category=fund.category,
        latest_nav=latest_nav_out(fund),
    )


def fund_details_out(fund: Fund) -> FundDetailsOut:
    return FundDetailsOut(
        fund_code=fund.code,
        fund_name=fund.name,
        amc=fund.amc,
        category=fund.category,
        scheme_type=fund.scheme_type,
        isin_growth=fund.isin_growth,
        isin_div_reinvestment=fund.isin_div_reinvestment,
        latest_nav=latest_nav_out(fund),
        last_synced_at=fund.last_synced_at.isoformat() if fund.last_synced_at else None,
    )


# ---------------------------------------------------------------------------
# FASTAPI APPLICATION
# ---------------------------------------------------------------------------

# app = FastAPI(title="Mutual Fund Analytics Platform")


# @app.on_event("startup")
# async def startup_event() -> None:
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)
#     await redis_client.set("sync_status", "idle")
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup logic ---
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await redis_client.set("sync_status", "idle")
    
    yield  # The app runs while it is yielded
    
    # --- Shutdown logic ---
    # (You can leave this empty or add teardown code here later)

app = FastAPI(title="Mutual Fund Analytics Platform", lifespan=lifespan)

# NOTE: /funds/rank MUST be registered before /funds/{code} so that FastAPI
# resolves the literal path segment "rank" before attempting integer coercion
# on the {code} path parameter.
@app.get("/funds/rank", response_model=RankResponseOut, tags=["Analytics"])
async def rank_funds(
    category: str = Query(..., description="Fund category to filter by (partial match)"),
    window: str = Query(..., pattern="^(1Y|3Y|5Y|10Y)$"),
    sort_by: str = Query("median_return", pattern="^(median_return|max_drawdown)$"),
    limit: int = Query(5, ge=1, le=50),
) -> RankResponseOut:
    async with AsyncSessionLocal() as session:
        query = (
            select(AnalyticsCache, Fund)
            .join(Fund, Fund.code == AnalyticsCache.fund_code)
            .where(AnalyticsCache.window == window)
            .where(Fund.category.ilike(f"%{category}%"))
            .where(Fund.is_active.is_(True))
        )
        if sort_by == "median_return":
            query = (
                query
                .where(AnalyticsCache.median_return.isnot(None))
                .order_by(AnalyticsCache.median_return.desc())
            )
        else:
            query = (
                query
                .where(AnalyticsCache.max_drawdown.isnot(None))
                .order_by(AnalyticsCache.max_drawdown.asc())
            )

        all_rows = (await session.execute(query)).all()
        ranked_rows = all_rows[:limit]

        metric_key = (
            f"median_return_{window.lower()}"
            if sort_by == "median_return"
            else f"max_drawdown_{window.lower()}"
        )
        funds: List[RankedFundOut] = []
        for idx, (cache, fund) in enumerate(ranked_rows, start=1):
            metrics: Dict[str, float] = {}
            if cache.median_return is not None:
                metrics[f"median_return_{window.lower()}"] = float(cache.median_return)
            if cache.max_drawdown is not None:
                metrics[f"max_drawdown_{window.lower()}"] = float(cache.max_drawdown)
            if metric_key not in metrics:
                continue
            funds.append(
                RankedFundOut(
                    rank=idx,
                    fund_code=fund.code,
                    fund_name=fund.name,
                    amc=fund.amc,
                    current_nav=float(fund.latest_nav) if fund.latest_nav is not None else None,
                    last_updated=fund.latest_nav_date.isoformat() if fund.latest_nav_date else None,
                    metrics=metrics,
                )
            )
        return RankResponseOut(
            category=category,
            window=window,
            sorted_by=sort_by,
            total_funds=len(all_rows),
            showing=len(funds),
            funds=funds,
        )


@app.get("/funds", tags=["Funds"])
async def list_tracked_funds(
    category: Optional[str] = None,
    amc: Optional[str] = None,
) -> Dict[str, Any]:
    async with AsyncSessionLocal() as session:
        query = (
            select(Fund)
            .where(Fund.is_active.is_(True))
            .order_by(Fund.amc.asc(), Fund.name.asc())
        )
        if category:
            query = query.where(Fund.category.ilike(f"%{category}%"))
        if amc:
            query = query.where(Fund.amc.ilike(f"%{amc}%"))
        result = await session.execute(query)
        funds = result.scalars().all()
        return {
            "total": len(funds),
            "funds": [fund_list_item_out(f).model_dump() for f in funds],
        }


@app.get("/funds/{code}", response_model=FundDetailsOut, tags=["Funds"])
async def get_fund_details(code: int) -> FundDetailsOut:
    async with AsyncSessionLocal() as session:
        fund = await session.get(Fund, code)
        if not fund or not fund.is_active:
            raise HTTPException(status_code=404, detail="Fund not found")
        return fund_details_out(fund)


@app.get("/funds/{code}/analytics", response_model=AnalyticsResponseOut, tags=["Analytics"])
async def get_fund_analytics(
    code: int,
    window: str = Query(..., pattern="^(1Y|3Y|5Y|10Y)$"),
) -> AnalyticsResponseOut:
    async with AsyncSessionLocal() as session:
        fund = await session.get(Fund, code)
        if not fund or not fund.is_active:
            raise HTTPException(status_code=404, detail="Fund not found")
        result = await session.execute(
            select(AnalyticsCache).where(
                AnalyticsCache.fund_code == code,
                AnalyticsCache.window == window,
            )
        )
        cache = result.scalar_one_or_none()
        if not cache:
            raise HTTPException(status_code=404, detail="Analytics not computed yet for this fund/window")
        payload = cache.payload
        return AnalyticsResponseOut.model_validate(
            {
                "fund_code": payload["fund_code"],
                "fund_name": payload["fund_name"],
                "category": payload["category"],
                "amc": payload["amc"],
                "window": payload["window"],
                "status": payload["status"],
                "reason": payload.get("reason"),
                "data_availability": payload["data_availability"],
                "rolling_periods_analyzed": payload.get("rolling_periods_analyzed", 0),
                "rolling_returns": payload.get("rolling_returns"),
                "max_drawdown": payload.get("max_drawdown"),
                "cagr": payload.get("cagr"),
                "computed_at": payload["computed_at"],
            }
        )


@app.post("/sync/trigger", tags=["Pipeline"])
async def trigger_data_sync(background_tasks: BackgroundTasks) -> Dict[str, str]:
    async with AsyncSessionLocal() as session:
        running = (
            await session.execute(
                select(SyncJob)
                .where(SyncJob.status == "RUNNING")
                .order_by(SyncJob.id.desc())
            )
        ).scalars().first()
        if running:
            return {"message": f"Pipeline is already running for job {running.id}."}

        resumable = (
            await session.execute(
                select(SyncJob)
                .where(SyncJob.status.in_(["FAILED", "PENDING"]))
                .order_by(SyncJob.id.desc())
            )
        ).scalars().first()

        if resumable:
            background_tasks.add_task(backfill_pipeline, resumable.id)
            return {"message": f"Resuming sync job {resumable.id}. Check /sync/status for progress."}

        job = SyncJob(status="PENDING", discovered_codes=[])
        session.add(job)
        await session.commit()
        await session.refresh(job)
        background_tasks.add_task(backfill_pipeline, job.id)
        return {"message": f"Sync job {job.id} triggered. Check /sync/status for progress."}


@app.get("/sync/status", tags=["Pipeline"])
async def get_sync_status() -> Dict[str, Any]:
    async with AsyncSessionLocal() as session:
        job = (
            await session.execute(select(SyncJob).order_by(SyncJob.id.desc()))
        ).scalars().first()
        if not job:
            return {"status": "idle", "job_id": None, "message": "No sync jobs have been created yet."}

        success_count = (
            await session.execute(
                select(func.count()).select_from(FundSyncState).where(
                    FundSyncState.last_job_id == job.id,
                    FundSyncState.sync_state == "SUCCESS",
                )
            )
        ).scalar_one()
        failed_count = (
            await session.execute(
                select(func.count()).select_from(FundSyncState).where(
                    FundSyncState.last_job_id == job.id,
                    FundSyncState.sync_state == "FAILED",
                )
            )
        ).scalar_one()
        pending_count = max(len(job.discovered_codes or []) - success_count - failed_count, 0)

        return {
            "status": job.status.lower(),
            "job_id": job.id,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "current_fund_code": job.current_fund_code,
            "discovered_funds": len(job.discovered_codes or []),
            "processed_funds": job.processed_funds,
            "failed_funds": job.failed_funds,
            "successful_funds": success_count,
            "pending_funds": pending_count,
            "last_error": job.last_error,
        }

