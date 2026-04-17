import asyncio
import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import List, Optional

import httpx
import pandas as pd
import redis.asyncio as aioredis
from fastapi import FastAPI, BackgroundTasks, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Date, Numeric, JSON, select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

# --- CONFIGURATION ---
DATABASE_URL = "postgresql+asyncpg://mf_user:mf_password@localhost:5432/mf_analytics"
REDIS_URL = "redis://localhost:6379"
MFAPI_BASE_URL = "https://api.mfapi.in/mf"

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- DATABASE SETUP ---
engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class Fund(Base):
    __tablename__ = 'funds'
    code = Column(Integer, primary_key=True)
    name = Column(String)
    amc = Column(String)
    category = Column(String)

class NAVHistory(Base):
    __tablename__ = 'nav_history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    fund_code = Column(Integer, index=True)
    date = Column(Date, index=True)
    nav = Column(Numeric(15, 5))

class AnalyticsCache(Base):
    __tablename__ = 'analytics_cache'
    id = Column(Integer, primary_key=True, autoincrement=True)
    fund_code = Column(Integer, index=True)
    window = Column(String)  # '1Y', '3Y', '5Y', '10Y'
    median_return = Column(Numeric)
    max_drawdown = Column(Numeric)
    payload = Column(JSON)  # Stores the full nested dictionary

# --- REDIS RATE LIMITER ---
redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)

RATE_LIMIT_LUA = """
local sec_key = KEYS[1]
local min_key = KEYS[2]
local hr_key = KEYS[3]

local current_sec = tonumber(redis.call('get', sec_key) or 0)
local current_min = tonumber(redis.call('get', min_key) or 0)
local current_hr = tonumber(redis.call('get', hr_key) or 0)

if current_sec >= 2 or current_min >= 50 or current_hr >= 300 then
    return 0
end

redis.call('incr', sec_key)
redis.call('expire', sec_key, 1)
redis.call('incr', min_key)
redis.call('expire', min_key, 60)
redis.call('incr', hr_key)
redis.call('expire', hr_key, 3600)

return 1
"""

async def wait_for_rate_limit():
    """Blocks until the Redis token bucket allows the request."""
    while True:
        now = datetime.utcnow()
        keys = [
            f"rl:sec:{now.strftime('%Y%m%d%H%M%S')}",
            f"rl:min:{now.strftime('%Y%m%d%H%M')}",
            f"rl:hr:{now.strftime('%Y%m%d%H')}"
        ]
        allowed = await redis_client.eval(RATE_LIMIT_LUA, 3, *keys)
        if allowed == 1:
            return
        await asyncio.sleep(0.5)

# --- DYNAMIC DISCOVERY ENGINE ---
async def discover_schemes() -> List[int]:
    """Fetches the master list and dynamically finds our 10 target funds."""
    logger.info("Starting Dynamic Scheme Discovery...")
    target_amcs = ["ICICI", "HDFC", "Axis", "SBI", "Kotak"]
    target_categories = ["Midcap", "Smallcap", "Mid Cap", "Small Cap"]
    
    discovered_codes = []
    found_matrix = {amc: {"Midcap": False, "Smallcap": False} for amc in target_amcs}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(MFAPI_BASE_URL)
        if response.status_code != 200:
            logger.error("Failed to fetch master list.")
            return []
            
        master_list = response.json()
        
    for item in master_list:
        name = item.get("schemeName", "").upper()
        
        # Must be Direct Growth to match assignment constraints
        if "DIRECT" not in name or "GROWTH" not in name:
            continue
            
        for amc in target_amcs:
            if amc.upper() in name:
                # Check Midcap
                if ("MIDCAP" in name or "MID CAP" in name) and not found_matrix[amc]["Midcap"]:
                    discovered_codes.append(item["schemeCode"])
                    found_matrix[amc]["Midcap"] = True
                # Check Smallcap
                elif ("SMALLCAP" in name or "SMALL CAP" in name) and not found_matrix[amc]["Smallcap"]:
                    discovered_codes.append(item["schemeCode"])
                    found_matrix[amc]["Smallcap"] = True

    logger.info(f"Discovered {len(discovered_codes)} matching schemes.")
    return discovered_codes

# --- ANALYTICS ENGINE ---
def compute_metrics(df: pd.DataFrame, window_years: int) -> dict:
    trading_days = int(window_years * 252)
    if len(df) <= trading_days:
        return {"error": "Insufficient data"}

    rolling_returns = (df['nav'].pct_change(periods=trading_days).dropna() * 100)
    cumulative_max = df['nav'].cummax()
    drawdown = ((df['nav'] - cumulative_max) / cumulative_max) * 100
    
    years_elapsed = len(df) / 252
    cagr = ((df['nav'].iloc[-1] / df['nav'].iloc[0]) ** (1 / years_elapsed) - 1) * 100

    return {
        "rolling_returns": {
            "min": round(rolling_returns.min(), 2),
            "max": round(rolling_returns.max(), 2),
            "median": round(rolling_returns.median(), 2),
            "p25": round(rolling_returns.quantile(0.25), 2),
            "p75": round(rolling_returns.quantile(0.75), 2)
        },
        "max_drawdown": round(drawdown.min(), 2),
        "cagr": round(cagr, 2)
    }

# --- PIPELINE WORKER ---
async def backfill_pipeline():
    await redis_client.set("sync_status", "running")
    
    try:
        codes = await discover_schemes()
        if not codes:
            raise Exception("No schemes discovered.")

        async with httpx.AsyncClient(timeout=20.0) as client:
            async with AsyncSessionLocal() as session:
                for code in codes:
                    logger.info(f"Fetching NAV history for scheme {code}...")
                    await wait_for_rate_limit()
                    
                    response = await client.get(f"{MFAPI_BASE_URL}/{code}")
                    if response.status_code != 200:
                        continue
                        
                    data = response.json()
                    if data.get("status") != "SUCCESS":
                        continue

                    # 1. Store Fund Metadata
                    meta = data["meta"]
                    fund = await session.get(Fund, code)
                    if not fund:
                        fund = Fund(
                            code=code, 
                            name=meta["scheme_name"], 
                            amc=meta["fund_house"], 
                            category=meta["scheme_category"]
                        )
                        session.add(fund)

                    # 2. Store NAV Data & Convert to Pandas DF for Analytics
                    nav_records = []
                    df_data = []
                    for item in data["data"]:
                        nav_date = datetime.strptime(item["date"], "%d-%m-%Y").date()
                        nav_val = float(item["nav"])
                        nav_records.append(NAVHistory(fund_code=code, date=nav_date, nav=nav_val))
                        df_data.append({"date": nav_date, "nav": nav_val})
                    
                    # Wipe old NAVs for this fund to avoid duplicates on resumability
                    await session.execute(NAVHistory.__table__.delete().where(NAVHistory.fund_code == code))
                    session.add_all(nav_records)
                    await session.commit()

                    # 3. Pre-compute and Store Analytics
                    df = pd.DataFrame(df_data).sort_values('date').set_index('date')
                    windows = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10}
                    
                    await session.execute(AnalyticsCache.__table__.delete().where(AnalyticsCache.fund_code == code))
                    
                    for win_label, win_years in windows.items():
                        metrics = compute_metrics(df, win_years)
                        if "error" not in metrics:
                            cache = AnalyticsCache(
                                fund_code=code,
                                window=win_label,
                                median_return=metrics["rolling_returns"]["median"],
                                max_drawdown=metrics["max_drawdown"],
                                payload=metrics
                            )
                            session.add(cache)
                    await session.commit()
                    logger.info(f"Successfully processed and pre-computed {code}")

    except Exception as e:
        logger.error(f"Pipeline crashed: {e}")
    finally:
        await redis_client.set("sync_status", "idle")
        await redis_client.set("sync_last_run", datetime.utcnow().isoformat())

# --- FASTAPI APP ---
app = FastAPI(title="Mutual Fund Analytics Platform")

@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await redis_client.set("sync_status", "idle")

@app.post("/sync/trigger", tags=["Pipeline"])
async def trigger_data_sync(background_tasks: BackgroundTasks):
    status = await redis_client.get("sync_status")
    if status == "running":
        return {"message": "Pipeline is already running."}
    
    background_tasks.add_task(backfill_pipeline)
    return {"message": "Sync triggered in the background. Check /sync/status for progress."}

@app.get("/sync/status", tags=["Pipeline"])
async def get_sync_status():
    status = await redis_client.get("sync_status")
    last_run = await redis_client.get("sync_last_run")
    return {"status": status, "last_run": last_run}

@app.get("/funds", tags=["Funds"])
async def list_tracked_funds(category: Optional[str] = None, amc: Optional[str] = None):
    async with AsyncSessionLocal() as session:
        query = select(Fund)
        if category:
            query = query.where(Fund.category.ilike(f"%{category}%"))
        if amc:
            query = query.where(Fund.amc.ilike(f"%{amc}%"))
        
        result = await session.execute(query)
        funds = result.scalars().all()
        return {"total": len(funds), "funds": funds}

@app.get("/funds/{code}", tags=["Funds"])
async def get_fund_details(code: int):
    async with AsyncSessionLocal() as session:
        fund = await session.get(Fund, code)
        if not fund:
            raise HTTPException(status_code=404, detail="Fund not found")
            
        nav_query = select(NAVHistory).where(NAVHistory.fund_code == code).order_by(NAVHistory.date.desc()).limit(1)
        nav_result = await session.execute(nav_query)
        latest_nav = nav_result.scalar_one_or_none()
        
        return {"fund": fund, "latest_nav": latest_nav}

@app.get("/funds/{code}/analytics", tags=["Analytics"])
async def get_fund_analytics(code: int, window: str = Query(..., pattern="^(1Y|3Y|5Y|10Y)$")):
    async with AsyncSessionLocal() as session:
        query = select(AnalyticsCache).where(AnalyticsCache.fund_code == code, AnalyticsCache.window == window)
        result = await session.execute(query)
        cache = result.scalar_one_or_none()
        
        if not cache:
            raise HTTPException(status_code=404, detail="Analytics not found or insufficient history")
            
        return {"fund_code": code, "window": window, "analytics": cache.payload}

@app.get("/funds/rank", tags=["Analytics"])
async def rank_funds(
    category: str = Query(...), 
    window: str = Query(..., pattern="^(1Y|3Y|5Y|10Y)$"),
    sort_by: str = Query("median_return", pattern="^(median_return|max_drawdown)$"),
    limit: int = Query(5, ge=1, le=10)
):
    async with AsyncSessionLocal() as session:
        # Join AnalyticsCache with Fund to filter by category
        query = select(AnalyticsCache, Fund.name).join(Fund, Fund.code == AnalyticsCache.fund_code)
        query = query.where(AnalyticsCache.window == window)
        query = query.where(Fund.category.ilike(f"%{category}%"))
        
        # Sort based on parameter
        if sort_by == "median_return":
            query = query.order_by(AnalyticsCache.median_return.desc())
        else:
            query = query.order_by(AnalyticsCache.max_drawdown.asc())
            
        query = query.limit(limit)
        result = await session.execute(query)
        
        ranked_list = []
        for cache, fund_name in result.all():
            ranked_list.append({
                "fund_code": cache.fund_code,
                "fund_name": fund_name,
                "median_return": float(cache.median_return),
                "max_drawdown": float(cache.max_drawdown)
            })
            
        return {"category": category, "window": window, "sorted_by": sort_by, "funds": ranked_list}