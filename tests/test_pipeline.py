"""
test_pipeline.py
================
Tests for pipeline resumability and crash-recovery behaviour.

Assignment requirements covered:
  1. Pipeline resumes only pending codes (SUCCESS codes skipped)
  2. discovered_codes are persisted before any fund is processed, so
     re-discovery is skipped on resume
  3. Full crash-and-resume simulation of backfill_pipeline

Every test uses the `db_session` fixture from conftest.py, which wraps the
test in a SAVEPOINT and rolls back afterwards — no data ever persists to the
production database.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from main import (
    AsyncSessionLocal,
    SyncJob,
    FundSyncState,
    Fund,
    pending_codes_for_job,
    backfill_pipeline,
    get_or_create_resumable_job,
)


# ---------------------------------------------------------------------------
# Helper: build a SyncJob + per-fund states inside the test session
# ---------------------------------------------------------------------------

async def _setup_job(session, codes, fund_states: dict) -> SyncJob:
    """
    Insert a FAILED SyncJob with the given discovered_codes and per-fund states.

    fund_states: dict mapping fund_code -> sync_state string.
                 Only codes listed here get a FundSyncState row;
                 unlisted codes have no row (they're truly undiscovered).
    """
    job = SyncJob(status="FAILED", discovered_codes=codes)
    session.add(job)
    await session.flush()  # populate job.id without committing

    for code, state_str in fund_states.items():
        state = FundSyncState(
            fund_code=code,
            sync_state=state_str,
            last_job_id=job.id,
        )
        session.add(state)

    await session.flush()
    return job


# ---------------------------------------------------------------------------
# 1. pending_codes_for_job — core resumability logic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_success_codes_excluded_from_pending(db_session):
    """
    Fund 101 has sync_state=SUCCESS for this job → must be excluded.
    Funds 102 (FAILED) and 103 (no state at all) → must be included.
    """
    job = await _setup_job(
        db_session,
        codes=[101, 102, 103],
        fund_states={101: "SUCCESS", 102: "FAILED"},
    )

    pending = await pending_codes_for_job(db_session, job)

    assert 101 not in pending, "A SUCCESS fund must not be re-processed"
    assert 102 in pending, "A FAILED fund must be re-processed"
    assert 103 in pending, "A fund with no state must be treated as pending"


@pytest.mark.asyncio
async def test_all_codes_pending_on_first_run(db_session):
    """
    When no fund_sync_state rows exist for the job (first-ever run),
    all discovered_codes must be returned as pending.
    """
    job = await _setup_job(
        db_session,
        codes=[201, 202, 203, 204, 205],
        fund_states={},  # nothing processed yet
    )

    pending = await pending_codes_for_job(db_session, job)

    assert set(pending) == {201, 202, 203, 204, 205}


@pytest.mark.asyncio
async def test_no_codes_pending_when_all_succeeded(db_session):
    """
    When every fund in discovered_codes has sync_state=SUCCESS,
    pending_codes_for_job must return an empty list.
    """
    codes = [301, 302, 303]
    job = await _setup_job(
        db_session,
        codes=codes,
        fund_states={c: "SUCCESS" for c in codes},
    )

    pending = await pending_codes_for_job(db_session, job)

    assert pending == [], "No codes should be pending when all have succeeded"


@pytest.mark.asyncio
async def test_only_success_state_excludes_code(db_session):
    """
    Only sync_state=SUCCESS causes a code to be excluded.
    RUNNING or FAILED states must still appear in the pending list.
    """
    job = await _setup_job(
        db_session,
        codes=[401, 402, 403, 404],
        fund_states={
            401: "SUCCESS",
            402: "RUNNING",   # mid-crash state → must be retried
            403: "FAILED",
            # 404 has no row → pending
        },
    )

    pending = await pending_codes_for_job(db_session, job)

    assert 401 not in pending
    assert 402 in pending
    assert 403 in pending
    assert 404 in pending


@pytest.mark.asyncio
async def test_pending_codes_empty_for_job_with_no_discovered_codes(db_session):
    """
    If discovered_codes is empty (discovery never ran), pending list must
    also be empty — there is nothing to process.
    """
    job = SyncJob(status="PENDING", discovered_codes=[])
    db_session.add(job)
    await db_session.flush()

    pending = await pending_codes_for_job(db_session, job)

    assert pending == []

@pytest.mark.asyncio
async def test_pending_codes_belong_to_correct_job():
    """
    SUCCESS rows from a *different* job must not exclude codes from the
    current job.  The filter is on last_job_id, not just sync_state.
    """
    from sqlalchemy import delete
    
    # 1. Clean up crosstalk from previous tests
    async with AsyncSessionLocal() as session:
        await session.execute(delete(FundSyncState))
        await session.execute(delete(SyncJob))
        await session.commit()
        
        # 2. Setup the test data
        job_a = SyncJob(status="SUCCESS", discovered_codes=[501])
        session.add(job_a)
        await session.commit()
        
        state_a = FundSyncState(fund_code=501, sync_state="SUCCESS", last_job_id=job_a.id)
        session.add(state_a)
        await session.commit()
        
        job_b = SyncJob(status="FAILED", discovered_codes=[501, 502])
        session.add(job_b)
        await session.commit()
        
        # 3. Assert the pending codes
        pending = await pending_codes_for_job(session, job_b)
        assert sorted(pending) == [501, 502]

# ---------------------------------------------------------------------------
# 2. discovered_codes persisted before fund processing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_discovered_codes_persisted_before_fund_processing():
    """
    discovered_codes must be saved to the DB immediately after scheme
    discovery, before any fund is processed.
    """
    # 1. FIX: Clean up DB so get_or_create_resumable_job doesn't grab a previous test's job
    from sqlalchemy import delete
    async with AsyncSessionLocal() as session:
        await session.execute(delete(FundSyncState))
        await session.execute(delete(SyncJob))
        await session.commit()

    fake_codes = [111, 222, 333]
    
    with patch("main.discover_schemes", new_callable=AsyncMock, return_value=fake_codes):
        with patch("main.process_scheme", new_callable=AsyncMock, side_effect=RuntimeError("simulated crash")):
            with patch("main.fetch_all_latest_navs", new_callable=AsyncMock, return_value={}):
                with patch("main.redis_client") as mock_redis:
                    mock_redis.set = AsyncMock()
                    mock_redis.delete = AsyncMock()

                    await backfill_pipeline(job_id=None)

    # Now check the DB to confirm discovered_codes were persisted.
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(SyncJob).order_by(SyncJob.id.desc())
        )
        job = result.scalars().first()
        assert job is not None, "A SyncJob must exist after the pipeline runs"
        assert set(job.discovered_codes) == set(fake_codes), (
            f"discovered_codes must be persisted before any fund is processed. "
            f"Expected {fake_codes}, got {job.discovered_codes}"
        )


@pytest.mark.asyncio
async def test_pipeline_resumes_from_failed_job_skipping_completed_funds():
    """
    Full crash-and-resume scenario:
    Run 1: discover [501, 502, 503]. Fund 501 succeeds, 502 crashes, 503 succeeds.
    Run 2: resume the FAILED job. Only 502 should be processed.
    """
    from sqlalchemy import delete
    
    # 1. Clean up crosstalk from previous tests
    async with AsyncSessionLocal() as session:
        await session.execute(delete(FundSyncState))
        await session.execute(delete(SyncJob))
        await session.commit()

    call_log = []

    async def fake_process(session, client, code, job_id, prefetched_latest=None):
        code_int = int(code)
        call_log.append(code_int)
        
        if code_int == 502 and call_log.count(502) == 1:
            raise RuntimeError("simulated crash on fund 502")
            
        from main import upsert_sync_state
        await upsert_sync_state(session, code_int, sync_state="SUCCESS", last_job_id=job_id)
        await session.commit()

    with patch("main.discover_schemes", new_callable=AsyncMock, return_value=[501, 502, 503]):
        with patch("main.process_scheme", side_effect=fake_process):
            with patch("main.fetch_all_latest_navs", new_callable=AsyncMock, return_value={}):
                with patch("main.redis_client") as mock_redis:
                    mock_redis.set = AsyncMock()
                    mock_redis.delete = AsyncMock()

                    # --- Run 1 ---
                    await backfill_pipeline(job_id=None)

    # After run 1, find the FAILED job.
    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(SyncJob).where(SyncJob.status == "FAILED").order_by(SyncJob.id.desc())
        )
        failed_job = result.scalars().first()
        assert failed_job is not None, "Job should be FAILED after fund 502 crash"

        # --- Run 2 ---
        with patch("main.discover_schemes", new_callable=AsyncMock, return_value=[501, 502, 503]):
            with patch("main.process_scheme", side_effect=fake_process):
                with patch("main.fetch_all_latest_navs", new_callable=AsyncMock, return_value={}):
                    with patch("main.redis_client") as mock_redis:
                        mock_redis.set = AsyncMock()
                        mock_redis.delete = AsyncMock()
                        
                        await backfill_pipeline(job_id=failed_job.id)

    # FIX 2: Correct assertion order. 
    # Run 1: 501, 502 (crash), 503. 
    # Run 2: 502.
    assert call_log == [501, 502, 503, 502]

@pytest.mark.asyncio
async def test_pipeline_skips_discovery_when_codes_already_persisted():
    """
    If discovered_codes already exist in the SyncJob row, the pipeline must
    NOT call discover_schemes again.  This prevents extra API calls during a
    resume and is the core of the resumability guarantee.
    """
    # Pre-create a FAILED job with discovered_codes already set.
    fake_codes = [601, 602]
    async with AsyncSessionLocal() as session:
        async with session.begin():
            job = SyncJob(status="FAILED", discovered_codes=fake_codes)
            session.add(job)
            await session.flush()
            job_id = job.id

    discover_mock = AsyncMock(return_value=fake_codes)

    with patch("main.discover_schemes", discover_mock):
        with patch("main.process_scheme", new_callable=AsyncMock):
            with patch("main.fetch_all_latest_navs", new_callable=AsyncMock, return_value={}):
                with patch("main.redis_client") as mock_redis:
                    mock_redis.set = AsyncMock()
                    mock_redis.delete = AsyncMock()

                    await backfill_pipeline(job_id=job_id)

    discover_mock.assert_not_called(), (
        "discover_schemes must NOT be called when discovered_codes are already persisted"
    )


@pytest.mark.asyncio
async def test_pipeline_final_status_failed_when_any_fund_fails():
    """
    If at least one fund fails processing, the final job status must be FAILED,
    even though the remaining funds were processed successfully.
    """
    async def fake_process(session, client, code, job_id, prefetched_latest=None):
        if code == 702:
            raise RuntimeError("fund 702 failed")

    with patch("main.discover_schemes", new_callable=AsyncMock, return_value=[701, 702, 703]):
        with patch("main.process_scheme", side_effect=fake_process):
            with patch("main.fetch_all_latest_navs", new_callable=AsyncMock, return_value={}):
                with patch("main.redis_client") as mock_redis:
                    mock_redis.set = AsyncMock()
                    mock_redis.delete = AsyncMock()

                    await backfill_pipeline(job_id=None)

    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(SyncJob).order_by(SyncJob.id.desc())
        )
        job = result.scalars().first()
        assert job.status == "FAILED", (
            f"Job status must be FAILED when any fund fails. Got: {job.status}"
        )
        assert job.failed_funds >= 1


@pytest.mark.asyncio
async def test_pipeline_final_status_success_when_all_funds_pass():
    """
    When all funds process successfully, the final job status must be SUCCESS.
    """
    with patch("main.discover_schemes", new_callable=AsyncMock, return_value=[801, 802]):
        with patch("main.process_scheme", new_callable=AsyncMock):
            with patch("main.fetch_all_latest_navs", new_callable=AsyncMock, return_value={}):
                with patch("main.redis_client") as mock_redis:
                    mock_redis.set = AsyncMock()
                    mock_redis.delete = AsyncMock()

                    await backfill_pipeline(job_id=None)

    async with AsyncSessionLocal() as session:
        from sqlalchemy import select
        result = await session.execute(
            select(SyncJob).order_by(SyncJob.id.desc())
        )
        job = result.scalars().first()
        assert job.status == "SUCCESS", (
            f"Job status must be SUCCESS when all funds pass. Got: {job.status}"
        )
