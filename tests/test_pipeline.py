import pytest
from datetime import datetime, timezone
from main import AsyncSessionLocal, SyncJob, FundSyncState, pending_codes_for_job

@pytest.mark.asyncio
async def test_pipeline_resumes_only_pending_codes():
    async with AsyncSessionLocal() as session:
        # 1. Setup a mocked failed job with 3 discovered codes
        job = SyncJob(
            status="FAILED", 
            discovered_codes=[101, 102, 103]
        )
        session.add(job)
        await session.flush()

        # 2. Mock that fund 101 was successfully synced before the crash
        state_101 = FundSyncState(
            fund_code=101, 
            sync_state="SUCCESS", 
            last_job_id=job.id
        )
        # Mock that fund 102 failed during the crash
        state_102 = FundSyncState(
            fund_code=102, 
            sync_state="FAILED", 
            last_job_id=job.id
        )
        session.add_all([state_101, state_102])
        await session.commit()

        # 3. Request pending codes for the job
        pending_codes = await pending_codes_for_job(session, job)

        # 4. Assert 101 is skipped, and 102/103 are resumed
        assert 101 not in pending_codes
        assert 102 in pending_codes
        assert 103 in pending_codes