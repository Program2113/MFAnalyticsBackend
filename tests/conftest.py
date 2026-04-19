import pytest_asyncio
from main import engine, Base

@pytest_asyncio.fixture(autouse=True)
async def setup_test_database():
    """
    Automatically creates all database tables before a test runs, 
    and drops them after the test finishes to ensure a completely clean slate.
    """
    # 1. Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield  # 2. Let the test run
    
    # 3. Clean up (drop all tables) so the next test starts fresh
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        
    # 4. CRITICAL FIX: Dispose of the SQLAlchemy connection pool!
    # This prevents the "Event loop is closed" error by forcing the 
    # global engine to create fresh connections in the new test's event loop.
    await engine.dispose()