# DESIGN_DECISIONS

## Rate limiting
The service enforces all three mfapi.in limits simultaneously using Redis sorted sets and a Lua script:
- 2 requests per rolling second
- 50 requests per rolling minute
- 300 requests per rolling hour

Each outgoing mfapi call goes through the same gate, including scheme discovery. The limiter trims expired timestamps from rolling windows, checks counts atomically, and records the new request only if all three windows are below threshold.

## Scheme discovery
Discovery is two-stage and deterministic:
1. Read the mfapi master list and collect candidate schemes using name-based filters for Direct/Growth plus rough AMC/category matching.
2. Validate each candidate by calling the scheme-history endpoint and checking metadata (`fund_house`, `scheme_category`, and scheme name) before accepting it.

Candidates are evaluated in deterministic order so the same verified code is selected on repeat runs.

## Backfill and incremental sync
The sync pipeline discovers the tracked schemes, then processes each scheme independently.

For each scheme:
1. Fetch full mfapi history.
2. Upsert fund metadata and latest NAV snapshot.
3. Insert only NAV rows newer than the last stored NAV date.
4. Recompute precomputed analytics for 1Y, 3Y, 5Y, and 10Y only when fresh data arrives.
5. Update fund sync state.

This keeps the local behavior incremental even though mfapi returns full history.

## Resumability
Two levels of state are persisted:
- `sync_jobs`: one record per sync run with discovered fund codes, progress counters, status, and current fund.
- `fund_sync_state`: per-fund progress, retry count, last synced NAV date, analytics timestamps, and the last job that touched the fund.

If a run crashes, `/sync/trigger` resumes the most recent failed or pending job instead of creating a fresh one. Funds already marked `SUCCESS` for that job are skipped.

## Storage schema
- `funds`: canonical metadata plus latest NAV snapshot
- `nav_history`: time-series NAV data with uniqueness on `(fund_code, date)`
- `analytics_cache`: precomputed analytics payload and ranking fields per `(fund_code, window)`
- `fund_sync_state`: per-fund resumability and operational state
- `sync_jobs`: job-level resumability and progress reporting

## Analytics semantics
Analytics are computed on real NAV dates, not row counts. For each window (1Y/3Y/5Y/10Y):
- rolling return uses the nearest available NAV on or before the target start date
- CAGR is computed for every valid rolling window and summarized as min/max/median
- max drawdown is computed inside each rolling window and the worst drawdown across all windows is stored

This avoids incorrect assumptions about fixed trading-day spacing.

## Precompute vs on-demand
Analytics are precomputed after ingestion rather than calculated during request handling. This keeps read latency low for:
- fund analytics endpoint
- ranking endpoint

The cache stores both display payload and sortable numeric fields (`median_return`, `max_drawdown`).

## Insufficient history
When a fund does not have enough history for a requested window, the analytics cache stores an `INSUFFICIENT_HISTORY` payload with availability metadata. The API still returns a structured analytics response for that fund/window so consumers can distinguish missing history from a missing fund.
