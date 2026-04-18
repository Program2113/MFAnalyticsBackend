# DESIGN_DECISIONS.md
## Mutual Fund Analytics Platform
**Stack:** Python / FastAPI · PostgreSQL · Redis · mfapi.in

---

## 1. Rate Limiting Strategy

### 1.1 Algorithm: Sliding-Window via Redis Sorted Sets

mfapi.in enforces three independent, simultaneously active limits: **2 requests/second**, **50 requests/minute**, and **300 requests/hour**. A fixed-window counter (e.g. an integer that resets at the top of each second) does not satisfy all three at once — a burst at a window boundary can double the effective rate within the allowed window.

We use **three Redis sorted sets**, one per time window. Each permitted request is recorded as a member with a score equal to its Unix timestamp in milliseconds. Before admitting a new request:

1. Stale members are evicted with `ZREMRANGEBYSCORE`
2. Remaining cardinality is checked with `ZCARD`
3. If **any** of the three windows is at capacity the request is denied; otherwise it is admitted and its timestamp is inserted into all three sets simultaneously

The entire check-and-admit sequence is expressed as a **single Lua script** executed atomically by Redis. Because Redis is single-threaded and Lua scripts are non-interruptible, there is no time-of-check / time-of-use (TOCTOU) race condition — even when multiple FastAPI coroutines call `wait_for_rate_limit()` concurrently.

### 1.2 Proof of Correctness

The following invariants hold at every point in time:

- `rl:rolling:sec` contains exactly the requests admitted in the last 1,000 ms (`ZREMRANGEBYSCORE` uses `now_ms - 1000` as the eviction boundary)
- `rl:rolling:min` contains exactly the requests admitted in the last 60,000 ms
- `rl:rolling:hr` contains exactly the requests admitted in the last 3,600,000 ms
- A request is admitted **only** when all three `ZCARD` values are strictly less than their respective limits (`< 2`, `< 50`, `< 300`). The Lua `OR` condition returns `0` (denied) as soon as any limit is reached
- After admission, the same member string is inserted into all three sets in the same Lua execution, so the three views are always consistent with each other

The member key is formed as `tostring(now_ms) .. '-' .. tostring(INCR rl:seq)`, which guarantees uniqueness even for requests arriving within the same millisecond. The `rl:seq` counter is refreshed with a 24-hour TTL on every call, preventing unbounded growth across Redis restarts.

### 1.3 Coordinating Three Concurrent Limits

All three limits are checked and enforced in **one Redis round-trip**. There is no separate per-second lock or token bucket that could conflict with the per-minute or per-hour check. The three sorted sets share identical member strings, so eviction of a member from the `sec` set does not remove it from the `min` or `hr` sets — each window manages its own independent view of the same request history.

If Redis returns `0` (throttled), the caller sleeps for 200 ms and retries. This polling interval is conservative but keeps the implementation simple. In the normal operating regime the pipeline never approaches the per-second limit because requests are issued sequentially.

`mfapi_get_json()` also wraps the rate-limited fetch with **exponential back-off on HTTP 429** responses (`sleep = min(2^attempt, 10) + 0.1s`, up to 5 retries). The 429 retry re-enters `wait_for_rate_limit()` at the top of the loop, so it cannot bypass the sliding-window check.

### 1.4 Rate Limiter State and Restarts

State is stored entirely in Redis. If the FastAPI process restarts, the sliding-window history survives in Redis as long as the Redis instance is running. The sorted sets have `PEXPIRE` timers slightly longer than their respective windows (e.g. 2,000 ms for the sec set) so they self-clean even if the process never runs again.

For full durability across Redis restarts, Redis should be configured with **AOF persistence** (`appendonly yes`). This is an operational concern outside the application code but is the expected deployment configuration.

---

## 2. Backfill Orchestration

### 2.1 Discovery Phase

Scheme codes are not known in advance. On first run, `discover_schemes()` resolves the 10 target (AMC, category) pairs to concrete mfapi.in scheme codes in two steps:

**Step 1 — full list download:** `GET /mf` returns every scheme on the platform (~10,000+ entries) in a single API call. Each entry is parsed as `MfapiSchemeListItem`, which captures `schemeCode`, `schemeName`, and both ISIN codes. This costs exactly **one API request**.

**Step 2 — candidate validation:** Entries are filtered in-process (zero additional API calls) for Direct Growth plans matching the target AMC and category tokens. For each of the 10 (AMC, category) buckets, candidates are sorted by name length then alphabetically. The first candidate in each bucket is validated by fetching its full history from `GET /mf/{code}`; the returned server-side meta block is checked against the filters. The loop breaks on the first confirmed match, so in the typical case this costs exactly **10 API calls** — one per bucket.

Discovered codes are persisted as a JSON array in `sync_jobs.discovered_codes` immediately after discovery completes. If the process crashes before any fund is processed, the next run resumes from this saved list, **skipping the full-list download and the 10 validation calls entirely**.

### 2.2 Incremental Sync Strategy

After the initial backfill, daily incremental syncs exploit the bulk `GET /mf/latest` endpoint, which returns the current NAV for **all schemes in a single request**. The pipeline calls this once at the start of each sync and distributes the prefetched `MfapiLatestNAVItem` objects to each `process_scheme()` call via the `prefetched_latest` parameter.

Inside `process_scheme()`, the shortcut logic is:

1. If a prefetched `MfapiLatestNAVItem` is available for this code, use it directly (zero additional API calls for the latest-NAV check)
2. Compare the prefetched `latest_nav_date` against `state.last_nav_date` stored in `fund_sync_state`
3. If `latest_nav_date <= last_nav_date` **and** analytics are already computed (`last_analytics_at IS NOT NULL`), return early — no further API calls, no analytics recomputation
4. If the date is newer, fall through to `GET /mf/{code}` for full history and recompute analytics

**API call budget per scenario:**

| Scenario | API calls | % of 300/hr quota |
|---|---|---|
| Initial backfill (first run) | 1 + 10 + 1 + 10 = **22** | 7.3% |
| Daily sync — no new NAVs | **1** (bulk `/mf/latest` only) | 0.3% |
| Daily sync — new NAVs for all 10 funds | 1 + 10 = **11** | 3.7% |
| Worst case — bulk prefetch fails, new NAVs | 10 + 10 = **20** | 6.7% |

Even in the worst case the pipeline consumes fewer than 7% of the hourly quota per run, leaving ample headroom for multiple daily syncs without hitting quota exhaustion.

### 2.3 Resumability After Crash

Crash resilience is built around two database tables:

- **`sync_jobs`:** Persists the job lifecycle (`PENDING / RUNNING / FAILED / SUCCESS`), the full list of `discovered_codes`, and running counters for `processed_funds` and `failed_funds`. If a job is in `RUNNING` or `FAILED` state when `POST /sync/trigger` is called, the existing job is resumed rather than a new one being created.

- **`fund_sync_state`:** Tracks per-fund state (`PENDING / RUNNING / SUCCESS / FAILED`) with the `last_job_id` foreign key. On resume, `pending_codes_for_job()` queries for funds that do not yet have a `SUCCESS` row for the current `job_id`, so already-completed funds are skipped.

This two-level design means a crash anywhere during the per-fund loop is recoverable with no data loss and no duplicate work. All NAV inserts use `ON CONFLICT DO NOTHING` and analytics writes use `ON CONFLICT DO UPDATE`, making all writes fully idempotent.

---

## 3. Storage Schema for Time-Series NAV Data

### 3.1 Table Design

| Table | Purpose | Key columns |
|---|---|---|
| `funds` | Fund master — one row per tracked scheme | `code` (PK), `name`, `amc`, `category`, `latest_nav`, `latest_nav_date` |
| `nav_history` | Immutable time-series of daily NAV values | `fund_code + date` (unique constraint), `nav` (Numeric 15,5) |
| `analytics_cache` | Pre-computed analytics, one row per (fund, window) | `fund_code + window` (unique), `median_return`, `max_drawdown`, `payload` (JSONB) |
| `fund_sync_state` | Per-fund pipeline state, survives restarts | `fund_code` (PK), `sync_state`, `last_nav_date`, `retry_count` |
| `sync_jobs` | Job-level pipeline metadata and resume state | `id` (PK), `status`, `discovered_codes` (JSON), processed/failed counters |

### 3.2 Key Design Decisions

**NAV precision:** NAV values are stored as `Numeric(15, 5)` — five decimal places matching the precision returned by mfapi.in. Using a fixed-precision numeric type rather than `float` prevents accumulation of binary floating-point rounding errors across 10 years of daily data.

**Unique constraint on `(fund_code, date)`:** The `uq_nav_fund_date` constraint on `nav_history` makes all NAV upserts idempotent. Any repeated backfill or re-sync simply does nothing for rows that already exist, with no application-level deduplication needed.

**Analytics payload as JSONB:** The full analytics result for each (fund, window) pair is stored as a JSON blob in `analytics_cache.payload`. This allows `GET /funds/{code}/analytics` to return the pre-computed result with a **single primary-key lookup** — no joins, no aggregations at query time. The structured columns `median_return` and `max_drawdown` are extracted from the payload specifically to support the `ORDER BY` clause in the ranking query without deserialising the JSON.

**Indexes for ranking queries:** Three composite indexes support the `GET /funds/rank` query pattern:

- `ix_fund_category_active` on `(category, is_active)` — eliminates a full table scan when filtering by category
- `ix_analytics_window_median` on `(window, median_return)` — allows `ORDER BY median_return DESC` to use an index scan rather than a sort
- `ix_analytics_window_drawdown` on `(window, max_drawdown)` — same benefit for the `sort_by=max_drawdown` path

---

## 4. Pre-Computation vs On-Demand Trade-offs

### 4.1 Why Pre-Compute

The analytics computation for a single fund and a single window iterates over up to 2,500 NAV rows. With four windows and 10 funds, a fully on-demand approach would run 40 such computations per analytics request — taking several seconds and making the 200 ms API response target impossible. Pre-computing at sync time and caching the results reduces analytics query latency to a **single database primary-key lookup**, well within 200 ms.

### 4.2 Vectorised Computation — O(n log n)

The analytics engine converts the NAV time-series to a numpy array and uses `np.searchsorted` for base-NAV lookup. For each trading day `t` the algorithm binary-searches for the NAV exactly `window_years × 365.25` days prior. This reduces lookup complexity from O(n) per row (pandas boolean mask scan) to O(log n) per row, reducing overall computation from **O(n²) to O(n log n)**.

For a 10-year history (~2,500 rows) the O(n²) approach requires roughly 6.25 million operations per window; the O(n log n) approach requires roughly 28,000.

### 4.3 Analytics Loaded from DB, Not API Response

After upserting new NAV rows, analytics are computed from the full `nav_history` table via `load_nav_dataframe()`, **not** from the raw API response data. This matters because mfapi.in may return a truncated history in some responses. Computing from the database ensures analytics always reflect the complete accumulated history regardless of what the API returned in any single call.

### 4.4 Metric Definitions

| Metric | Definition |
|---|---|
| `rolling_returns` | For each trading day `t`, total percentage return of the fund over the preceding `window` years. Distribution (min, max, median, p25, p75) across all `t`. |
| `cagr` | Compound Annual Growth Rate for each rolling period: `((nav_end / nav_start) ^ (1/years) - 1) × 100`. Distribution (min, max, median). |
| `max_drawdown` | Worst peak-to-trough decline (as a negative %) across all rolling windows. Computed via vectorised `cummax` on the NAV series. |

---

## 5. Schemes With Insufficient History

### 5.1 Detection

`compute_metrics()` returns `status: INSUFFICIENT_HISTORY` when the NAV series does not contain any trading day with a corresponding data point `window_years × 365.25` days earlier. This applies to recently launched funds (a fund launched in 2022 cannot have a 5Y or 10Y window) or any fund where historical data was unavailable at time of backfill.

### 5.2 API Behaviour

`GET /funds/{code}/analytics` returns the cached payload including the `INSUFFICIENT_HISTORY` status and a human-readable `reason` string rather than a `404` or `500` error. The `data_availability` block is always populated so callers can inspect `start_date`, `end_date`, and `nav_data_points` to understand how much history is available.

### 5.3 Ranking Behaviour

`GET /funds/rank` filters out funds with insufficient history by querying only rows where `analytics_cache.median_return IS NOT NULL` (or `max_drawdown IS NOT NULL` for the drawdown sort). Funds without sufficient history are excluded from ranked results rather than appearing with null or zero values. The `total_funds` field reflects only funds with valid analytics for the requested window.

### 5.4 Partial Windows

A fund with 2.5 years of history when a 3Y window is requested will have some rolling periods computed and others skipped. `rolling_periods_analyzed` reflects only the periods for which a valid base NAV existed. Status remains `SUCCESS` as long as at least one rolling period could be computed, allowing partial but valid analytics to be served.

---

## 6. Caching Strategy

### 6.1 Two-Layer Cache

| Layer | Technology | What is cached | Invalidation |
|---|---|---|---|
| Analytics cache | PostgreSQL (`analytics_cache`) | Full pre-computed analytics payload per (fund, window) | Replaced on every sync that inserts new NAV data |
| Pipeline state | Redis (string keys) | `sync_status`, current scheme code, job ID, last run timestamp | Updated at pipeline start/end; no TTL |
| Rate limiter state | Redis (sorted sets) | Timestamps of admitted requests per window | Self-evicting via `ZREMRANGEBYSCORE`; sets expire via `PEXPIRE` |

### 6.2 No HTTP-Layer Response Cache

API responses are not cached at the HTTP layer. With only 10 funds and pre-computed analytics already in PostgreSQL, a database read per request is fast enough (~1–5 ms) and eliminates cache invalidation complexity. Adding an HTTP cache layer would require careful coordination with the sync pipeline and is not warranted at this scale.

---

## 7. Failure Handling

### 7.1 Per-Fund Isolation

Each fund is processed inside its own `try/except` block in `backfill_pipeline()`. A failure on one fund increments `failed_funds` and records the error in `fund_sync_state.last_error` but does **not** abort the pipeline. Remaining funds continue processing. The final job status is set to `FAILED` only if any fund failed; if all succeed the job is marked `SUCCESS`.

### 7.2 Session Rollback on Shortcut Failure

The `/latest` shortcut in `process_scheme()` runs inside a `try/except` that calls `session.rollback()` on failure before falling back to the full history path. This prevents partial writes (e.g. a partially-updated `Fund` row) from being committed when the shortcut fails mid-way. After rollback, `state` and `fund` are re-fetched from the database to ensure the fallback path works with a clean session.

### 7.3 Retry Tracking

`fund_sync_state.retry_count` is incremented on every failure. This provides visibility into repeatedly failing funds and provides a foundation for future max-retry logic (e.g. skip funds that have failed more than 5 times until manually reset). The `last_error` column stores the exception message for debugging.

### 7.4 Discovery Failure

If `discover_schemes()` cannot find a verified scheme code for any of the 10 target (AMC, category) pairs, it raises `RuntimeError` with the list of missing keys. This is treated as a **hard pipeline failure** — the job is marked `FAILED` and no funds are processed. Partial discovery (e.g. 8 of 10 funds found) could lead to a permanently incomplete dataset, so we treat it as all-or-nothing.

---

## 8. Design Choices — Explicit Reasoning

| Area | Choice made | Alternatives considered | Why this choice |
|---|---|---|---|
| **Rate limiter algorithm** | Sliding-window via Redis sorted sets + atomic Lua script | Token bucket (Redis string + DECRBY); fixed window (Redis TTL counter); in-process `asyncio.Semaphore` | Sliding window is the only algorithm that correctly enforces all three overlapping time windows simultaneously. Token bucket doesn't model a "per-minute" limit accurately at window boundaries. In-process locks don't survive restarts or scale across workers. |
| **Storage** | PostgreSQL for persistent data, Redis for transient state | SQLite (simpler); MongoDB (flexible schema); pure Redis | PostgreSQL gives ACID transactions for upsert patterns, strong `Numeric` precision for NAV values, and efficient indexed `ORDER BY` for ranking. Redis is ideal for the rate limiter (atomic sorted-set ops) and pipeline status (fast key-value reads). |
| **Analytics: pre-compute vs on-demand** | Pre-compute at sync time; serve from cache | On-demand per API request; partial pre-compute (summary stats only) | On-demand for 10 years of daily data at O(n log n) per window takes ~100–500 ms per fund — too slow for a 200 ms API budget. Pre-computing in the async background pipeline has no latency constraint. |
| **Backfill strategy** | Sequential per-fund, rate-limited, with bulk `/mf/latest` prefetch | Concurrent fetches with semaphore; separate backfill vs incremental jobs | Sequential processing naturally satisfies the per-second limit (one request every ~0.5 s). The bulk `/mf/latest` endpoint eliminates per-fund latest-NAV calls on daily syncs, achieving the efficiency benefits of concurrency without its complexity. |
| **Caching strategy** | Analytics in PostgreSQL; no HTTP-layer cache | Redis cache for API responses; in-process LRU | With 10 funds and pre-computed results in PostgreSQL, a DB read per analytics request is fast enough without an extra cache layer. Avoiding it eliminates cache-invalidation bugs and simplifies the architecture. |
| **Failure handling** | Per-fund isolation with retry count; hard fail on discovery errors | Abort entire pipeline on first fund failure; silent skip on all errors | Aborting on first failure would leave 9 funds unprocessed after a transient network error on fund 1. Silent skip would make discovery failures invisible. The chosen design surfaces failures clearly while maximising useful work completed per run. |

---

## 9. Extensibility and Evolvability

The system is built to evolve beyond the 10 specified funds and 4 analytics windows:

- **`WINDOWS` dict:** Adding a new analytics window (e.g. `15Y`) requires only a new entry in the constant. The pipeline loop and all analytics infrastructure pick it up automatically.

- **`TARGET_AMCS` / `TARGET_CATEGORIES`:** The target universe is configuration, not code. Adding a new AMC or category requires a new entry in these constants and a re-run of discovery.

- **Event-driven analytics:** Because `nav_history` is a standard time-series table with a date index, any time-bounded query is straightforward. Answering questions like *"How did this fund perform post-COVID (March 2020 onwards)?"* or *"What was the 1Y return in the 12 months following a specific RBI rate decision?"* can be done by filtering `nav_history` by date range and passing the resulting slice to `compute_metrics()`.

- **Payload JSONB:** `analytics_cache.payload` stores the full result as JSONB. New metrics (e.g. Sharpe ratio, Sortino ratio, beta) can be added to the payload without a schema migration. The API response models use `Optional` fields, so new keys are transparent to existing clients.

- **Sync pipeline framework:** The `SyncJob` and `FundSyncState` tables form a generic job framework. Multiple independent pipelines (e.g. equity funds, debt funds) can coexist by using separate job IDs.

---

## Appendix: mfapi.in API Endpoint Usage

| Endpoint | Used in | Purpose |
|---|---|---|
| `GET /mf` | `discover_schemes()` | Download full scheme list for candidate filtering. One call per discovery run. |
| `GET /mf/{code}` | `fetch_scheme_history()`, discovery validation | Full NAV history + metadata. Used for initial backfill and discovery validation. |
| `GET /mf/{code}/latest` | `fetch_scheme_latest()` | Single-fund latest NAV. Fallback when bulk prefetch fails. |
| `GET /mf/latest` | `fetch_all_latest_navs()` | Bulk latest NAV for all schemes. One call covers all 10 funds during daily incremental sync. |

> **Note on `GET /mf/search`:** This endpoint is not used. The full `/mf` list download followed by in-process filtering costs the same one API call as a search query, but avoids the ambiguity of partial-match search results and gives a deterministic, reproducible candidate set for each (AMC, category) bucket.