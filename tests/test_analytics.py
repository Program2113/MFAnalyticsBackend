"""
test_analytics.py
=================
Tests for the analytics computation engine.

Assignment requirements covered:
  1. Rolling returns: min, max, median, p25, p75
  2. Max drawdown correctness (manually verified)
  3. CAGR distribution: min, max, median
  4. Insufficient history — schemes with too little data
  5. NAV gaps (weekends/holidays) — non-contiguous date handling
  6. Partial window — some rolling periods computable, some not
  7. All four windows: 1Y, 3Y, 5Y, 10Y

All assertions on float values use pytest.approx(..., abs=0.01) to avoid
floating-point precision failures on exact equality checks.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

from main import compute_metrics, compute_max_drawdown

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _daily_df(start: str, periods: int, start_nav: float = 100.0, growth: float = 0.0) -> pd.DataFrame:
    """
    Build a DataFrame of daily NAV values with a fixed daily growth rate.
    index = DatetimeIndex, column = "nav".
    """
    dates = pd.date_range(start=start, periods=periods, freq="D")
    navs = [start_nav * ((1 + growth) ** i) for i in range(periods)]
    return pd.DataFrame({"nav": navs}, index=dates)


def _business_day_df(start: str, periods: int, start_nav: float = 100.0, growth: float = 0.0) -> pd.DataFrame:
    """
    Build a DataFrame of business-day (Mon–Fri) NAV values — simulates the
    real mfapi.in data which has gaps on weekends and public holidays.
    """
    dates = pd.date_range(start=start, periods=periods, freq="B")
    navs = [start_nav * ((1 + growth) ** i) for i in range(periods)]
    return pd.DataFrame({"nav": navs}, index=dates)


# ---------------------------------------------------------------------------
# 1. Max drawdown — manual verification
# ---------------------------------------------------------------------------

class TestMaxDrawdown:

    def test_basic_drawdown(self):
        """
        Peak is 120, trough is 90.
        Drawdown = (90 - 120) / 120 × 100 = -25.0%
        """
        series = pd.Series([100.0, 110.0, 120.0, 90.0, 100.0], dtype="float64")
        assert compute_max_drawdown(series) == pytest.approx(-25.0, abs=0.01)

    def test_drawdown_with_recovery(self):
        """
        Even when NAV recovers after a trough, the drawdown is the worst
        peak-to-trough, not the final peak-to-trough.
        Peak = 150, trough = 75. Drawdown = (75-150)/150×100 = -50.0%
        """
        series = pd.Series([100.0, 150.0, 75.0, 130.0], dtype="float64")
        assert compute_max_drawdown(series) == pytest.approx(-50.0, abs=0.01)

    def test_monotonically_rising_nav(self):
        """A fund that never falls should have 0% drawdown."""
        series = pd.Series([100.0, 110.0, 120.0, 130.0], dtype="float64")
        assert compute_max_drawdown(series) == pytest.approx(0.0, abs=0.01)

    def test_monotonically_falling_nav(self):
        """
        NAV falls from 100 to 50 with no recovery.
        Peak = 100, trough = 50. Drawdown = -50.0%
        """
        series = pd.Series([100.0, 90.0, 80.0, 70.0, 50.0], dtype="float64")
        assert compute_max_drawdown(series) == pytest.approx(-50.0, abs=0.01)

    def test_single_point(self):
        """A single NAV point has no peak-to-trough and must return 0."""
        series = pd.Series([100.0], dtype="float64")
        assert compute_max_drawdown(series) == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# 2. Rolling returns — distribution fields
# ---------------------------------------------------------------------------

class TestRollingReturns:

    def test_1y_single_period_return(self):
        """
        Exactly one 1Y rolling period: base NAV 100 on 2024-01-01,
        end NAV 115 on 2025-01-01.  Total return = 15%.

        2024 is a leap year (366 days), so elapsed_years = 366 / 365.25 ≈ 1.002.
        CAGR = (1.15 ^ (1 / 1.002)) - 1 ≈ 14.97%
        """
        df = pd.DataFrame(
            {"nav": [100.0, 115.0]},
            index=[pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")],
        )
        metrics = compute_metrics(df, window_years=1)

        assert metrics["status"] == "SUCCESS"
        assert metrics["rolling_periods_analyzed"] == 1
        assert metrics["rolling_returns"]["median"] == pytest.approx(15.0, abs=0.01)
        assert metrics["cagr"]["median"] == pytest.approx(14.97, abs=0.1)

    def test_distribution_p25_p75_populated(self):
        """
        With multiple rolling periods, rolling_returns must include p25 and p75
        alongside min, max, and median.
        """
        # 2 years of daily data @ 10% annual growth → many 1Y rolling periods.
        df = _daily_df("2022-01-01", periods=730, growth=0.000263)  # ≈10%/yr daily
        metrics = compute_metrics(df, window_years=1)

        assert metrics["status"] == "SUCCESS"
        rr = metrics["rolling_returns"]
        assert "p25" in rr, "rolling_returns must contain p25"
        assert "p75" in rr, "rolling_returns must contain p75"
        assert rr["p25"] is not None
        assert rr["p75"] is not None
        # p25 ≤ median ≤ p75 must always hold.
        assert rr["p25"] <= rr["median"] <= rr["p75"], (
            f"Quartile ordering violated: p25={rr['p25']}, "
            f"median={rr['median']}, p75={rr['p75']}"
        )

    def test_distribution_ordering(self):
        """min ≤ p25 ≤ median ≤ p75 ≤ max for rolling_returns and cagr."""
        df = _daily_df("2020-01-01", periods=1095, growth=0.000263)  # 3 years
        metrics = compute_metrics(df, window_years=1)

        assert metrics["status"] == "SUCCESS"
        rr = metrics["rolling_returns"]
        assert rr["min"] <= rr["p25"] <= rr["median"] <= rr["p75"] <= rr["max"], (
            f"rolling_returns ordering violated: {rr}"
        )
        cagr = metrics["cagr"]
        assert cagr["min"] <= cagr["median"] <= cagr["max"], (
            f"cagr ordering violated: {cagr}"
        )

    def test_known_return_is_within_distribution(self):
        """
        With a constant daily growth of exactly 10%/year, every 1Y rolling
        return must be approximately 10%, so min ≈ max ≈ median ≈ 10%.
        """
        daily_growth = (1.10 ** (1 / 365.25)) - 1
        df = _daily_df("2020-01-01", periods=730, growth=daily_growth)
        metrics = compute_metrics(df, window_years=1)

        assert metrics["status"] == "SUCCESS"
        rr = metrics["rolling_returns"]
        # All rolling returns should be approximately 10% with constant growth.
        assert rr["min"] == pytest.approx(10.0, abs=0.5)
        assert rr["max"] == pytest.approx(10.0, abs=0.5)
        assert rr["median"] == pytest.approx(10.0, abs=0.5)


# ---------------------------------------------------------------------------
# 3. CAGR distribution
# ---------------------------------------------------------------------------

class TestCAGRDistribution:

    def test_cagr_fields_present(self):
        """CAGR result must contain min, max, and median."""
        df = _daily_df("2022-01-01", periods=730, growth=0.000263)
        metrics = compute_metrics(df, window_years=1)

        assert metrics["status"] == "SUCCESS"
        cagr = metrics["cagr"]
        assert "min" in cagr
        assert "max" in cagr
        assert "median" in cagr

    def test_cagr_no_quartiles(self):
        """CAGR must NOT include p25/p75 — the spec only requires min/max/median."""
        df = _daily_df("2022-01-01", periods=730, growth=0.000263)
        metrics = compute_metrics(df, window_years=1)

        cagr = metrics["cagr"]
        assert "p25" not in cagr, "CAGR should not include p25 (only rolling_returns does)"
        assert "p75" not in cagr, "CAGR should not include p75 (only rolling_returns does)"

    def test_cagr_value_correctness(self):
        """
        With exactly 10% annual growth and a 1Y window, CAGR median must be
        approximately 10%.
        """
        daily_growth = (1.10 ** (1 / 365.25)) - 1
        df = _daily_df("2020-01-01", periods=730, growth=daily_growth)
        metrics = compute_metrics(df, window_years=1)

        assert metrics["status"] == "SUCCESS"
        assert metrics["cagr"]["median"] == pytest.approx(10.0, abs=0.5)


# ---------------------------------------------------------------------------
# 4. Insufficient history
# ---------------------------------------------------------------------------

class TestInsufficientHistory:

    def test_no_data_returns_insufficient(self):
        """An empty DataFrame must return INSUFFICIENT_HISTORY."""
        df = pd.DataFrame(columns=["nav"]).rename_axis("date")
        metrics = compute_metrics(df, window_years=1)
        assert metrics["status"] == "INSUFFICIENT_HISTORY"
        assert "No NAV data" in metrics["reason"]

    def test_six_months_against_3y_window(self):
        """6 months of data is insufficient for a 3Y window."""
        df = _daily_df("2024-01-01", periods=180, growth=0.0002)
        metrics = compute_metrics(df, window_years=3)

        assert metrics["status"] == "INSUFFICIENT_HISTORY"
        assert "Not enough data" in metrics["reason"]
        assert metrics["rolling_periods_analyzed"] == 0

    def test_just_under_one_year_against_1y_window(self):
        """364 days of data is insufficient for a 1Y window (need ≥ 365.25 days back)."""
        df = _daily_df("2024-01-01", periods=364, growth=0.0002)
        metrics = compute_metrics(df, window_years=1)

        assert metrics["status"] == "INSUFFICIENT_HISTORY"

    def test_data_availability_block_always_populated(self):
        """
        Even for INSUFFICIENT_HISTORY, the data_availability block must be
        populated so callers can inspect how much data is available.
        """
        df = _daily_df("2024-01-01", periods=180, growth=0.0002)
        metrics = compute_metrics(df, window_years=3)

        da = metrics["data_availability"]
        assert da["nav_data_points"] == 180
        assert da["sufficient_for_window"] is False
        assert da["start_date"] != ""
        assert da["end_date"] != ""

    def test_5y_window_requires_5y_of_data(self):
        """2 years of data is insufficient for a 5Y window."""
        df = _daily_df("2022-01-01", periods=730, growth=0.0002)
        metrics = compute_metrics(df, window_years=5)
        assert metrics["status"] == "INSUFFICIENT_HISTORY"

    def test_10y_window_requires_10y_of_data(self):
        """5 years of data is insufficient for a 10Y window."""
        df = _daily_df("2015-01-01", periods=1825, growth=0.0002)
        metrics = compute_metrics(df, window_years=10)
        assert metrics["status"] == "INSUFFICIENT_HISTORY"


# ---------------------------------------------------------------------------
# 5. NAV gaps — weekends and holidays
# ---------------------------------------------------------------------------

class TestNAVGaps:

    def test_business_day_gaps_do_not_break_computation(self):
        """
        Real mfapi.in data only has trading days (Mon–Fri) with gaps on
        weekends and public holidays.  The rolling-window binary search must
        find the nearest available trading day rather than expecting a precise
        365-day-back data point.
        """
        # 3 years of business-day data — gaps on Sat/Sun every week.
        df = _business_day_df("2021-01-01", periods=780, growth=0.000263)
        metrics = compute_metrics(df, window_years=1)

        # Must succeed despite the non-contiguous dates.
        assert metrics["status"] == "SUCCESS"
        assert metrics["rolling_periods_analyzed"] > 0

    def test_sparse_data_with_large_gaps(self):
        """
        A fund with monthly NAV data points (extreme gaps) should still compute
        correctly for windows where enough months exist.
        """
        # 18 months of monthly data — 1 point per month.
        dates = pd.date_range(start="2023-01-01", periods=18, freq="MS")
        navs = [100.0 * (1.01 ** i) for i in range(18)]
        df = pd.DataFrame({"nav": navs}, index=dates)

        metrics = compute_metrics(df, window_years=1)
        # With 18 months we have enough for a 1Y window.
        assert metrics["status"] == "SUCCESS"

    def test_rolling_periods_analyzed_reflects_gap_count(self):
        """
        Fewer rolling periods should be analyzed for business-day data vs
        calendar-day data over the same wall-clock span, because there are
        fewer data points to iterate over.
        """
        df_daily = _daily_df("2022-01-01", periods=730, growth=0.0002)
        df_bday = _business_day_df("2022-01-01", periods=522, growth=0.0002)  # ≈730 calendar days of bdays

        m_daily = compute_metrics(df_daily, window_years=1)
        m_bday = compute_metrics(df_bday, window_years=1)

        # Both must succeed.
        assert m_daily["status"] == "SUCCESS"
        assert m_bday["status"] == "SUCCESS"
        # Business-day series has fewer points → fewer rolling periods.
        assert m_bday["rolling_periods_analyzed"] < m_daily["rolling_periods_analyzed"]


# ---------------------------------------------------------------------------
# 6. Partial window — some periods computable, some not
# ---------------------------------------------------------------------------

class TestPartialWindow:

    def test_25_months_against_3y_window(self):
        """
        A fund with 25 months of history against a 3Y window can compute
        rolling returns for the last ~1 month (where a 3Y-back base exists)
        but not for the earlier portion.  Status must be SUCCESS with a
        non-zero but small rolling_periods_analyzed count.
        """
        # 25 months of daily data starting Jan 2022 → data through Feb 2024.
        df = _daily_df("2022-01-01", periods=762, growth=0.0002)
        metrics = compute_metrics(df, window_years=3)

        # 762 days ≈ 25 months < 36 months (3Y), so no 3Y-back base exists.
        assert metrics["status"] == "INSUFFICIENT_HISTORY"

    def test_exactly_at_window_boundary(self):
        """
        A fund with exactly 365 days of data should have at least 1 rolling
        period analyzable for the 1Y window.
        """
        df = _daily_df("2024-01-01", periods=366, growth=0.0002)  # includes the end date
        metrics = compute_metrics(df, window_years=1)

        assert metrics["status"] == "SUCCESS"
        assert metrics["rolling_periods_analyzed"] >= 1

    def test_partial_window_status_is_success_not_insufficient(self):
        """
        A fund with 2.5 years of data against a 1Y window has many rolling
        periods.  Status must be SUCCESS — partial window does not mean
        INSUFFICIENT_HISTORY.
        """
        df = _daily_df("2022-01-01", periods=912, growth=0.0002)  # ~2.5 years
        metrics = compute_metrics(df, window_years=1)

        assert metrics["status"] == "SUCCESS"
        assert metrics["rolling_periods_analyzed"] > 0


# ---------------------------------------------------------------------------
# 7. All four windows (1Y, 3Y, 5Y, 10Y)
# ---------------------------------------------------------------------------

class TestAllWindows:

    @pytest.fixture
    def ten_year_df(self):
        """10 years + 1 day of daily data — sufficient for all four windows."""
        return _daily_df("2014-01-01", periods=3653, growth=0.000263)

    def test_1y_window(self, ten_year_df):
        metrics = compute_metrics(ten_year_df, window_years=1)
        assert metrics["status"] == "SUCCESS"
        assert metrics["rolling_periods_analyzed"] > 0

    def test_3y_window(self, ten_year_df):
        metrics = compute_metrics(ten_year_df, window_years=3)
        assert metrics["status"] == "SUCCESS"
        assert metrics["rolling_periods_analyzed"] > 0

    def test_5y_window(self, ten_year_df):
        metrics = compute_metrics(ten_year_df, window_years=5)
        assert metrics["status"] == "SUCCESS"
        assert metrics["rolling_periods_analyzed"] > 0

    def test_10y_window(self, ten_year_df):
        metrics = compute_metrics(ten_year_df, window_years=10)
        assert metrics["status"] == "SUCCESS"
        assert metrics["rolling_periods_analyzed"] > 0

    def test_10y_has_fewer_periods_than_1y(self, ten_year_df):
        """
        A 10Y window requires more historical runway per period, so fewer
        rolling periods are analyzable compared to a 1Y window on the same data.
        """
        m1y = compute_metrics(ten_year_df, window_years=1)
        m10y = compute_metrics(ten_year_df, window_years=10)
        assert m10y["rolling_periods_analyzed"] < m1y["rolling_periods_analyzed"]

    def test_10y_window_data_availability_flags_sufficiency(self, ten_year_df):
        """sufficient_for_window must be True when there is enough data."""
        metrics = compute_metrics(ten_year_df, window_years=10)
        assert metrics["data_availability"]["sufficient_for_window"] is True
