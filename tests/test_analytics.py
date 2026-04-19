import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from decimal import Decimal
from main import compute_metrics, compute_max_drawdown

def test_max_drawdown_correctness():
    # Peak is 120, Trough is 90. Drop is 30.
    # Drawdown = (90 - 120) / 120 = -25.0%
    navs = [100.0, 110.0, 120.0, 90.0, 100.0]
    series = pd.Series(navs, dtype="float64")
    
    drawdown = compute_max_drawdown(series)
    assert drawdown == -25.0

def test_compute_metrics_1y_cagr_and_returns():
    # Create exactly 1 year of data with a known return.
    # Start NAV: 100. End NAV: 115. Return: 15%. 
    # Note: 2024 is a leap year (366 days). 366 / 365.25 = 1.002 years.
    # CAGR = (1.15 ^ (1/1.002)) - 1 = 14.97%
    dates = [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-07-01"),
        pd.Timestamp("2025-01-01")
    ]
    df = pd.DataFrame({"nav": [100.0, 105.0, 115.0]}, index=dates)
    
    # Analyze for a 1Y window
    metrics = compute_metrics(df, window_years=1)
    
    assert metrics["status"] == "SUCCESS"
    assert metrics["rolling_periods_analyzed"] == 1
    assert metrics["rolling_returns"]["median"] == 15.0
    assert metrics["cagr"]["median"] == 14.97

def test_insufficient_history():
    # Only 6 months of data, but requesting a 3Y window
    dates = pd.date_range(start="2024-01-01", periods=180, freq="D")
    df = pd.DataFrame({"nav": np.random.uniform(100, 150, 180)}, index=dates)
    
    metrics = compute_metrics(df, window_years=3)
    
    assert metrics["status"] == "INSUFFICIENT_HISTORY"
    assert "Not enough data" in metrics["reason"]
