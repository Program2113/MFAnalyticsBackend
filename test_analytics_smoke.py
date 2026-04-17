from datetime import date

import pandas as pd

from main import compute_metrics, is_direct_growth_scheme, normalize_assignment_category, verified_target_key, MfapiSchemeMeta


def test_compute_metrics_insufficient_history():
    df = pd.DataFrame(
        [{"date": date(2025, 1, 1), "nav": 10.0}, {"date": date(2025, 6, 1), "nav": 11.0}]
    ).set_index("date")
    out = compute_metrics(df, 3)
    assert out["status"] == "INSUFFICIENT_HISTORY"
    assert out["rolling_periods_analyzed"] == 0
    assert out["data_availability"]["sufficient_for_window"] is False


def test_compute_metrics_success_and_drawdown():
    rows = [
        {"date": date(2020, 1, 1), "nav": 10.0},
        {"date": date(2021, 1, 1), "nav": 20.0},
        {"date": date(2021, 6, 1), "nav": 10.0},
        {"date": date(2022, 1, 1), "nav": 30.0},
        {"date": date(2023, 1, 1), "nav": 40.0},
    ]
    df = pd.DataFrame(rows).set_index("date")
    out = compute_metrics(df, 1)
    assert out["status"] == "SUCCESS"
    assert out["rolling_periods_analyzed"] > 0
    assert "median" in out["rolling_returns"]
    assert "median" in out["cagr"]
    assert out["max_drawdown"] <= 0


def test_scheme_verification_helpers():
    assert is_direct_growth_scheme("Axis Small Cap Fund - Direct Plan - Growth") is True
    assert normalize_assignment_category(None, "Axis Mid Cap Fund - Direct Growth") == "Equity: Mid Cap"
    meta = MfapiSchemeMeta(
        fund_house="Kotak Mahindra Mutual Fund",
        scheme_type="Open Ended Schemes",
        scheme_category="Equity Scheme - Small Cap Fund",
        scheme_code=123,
        scheme_name="Kotak Small Cap Fund - Direct Plan - Growth",
        isin_growth=None,
        isin_div_reinvestment=None,
    )
    assert verified_target_key(meta) == ("Kotak Mahindra", "Equity: Small Cap")
