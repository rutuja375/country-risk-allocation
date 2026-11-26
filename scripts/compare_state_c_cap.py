"""Compare State C caps for the ten-year ETF tangency portfolio."""

import pandas as pd

from etf_tangency_sensitivity import (
    WINDOW_STARTS,
    load_matched_returns,
    max_return_to_volatility,
)
from project_paths import DATA_DIR, TABLES_DIR


STATE_PERIOD = "2022-2024"
STATE_C_CAPS = (0.55, 0.50, 0.45)
MEXICO_TEST_STATE_C_CAP = 0.50
MEXICO_TEST_CAP = 0.03
SHRINKAGE = 0.50
REPLAY_YEAR = "2022"


def load_stress_returns(countries):
    prices = pd.read_csv(
        DATA_DIR / "daily_etf_adjusted_prices_usd.csv",
        index_col="trading_date",
        parse_dates=["trading_date"],
    ).sort_index()
    # Use every ETF week in the replay, even when a local index is missing.
    weekly_prices = prices.loc[:, countries].resample("W-FRI").last()
    returns = weekly_prices.pct_change(fill_method=None).loc[REPLAY_YEAR]
    expected_weeks = pd.date_range(f"{REPLAY_YEAR}-01-01", f"{REPLAY_YEAR}-12-31", freq="W-FRI")
    if not returns.index.equals(expected_weeks) or returns.isna().any().any():
        raise ValueError(f"The stress replay requires complete ETF returns for {REPLAY_YEAR}.")
    return returns


def summarize(weights, expected_return, volatility, state_c, stress_returns):
    weekly_returns = stress_returns @ weights
    opening = pd.Series(1.0, index=[weekly_returns.index[0] - pd.Timedelta(weeks=1)])
    wealth = pd.concat([opening, (1 + weekly_returns).cumprod()])
    drawdown = wealth / wealth.cummax() - 1
    return {
        "state_c_weight_pct": 100 * weights.loc[state_c].sum(),
        "expected_annual_return_pct": 100 * expected_return,
        "annual_volatility_pct": 100 * volatility,
        "return_to_volatility": expected_return / volatility,
        "replay_weeks": len(weekly_returns),
        "replay_total_return_pct": 100 * (wealth.iloc[-1] - 1),
        "replay_max_drawdown_pct": 100 * drawdown.min(),
        "replay_trough_week": drawdown.idxmin().date().isoformat(),
    }


def main():
    sample = load_matched_returns()["etf"].loc[WINDOW_STARTS[10]:]
    states = pd.read_csv(TABLES_DIR / "country_risk_state_migration.csv", index_col="country")
    if not states.index.is_unique or set(states.index) != set(sample.columns):
        raise ValueError("Country-risk states and ETF returns must cover the same countries.")
    states = states.loc[sample.columns, STATE_PERIOD]
    if not states.isin(["A", "B", "C", "D"]).all():
        raise ValueError(f"Invalid country-risk state in {STATE_PERIOD}.")
    state_c = states.index[states.eq("C")]
    stress_returns = load_stress_returns(sample.columns)
    covariance = sample.cov()

    comparison = states.rename("state").to_frame()
    results = []
    scenarios = [("Baseline", "baseline", None, None)]
    scenarios.extend((f"{cap:.0%} cap", f"cap_{cap * 100:.0f}", cap, None) for cap in STATE_C_CAPS)
    scenarios.append((
        f"{MEXICO_TEST_STATE_C_CAP:.0%} State C cap + {MEXICO_TEST_CAP:.0%} Mexico cap",
        f"cap_{MEXICO_TEST_STATE_C_CAP * 100:.0f}_mexico_{MEXICO_TEST_CAP * 100:.0f}",
        MEXICO_TEST_STATE_C_CAP,
        MEXICO_TEST_CAP,
    ))
    for name, prefix, cap, mexico_cap in scenarios:
        allocation, expected_return, volatility = max_return_to_volatility(
            sample, SHRINKAGE, group_cap=None if cap is None else (state_c, cap),
            country_caps=None if mexico_cap is None else {"Mexico": mexico_cap},
        )
        weights = pd.Series(allocation, index=sample.columns)
        # Each country gets its own variance plus half of each pair's covariance term.
        variance_contributions = weights * (covariance @ weights)
        variance_shares = 100 * variance_contributions / variance_contributions.sum()
        comparison[f"{prefix}_weight_pct"] = 100 * weights
        comparison[f"{prefix}_variance_share_pct"] = variance_shares
        results.append({
            "scenario": name,
            "matched_input_weeks": len(sample),
            "state_c_cap_pct": None if cap is None else 100 * cap,
            "mexico_cap_pct": None if mexico_cap is None else 100 * mexico_cap,
            "state_c_variance_share_pct": variance_shares.loc[state_c].sum(),
            **summarize(weights, expected_return, volatility, state_c, stress_returns),
        })

    summary = pd.DataFrame(results)
    summary.to_csv(TABLES_DIR / "state_c_cap_summary.csv", index=False, float_format="%.8f")
    comparison.to_csv(TABLES_DIR / "state_c_cap_weights.csv", index_label="country", float_format="%.8f")
    print(summary.to_string(index=False, float_format="{:.4f}".format))
    print()
    print(comparison.to_string(float_format="{:.4f}".format))


if __name__ == "__main__":
    main()
