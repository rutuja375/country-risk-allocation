from itertools import product

import pandas as pd

import compare_weekly_correlations as mvo
from project_paths import DATA_DIR, TABLES_DIR


def main():
    # Include one extra price to calculate the first 10-year weekly return.
    mvo.EXPECTED_WEEKS = pd.date_range("2016-10-28", "2026-10-30", freq="W-FRI")

    etf = mvo.load_weekly_returns(DATA_DIR / "daily_etf_adjusted_prices_usd.csv")
    index = mvo.load_weekly_returns(DATA_DIR / "daily_index_levels_usd.csv")

    if set(etf.columns) != set(index.columns):
        raise ValueError("ETF and index data must contain the same countries.")

    # Use the same week-matching rule as the original comparison.
    # The optimizer uses ETF returns for every scenario.
    shared = etf.notna().all(axis=1) & index.notna().all(axis=1)
    etf = etf.loc[shared]

    window_starts = {3: "2023-11-03", 5: "2021-11-05", 10: "2016-11-04"}
    risk_values = (2.0, 4.0, 8.0)
    shrinkage_values = (0.25, 0.50, 0.75)
    mvo.COUNTRY_CAP = 0.20

    rows = []
    for years, risk_aversion, shrinkage in product(
        window_starts, risk_values, shrinkage_values
    ):
        sample = etf.loc[window_starts[years]:]
        mvo.RISK_AVERSION = risk_aversion
        mvo.MEAN_SHRINKAGE = shrinkage
        weights, _, _ = mvo.run_mvo(sample)

        rows.append({
            "lookback_years": years,
            "risk_aversion": risk_aversion,
            "shrinkage": shrinkage,
            "matched_weeks": len(sample),
            **weights.mul(100).clip(lower=0).to_dict(),
        })

    scenarios = pd.DataFrame(rows)
    weights_pct = scenarios[etf.columns]

    summary = pd.DataFrame({
        "runs_at_or_above_1pct": weights_pct.ge(1 - 1e-7).sum(),
        "median_weight_pct": weights_pct.median(),
        "min_weight_pct": weights_pct.min(),
        "max_weight_pct": weights_pct.max(),
    }).sort_values("runs_at_or_above_1pct", ascending=False)
    summary.index.name = "country"

    scenarios.to_csv(TABLES_DIR / "mvo_sensitivity_scenarios.csv", index=False)
    summary.to_csv(TABLES_DIR / "mvo_sensitivity_summary.csv")

    print(f"Ran {len(scenarios)} scenarios.")
    print(scenarios.groupby("lookback_years")["matched_weeks"].first().to_string())
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()