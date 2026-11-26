"""Replay fixed oil and gold allocation what-ifs without optimizing weights."""

import numpy as np
import pandas as pd

from project_paths import DATA_DIR, TABLES_DIR


PORTFOLIO_USD = 300_000_000
MONTHS_PER_TAIL = 12
CANADA_CUT = 0.02
HEDGE_WEIGHT = 0.05
BASELINE = "Baseline"
OIL_CASE = "Canada cut + cash"
GOLD_CASE = "5% GLD"
CASH_CASE = "5% cash"


def main():
    weights = pd.read_csv(TABLES_DIR / "state_c_cap_weights.csv", index_col="country").cap_50_weight_pct / 100
    returns = pd.read_csv(DATA_DIR / "monthly_etf_returns_usd.csv", index_col="month_end")
    returns.index = pd.PeriodIndex(returns.index, freq="M", name="month")
    prices = pd.read_csv(DATA_DIR / "theme_monthly_prices.csv", index_col="month")
    prices.index = pd.PeriodIndex(prices.index, freq="M", name="month")
    if not (
        len(weights) == 12 and weights.index.is_unique and weights.between(0, 1).all()
        and np.isclose(weights.sum(), 1) and set(weights.index) == set(returns.columns)
        and returns.index.equals(pd.period_range("2016-11", "2026-10", freq="M"))
        and prices.index.equals(pd.period_range("2016-10", "2026-10", freq="M"))
        and np.isfinite(returns.to_numpy()).all() and returns.ge(-1).all().all()
        and np.isfinite(prices.to_numpy()).all() and prices.gt(0).all().all()
        and weights.get("Canada", 0) >= CANADA_CUT
    ):
        raise ValueError("Invalid weights, mismatched dates, or missing price or return data.")

    changes = prices.pct_change(fill_method=None).iloc[1:]
    brent = changes.brent_monthly_average_usd
    assets = returns.assign(GLD=changes.gld_adjusted_close_usd, cash=0.0)
    baseline = weights.reindex(assets.columns, fill_value=0.0)
    oil_weights = baseline.copy()
    oil_weights.at["Canada"] -= CANADA_CUT
    oil_weights.at["cash"] = CANADA_CUT
    allocations = pd.DataFrame({
        BASELINE: baseline,
        OIL_CASE: oil_weights,
        GOLD_CASE: baseline * (1 - HEDGE_WEIGHT),
        CASH_CASE: baseline * (1 - HEDGE_WEIGHT),
    })
    allocations.at["GLD", GOLD_CASE] = HEDGE_WEIGHT
    allocations.at["cash", CASH_CASE] = HEDGE_WEIGHT
    if not np.allclose(allocations.sum(), 1) or allocations.lt(0).any().any():
        raise ValueError("Each scenario must have nonnegative weights summing to 100%.")

    monthly_results = assets @ allocations
    summaries, episodes, country_results = [], [], []
    for tail, selected in (("Brent up", brent.nlargest(MONTHS_PER_TAIL)),
                           ("Brent down", brent.nsmallest(MONTHS_PER_TAIL))):
        months = selected.index.sort_values()
        sample = monthly_results.loc[months]
        improvement = sample.subtract(sample[BASELINE], axis=0)
        gold_vs_cash = sample[GOLD_CASE] - sample[CASH_CASE]
        for scenario, result in sample.items():
            gain = improvement[scenario]
            summaries.append({
                "oil_window": tail, "scenario": scenario, "months": len(result),
                "average_return_pct": 100 * result.mean(),
                "average_pnl_usd": PORTFOLIO_USD * result.mean(),
                "worst_month": str(result.idxmin()),
                "worst_return_pct": 100 * result.min(),
                "worst_pnl_usd": PORTFOLIO_USD * result.min(),
                "average_change_vs_baseline_usd": PORTFOLIO_USD * gain.mean(),
                "months_beat_baseline": int(gain.gt(1e-10).sum()),
                "average_change_vs_cash_usd": PORTFOLIO_USD * gold_vs_cash.mean() if scenario == GOLD_CASE else None,
                "months_beat_cash": int(gold_vs_cash.gt(1e-10).sum()) if scenario == GOLD_CASE else None,
            })
        for month, row in sample.iterrows():
            episodes.extend({
                "oil_window": tail, "month": str(month), "brent_change_pct": 100 * brent[month],
                "scenario": scenario, "return_pct": 100 * value,
                "pnl_usd": PORTFOLIO_USD * value,
                "change_vs_baseline_usd": PORTFOLIO_USD * improvement.at[month, scenario],
            } for scenario, value in row.items())
        country_returns = returns.loc[months, weights.index]
        country_results.append(pd.DataFrame({
            "oil_window": tail,
            "portfolio_weight_pct": 100 * weights,
            "average_etf_return_pct": 100 * country_returns.mean(),
            "negative_months": country_returns.lt(0).sum(),
            "average_contribution_pp": 100 * country_returns.mean() * weights,
            "average_contribution_usd": PORTFOLIO_USD * country_returns.mean() * weights,
        }).rename_axis("country").reset_index())

    allocation_table = allocations.rename_axis(index="holding", columns="scenario").stack().rename("weight")
    allocation_table = allocation_table.to_frame().assign(
        weight_pct=lambda frame: 100 * frame.weight,
        allocation_usd=lambda frame: PORTFOLIO_USD * frame.weight,
    ).drop(columns="weight").reset_index()
    summary = pd.DataFrame(summaries)
    tables = {
        "theme_allocation_weights": allocation_table,
        "theme_allocation_summary": summary,
        "theme_allocation_monthly_replay": pd.DataFrame(episodes),
        "theme_allocation_oil_country_evidence": pd.concat(country_results, ignore_index=True),
    }
    for name, table in tables.items():
        table.to_csv(TABLES_DIR / f"{name}.csv", index=False, float_format="%.8f")
    print(summary.drop(columns=["average_pnl_usd", "worst_pnl_usd"]).to_string(
        index=False, float_format="{:,.2f}".format, na_rep="—",
    ))
    print(f"Wrote four comparison tables to {TABLES_DIR}")


if __name__ == "__main__":
    main()
