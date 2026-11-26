"""Run ETF tangency sensitivity and provide the same calculation for indices."""

from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

import compare_weekly_correlations as weekly
from project_paths import DATA_DIR, TABLES_DIR


COUNTRY_CAP = weekly.COUNTRY_CAP
WEEKS_PER_YEAR = weekly.WEEKS_PER_YEAR
WINDOW_STARTS = {3: "2023-11-03", 5: "2021-11-05", 10: "2016-11-04"}
SHRINKAGE_VALUES = (0.25, 0.50, 0.75)
MEANINGFUL_WEIGHT_PCT = 1.0


def load_matched_returns():
    original_weeks = weekly.EXPECTED_WEEKS
    # Include the starting price needed for the first 10-year weekly return.
    weekly.EXPECTED_WEEKS = pd.date_range("2016-10-28", "2026-10-30", freq="W-FRI")
    try:
        etf = weekly.load_weekly_returns(DATA_DIR / "daily_etf_adjusted_prices_usd.csv")
        index = weekly.load_weekly_returns(DATA_DIR / "daily_index_levels_usd.csv")
    finally:
        weekly.EXPECTED_WEEKS = original_weeks

    if set(etf.columns) != set(index.columns):
        raise ValueError("ETF and local-index data must contain the same countries.")
    matched = etf.notna().all(axis=1) & index.notna().all(axis=1)
    return {"etf": etf.loc[matched], "index": index.loc[matched, etf.columns]}


def max_return_to_volatility(returns, shrinkage, *, group_cap=None, country_caps=None):
    """Solve for tangency with optional group and tighter individual caps."""
    means = returns.mean().to_numpy() * WEEKS_PER_YEAR
    expected = (1 - shrinkage) * means + shrinkage * means.mean()
    covariance = returns.cov().to_numpy() * WEEKS_PER_YEAR
    count = len(expected)

    def variance(scaled_weights):
        return scaled_weights @ covariance @ scaled_weights

    # The return/volatility ratio is unchanged by positive scaling.
    # Fix scaled expected return and minimize variance, then normalize to 100%.
    scaled_returns = expected / 0.10
    individual_limits = np.full(count, COUNTRY_CAP)
    for country, limit in (country_caps or {}).items():
        if country not in returns.columns or not 0 <= limit <= COUNTRY_CAP:
            raise ValueError(f"Invalid individual cap for {country}: {limit}.")
        individual_limits[returns.columns.get_loc(country)] = limit
    # Enforce scaled_weight[i] <= individual_limits[i] * sum(scaled_weights).
    cap_matrix = individual_limits[:, None] - np.eye(count)
    start = np.full(count, 1 / count)
    if group_cap is not None:
        countries, limit = group_cap
        members = returns.columns.isin(countries).astype(float)
        if not 0 <= limit <= 1 or not members.any() or set(countries) - set(returns.columns):
            raise ValueError("Group cap requires known countries and a limit from 0 to 1.")
        # Since w = y / sum(y), group_weight <= limit becomes
        # sum(y in group) <= limit * sum(y).
        cap_matrix = np.vstack((cap_matrix, limit - members))
    if group_cap is not None or country_caps:
        if np.any(cap_matrix @ start < -1e-12) or scaled_returns @ start <= 0:
            feasible = linprog(
                -scaled_returns, A_ub=-cap_matrix, b_ub=np.zeros(len(cap_matrix)),
                A_eq=np.ones((1, count)), b_eq=[1], bounds=(0, None), method="highs",
            )
            if not feasible.success:
                raise ValueError(f"Allocation constraints are infeasible: {feasible.message}")
            start = feasible.x

    initial_return = scaled_returns @ start
    if initial_return <= 0:
        raise ValueError("This solver requires a positive expected return at the starting weights.")
    result = minimize(
        variance,
        x0=start / initial_return,
        jac=lambda y: 2 * covariance @ y,
        method="SLSQP",
        bounds=[(0, None)] * count,
        constraints=[
            {"type": "eq", "fun": lambda y: scaled_returns @ y - 1,
             "jac": lambda y: scaled_returns},
            {"type": "ineq", "fun": lambda y: cap_matrix @ y,
             "jac": lambda y: cap_matrix},
        ],
        options={"ftol": 1e-12, "maxiter": 600},
    )
    if not result.success:
        raise RuntimeError(f"Tangency optimization failed: {result.message}")

    weights = result.x / result.x.sum()
    if not np.isfinite(weights).all() or weights.min() < -1e-7 or np.any(cap_matrix @ weights < -1e-7):
        raise RuntimeError("Tangency weights violate the allocation bounds.")

    annual_return = expected @ weights
    annual_volatility = np.sqrt(variance(weights))
    if not np.isfinite(annual_volatility) or annual_volatility <= 0:
        raise RuntimeError("Tangency volatility must be finite and positive.")

    return weights, annual_return, annual_volatility


def run_sensitivity(dataset):
    returns = load_matched_returns()[dataset]
    rows = []
    for (years, start), shrinkage in product(WINDOW_STARTS.items(), SHRINKAGE_VALUES):
        sample = returns.loc[start:]
        if len(sample) < 2:
            raise ValueError(f"Too few matched returns for the {years}-year window.")
        weights, annual_return, annual_volatility = max_return_to_volatility(
            sample, shrinkage
        )
        rows.append({
            "lookback_years": years,
            "shrinkage_pct": shrinkage * 100,
            "matched_weeks": len(sample),
            "expected_return_pct": annual_return * 100,
            "annual_volatility_pct": annual_volatility * 100,
            "return_to_volatility": annual_return / annual_volatility,
            **dict(zip(returns.columns, np.maximum(weights, 0) * 100)),
        })

    scenarios = pd.DataFrame(rows)
    weights_pct = scenarios[returns.columns]
    summary = pd.DataFrame({
        "runs_at_or_above_1pct": weights_pct.ge(MEANINGFUL_WEIGHT_PCT - 1e-7).sum(),
        "min_weight_pct": weights_pct.min(),
        "max_weight_pct": weights_pct.max(),
    })
    summary.index.name = "country"

    output_prefix = TABLES_DIR / f"{dataset}_tangency_sensitivity"
    scenarios.to_csv(output_prefix.with_name(f"{output_prefix.name}_scenarios.csv"), index=False)
    summary.to_csv(output_prefix.with_name(f"{output_prefix.name}_summary.csv"))
    print(f"{dataset.upper()}: ran {len(scenarios)} scenarios.")
    print(scenarios.groupby("lookback_years")["matched_weeks"].first().to_string())
    print(summary.round(2).to_string())


if __name__ == "__main__":
    run_sensitivity("etf")
