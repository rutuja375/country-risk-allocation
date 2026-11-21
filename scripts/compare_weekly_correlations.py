import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from project_paths import CHARTS_DIR, DATA_DIR, TABLES_DIR

WEEKS_PER_YEAR = 52
COUNTRY_CAP = 0.20
RISK_AVERSION = 4.0
MEAN_SHRINKAGE = 0.50

# Include the starting price for the first weekly return.
EXPECTED_WEEKS = pd.date_range("2021-10-29", "2026-10-30", freq="W-FRI")


def load_weekly_returns(filename):
    daily = pd.read_csv(
        filename,
        index_col="trading_date",
        parse_dates=["trading_date"],
    ).sort_index()

    # Use each country's last available price in each Friday-ending week.
    weekly = daily.resample("W-FRI").last().reindex(EXPECTED_WEEKS)

    if weekly.shape[1] != 12:
        raise ValueError(f"{filename}: expected 12 country columns.")

    if np.isinf(weekly.to_numpy(dtype=float)).any() or weekly.le(0).any().any():
        raise ValueError(f"{filename}: found infinite or non-positive weekly prices.")

    # Calculate before dropping gaps so each return spans exactly one week.
    return weekly.pct_change(fill_method=None).iloc[1:]


def run_mvo(returns):
    historical_mean = returns.mean() * WEEKS_PER_YEAR
    expected = (
        (1 - MEAN_SHRINKAGE) * historical_mean
        + MEAN_SHRINKAGE * historical_mean.mean()
    ).to_numpy()
    covariance = (returns.cov() * WEEKS_PER_YEAR).to_numpy()
    country_count = returns.shape[1]

    def objective(weights):
        portfolio_return = weights @ expected
        portfolio_variance = weights @ covariance @ weights
        return -portfolio_return + (RISK_AVERSION / 2) * portfolio_variance

    def gradient(weights):
        return -expected + RISK_AVERSION * (covariance @ weights)

    result = minimize(
        objective,
        x0=np.full(country_count, 1 / country_count),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, COUNTRY_CAP)] * country_count,
        constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1}],
        options={"ftol": 1e-10, "maxiter": 500},
    )

    weights = result.x
    if (
        not result.success
        or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-7)
        or weights.min() < -1e-7
        or weights.max() > COUNTRY_CAP + 1e-7
    ):
        raise RuntimeError(f"MVO failed or violated its constraints: {result.message}")

    return (
        pd.Series(weights, index=returns.columns),
        weights @ expected,
        np.sqrt(weights @ covariance @ weights),
    )


def plot_weights(comparison, matched_count):
    order = comparison["ETF minus index (pp)"].abs().sort_values(ascending=False)
    chart = comparison.loc[order.index]
    rows = np.arange(len(chart))
    etf_pct = chart["ETF weight (%)"]
    index_pct = chart["Index weight (%)"]

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    ax.hlines(rows, etf_pct, index_pct, colors="#aab4c3", linewidth=2)
    ax.scatter(
        etf_pct, rows, s=95, facecolors="none", edgecolors="#2459a5",
        linewidths=2, label="ETF", zorder=3,
    )
    ax.scatter(
        index_pct, rows, s=32, marker="s", color="#b54b26",
        label="Local index", zorder=4,
    )
    ax.axvline(
        COUNTRY_CAP * 100, color="#737373", linestyle="--",
        label=f"{COUNTRY_CAP:.0%} country cap",
    )

    ax.set_yticks(rows, chart.index)
    ax.invert_yaxis()
    ax.set_xlim(-0.65, COUNTRY_CAP * 100 + 1)
    ticks = np.linspace(0, COUNTRY_CAP * 100, 5)
    ax.set_xticks(ticks, [f"{tick:g}%" for tick in ticks])
    ax.set_xlabel("Portfolio weight")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)

    fig.suptitle(f"Weekly MVO allocation comparison ({matched_count} matched weeks)")
    fig.savefig(CHARTS_DIR / "mvo_weekly_weights.png", dpi=180)
    plt.show()


def main():
    etf = load_weekly_returns(DATA_DIR / "daily_etf_adjusted_prices_usd.csv")
    index = load_weekly_returns(DATA_DIR / "daily_index_levels_usd.csv")

    if set(etf.columns) != set(index.columns):
        raise ValueError("The two datasets contain different countries.")

    # Keep the same complete return weeks in both datasets.
    shared_weeks = etf.notna().all(axis=1) & index.notna().all(axis=1)
    matched_count = int(shared_weeks.sum())

    if matched_count < 2:
        raise ValueError("The datasets do not have enough shared weekly returns.")

    print(f"Using {matched_count} of {len(shared_weeks)} matched weekly returns.")

    etf = etf.loc[shared_weeks]
    index = index.loc[shared_weeks, etf.columns]

    for name, returns in {"etf": etf, "index": index}.items():
        output = TABLES_DIR / f"{name}_correlation_weekly_5y.csv"
        returns.corr().to_csv(output)
        print(f"{name.upper()}: {len(returns)} weekly returns x {returns.shape[1]} countries")
        print(f"Saved {output}")

    etf_weights, etf_expected, etf_volatility = run_mvo(etf)
    index_weights, index_expected, index_volatility = run_mvo(index)

    comparison = pd.DataFrame({
        "ETF weight (%)": etf_weights * 100,
        "Index weight (%)": index_weights * 100,
        "ETF minus index (pp)": (etf_weights - index_weights) * 100,
    })
    comparison.index.name = "country"
    comparison.to_csv(TABLES_DIR / "mvo_weights_comparison_weekly_5y.csv")

    print(f"\nMVO using {matched_count} matched weekly returns")
    print(comparison.round(2).to_string())
    print(f"\nETF estimate:   return {etf_expected:.2%}, volatility {etf_volatility:.2%}")
    print(f"Index estimate: return {index_expected:.2%}, volatility {index_volatility:.2%}")
    print("\nSaved mvo_weights_comparison_weekly_5y.csv and mvo_weekly_weights.png")

    plot_weights(comparison, matched_count)


if __name__ == "__main__":
    main()