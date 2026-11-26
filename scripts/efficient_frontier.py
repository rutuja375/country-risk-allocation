"""Plot the efficient frontiers and maximize expected return / volatility."""

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.optimize import linprog, minimize

from compare_weekly_correlations import (
    COUNTRY_CAP,
    MEAN_SHRINKAGE,
    WEEKS_PER_YEAR,
    load_weekly_returns,
)
from project_paths import CHARTS_DIR, DATA_DIR
from etf_tangency_sensitivity import max_return_to_volatility


def solve_frontier(sample, shrinkage=MEAN_SHRINKAGE, point_count=201, *, group_cap=None):
    """Return frontier coordinates, corresponding weights, and reference portfolios."""
    raw = sample.mean().to_numpy() * WEEKS_PER_YEAR
    mu = (1 - shrinkage) * raw + shrinkage * raw.mean()
    cov = sample.cov().to_numpy() * WEEKS_PER_YEAR
    n = len(mu)
    unit = np.ones(n)
    bounds = [(0, COUNTRY_CAP)] * n
    budget = {"type": "eq", "fun": lambda w: unit @ w - 1, "jac": lambda w: unit}
    constraints = [budget]
    cap_args = {}
    if group_cap is not None:
        countries, limit = group_cap
        members = sample.columns.isin(countries).astype(float)
        cap_args = {"A_ub": members[None, :], "b_ub": [limit]}
        constraints.append({
            "type": "ineq", "fun": lambda w: limit - members @ w, "jac": lambda w: -members,
        })
    opts = {"ftol": 1e-12, "maxiter": 600}

    def variance(w):
        return w @ cov @ w

    def grad(w):
        return 2 * (cov @ w)

    tangency, _, _ = max_return_to_volatility(sample, shrinkage, group_cap=group_cap)
    maxret = linprog(-mu, A_eq=unit[None, :], b_eq=[1], bounds=bounds, method="highs", **cap_args)
    assert maxret.success, maxret.message
    minvar = minimize(
        variance, tangency, jac=grad, bounds=bounds, constraints=constraints, method="SLSQP", options=opts,
    )
    assert minvar.success, minvar.message

    def point(w):
        return np.sqrt(variance(w)), mu @ w

    extrema = (point(minvar.x), point(maxret.x))
    best = point(tangency)
    targets = np.unique(
        np.append(np.linspace(extrema[0][1], extrema[1][1], point_count), best[1])
    )
    curve = []
    portfolios = []
    for target in targets:
        if target == best[1]:
            curve.append(best)
            portfolios.append(tangency)
            continue
        if target == extrema[0][1]:
            curve.append(extrema[0])
            portfolios.append(minvar.x)
            continue
        if target == extrema[1][1]:
            curve.append(extrema[1])
            portfolios.append(maxret.x)
            continue
        mix = (target - extrema[0][1]) / (extrema[1][1] - extrema[0][1])
        start = (1 - mix) * minvar.x + mix * maxret.x
        constraint = {"type": "eq", "fun": lambda w: mu @ w - target, "jac": lambda w: mu}
        fit = minimize(
            variance, start, jac=grad, bounds=bounds, constraints=[*constraints, constraint],
            method="SLSQP", options=opts,
        )
        assert fit.success, fit.message
        assert abs(mu @ fit.x - target) < 1e-8
        curve.append(point(fit.x))
        portfolios.append(fit.x)
    vol, ret = np.asarray(curve).T
    assert np.all(np.diff(vol) > -1e-7)
    assert abs(np.max(ret / vol) - best[1] / best[0]) < 1e-6
    assert abs(tangency.sum() - 1) < 1e-8 and tangency.max() <= COUNTRY_CAP + 1e-7

    return vol, ret, np.asarray(portfolios), extrema, best, tangency


def main():
    etf = load_weekly_returns(DATA_DIR / "daily_etf_adjusted_prices_usd.csv")
    index = load_weekly_returns(DATA_DIR / "daily_index_levels_usd.csv")
    if set(etf.columns) != set(index.columns):
        raise ValueError("The ETF and index datasets must contain the same countries.")
    shared = etf.notna().all(axis=1) & index.notna().all(axis=1)
    matched_weeks = int(shared.sum())
    if matched_weeks < 2:
        raise ValueError("At least two matched weekly returns are required.")
    etf, index = etf.loc[shared], index.loc[shared, etf.columns]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained", sharex=True, sharey=True)
    fig.suptitle("Efficient frontiers and maximum return-to-volatility portfolios", fontsize=17)
    for ax, label, sample, color in zip(axes, ("US-listed ETFs", "Local indices in USD"), (etf, index), ("#235789", "#137a70")):
        vol, ret, _, extrema, best, tangency = solve_frontier(sample)
        ax.plot(vol, ret, color=color, linewidth=2.7, label="Efficient frontier")
        ax.scatter(*np.asarray(extrema).T, color=color, s=43, marker="D", zorder=3, label="Frontier endpoints")
        ax.scatter(*best, color="#b34a1c", s=195, marker="*", zorder=4, label="Maximum return / volatility")
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Estimated annual volatility")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=9)
        print(f"\n{label} | {matched_weeks} matched weeks")
        for point_label, (volatility, expected_return) in zip(
            ("Minimum volatility", "Maximum expected return", "Tangency portfolio"),
            (*extrema, best),
        ):
            print(f"{point_label}: return {expected_return:.2%}, volatility {volatility:.2%}")
        print(f"Expected return / volatility: {best[1] / best[0]:.4f}")
        for country, w in zip(sample.columns, tangency):
            print(f"{country}: {w:.6%}")

    axes[0].set_ylabel("Estimated annual return")
    fig.supxlabel(
        f"{matched_weeks} matched weeks | Long only | {COUNTRY_CAP:.0%} country cap"
        f" | {MEAN_SHRINKAGE:.0%} return shrinkage",
        fontsize=10,
    )
    fig.savefig(CHARTS_DIR / "efficient_frontier.png", dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
