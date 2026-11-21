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
    raw = sample.mean().to_numpy() * WEEKS_PER_YEAR
    mu = (1 - MEAN_SHRINKAGE) * raw + MEAN_SHRINKAGE * raw.mean()
    cov = sample.cov().to_numpy() * WEEKS_PER_YEAR
    n = len(mu)
    unit = np.ones(n)
    bounds = [(0, COUNTRY_CAP)] * n
    budget = {"type": "eq", "fun": lambda w: unit @ w - 1, "jac": lambda w: unit}
    opts = {"ftol": 1e-12, "maxiter": 600}

    def variance(w):
        return w @ cov @ w

    def grad(w):
        return 2 * (cov @ w)

    minvar = minimize(variance, unit / n, jac=grad, bounds=bounds, constraints=[budget], method="SLSQP", options=opts)
    maxret = linprog(-mu, A_eq=unit[None, :], b_eq=[1], bounds=bounds, method="highs")
    assert minvar.success and maxret.success

    # Normalize excess-free return to a positive constant via a scale-free
    # variable. This makes the maximum return/sigma search a convex QP.
    scaled_mu = mu / 0.1
    cap_matrix = COUNTRY_CAP * np.ones((n, n)) - np.eye(n)
    normalized = minimize(
        variance, minvar.x / (scaled_mu @ minvar.x), jac=grad,
        bounds=[(0, None)] * n,
        constraints=[
            {"type": "eq", "fun": lambda y: scaled_mu @ y - 1, "jac": lambda y: scaled_mu},
            {"type": "ineq", "fun": lambda y: cap_matrix @ y, "jac": lambda y: cap_matrix},
        ],
        method="SLSQP", options=opts,
    )
    assert normalized.success, normalized.message
    tangency = normalized.x / normalized.x.sum()
    def point(w):
        return np.sqrt(variance(w)), mu @ w

    extrema = (point(minvar.x), point(maxret.x))
    best = point(tangency)
    targets = np.unique(np.append(np.linspace(extrema[0][1], extrema[1][1], 201), best[1]))
    curve = []
    for target in targets:
        mix = (target - extrema[0][1]) / (extrema[1][1] - extrema[0][1])
        start = (1 - mix) * minvar.x + mix * maxret.x
        constraint = {"type": "eq", "fun": lambda w: mu @ w - target, "jac": lambda w: mu}
        fit = minimize(variance, start, jac=grad, bounds=bounds, constraints=[budget, constraint], method="SLSQP", options=opts)
        assert fit.success, fit.message
        assert abs(mu @ fit.x - target) < 1e-8
        curve.append(point(fit.x))
    vol, ret = np.asarray(curve).T
    assert np.all(np.diff(vol) > -1e-7)
    assert abs(np.max(ret / vol) - best[1] / best[0]) < 1e-6
    assert abs(tangency.sum() - 1) < 1e-8 and tangency.max() <= COUNTRY_CAP + 1e-7

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
