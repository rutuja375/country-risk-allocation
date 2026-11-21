
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from project_paths import CHARTS_DIR, TABLES_DIR

scenario_columns = ["lookback_years", "shrinkage_pct"]
scenarios = pd.MultiIndex.from_product(
    [[3, 5, 10], [25, 50, 75]],
    names=scenario_columns,
)
metadata = [
    "matched_weeks",
    "expected_return_pct",
    "annual_volatility_pct",
    "return_to_volatility",
]


def load_weights(filename):
    data = pd.read_csv(TABLES_DIR / filename).set_index(scenario_columns)
    if not data.index.is_unique or set(data.index) != set(scenarios):
        raise ValueError(f"{filename}: expected exactly the nine sensitivity settings.")

    # Put countries on rows and scenarios on columns.
    return data.drop(columns=metadata).reindex(scenarios).T


etf = load_weights("etf_tangency_sensitivity_scenarios.csv")
index = load_weights("index_tangency_sensitivity_scenarios.csv")

if set(etf.index) != set(index.index):
    raise ValueError("The ETF and index files contain different countries.")

countries = etf.index
xlabels = [f"{years}y\n{shrinkage}%" for years, shrinkage in scenarios]

fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True, layout="constrained")
cmap = plt.get_cmap("Blues")

for ax, (title, weights) in zip(
    axes, [("US-listed ETFs", etf), ("Local indices in USD", index.loc[countries])]
):
    values = weights.to_numpy(dtype=float)
    heatmap = ax.imshow(values, cmap=cmap, vmin=0, vmax=20, aspect="auto")

    ax.set_title(title, pad=10)
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(countries)), countries)
    ax.set_xlabel("Lookback window and return shrinkage")
    ax.tick_params(length=0)

    # Separate the three lookback groups.
    for boundary in (2.5, 5.5):
        ax.axvline(boundary, color="white", linewidth=2)

    for (row, col), value in np.ndenumerate(values):
        ax.text(
            col, row, f"{value:.1f}",
            ha="center", va="center", fontsize=8,
            color="white" if value >= 12 else "#162638",
        )

fig.suptitle("Tangency portfolio weights across nine settings")
colorbar = fig.colorbar(heatmap, ax=axes, ticks=[0, 5, 10, 15, 20])
colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
colorbar.set_label("Country weight")

output = CHARTS_DIR / "tangency_sensitivity_heatmaps.png"
fig.savefig(output, dpi=200)
print(f"Saved {output}")
plt.show()