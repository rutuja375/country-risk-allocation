import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from project_paths import CHARTS_DIR, TABLES_DIR

etf = pd.read_csv(TABLES_DIR / "etf_correlation_weekly_5y.csv", index_col=0)
index = pd.read_csv(TABLES_DIR / "index_correlation_weekly_5y.csv", index_col=0)
countries = etf.index.tolist()
etf = etf.loc[countries, countries]
index = index.loc[countries, countries]
hide_cells = np.triu(np.ones(etf.shape, dtype=bool))

fig, axes = plt.subplots(1, 2, figsize=(17, 8), layout="constrained")

for ax, (title, matrix) in zip(
    axes,
    [("US-listed ETF returns", etf), ("Local index returns in USD", index)],
):
    values = matrix.to_numpy()
    visible_values = np.ma.masked_where(hide_cells, values)

    heatmap = ax.imshow(
        visible_values,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    positions = range(len(countries))
    ax.set_xticks(positions, countries, rotation=45, ha="right")
    ax.set_yticks(positions, countries)
    ax.set_title(title, pad=12)
    ax.tick_params(length=0)

    for row in positions:
        for col in range(row):
            value = values[row, col]
            ax.text(
                col, row, f"{value:.2f}",
                ha="center", va="center",
                color="white" if abs(value) >= 0.65 else "black",
                fontsize=8,
            )

fig.suptitle("Weekly country equity correlations · Nov 2021–Oct 2026")
fig.colorbar(heatmap, ax=axes, shrink=0.75, label="Pearson correlation")
fig.savefig(CHARTS_DIR / "country_correlations_weekly_5y.png", dpi=180, bbox_inches="tight")
plt.show()