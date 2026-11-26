"""Plot the four country risk inputs on a shared color scale."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from project_paths import CHARTS_DIR, TABLES_DIR


PANELS = {
    "debt_z": "Government debt",
    "fiscal_balance_risk_z": "Fiscal balance",
    "current_account_risk_z": "Current account",
    "governance_risk_z": "Combined governance",
}


def plot_heatmaps(scores):
    countries = scores["country"].drop_duplicates().tolist()
    periods = scores[["start_year", "end_year"]].drop_duplicates().sort_values("start_year")
    years = periods["start_year"].to_list()
    labels = [f"{start}\N{EN DASH}{end}" for start, end in periods.itertuples(index=False, name=None)]
    limit = np.ceil(scores[list(PANELS)].abs().to_numpy().max())
    norm = Normalize(vmin=-limit, vmax=limit)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), layout="constrained")
    for ax, (column, title) in zip(axes.flat, PANELS.items()):
        matrix = scores.pivot(index="country", columns="start_year", values=column)
        values = matrix.loc[countries, years].to_numpy()
        heatmap = ax.imshow(values, cmap="RdBu_r", norm=norm, aspect="auto")
        ax.set_xticks(range(len(labels)), labels, fontsize=9)
        ax.set_yticks(range(len(countries)), countries, fontsize=9)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=9)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for (row, col), value in np.ndenumerate(values):
            red, green, blue, _ = heatmap.cmap(norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "white" if luminance < 0.5 else "#17243b"
            ax.text(col, row, f"{value:+.2f}", ha="center", va="center", color=color, fontsize=9)

    fig.suptitle("Country risk across five periods", fontsize=19, fontweight="bold")
    fig.colorbar(
        heatmap, ax=axes, location="bottom", shrink=0.7, aspect=45, pad=0.04,
        label="Robust z-score   |   0 = pooled median   |   Higher = greater vulnerability on this factor",
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true", help="Save without opening a plot window.")
    args = parser.parse_args()

    scores = pd.read_csv(TABLES_DIR / "country_risk_standardized_by_period.csv")
    fig = plot_heatmaps(scores)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    output = CHARTS_DIR / "country_risk_heatmaps.png"
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")
    if not args.save_only:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
