"""Plot country risk state transitions and the fitted clusters."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cluster_country_risk import PROFILE_TERMS
from project_paths import CHARTS_DIR, TABLES_DIR


STATE_COLORS = {
    "A": "#b9dfdc",
    "B": "#b9d2ef",
    "C": "#f6c8a3",
    "D": "#d8caed",
}
STATE_MARKERS = dict(zip(STATE_COLORS, ("o", "s", "^", "D")))


def plot_risk_states(migration, profiles):
    color_index = {state: index for index, state in enumerate(STATE_COLORS)}
    if set(profiles.index) != set(STATE_COLORS) or not migration.isin(list(color_index)).all().all():
        raise ValueError("Expected migration cells and profiles for the four reference states A–D.")

    fig, ax = plt.subplots(figsize=(10.5, 7.5), layout="constrained")
    values = migration.to_numpy()
    color_values = migration.apply(lambda column: column.map(color_index)).to_numpy()
    ax.imshow(
        color_values, cmap=ListedColormap(list(STATE_COLORS.values())),
        vmin=-0.5, vmax=len(STATE_COLORS) - 0.5, aspect="auto", interpolation="nearest",
    )

    ax.set_xticks(range(migration.shape[1]), migration.columns.str.replace("-", "–"))
    ax.set_yticks(range(migration.shape[0]), migration.index)
    ax.xaxis.tick_top()
    ax.tick_params(which="both", length=0, pad=8)
    ax.set_xticks(np.arange(migration.shape[1]) - 0.5, minor=True)
    ax.set_yticks(np.arange(migration.shape[0]) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (row, col), state in np.ndenumerate(values):
        ax.text(col, row, state, ha="center", va="center", color="#17243b", fontweight="bold")

    fig.suptitle("Country risk states across five periods", fontsize=18, fontweight="bold")
    handles = [
        Patch(facecolor=color, label=f"{state}  {profiles.at[state, 'profile']}")
        for state, color in STATE_COLORS.items()
    ]
    ax.legend(handles=handles, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.035))
    return fig


def plot_cluster_scatter(standardized, migration, profiles):
    assignments = migration.reset_index().melt(id_vars="country", var_name="years", value_name="state")
    observations = standardized.assign(
        years=standardized["start_year"].astype(str) + "-" + standardized["end_year"].astype(str),
    ).merge(assignments, on=["country", "years"], validate="one_to_one")
    values = observations[list(PROFILE_TERMS)].to_numpy()
    if len(observations) != len(standardized) or len(observations) != len(assignments):
        raise ValueError("The standardized rows and migration assignments must match.")
    if not np.isfinite(values).all():
        raise ValueError("Expected finite standardized clustering inputs.")

    # Project the four model inputs for display while retaining the fitted assignments.
    centered = values - values.mean(axis=0)
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    points = centered @ axes[:2].T
    variance_share = singular_values[:2] ** 2 / np.square(singular_values).sum()

    fig, ax = plt.subplots(figsize=(10.5, 7.5), layout="constrained")
    for state, color in STATE_COLORS.items():
        group = points[observations["state"].eq(state)]
        ax.scatter(
            *group.T, color=color, marker=STATE_MARKERS[state], s=85,
            edgecolors="#34445b", linewidths=0.7, alpha=0.92,
            label=f"{state}  {profiles.at[state, 'profile']}", zorder=3,
        )
        ax.scatter(
            *group.mean(axis=0), color=color, marker="X", s=180,
            edgecolors="#17243b", linewidths=1.2, zorder=4,
        )

    ax.set_xlabel(f"Principal component 1 ({variance_share[0]:.1%} of variance)")
    ax.set_ylabel(f"Principal component 2 ({variance_share[1]:.1%} of variance)")
    ax.set_title("Each point represents one country in one period", loc="left", fontsize=11, pad=10)
    ax.grid(color="#e4e9ef", linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.09)
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="#17243b", marker="X", linestyle="none", label="Cluster center"))
    ax.legend(handles=handles, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("K-means country risk clusters", fontsize=18, fontweight="bold")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true", help="Save without opening a plot window.")
    args = parser.parse_args()

    migration = pd.read_csv(TABLES_DIR / "country_risk_state_migration.csv", index_col="country")
    profiles = pd.read_csv(TABLES_DIR / "country_risk_state_profiles.csv", index_col="state")
    standardized = pd.read_csv(TABLES_DIR / "country_risk_standardized_by_period.csv")
    charts = {
        "country_risk_state_migration.png": plot_risk_states(migration, profiles),
        "country_risk_cluster_scatter.png": plot_cluster_scatter(standardized, migration, profiles),
    }
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, fig in charts.items():
        output = CHARTS_DIR / name
        fig.savefig(output, dpi=180)
        print(f"Wrote {output}")
    if not args.save_only:
        plt.show()
    for fig in charts.values():
        plt.close(fig)


if __name__ == "__main__":
    main()
