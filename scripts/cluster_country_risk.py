"""Fit and compare country risk states across all five periods."""

import numpy as np
import pandas as pd
from scipy.cluster.vq import ClusterError, kmeans2
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from build_country_comparison import INDICATORS
from project_paths import TABLES_DIR
from standardize_country_risk import IDENTIFIERS


PROFILE_TERMS = {
    "debt_z": ("Lower debt", "Higher debt"),
    "fiscal_balance_risk_z": ("Stronger fiscal balance", "Weaker fiscal balance"),
    "current_account_risk_z": ("Stronger current account", "Weaker current account"),
    "governance_risk_z": ("Stronger governance", "Weaker governance"),
}
REFERENCE_CLUSTERS = 4


def fit_kmeans(values, n_clusters):
    rng = np.random.default_rng(42)
    best = None
    for _ in range(50):
        try:
            centers, labels = kmeans2(values, n_clusters, minit="++", iter=100, missing="raise", rng=rng)
        except ClusterError:
            continue
        if np.unique(labels).size != n_clusters:
            continue
        inertia = float(np.square(values - centers[labels]).sum())
        if best is None or inertia < best[0]:
            best = (inertia, labels)
    if best is None:
        raise ValueError(f"No valid K-means run found for {n_clusters} clusters.")
    return best


def mean_silhouette(values, labels):
    distances = cdist(values, values)
    clusters = np.unique(labels)
    scores = np.zeros(len(labels))
    for row, label in enumerate(labels):
        same = labels == label
        if same.sum() == 1:
            continue
        within = distances[row, same].sum() / (same.sum() - 1)
        nearest = min(distances[row, labels == other].mean() for other in clusters if other != label)
        scale = max(within, nearest)
        if scale:
            scores[row] = (nearest - within) / scale
    return scores.mean()


def compare_cluster_counts(values):
    capped = np.clip(values, -3, 3)
    diagnostics = []
    reference_labels = None
    for n_clusters in (3, 4, 5):
        inertia, labels = fit_kmeans(values, n_clusters)
        _, capped_labels = fit_kmeans(capped, n_clusters)
        matches = pd.crosstab(labels, capped_labels).to_numpy()
        rows, columns = linear_sum_assignment(matches, maximize=True)
        diagnostics.append({
            "clusters": n_clusters,
            "uncapped_inertia": inertia,
            "uncapped_silhouette": mean_silhouette(values, labels),
            "capped_silhouette": mean_silhouette(capped, capped_labels),
            "same_assignments_after_cap": matches[rows, columns].sum(),
            "country_period_rows": len(labels),
            "smallest_uncapped_cluster": np.bincount(labels).min(),
        })
        if n_clusters == REFERENCE_CLUSTERS:
            reference_labels = labels
    return pd.DataFrame(diagnostics), reference_labels


def summarize_states(data, labels):
    data = data.assign(cluster=labels)
    grouped = data.groupby("cluster")
    profiles = grouped[INDICATORS + list(PROFILE_TERMS)].mean().sort_values("debt_z")
    profiles["observations"] = grouped.size()
    profiles["countries"] = grouped["iso3"].nunique()
    profiles["state"] = [chr(ord("A") + i) for i in range(len(profiles))]

    descriptions = []
    for _, profile in profiles[list(PROFILE_TERMS)].iterrows():
        strongest = profile.abs().nlargest(2).index
        descriptions.append(" / ".join(PROFILE_TERMS[factor][int(profile[factor] > 0)] for factor in strongest))
    profiles["profile"] = descriptions
    profiles = profiles[["state", "profile", "observations", "countries", *INDICATORS, *PROFILE_TERMS]]

    data["state"] = data["cluster"].map(profiles["state"])
    data["years"] = data["start_year"].astype(str) + "-" + data["end_year"].astype(str)
    countries = data["country"].drop_duplicates()
    migration = data.pivot(index="country", columns="years", values="state").loc[countries]
    if migration.isna().any().any():
        raise ValueError("Missing country-period risk state.")
    return profiles.reset_index(drop=True), migration


def main():
    standardized = pd.read_csv(TABLES_DIR / "country_risk_standardized_by_period.csv")
    raw = pd.read_csv(TABLES_DIR / "country_factor_averages_by_period.csv")
    data = standardized.merge(raw[IDENTIFIERS + INDICATORS], on=IDENTIFIERS, validate="one_to_one")
    values = data[list(PROFILE_TERMS)].to_numpy()
    if len(data) != len(standardized) or len(data) != len(raw) or not np.isfinite(values).all():
        raise ValueError("The standardized inputs and raw period averages must cover the same rows.")

    diagnostics, labels = compare_cluster_counts(values)
    profiles, migration = summarize_states(data, labels)
    diagnostics.to_csv(TABLES_DIR / "country_risk_cluster_diagnostics.csv", index=False, float_format="%.6f")
    profiles.to_csv(TABLES_DIR / "country_risk_state_profiles.csv", index=False, float_format="%.6f")
    migration.to_csv(TABLES_DIR / "country_risk_state_migration.csv")
    print(diagnostics.round(4).to_string(index=False))
    print(f"Wrote the {REFERENCE_CLUSTERS}-state profiles and migration table to {TABLES_DIR}")


if __name__ == "__main__":
    main()
