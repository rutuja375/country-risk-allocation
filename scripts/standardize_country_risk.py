"""Put the country-period clustering inputs on a common robust scale."""

import numpy as np
import pandas as pd

from build_country_comparison import FACTORS, GOVERNANCE_FACTORS, PERIODS
from prepare_country_risk_data import COUNTRIES
from project_paths import TABLES_DIR


IDENTIFIERS = ["iso3", "country", "etf", "period", "start_year", "end_year"]
FINANCIAL_FACTORS = {
    "gross_government_debt_pct_gdp": ("debt_z", 1),
    "government_balance_pct_gdp": ("fiscal_balance_risk_z", -1),
    "current_account_balance_pct_gdp": ("current_account_risk_z", -1),
}


def robust_z_scores(values):
    centered = values - values.median()
    scale = 1.4826 * centered.abs().median()
    if scale.eq(0).any():
        raise ValueError(f"Zero median absolute deviation: {scale.index[scale.eq(0)].tolist()}")
    return centered / scale


def build_standardized_inputs(period_averages):
    """Return four signed features for each of the 60 country-period rows."""
    keys = ["iso3", "period", "start_year", "end_year"]
    expected = pd.MultiIndex.from_tuples(
        [(code, period, start, end) for period, (start, end) in PERIODS.items() for code in COUNTRIES],
        names=keys,
    )
    observed = pd.MultiIndex.from_frame(period_averages[keys])
    if not observed.is_unique or not expected.sort_values().equals(observed.sort_values()):
        raise ValueError("Expected one row for each project country in every comparison period.")

    values = period_averages[FACTORS].apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(values)
    if invalid.any().any():
        raise ValueError(f"Non-numeric or missing values in: {invalid.columns[invalid.any()].tolist()}")

    scores = robust_z_scores(values)
    output = period_averages[IDENTIFIERS].copy()
    for factor, (label, direction) in FINANCIAL_FACTORS.items():
        output[label] = scores[factor] * direction

    governance = -scores[GOVERNANCE_FACTORS].mean(axis=1)
    output["governance_risk_z"] = robust_z_scores(governance.to_frame()).iloc[:, 0]
    return output


def main():
    source = TABLES_DIR / "country_factor_averages_by_period.csv"
    output_path = TABLES_DIR / "country_risk_standardized_by_period.csv"
    output = build_standardized_inputs(pd.read_csv(source))
    output.to_csv(output_path, index=False, float_format="%.8f")

    features = output.drop(columns=IDENTIFIERS)
    print(f"Standardized {len(output)} country-period rows across {len(features.columns)} features.")
    print("Pearson correlations among the four inputs:")
    print(features.corr().round(2).to_string())
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
