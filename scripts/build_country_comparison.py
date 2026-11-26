"""Compare country risk factors across historical periods."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from country_total_return_usd import FUNDS
from prepare_country_risk_data import COUNTRIES
from project_paths import TABLES_DIR


YEARS = range(2002, 2026)
PERIODS = {
    "Pre-crisis": (2002, 2007),
    "Global financial crisis": (2008, 2009),
    "Post-crisis to pre-pandemic": (2010, 2019),
    "Pandemic": (2020, 2021),
    "Recent comparison": (2022, 2024),
}
FACTOR_COLUMNS = {
    "Government debt (% of GDP)": "gross_government_debt_pct_gdp",
    "Fiscal balance (% of GDP)": "government_balance_pct_gdp",
    "Current account (% of GDP)": "current_account_balance_pct_gdp",
    "Political stability (0–100)": "political_stability_score",
    "Government effectiveness (0–100)": "government_effectiveness_score",
}
CONTEXT_COLUMNS = {
    "Real GDP growth (%)": "real_gdp_growth_pct",
    "CPI inflation (%)": "average_cpi_inflation_pct",
}
WORKBOOK_COLUMNS = {
    "Country code": "iso3",
    "Year": "year",
    "Country": "country",
    **FACTOR_COLUMNS,
    **CONTEXT_COLUMNS,
    "IMF estimated indicators": "weo_estimated_indicators",
}
FACTORS = list(FACTOR_COLUMNS.values())
INDICATORS = FACTORS + list(CONTEXT_COLUMNS.values())
GOVERNANCE_FACTORS = ["political_stability_score", "government_effectiveness_score"]


def build_comparison(annual):
    """Validate the annual data and return one row per country and period."""
    duplicate_rows = annual.loc[annual.duplicated(["iso3", "year"], keep=False)]
    if not duplicate_rows.empty:
        keys = duplicate_rows[["iso3", "year"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Duplicate country-year rows: {keys}")

    expected = pd.MultiIndex.from_product([COUNTRIES, YEARS], names=["iso3", "year"])
    observed = annual.set_index(["iso3", "year"]).index
    missing = expected.difference(observed)
    unexpected = observed.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(f"Country-year coverage mismatch. Missing: {list(missing)}. Unexpected: {list(unexpected)}.")
    if not annual["country"].eq(annual["iso3"].map(COUNTRIES)).all():
        raise ValueError("Workbook country names do not match their country codes.")
    if set(FUNDS) != set(COUNTRIES.values()):
        raise ValueError("Workbook countries and market ETF countries differ.")

    values = annual[INDICATORS].apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(values)
    if invalid.any().any():
        details = annual.loc[invalid.any(axis=1), ["iso3", "year"]].copy()
        details["missing_or_invalid_factors"] = invalid.loc[details.index].apply(
            lambda row: ", ".join(row.index[row]), axis=1,
        )
        raise ValueError(f"Missing or invalid factor values: {details.to_dict('records')}")
    if not values[GOVERNANCE_FACTORS].apply(lambda series: series.between(0, 100)).all().all():
        raise ValueError("Governance scores must be between 0 and 100.")

    estimates = annual["weo_estimated_indicators"].astype("string").fillna("").str.split(";")
    estimated = estimates.map(lambda fields: any(field.strip() in INDICATORS for field in fields))
    comparisons = []
    for period, (start, end) in PERIODS.items():
        in_window = annual["year"].between(start, end)
        if (in_window & estimated).any():
            raise ValueError(f"The {period} period contains IMF estimates for comparison indicators.")
        comparison = values.loc[in_window].groupby(annual.loc[in_window, "iso3"]).mean()
        comparison = comparison.reindex(COUNTRIES).reset_index()
        comparison.insert(1, "country", comparison["iso3"].map(COUNTRIES))
        comparison.insert(2, "etf", comparison["country"].map(FUNDS))
        comparison.insert(3, "period", period)
        comparison.insert(4, "start_year", start)
        comparison.insert(5, "end_year", end)
        comparisons.append(comparison)
    return pd.concat(comparisons, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, required=True)
    args = parser.parse_args()

    annual = pd.read_excel(
        args.workbook, sheet_name="Country risk data", usecols=list(WORKBOOK_COLUMNS),
    ).rename(columns=WORKBOOK_COLUMNS)
    comparison = build_comparison(annual)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output = TABLES_DIR / "country_factor_averages_by_period.csv"
    comparison.to_csv(output, index=False, float_format="%.8f")

    print(f"Verified {len(annual)} annual observations for {len(COUNTRIES)} countries.")
    print(f"Calculated {len(comparison)} country-period averages across {len(PERIODS)} periods.")
    estimate_rows = annual.loc[annual["weo_estimated_indicators"].fillna("").ne("")]
    for row in estimate_rows.itertuples(index=False):
        print(f"IMF estimates in source: {row.country}, {row.year}: {row.weo_estimated_indicators}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
