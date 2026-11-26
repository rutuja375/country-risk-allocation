"""Build comparable country-risk inputs from the IMF WEO and World Bank WGI releases."""

import argparse
import re
from pathlib import Path

import pandas as pd

from project_paths import DATA_DIR


COUNTRIES = {
    "USA": "United States", "CAN": "Canada", "GBR": "United Kingdom",
    "DEU": "Germany", "JPN": "Japan", "AUS": "Australia", "IND": "India",
    "CHN": "China", "BRA": "Brazil", "MEX": "Mexico", "IDN": "Indonesia",
    "ZAF": "South Africa",
}
YEARS = list(range(2016, 2026))
BASE_YEAR = 2024
WEO_FIELDS = {
    "NGDP_RPCH": "real_gdp_growth_pct",
    "PCPIPCH": "average_cpi_inflation_pct",
    "GGXWDG_NGDP": "gross_government_debt_pct_gdp",
    "GGXCNL_NGDP": "government_balance_pct_gdp",
    "BCA_NGDPD": "current_account_balance_pct_gdp",
}
WGI_FIELDS = {
    "pv": "political_stability_score",
    "ge": "government_effectiveness_score",
}


def read_weo(path):
    key_columns = {"COUNTRY.ID", "INDICATOR.ID", "LATEST_ACTUAL_ANNUAL_DATA"}
    source = pd.read_excel(
        path, sheet_name="Countries", usecols=lambda name: name in key_columns or name in YEARS,
    )
    source = source.loc[
        source["COUNTRY.ID"].isin(COUNTRIES) & source["INDICATOR.ID"].isin(WEO_FIELDS)
    ].rename(columns={"COUNTRY.ID": "iso3", "INDICATOR.ID": "indicator"})
    long = source.melt(
        id_vars=["iso3", "indicator", "LATEST_ACTUAL_ANNUAL_DATA"],
        value_vars=YEARS,
        var_name="year", value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="raise")
    # In the source columns FY2024/25 is labeled 2024, the fiscal start year.
    last_actual = long["LATEST_ACTUAL_ANNUAL_DATA"].astype(str).map(
        lambda label: int(re.search(r"\d{4}", label).group())
    )
    long["indicator"] = long["indicator"].map(WEO_FIELDS)
    estimated = long.loc[long["year"] > last_actual]
    flags = estimated.groupby(["iso3", "year"])["indicator"].agg(";".join)
    return long.pivot(index=["iso3", "year"], columns="indicator", values="value"), flags


def read_wgi(path):
    columns = ["Economy (code)", "Year", "Governance score (0-100)"]
    sheets = pd.read_excel(path, sheet_name=list(WGI_FIELDS), usecols=columns)
    result = {}
    for sheet, name in WGI_FIELDS.items():
        data = sheets[sheet].rename(
            columns={columns[0]: "iso3", columns[1]: "year", columns[2]: name},
        )
        selected = data.loc[data["iso3"].isin(COUNTRIES) & data["year"].isin(YEARS)]
        result[name] = selected.set_index(["iso3", "year"])[name]
    return pd.DataFrame(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weo", type=Path, required=True)
    parser.add_argument("--wgi", type=Path, required=True)
    args = parser.parse_args()

    weo, flags = read_weo(args.weo)
    index = pd.MultiIndex.from_product([COUNTRIES, YEARS], names=["iso3", "year"])
    indicators = [*WEO_FIELDS.values(), *WGI_FIELDS.values()]
    annual = weo.join(read_wgi(args.wgi), how="outer").reindex(index)[indicators]
    if annual.isna().any().any():
        raise ValueError(f"Missing values by indicator: {annual.isna().sum().to_dict()}")
    if not annual[list(WGI_FIELDS.values())].apply(lambda s: s.between(0, 100).all()).all():
        raise ValueError("WGI scores must be between 0 and 100.")
    annual["weo_estimate_fields"] = flags.reindex(index, fill_value="")
    annual.insert(0, "country", annual.index.get_level_values("iso3").map(COUNTRIES))

    endpoint = annual.xs(BASE_YEAR, level="year")
    recent = annual.loc[annual.index.get_level_values("year").isin(range(BASE_YEAR - 2, BASE_YEAR + 1))]
    if recent["weo_estimate_fields"].ne("").any():
        raise ValueError("The baseline window contains IMF estimates.")
    debt = WEO_FIELDS["GGXWDG_NGDP"]
    averaged_fields = [field for field in WEO_FIELDS.values() if field != debt]
    averages = recent.groupby(level="iso3")[averaged_fields].mean()
    snapshot = endpoint[["country", debt, *WGI_FIELDS.values()]].join(
        averages.add_prefix("three_year_mean_"),
    )
    snapshot.insert(1, "reference_year", BASE_YEAR)
    annual.reset_index().to_csv(DATA_DIR / "country_risk_annual.csv", index=False, float_format="%.8f")
    snapshot.reset_index().to_csv(DATA_DIR / "country_risk_inputs.csv", index=False, float_format="%.8f")
    print(f"Wrote {len(annual)} country-year rows and {len(snapshot)} scoring-input rows.")


if __name__ == "__main__":
    main()
