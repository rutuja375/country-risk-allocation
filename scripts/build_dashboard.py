"""Refresh the embedded data in the static GitHub Pages dashboard."""

import json
import re

import pandas as pd

from build_country_comparison import CONTEXT_COLUMNS, WORKBOOK_COLUMNS
from compare_state_c_cap import SHRINKAGE, STATE_PERIOD
from compare_weekly_correlations import COUNTRY_CAP
from efficient_frontier import solve_frontier
from etf_tangency_sensitivity import SHRINKAGE_VALUES, WINDOW_STARTS, load_matched_returns
from project_paths import DATA_DIR, PROJECT_ROOT, TABLES_DIR


FINAL_STATE_C_CAP = 0.50
Z_COLUMNS = ("debt_z", "fiscal_balance_risk_z", "current_account_risk_z", "governance_risk_z")


def percentage(value):
    return round(float(value) * 100, 4)


def frontier_data(sample, shrinkage, *, group_cap=None):
    vol, ret, weights, _, best, _ = solve_frontier(sample, shrinkage, 51, group_cap=group_cap)
    return {
        "weeks": len(sample),
        "best": int(abs(ret - best[1]).argmin()),
        "points": [
            [percentage(risk), percentage(gain), [percentage(max(weight, 0)) for weight in row]]
            for risk, gain, row in zip(vol, ret, weights, strict=True)
        ],
    }


def build_country_data(sample):
    countries = sample.columns
    migration = pd.read_csv(TABLES_DIR / "country_risk_state_migration.csv", index_col="country").loc[countries]
    scores = pd.read_csv(TABLES_DIR / "country_risk_standardized_by_period.csv")
    periods = scores["start_year"].astype(str) + "-" + scores["end_year"].astype(str)
    scores = scores.assign(years=periods).set_index(["years", "country"])
    profiles = pd.read_csv(TABLES_DIR / "country_risk_state_profiles.csv").set_index("state")
    annual = pd.read_excel(
        DATA_DIR / "country_risk_data_2002_2025.xlsx",
        sheet_name="Country risk data",
        usecols=["Country", "Year", *CONTEXT_COLUMNS],
    ).rename(columns=WORKBOOK_COLUMNS)
    macro_columns = list(CONTEXT_COLUMNS.values())
    median = annual.groupby("country")[macro_columns].median()
    latest = annual.loc[annual.year.eq(2025)].set_index("country")
    trials = pd.read_csv(TABLES_DIR / "state_c_cap_summary.csv")
    weights = pd.read_csv(TABLES_DIR / "state_c_cap_weights.csv", index_col="country").loc[countries]
    state_c = countries[migration[STATE_PERIOD].eq("C")]
    constrained = frontier_data(sample, SHRINKAGE, group_cap=(state_c, FINAL_STATE_C_CAP))
    final_weights = constrained["points"][constrained["best"]][2]
    if any(abs(w - saved) > 0.0001 for w, saved in zip(final_weights, weights.cap_50_weight_pct, strict=True)):
        raise ValueError("Country-risk frontier tangency weights differ from the saved final allocation.")
    return {
        "periods": migration.columns.tolist(),
        "latestPeriod": STATE_PERIOD,
        "migration": migration.values.tolist(),
        "scores": {
            period: scores.loc[period, list(Z_COLUMNS)].loc[countries].round(2).values.tolist()
            for period in migration
        },
        "profiles": profiles[["profile", "observations", "gross_government_debt_pct_gdp",
                              "government_balance_pct_gdp", "current_account_balance_pct_gdp"]]
        .round(2).to_dict(orient="index"),
        "macroMedian": median.loc[countries, macro_columns].round(2).values.tolist(),
        "macroLatest": latest.loc[countries, macro_columns].round(2).values.tolist(),
        "capTests": trials[["scenario", "state_c_weight_pct", "expected_annual_return_pct",
                            "annual_volatility_pct", "return_to_volatility"]].round(4).values.tolist(),
        "cap": percentage(FINAL_STATE_C_CAP),
        "frontier": constrained,
        "varianceShares": weights.cap_50_variance_share_pct.round(2).tolist(),
    }


def build_data():
    samples = load_matched_returns()
    countries = samples["etf"].columns.tolist()
    data = {
        "countries": countries,
        "countryCap": percentage(COUNTRY_CAP),
        "endDate": samples["etf"].index[-1].date().isoformat(),
        "datasets": {},
    }
    for key, returns in samples.items():
        scenarios = {}
        for years, start in WINDOW_STARTS.items():
            sample = returns.loc[start:]
            for shrinkage in SHRINKAGE_VALUES:
                scenarios[f"{years}-{percentage(shrinkage):g}"] = frontier_data(sample, shrinkage)

        matrices = {}
        for period, suffix in (("weekly", "_weekly"), ("monthly", "")):
            frame = pd.read_csv(
                TABLES_DIR / f"{key}_correlation{suffix}_5y.csv", index_col=0
            )
            matrices[period] = frame.loc[countries, countries].round(4).values.tolist()

        data["datasets"][key] = {
            "scenarios": scenarios,
            "correlations": matrices,
        }
    data["countryRisk"] = build_country_data(samples["etf"].loc[WINDOW_STARTS[10]:])
    return data


def main():
    page = PROJECT_ROOT / "docs" / "index.html"
    html = page.read_text(encoding="utf-8")
    payload = json.dumps(build_data(), separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", r"\u003c")
    pattern = r'(<script type="application/json" id="model-data">).*?(</script>)'
    html, count = re.subn(
        pattern, lambda match: f"{match[1]}{payload}{match[2]}", html, flags=re.DOTALL
    )
    if count != 1:
        raise ValueError("Expected one model-data script in docs/index.html.")
    page.write_text(html, encoding="utf-8", newline="\n")
    print(f"Updated {page} with {len(payload):,} bytes of model data.")


if __name__ == "__main__":
    main()
