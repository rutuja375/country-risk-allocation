"""Build descriptive theme comparisons using the final portfolio weights."""

import argparse
import json

import numpy as np
import pandas as pd

from project_paths import DATA_DIR, PROJECT_ROOT, TABLES_DIR


WAR_EVENTS = [
    ("Russia's full-scale invasion of Ukraine", "2022-02-24",
     "https://www.un.org/sg/en/content/sg/statements/2022-02-24/statement-the-secretary-general-ukraine"),
    ("October 7 attacks and outbreak of the Israel–Hamas war", "2023-10-07",
     "https://www.un.org/unispal/document/unrwa-situation-report-1-on-the-situation-in-the-gaza-strip/"),
]


def write_review_tables(inputs, wars):
    chips = inputs[[
        "country", "portfolio_weight_pct", "imports_pct_gdp", "exports_pct_gdp",
        "top_reported_partner", "top_partner_share_pct", "chip_imports_thousand_usd",
        "chip_exports_thousand_usd", "gdp_million_usd", "chip_partner_source",
    ]].assign(trade_year=2024)
    data_centers = inputs[[
        "country", "portfolio_weight_pct", "live_data_center_mw_rounded_down",
    ]].assign(
        snapshot_month="2026-11",
        source="https://dcatlas.io/en/answers/data-center-capacity",
    )

    war_tables = []
    for event, (_, date, _) in zip(wars, WAR_EVENTS, strict=True):
        countries = inputs[["country", "portfolio_weight_pct"]].copy()
        countries["return_pct"] = event["returnsPct"]
        countries["portfolio_contribution_pp"] = countries.portfolio_weight_pct * countries.return_pct / 100
        portfolio_return = countries.portfolio_contribution_pp.sum()
        countries.loc[len(countries)] = ["Portfolio total", 100, portfolio_return, portfolio_return]
        war_tables.append(countries.assign(
            event=event["label"], event_date=date, start_close=event["start"],
            end_close=event["end"], source=event["source"],
        ))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("chips", chips), ("data_centers", data_centers), ("war", pd.concat(war_tables, ignore_index=True)),
    ):
        path = TABLES_DIR / f"theme_{name}_review.csv"
        table.to_csv(path, index=False, float_format="%.6f")
        print(f"Wrote {path} ({len(table)} rows).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", action="store_true", help="Export chips, data centers, and war to CSV.")
    args = parser.parse_args()

    weights = pd.read_csv(TABLES_DIR / "state_c_cap_weights.csv", index_col="country").cap_50_weight_pct / 100
    inputs = pd.read_csv(DATA_DIR / "theme_country_inputs.csv", index_col="country")
    monthly = pd.read_csv(DATA_DIR / "monthly_etf_returns_usd.csv", index_col="month_end")
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    expected_months = pd.period_range("2016-11", "2026-10", freq="M")
    prices = pd.read_csv(DATA_DIR / "theme_monthly_prices.csv", index_col="month")
    prices.index = pd.PeriodIndex(prices.index, freq="M")
    if not (
        len(weights) == 12 and weights.index.is_unique and weights.between(0, 1).all()
        and np.isclose(weights.sum(), 1) and inputs.index.is_unique
        and set(weights.index) == set(inputs.index) == set(monthly.columns)
        and monthly.index.equals(expected_months)
        and prices.index.equals(pd.period_range("2016-10", "2026-10", freq="M"))
        and np.isfinite(monthly.to_numpy()).all()
        and np.isfinite(prices.to_numpy()).all() and prices.gt(0).all().all()
    ):
        raise ValueError("Missing, duplicate, or inconsistent data in the theme analysis inputs.")
    inputs = inputs.loc[weights.index]
    if not (inputs.gdp_million_usd.gt(0) & inputs.chip_imports_thousand_usd.gt(0)
            & inputs.chip_exports_thousand_usd.ge(0) & inputs.etf_energy_pct.between(0, 100)
            & inputs.top_partner_imports_thousand_usd.between(0, inputs.chip_imports_thousand_usd)
            & inputs.live_data_center_mw_rounded_down.gt(0)).all():
        raise ValueError("Invalid chip, data-center, or ETF energy data.")

    inputs["imports_pct_gdp"] = inputs.chip_imports_thousand_usd / inputs.gdp_million_usd / 10
    inputs["exports_pct_gdp"] = inputs.chip_exports_thousand_usd / inputs.gdp_million_usd / 10
    inputs["top_partner_share_pct"] = 100 * inputs.top_partner_imports_thousand_usd / inputs.chip_imports_thousand_usd
    inputs["energy_contribution_pct"] = weights * inputs.etf_energy_pct
    inputs["portfolio_weight_pct"] = 100 * weights
    inputs = inputs.reset_index()
    changes = prices.pct_change(fill_method=None).iloc[1:]
    brent = changes.brent_monthly_average_usd
    gold = changes.gld_adjusted_close_usd
    mix = monthly @ weights
    oil = []
    for label, window, selected in (
        ("Largest Brent rises", "Brent up", brent.nlargest(12)),
        ("Largest Brent falls", "Brent down", brent.nsmallest(12)),
    ):
        months = selected.index
        oil.append({
            "label": label,
            "window": window,
            "firstCutoffPct": round(float(100 * selected.iloc[-1]), 4),
            "returnsPct": (100 * monthly.loc[months, weights.index].mean()).round(6).tolist(),
            "months": [{
                "month": str(month),
                "brent": 100 * brent[month],
                "portfolio": 100 * mix[month],
                "gold": 100 * (0.95 * mix[month] + 0.05 * gold[month]),
                "cash": 100 * 0.95 * mix[month],
            } for month in months],
        })

    daily = pd.read_csv(DATA_DIR / "daily_etf_adjusted_prices_usd.csv", index_col="trading_date", parse_dates=True)
    daily = daily.sort_index().loc[:, weights.index]
    wars = []
    for label, date, source in WAR_EVENTS:
        event_day = pd.Timestamp(date)
        before = daily.loc[daily.index < event_day].tail(1)
        after = daily.loc[daily.index >= event_day].head(5)
        if before.empty or len(after) != 5:
            raise ValueError(f"Incomplete ETF window for {date}.")
        event_returns = 100 * (after.iloc[-1] / before.iloc[0] - 1)
        if not np.isfinite(event_returns).all():
            raise ValueError(f"Missing ETF return for the event on {date}.")
        wars.append({
            "label": label,
            "start": before.index[0].date().isoformat(),
            "end": after.index[-1].date().isoformat(),
            "source": source,
            "returnsPct": event_returns.round(6).tolist(),
        })
    if args.review:
        write_review_tables(inputs, wars)
        return

    allocation_tables = {}
    for key in ("summary", "weights"):
        table = pd.read_csv(TABLES_DIR / f"theme_allocation_{key}.csv")
        allocation_tables[key] = json.loads(table.to_json(orient="records"))
    data = {
        "countries": inputs.to_dict(orient="records"), "oil": oil, "wars": wars,
        "allocations": allocation_tables,
    }
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    path = PROJECT_ROOT / "docs" / "theme-data.js"
    path.write_text(f"const themeData = {payload};\n", encoding="utf-8")
    print(f"Built {path.name} for {len(weights)} countries, {len(monthly)} months, and {len(wars)} war windows.")


if __name__ == "__main__":
    main()
