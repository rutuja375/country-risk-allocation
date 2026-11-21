import numpy as np
import pandas as pd

from project_paths import DATA_DIR, TABLES_DIR

expected_months = pd.date_range("2016-11-30", "2026-10-31", freq="ME")


def load_returns(filename):
    returns = pd.read_csv(
        filename, index_col="month_end", parse_dates=["month_end"]
    ).sort_index()
    returns = returns.apply(pd.to_numeric, errors="raise")

    if not returns.index.equals(expected_months):
        raise SystemExit(f"{filename}: expected 120 consecutive month-end dates.")
    if returns.shape[1] != 12:
        raise SystemExit(f"{filename}: expected 12 country columns.")
    if not np.isfinite(returns.to_numpy()).all():
        raise SystemExit(f"{filename}: found missing or non-finite returns.")

    return returns


etf = load_returns(DATA_DIR / "monthly_etf_returns_usd.csv")
index = load_returns(DATA_DIR / "monthly_index_returns_usd.csv")

if set(etf.columns) != set(index.columns):
    raise SystemExit("The ETF and index files contain different countries.")

index = index[etf.columns]

etf_5y = etf.loc["2021-11-01":"2026-10-31"]
index_5y = index.loc["2021-11-01":"2026-10-31"]


for name, returns in {"etf": etf_5y, "index": index_5y}.items():
    correlation = returns.corr()
    output_file = TABLES_DIR / f"{name}_correlation_5y.csv"
    correlation.to_csv(output_file)

    print(f"\n{name.upper()}: {len(returns)} months × {returns.shape[1]} countries")
    print(f"Saved {output_file}")
    print(correlation.round(2).to_string())