import pandas as pd
import yfinance as yf

from project_paths import DATA_DIR

tickers = {
    "United States": "VTI",
    "Canada": "EWC",
    "United Kingdom": "EWU",
    "Germany": "EWG",
    "Japan": "EWJ",
    "Australia": "EWA",
    "India": "INDA",
    "China": "MCHI",
    "Brazil": "EWZ",
    "Mexico": "EWW",
    "Indonesia": "EIDO",
    "South Africa": "EZA",
}

expected_months = pd.date_range("2016-10-31", "2026-10-31", freq="ME")
prices_by_country = {}

for country, ticker in tickers.items():
    try:
        history = yf.Ticker(ticker).history(
            start="2016-10-01",
            end="2026-11-01",  # Exclusive, so October 2026 is included.
            interval="1d",
            auto_adjust=True,
        )
    except Exception as error:
        print(f"{country} ({ticker}): download failed: {error}")
        continue

    if history.empty:
        print(f"{country} ({ticker}): Yahoo returned no data")
        continue

    # With auto_adjust=True, Close is adjusted for dividends and splits.
    monthly_prices = history["Close"].resample("ME").last()
    monthly_prices.index = monthly_prices.index.tz_localize(None)
    monthly_prices = monthly_prices.reindex(expected_months)

    missing_months = monthly_prices[monthly_prices.isna()].index
    print(
        f"{country} ({ticker}): "
        f"{monthly_prices.count()}/{len(expected_months)} prices"
    )

    if len(missing_months) > 0:
        print(f"  Missing months: {', '.join(missing_months.strftime('%Y-%m'))}")
        continue

    prices_by_country[country] = monthly_prices

if len(prices_by_country) != len(tickers):
    raise SystemExit("At least one country has incomplete data. No tables were saved.")

prices = pd.DataFrame(prices_by_country)
prices.index.name = "month_end"

# Each return is this month's price divided by last month's price, minus one.
# The first price has no prior month, so the return table starts one month later.
returns = prices.pct_change(fill_method=None).iloc[1:]

prices.to_csv(DATA_DIR / "monthly_adjusted_prices.csv")
returns.to_csv(DATA_DIR / "monthly_returns.csv")

print(f"\nPrice table: {prices.shape[0]} months × {prices.shape[1]} countries")
print(f"Returns table: {returns.shape[0]} months × {returns.shape[1]} countries")
print(f"Return period: {returns.index[0].date()} to {returns.index[-1].date()}")
print("\nLast three months of returns (%):")
print(returns.tail(3).mul(100).round(2).to_string())