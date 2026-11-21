"""Download adjusted USD prices for US-listed country equity ETF proxies."""

import numpy as np
import pandas as pd
import yfinance as yf

from project_paths import DATA_DIR, WORKBOOKS_DIR

START = "2016-10-01"
END = "2026-11-01"  # Exclusive. Includes all of October 2026.
EXPECTED_MONTHS = pd.date_range("2016-10-31", "2026-10-31", freq="ME")

# The funds provide country equity exposure. Their benchmark rules vary.
FUNDS = {
    "United States": "SPY",
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


def get_adjusted_close(ticker: str) -> pd.Series:
    history = yf.Ticker(ticker).history(
        start=START,
        end=END,
        interval="1d",
        auto_adjust=True,
    )
    if history.empty or "Close" not in history:
        raise ValueError(f"Yahoo returned no adjusted closing prices for {ticker}")

    close = history["Close"].astype(float).sort_index()
    close.index = close.index.tz_localize(None).normalize()
    close = close.where(np.isfinite(close) & close.gt(0)).dropna()

    if close.empty:
        raise ValueError(f"{ticker} has no usable adjusted closing prices")
    if close.index.has_duplicates:
        raise ValueError(f"{ticker} has duplicate trading dates")
    return close


def main():
    prices_by_country = {}
    monthly_by_country = {}
    issues = []

    for country, ticker in FUNDS.items():
        try:
            daily = get_adjusted_close(ticker)
            monthly = daily.resample("ME").last().reindex(EXPECTED_MONTHS)
            missing_months = monthly[monthly.isna()].index.strftime("%Y-%m")

            print(
                f"{country} ({ticker}): {len(daily):,} daily adjusted prices | "
                f"{daily.index[0].date()} to {daily.index[-1].date()} | "
                f"{monthly.count()}/{len(EXPECTED_MONTHS)} month-end prices"
            )

            if len(missing_months):
                issues.append(
                    f"{country}: missing months {', '.join(missing_months)}"
                )

            prices_by_country[country] = daily
            monthly_by_country[country] = monthly

        except Exception as error:
            issues.append(f"{country} ({ticker}): {error}")
            print(f"{country} ({ticker}): FAILED — {error}")

    if issues:
        print("\nCoverage issues:")
        for issue in issues:
            print(f"  {issue}")
        raise SystemExit("ETF coverage check failed. No new files were written.")

    daily_prices = pd.DataFrame(prices_by_country).sort_index()
    monthly_prices = pd.DataFrame(monthly_by_country)
    daily_returns = daily_prices.pct_change(fill_method=None).iloc[1:]
    monthly_returns = monthly_prices.pct_change(fill_method=None).iloc[1:]

    years = len(monthly_returns) / 12
    cumulative_return = monthly_prices.iloc[-1] / monthly_prices.iloc[0] - 1
    annualized_return = (1 + cumulative_return).pow(1 / years) - 1
    annualized_vol = monthly_returns.std() * np.sqrt(12)
    drawdown = (monthly_prices / monthly_prices.cummax() - 1).min()

    summary = pd.DataFrame(
        {
            "ETF": FUNDS,
            "Cumulative return (USD %)": cumulative_return * 100,
            "Annualized return (USD %)": annualized_return * 100,
            "Annualized volatility (%)": annualized_vol * 100,
            "Month-end max drawdown (%)": drawdown * 100,
        }
    ).sort_values("Annualized return (USD %)", ascending=False)

    yearly_returns = (
        monthly_prices.resample("YE").last().pct_change(fill_method=None)
        .iloc[1:].loc[:"2025-12-31"].mul(100)
    )

    with pd.ExcelWriter(WORKBOOKS_DIR / "country_total_return_usd.xlsx", engine="openpyxl") as workbook:
        summary.round(2).to_excel(workbook, sheet_name="Summary")
        yearly_returns.round(2).to_excel(workbook, sheet_name="Calendar-year returns %")
        monthly_prices.to_excel(workbook, sheet_name="Month-end adjusted prices")
        daily_prices.to_excel(workbook, sheet_name="Daily adjusted prices")
        (daily_returns * 100).to_excel(workbook, sheet_name="Daily USD returns %")
        monthly_returns.corr().round(4).to_excel(workbook, sheet_name="Monthly correlation")

    daily_prices.to_csv(DATA_DIR / "daily_etf_adjusted_prices_usd.csv", index_label="trading_date")
    daily_returns.to_csv(DATA_DIR / "daily_etf_returns_usd.csv", index_label="trading_date")
    monthly_returns.to_csv(DATA_DIR / "monthly_etf_returns_usd.csv", index_label="month_end")

    print("\nMissing daily prices across the combined trading dates:")
    print(daily_prices.isna().sum().to_string())
    print(f"\nMonthly USD returns: {monthly_returns.shape[0]} × {monthly_returns.shape[1]}")
    print(summary.round(2).to_string())
    print("\nSaved country_total_return_usd.xlsx and three ETF CSVs.")


if __name__ == "__main__":
    main()