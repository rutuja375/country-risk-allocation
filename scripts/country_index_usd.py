"""Download local country indices and convert their levels to USD."""

import numpy as np
import pandas as pd
import yfinance as yf

from project_paths import DATA_DIR, WORKBOOKS_DIR

START = "2016-10-01"
END = "2026-11-01"  # Exclusive. Includes all of October 2026.
FX_START = "2016-09-26"
MAX_FX_AGE = pd.Timedelta(days=4)
EXPECTED_MONTHS = pd.date_range("2016-10-31", "2026-10-31", freq="ME")

# FX is quoted as USD per one unit of local currency.
INDICES = {
    "United States": ("^GSPC", "S&P 500", "Price", None),
    "Canada": ("^GSPTSE", "S&P/TSX Composite", "Price", "CADUSD=X"),
    "United Kingdom": ("^FTSE", "FTSE 100", "Price", "GBPUSD=X"),
    "Germany": ("^GDAXI", "DAX", "Total return", "EURUSD=X"),
    "Japan": ("^N225", "Nikkei 225", "Price", "JPYUSD=X"),
    "Australia": ("^AXJO", "S&P/ASX 200", "Price", "AUDUSD=X"),
    "India": ("^NSEI", "Nifty 50", "Price", "INRUSD=X"),
    "China": ("000001.SS", "Shanghai Composite", "Price", "CNYUSD=X"),
    "Brazil": ("^BVSP", "Ibovespa", "Total return", "BRLUSD=X"),
    "Mexico": ("^MXX", "IPC Mexico", "Price", "MXNUSD=X"),
    "Indonesia": ("^JKSE", "Jakarta Composite", "Price", "IDRUSD=X"),
    "South Africa": ("^J203.JO", "FTSE/JSE All Share", "Price", "ZARUSD=X"),
}


def get_close(ticker: str, start: str = START) -> pd.Series:
    history = yf.Ticker(ticker).history(
        start=start,
        end=END,
        interval="1d",
        auto_adjust=False,
    )
    if history.empty or "Close" not in history:
        raise ValueError(f"Yahoo returned no closing prices for {ticker}")

    close = history["Close"].astype(float).sort_index()
    close.index = close.index.tz_localize(None).normalize()
    close = close.where(np.isfinite(close) & close.gt(0)).dropna()

    if close.empty:
        raise ValueError(f"{ticker} has no usable closing prices")
    if close.index.has_duplicates:
        raise ValueError(f"{ticker} has duplicate trading dates")
    return close


def month_end_levels(levels: pd.Series) -> pd.Series:
    # Keep the level from the actual final trading date of each month.
    # If FX is missing on that date, the month remains missing.
    last_dates = levels.groupby(levels.index.to_period("M")).tail(1)
    last_dates.index = last_dates.index + pd.offsets.MonthEnd(0)
    return last_dates.reindex(EXPECTED_MONTHS)


def main():
    local_by_country = {}
    usd_by_country = {}
    fx_by_country = {}
    monthly_by_country = {}
    issues = []

    for country, (ticker, _, _, fx_ticker) in INDICES.items():
        try:
            local = get_close(ticker)
            if fx_ticker is None:
                usd = local.copy()
            else:
                fx = get_close(fx_ticker, start=FX_START)
                matched_fx = fx.reindex(
                    local.index, method="ffill", tolerance=MAX_FX_AGE
                )
                fx_by_country[country] = matched_fx
                usd = local * matched_fx

            monthly = month_end_levels(usd)
            missing_months = monthly[monthly.isna()].index.strftime("%Y-%m")
            unmatched_dates = usd[usd.isna()].index

            print(
                f"{country} ({ticker}, {fx_ticker or 'USD'}): "
                f"{len(local):,} daily index closes | "
                f"{local.index[0].date()} to {local.index[-1].date()} | "
                f"{monthly.count()}/{len(EXPECTED_MONTHS)} USD month-end levels"
            )

            if len(unmatched_dates):
                print(f"  Trading dates without usable FX: {len(unmatched_dates)}")
                print(f"  First unmatched date: {unmatched_dates[0].date()}")

            if len(missing_months):
                issues.append(
                    f"{country}: missing USD month-end levels for "
                    f"{', '.join(missing_months)}"
                )

            local_by_country[country] = local
            usd_by_country[country] = usd
            monthly_by_country[country] = monthly

        except Exception as error:
            issues.append(f"{country} ({ticker}): {error}")
            print(f"{country} ({ticker}): FAILED — {error}")

    if issues:
        print("\nCoverage issues:")
        for issue in issues:
            print(f"  {issue}")
        raise SystemExit("Index coverage check failed. No new files were written.")

    local_daily = pd.DataFrame(local_by_country).sort_index()
    usd_daily = pd.DataFrame(usd_by_country).sort_index()
    usd_monthly = pd.DataFrame(monthly_by_country)
    local_monthly = pd.DataFrame(
        {country: month_end_levels(levels)
         for country, levels in local_by_country.items()}
    )

    # Calculate daily returns on each market's own trading calendar.
    # Closed markets show blanks when the series are combined.
    daily_returns = pd.DataFrame(
        {
            country: levels.pct_change(fill_method=None)
            for country, levels in usd_by_country.items()
        }
    ).sort_index()
    monthly_returns = usd_monthly.pct_change(fill_method=None).iloc[1:]

    years = len(monthly_returns) / 12
    cumulative_usd = usd_monthly.iloc[-1] / usd_monthly.iloc[0] - 1
    cumulative_local = local_monthly.iloc[-1] / local_monthly.iloc[0] - 1
    annualized_usd = (1 + cumulative_usd).pow(1 / years) - 1
    annualized_local = (1 + cumulative_local).pow(1 / years) - 1
    annualized_vol = monthly_returns.std() * np.sqrt(12)
    drawdown = (usd_monthly / usd_monthly.cummax() - 1).min()

    summary = pd.DataFrame(
        {
            "Ticker": {country: details[0] for country, details in INDICES.items()},
            "Index": {country: details[1] for country, details in INDICES.items()},
            "Type": {country: details[2] for country, details in INDICES.items()},
            "Annualized return (local %)": annualized_local * 100,
            "Annualized return (USD %)": annualized_usd * 100,
            "USD minus local (pp/yr)": (annualized_usd - annualized_local) * 100,
            "Cumulative return (USD %)": cumulative_usd * 100,
            "Annualized volatility (%)": annualized_vol * 100,
            "Month-end max drawdown (%)": drawdown * 100,
        }
    ).sort_values("Annualized return (USD %)", ascending=False)

    yearly_returns = (
        usd_monthly.resample("YE").last().pct_change(fill_method=None)
        .iloc[1:].loc[:"2025-12-31"].mul(100)
    )

    with pd.ExcelWriter(WORKBOOKS_DIR / "country_index_usd.xlsx", engine="openpyxl") as workbook:
        summary.round(2).to_excel(workbook, sheet_name="Summary")
        yearly_returns.round(2).to_excel(workbook, sheet_name="Calendar-year returns %")
        usd_monthly.to_excel(workbook, sheet_name="Month-end USD levels")
        usd_daily.to_excel(workbook, sheet_name="Daily USD levels")
        local_daily.to_excel(workbook, sheet_name="Daily local levels")
        (daily_returns * 100).to_excel(workbook, sheet_name="Daily USD returns %")
        pd.DataFrame(fx_by_country).sort_index().to_excel(workbook, sheet_name="FX used")
        monthly_returns.corr().round(4).to_excel(workbook, sheet_name="Monthly correlation")

    usd_daily.to_csv(DATA_DIR / "daily_index_levels_usd.csv", index_label="trading_date")
    local_daily.to_csv(DATA_DIR / "daily_index_levels_local.csv", index_label="trading_date")
    daily_returns.to_csv(DATA_DIR / "daily_index_returns_usd.csv", index_label="trading_date")
    monthly_returns.to_csv(DATA_DIR / "monthly_index_returns_usd.csv", index_label="month_end")

    print(f"\nMonthly USD returns: {monthly_returns.shape[0]} × {monthly_returns.shape[1]}")
    print(summary.round(2).to_string())
    print("\nSaved country_index_usd.xlsx and four index CSVs.")
    print("The index comparison mixes price and total-return benchmarks.")


if __name__ == "__main__":
    main()