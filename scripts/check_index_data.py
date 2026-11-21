import pandas as pd
import yfinance as yf

indices = {
    "MSCI USA": "^984000-USD-STRD",
    "MSCI Japan": "^939200-USD-STRD",
}

expected_months = pd.date_range("2016-10-31", "2026-10-31", freq="ME")

for name, ticker in indices.items():
    history = yf.Ticker(ticker).history(
        start="2016-10-01",
        end="2026-11-01",
        interval="1d",
        auto_adjust=False,
    )

    if history.empty:
        print(f"{name} ({ticker}): Yahoo returned no data")
        continue

    monthly = history["Close"].resample("ME").last()
    monthly.index = monthly.index.tz_localize(None)
    monthly = monthly.reindex(expected_months)

    missing = monthly[monthly.isna()].index.strftime("%Y-%m").tolist()

    print(f"{name} ({ticker}): {monthly.count()}/{len(expected_months)} months")
    print(f"  First and last dates returned: {history.index[0].date()} to {history.index[-1].date()}")
    print(f"  Missing months: {', '.join(missing) if missing else 'none'}")