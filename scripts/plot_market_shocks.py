"""Plot weekly ETF portfolio shocks, drawdowns, and calendar-month returns."""

import argparse
import calendar
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

from etf_tangency_sensitivity import (
    WINDOW_STARTS,
    load_matched_returns,
    max_return_to_volatility,
)
from project_paths import DATA_DIR, PROJECT_ROOT


SHRINKAGE = 0.50
GAIN_COLOR = "#127a69"
LOSS_COLOR = "#b74433"
DRAWDOWN_COLOR = "#244e78"
CHART_DIR = PROJECT_ROOT / "docs" / "charts"


def annotate_event(ax, values, week, label, offset, color):
    date = pd.Timestamp(week)
    value = values.loc[date]
    ax.scatter(date, value, color=color, s=24, zorder=4)
    ax.annotate(
        f"{label}  {value:+.1%}",
        (date, value),
        xytext=offset,
        textcoords="offset points",
        ha="left" if offset[0] > 0 else "right",
        va="center",
        color=color,
        weight="bold",
        arrowprops={"arrowstyle": "-", "color": color},
    )


def create_chart(title, y_label, footer):
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    fig.canvas.manager.set_window_title(title)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.20)
    fig.text(0.10, 0.91, title, fontsize=23, weight="bold")
    fig.text(0.10, 0.055, footer, fontsize=9.5, color="#476078")
    ax.set_ylabel(y_label, fontsize=11)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0, pad=7, labelsize=11)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dce4eb", linewidth=0.75)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.axhline(0, color="#8193a4", linewidth=0.8)
    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-only", action="store_true", help="Save the charts without opening windows.")
    args = parser.parse_args()
    if args.save_only:
        plt.switch_backend("Agg")
    first_week = pd.Timestamp(WINDOW_STARTS[10])
    matched = load_matched_returns()["etf"].loc[first_week:]
    weights, _, _ = max_return_to_volatility(matched, SHRINKAGE)
    allocation = pd.Series(weights, index=matched.columns)

    starting_price_week = first_week - pd.Timedelta(weeks=1)
    last_week = matched.index[-1]
    prices = pd.read_csv(
        DATA_DIR / "daily_etf_adjusted_prices_usd.csv",
        index_col="trading_date",
        parse_dates=["trading_date"],
    ).sort_index()
    weekly_prices = prices.resample("W-FRI").last().loc[starting_price_week:last_week]
    weekly = weekly_prices.pct_change(fill_method=None).iloc[1:]
    monthly = prices.resample("ME").last().pct_change(fill_method=None)
    # Compare only complete calendar months within the weekly display window.
    monthly = monthly.loc[first_week + pd.offsets.MonthEnd(0):last_week + pd.offsets.MonthEnd(0)]
    if weekly.isna().any().any() or monthly.isna().any().any():
        raise ValueError("The chart requires complete ETF returns for all displayed periods.")

    weekly_returns = weekly @ allocation
    monthly_returns = monthly @ allocation
    wealth = pd.concat([
        pd.Series(1.0, index=[starting_price_week]),
        (1 + weekly_returns).cumprod(),
    ])
    drawdown = wealth / wealth.cummax() - 1
    month_means = monthly_returns.groupby(monthly_returns.index.month).mean()
    sample_period = f"{first_week:%b %Y} to {last_week:%b %Y} | Fixed 10-year ETF weights"
    weekly_footer = f"{sample_period} | {len(weekly_returns)} weeks | USD total returns"
    weekly_fig, weekly_ax = create_chart("Weekly portfolio returns", "Weekly return", weekly_footer)
    drawdown_fig, drawdown_ax = create_chart("Portfolio drawdown", "Below previous peak", weekly_footer)
    monthly_fig, monthly_ax = create_chart(
        "Average return by calendar month", "Average monthly return",
        f"{sample_period} | {len(monthly_returns)} months | USD total returns",
    )

    for ax in (weekly_ax, drawdown_ax):
        ax.set_xlim(starting_price_week - pd.Timedelta(weeks=6), last_week + pd.Timedelta(weeks=6))
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    weekly_ax.bar(
        weekly_returns.index, weekly_returns, width=5.4,
        color=[GAIN_COLOR if value >= 0 else LOSS_COLOR for value in weekly_returns],
    )
    weekly_ax.set_ylim(-0.16, 0.15)
    annotate_event(weekly_ax, weekly_returns, "2020-03-20", "Mar 2020 crash", (-28, 0), LOSS_COLOR)
    annotate_event(weekly_ax, weekly_returns, "2020-04-10", "Apr 2020 rebound", (30, 0), GAIN_COLOR)
    annotate_event(weekly_ax, weekly_returns, "2025-04-04", "Apr 2025 tariff sell-off", (-25, -18), LOSS_COLOR)

    drawdown_ax.fill_between(drawdown.index, drawdown, 0, color="#dce8f3")
    drawdown_ax.plot(drawdown.index, drawdown, color=DRAWDOWN_COLOR, linewidth=1.4)
    drawdown_ax.set_ylim(-0.38, 0.018)
    annotate_event(drawdown_ax, drawdown, "2020-03-20", "Mar 2020", (26, 0), DRAWDOWN_COLOR)
    annotate_event(drawdown_ax, drawdown, "2022-10-14", "Oct 2022", (23, -11), DRAWDOWN_COLOR)

    monthly_ax.bar(
        month_means.index, month_means, width=0.64,
        color=[GAIN_COLOR if value >= 0 else LOSS_COLOR for value in month_means],
    )
    monthly_ax.set_xticks(month_means.index, [calendar.month_abbr[month] for month in month_means.index])
    for month in (3, 7, 9, 11):
        value = month_means.loc[month]
        monthly_ax.annotate(
            f"{value:+.1%}", (month, value),
            xytext=(0, 5 if value >= 0 else -5), textcoords="offset points",
            ha="center", va="bottom" if value >= 0 else "top",
        )
    monthly_ax.margins(y=0.2)

    charts = (
        ("weekly-returns.png", weekly_fig),
        ("portfolio-drawdown.png", drawdown_fig),
        ("monthly-seasonality.png", monthly_fig),
    )
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    for filename, figure in charts:
        output = CHART_DIR / filename
        figure.savefig(output, dpi=160, facecolor="white")
        print(f"Saved {output}", flush=True)
    if not args.save_only:
        plt.show()
    for _, figure in charts:
        plt.close(figure)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if os.name != "nt":
            raise
        from tkinter import TclError, TclVersion, TkVersion

        tcl_dir = Path(sys.base_prefix) / "tcl"
        if (
            not isinstance(error, TclError)
            or "init.tcl" not in str(error)
            or Path.cwd() == tcl_dir
            or not (tcl_dir / f"tcl{TclVersion}" / "init.tcl").is_file()
            or not (tcl_dir / f"tk{TkVersion}" / "tk.tcl").is_file()
        ):
            raise

        # Tk can fail to resolve its startup files by absolute path on Windows.
        # Start the child beside the installed libraries with relative paths.
        env = {**os.environ, "TCL_LIBRARY": f"tcl{TclVersion}", "TK_LIBRARY": f"tk{TkVersion}"}
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=tcl_dir,
            env=env,
            check=False,
        )
        raise SystemExit(result.returncode)
