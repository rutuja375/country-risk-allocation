# Country Risk Allocation

Compare country-level returns from US-listed ETFs and USD-converted local indices. The analysis covers correlations, mean-variance allocations, efficient frontiers, and sensitivity to lookback windows and return shrinkage.

| Folder | Contents |
| --- | --- |
| `scripts/` | Market data collection, analysis, and plotting scripts |
| `docs/` | Static GitHub Pages dashboard for exploring the results |
| `data/` | Daily and monthly market data CSVs used by the analysis |
| `outputs/tables/` | Correlation matrices, allocation comparisons, and sensitivity results |
| `outputs/charts/` | Heatmaps, allocation comparisons, and efficient frontiers |
| `outputs/workbooks/` | Excel exports of the ETF and local-index datasets |

## Run the analysis

From the project root, use the project virtual environment when available and install dependencies with `python -m pip install -r requirements.txt`.

Run these commands in order to regenerate the analysis from the included market data. Plotting scripts save the chart and can open a plot window.

```text
python scripts/compare_market_data.py
python scripts/plot_correlations.py
python scripts/compare_weekly_correlations.py
python scripts/plot_weekly_correlations.py
python scripts/efficient_frontier.py
python scripts/mvo_sensitivity.py
python scripts/etf_tangency_sensitivity.py
python scripts/index_tangency_sensitivity.py
python scripts/plot_tangency_sensitivity.py
```

To refresh the underlying datasets and Excel exports, run `python scripts/country_total_return_usd.py` and `python scripts/country_index_usd.py`. These two scripts request historical data from Yahoo Finance.

## Dashboard

Open `docs/index.html` in a browser. To regenerate the embedded chart data after rerunning the analysis, run `python scripts/build_dashboard.py`. For GitHub Pages, select `main` and `/docs` as the branch and folder to publish.

The **Market shocks** tab uses three separate images in `docs/charts/`. Run `python scripts/plot_market_shocks.py` to regenerate them and open the three Matplotlib figures. Add `--save-only` to regenerate the images without opening windows.
