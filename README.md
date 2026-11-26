# Country Risk Allocation

The project proposes a $300 million country equity allocation across 12 markets. First, it compares market-only tangency portfolios for US-listed ETFs and USD-converted local indices. Then it uses country-risk states to test allocation limits and selects the ten-year ETF portfolio with a 50% State C cap.

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

The market-only model maximizes annual expected return divided by annual volatility. It is fully invested and long only, with a 20% cap per country. The matched weekly data run from November 2016 through October 2026. The sensitivity analysis uses 3-, 5-, and 10-year windows and 25%, 50%, and 75% mean shrinkage. The final recommendation uses ten-year ETF data and 50% mean shrinkage.

To regenerate the market analysis from the included prices, run the following commands. The plotting scripts can open Matplotlib windows.

```text
python scripts/compare_market_data.py
python scripts/plot_correlations.py
python scripts/compare_weekly_correlations.py
python scripts/plot_weekly_correlations.py
python scripts/efficient_frontier.py
python scripts/etf_tangency_sensitivity.py
python scripts/index_tangency_sensitivity.py
python scripts/plot_tangency_sensitivity.py
```

To refresh the underlying datasets and Excel exports, run `python scripts/country_total_return_usd.py` and `python scripts/country_index_usd.py`. These two scripts request historical data from Yahoo Finance.

## Dashboard

Open `docs/index.html` in a browser. The market tabs show market-only results. The Country risk tab shows standardized indicators, state assignments, the cap comparison, the constrained efficient frontier, final weights, and the GDP and inflation review. To regenerate the embedded chart data after rerunning the analysis, run `python scripts/build_dashboard.py`. GitHub Pages serves `/docs` from `main`.

The **Market shocks** tab uses three separate images in `docs/charts/`. Run `python scripts/plot_market_shocks.py` to regenerate them and open the three Matplotlib figures. Add `--save-only` to regenerate the images without opening windows.

## Portfolio themes

Run the two commands below to reproduce the theme comparisons in `outputs/tables/`. Both use the selected portfolio as the baseline and compare the 12 country ETFs.

```powershell
python scripts/build_theme_analysis.py --review
python scripts/compare_theme_allocations.py
```

- **Chips:** 2024 integrated-circuit trade (HS 8542) from [UN Comtrade via WITS](https://wits.worldbank.org/trade/comtrade/en/country/ALL/year/2024/tradeflow/Imports/partner/WLD/product/8542), scaled by 2024 [World Bank GDP](https://datacatalogfiles.worldbank.org/ddh-published/0038130/DR0046441/GDP.pdf). Supplier shares exclude aggregate and unspecified partner rows from the supplier selection and use total reported imports as the denominator.
- **Data centers:** Rounded-down estimates of energized capacity from [DC Atlas](https://dcatlas.io/en/answers/data-center-capacity), retrieved November 2026. The estimates cover all uses, and source coverage varies.
- **Oil:** The 12 largest increases and decreases in monthly average [Brent prices](https://fred.stlouisfed.org/series/MCOILBRENTEU) over the 120-month sample. Fund-provider Energy weights are dated snapshots. The allocation comparison tests shifting two percentage points from Canada to USD cash.
- **War events:** Two named events using USD ETF prices from the last close before the event to the fifth US trading session on or after it. These are separate historical examples.
- **Gold:** The same oil months with 5% of the fixed portfolio shifted proportionally to [GLD](https://www.digrin.com/stocks/detail/GLD/price) or zero-return USD cash. Each observation starts at the stated weights and omits trading costs.

After regenerating the tables, run `python scripts/build_theme_analysis.py` to refresh the dashboard's `docs/theme-data.js`.

## Country-risk data

Build the historical country comparison in [the analysis plan](scripts/instructions.md) from `data/country_risk_data_2002_2025.xlsx`.

```text
python scripts/build_country_comparison.py --workbook data/country_risk_data_2002_2025.xlsx
```

The script validates the annual panel and writes `outputs/tables/country_factor_averages_by_period.csv`. Each row contains a country's seven indicator averages for one period: 2002–2007, 2008–2009, 2010–2019, 2020–2021, or 2022–2024. The two growth and inflation measures provide context for the five core risk factors. The historical rows come from the October 2026 IMF and 2026 World Bank releases linked below. The three 2025 rows flagged with IMF estimates fall outside the comparison periods.

Run `python scripts/standardize_country_risk.py` to convert all 60 country-period rows to four comparable inputs in `outputs/tables/country_risk_standardized_by_period.csv`. The script uses pooled median and median absolute deviation, combines the two governance indicators after scaling them, and orients each column so higher values indicate more vulnerability on that dimension.

Run `python scripts/cluster_country_risk.py` to compare three, four, and five K-means clusters and record the sensitivity to capping inputs at ±3. The reference fit uses four uncapped clusters because it has the highest uncapped silhouette score in this dataset and its assignments match the capped fit. See `outputs/tables/country_risk_state_migration.csv` for the period assignments and `outputs/tables/country_risk_state_profiles.csv` for the factor averages behind each descriptive state. Cluster letters identify profiles and carry no risk ranking.

Run `python scripts/compare_state_c_cap.py` to reproduce the allocation trials. The State C countries are selected from the 2022–2024 assignments. The market-only portfolio allocated 60% to State C. The selected 50% cap moves $30 million from the UK and China to Australia and Mexico. The GDP and inflation review uses 2025 observations and each country's 2002–2025 annual medians. A 3% cap on Mexico was evaluated as a sensitivity test and was not adopted.

### Source-release exports

The workbook contains annual observations for 2002–2025. IMF fiscal-year labels follow the source conventions, so some reporting periods differ from calendar years. The workbook flags estimates. The indicator sources are the [IMF October 2026 World Economic Outlook](https://data.imf.org/datasets/WEO) and the [World Bank Worldwide Governance Indicators 2026 release](https://datacatalog.worldbank.org/search/dataset/0038026/worldwide-governance-indicators).
