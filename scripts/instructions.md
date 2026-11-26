# Country risk and portfolio allocation

## 1. Start with the project question

We have a market-based allocation for a hypothetical $300 million portfolio across 12 country equity ETFs. Use the country data to answer three questions in order:

1. Where is each country vulnerable relative to the other 11?
2. Do several holdings share an exposure that matters for the portfolio?
3. Would changing the allocation reduce that exposure enough to justify the trade-off?

Compare market-only tangency portfolios with country-risk constraints after completing the country assessment. Keep results grounded in the data and distinguish findings from working assumptions.

## 2. Prepare the five-factor country comparison

Use `data/country_risk_data_2002_2025.xlsx` as the source for the annual country data. Match countries using their country codes and the project's ETF mapping. Check that every country has one observation per year for each required factor, and flag missing values, duplicate rows, and IMF estimates.

Use these five factors:

1. Government debt / GDP
2. Fiscal balance / GDP
3. Current account balance / GDP
4. Political stability
5. Government effectiveness

Calculate each country's simple average for all seven indicators within each of these periods. Include GDP growth and CPI inflation as context for the five core factors.

| Years | What to compare |
| --- | --- |
| 2002–2007 | Where countries stood before the global financial crisis |
| 2008–2009 | What changed during the global financial crisis |
| 2010–2019 | How conditions evolved between the financial crisis and the pandemic |
| 2020–2021 | What changed during the pandemic |
| 2022–2024 | Where countries stood in the recent comparison window |

Review the period averages alongside the annual values before standardizing. Compare each stress period with the period before it, then inspect the annual values to see how large the deterioration was and how much it reversed. Period averages can hide the worst year and the pace of recovery. Use weekly ETF returns separately to examine portfolio drawdowns where market history is available. Label any estimates shown in the annual review.

Use all five periods to compare country profiles over time. The comparison ends in 2024 because some 2025 debt and fiscal balance values are IMF estimates.

## 3. Standardize the inputs

For each core factor, subtract the pooled median across the **60 country-period rows** and divide by 1.4826 times the median absolute deviation. Use this common scale for all periods.

Keep the debt sign and reverse the signs for fiscal balance, current account, political stability, and government effectiveness. Average the two standardized governance scores and scale that composite once more with the same method. The four inputs are debt, fiscal balance, current account, and combined governance. Higher values indicate greater vulnerability on that dimension relative to the pooled sample.

Check that each standardized column has a median near zero and that 1.4826 times its median absolute deviation is near one. Review correlations and extreme observations. Keep uncapped scores as the starting point and use the raw values alongside them.

## 4. Build country risk states

Fit K-means on the four standardized columns across all 60 rows. Compare three, four, and five clusters with a fixed seed and multiple starting points. Use the original factor averages to describe what each group shares. Repeat with inputs capped at ±3 to see whether extreme observations change the result. Report any groupings that depend on that choice.

Show one row per country and one column per period in a migration table. Read the results as five snapshots of unequal length. Look for weaknesses that persist or appear during stress.

## 5. Use GDP growth and inflation to explain the country findings

For each country, subtract its own **2010–2019 median** from its **2022–2024 average** for both real GDP growth and CPI inflation. Report the gaps in percentage points. Show the annual values alongside the gaps so a recent reversal is visible.

Use these two measures to explain or challenge the cluster profiles. For example, weak growth relative to a country's history may make a deteriorating fiscal position more important to investigate. Read the inflation gap alongside the actual rate and trend.

**Decision after reviewing the results:** Use quarterly GDP or monthly inflation only if knowing when conditions changed would clarify a material finding. The annual comparison is the starting point. Keep growth and inflation as context alongside the cluster inputs.

## 6. Optional extension: oil stress test

Use the common market-data window from **2016 onward**. Average daily global Geopolitical Risk Index values by week. Flag weeks above the **95th percentile** of that historical window. Use the first qualifying week to start a **four-week episode**, count overlapping spikes once, and report every qualifying episode with enough data to measure returns. Label this a retrospective study and check whether the findings hold with an eight-week window.

Compare episodes when **Brent rose** with those when it was **flat or fell**. For each country, look at:

- Net oil imports relative to domestic oil consumption, to understand the economy’s exposure.
- The ETF’s energy-sector weight, to understand what the equity holding is exposed to. Label the date of the holdings data and distinguish present-day holdings from holdings during historical episodes.
- The ETF’s USD return, its return relative to a broad global equity benchmark, and its contribution to the portfolio’s gain or loss. Where useful, compare the same index in local currency and USD to see the currency effect.

Look for patterns across episodes. Report the number of episodes in each group and treat one-off findings or small samples cautiously. Use country-specific monthly GPR only if it helps explain a particular episode.

Keep **2024 chip imports as a share of GDP** visible in the country table. This measures the scale of chip imports. A supplier-specific stress scenario also needs trade by supplier and evidence linking the disruption to ETF holdings. Revisit gold only after the oil test identifies a loss we want to hedge. If we reach that stage, compare the original portfolio with versions that shift 5% to gold and 5% to cash.

## 7. Test the portfolio allocation

Start with the ten-year ETF tangency portfolio at 50% mean shrinkage. Use the latest state assignments to identify State C holdings and test portfolio caps of 55%, 50%, and 45%. Keep the historical inputs, objective, and individual 20% country limits fixed.

The selected 50% State C cap moves $30 million out of the shared fiscal-strain group at a small cost in expected return per unit of volatility. The growth and inflation review prompted a separate 3% cap trial for Mexico. Keep the 50% group cap as the sole additional constraint in the final allocation.
