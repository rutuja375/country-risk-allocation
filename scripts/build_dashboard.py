"""Refresh the embedded data in the static GitHub Pages dashboard."""

import json
import re

import pandas as pd

from compare_weekly_correlations import COUNTRY_CAP
from efficient_frontier import solve_frontier
from etf_tangency_sensitivity import SHRINKAGE_VALUES, WINDOW_STARTS, load_matched_returns
from project_paths import PROJECT_ROOT, TABLES_DIR


def percentage(value):
    return round(float(value) * 100, 4)


def build_data():
    samples = load_matched_returns()
    countries = samples["etf"].columns.tolist()
    data = {
        "countries": countries,
        "countryCap": percentage(COUNTRY_CAP),
        "endDate": samples["etf"].index[-1].date().isoformat(),
        "datasets": {},
    }
    allocation = pd.read_csv(
        TABLES_DIR / "mvo_weights_comparison_weekly_5y.csv", index_col="country"
    ).loc[countries]

    for key, returns in samples.items():
        scenarios = {}
        for years, start in WINDOW_STARTS.items():
            sample = returns.loc[start:]
            for shrinkage in SHRINKAGE_VALUES:
                vol, ret, weights, _, best, _ = solve_frontier(sample, shrinkage, 51)
                scenarios[f"{years}-{percentage(shrinkage):g}"] = {
                    "weeks": len(sample),
                    "best": int(abs(ret - best[1]).argmin()),
                    "points": [
                        [percentage(risk), percentage(gain),
                         [percentage(max(weight, 0)) for weight in row]]
                        for risk, gain, row in zip(vol, ret, weights, strict=True)
                    ],
                }

        matrices = {}
        for period, suffix in (("weekly", "_weekly"), ("monthly", "")):
            frame = pd.read_csv(
                TABLES_DIR / f"{key}_correlation{suffix}_5y.csv", index_col=0
            )
            matrices[period] = frame.loc[countries, countries].round(4).values.tolist()

        data["datasets"][key] = {
            "scenarios": scenarios,
            "correlations": matrices,
            "mvo": allocation[f"{'ETF' if key == 'etf' else 'Index'} weight (%)"]
            .clip(lower=0).round(4).tolist(),
        }
    return data


def main():
    page = PROJECT_ROOT / "docs" / "index.html"
    html = page.read_text(encoding="utf-8")
    payload = json.dumps(build_data(), separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", r"\u003c")
    pattern = r'(<script type="application/json" id="model-data">).*?(</script>)'
    html, count = re.subn(
        pattern, lambda match: f"{match[1]}{payload}{match[2]}", html, flags=re.DOTALL
    )
    if count != 1:
        raise ValueError("Expected one model-data script in docs/index.html.")
    page.write_text(html, encoding="utf-8", newline="\n")
    print(f"Updated {page} with {len(payload):,} bytes of model data.")


if __name__ == "__main__":
    main()
