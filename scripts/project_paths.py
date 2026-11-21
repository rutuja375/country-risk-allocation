from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
CHARTS_DIR = OUTPUTS_DIR / "charts"
WORKBOOKS_DIR = OUTPUTS_DIR / "workbooks"
