"""Project-wide paths for fraud lab utilities."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
KAGGLE_DIR = DATA_DIR / "kaggle"
KAGGLE_DATASET_PATH = KAGGLE_DIR / "online_fraud.csv"
