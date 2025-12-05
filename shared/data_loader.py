"""Shared dataset loader for dashboards."""

from functools import lru_cache
import pandas as pd

from fraud_lab import config


def _normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize the fraud dataset to avoid downstream crashes.

    - Parse datetime-ish columns
    - Fill numeric NaNs with column medians
    - Fill categorical NaNs with mode / 'unknown'
    - Fill datetime NaNs via forward/backward fill
    """
    df = df.copy()

    # Convert any time/date columns to datetime
    for col in df.columns:
        low = col.lower()
        if "time" in low or "date" in low or "ts" in low:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Numeric columns -> median
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        medians = df[num_cols].median()
        df[num_cols] = df[num_cols].fillna(medians)

    # Categorical/object/bool -> mode or "unknown"
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    for col in cat_cols:
        if df[col].isna().any():
            mode = df[col].mode()
            fallback = mode.iloc[0] if not mode.empty else "unknown"
            df[col] = df[col].fillna(fallback)

    # Datetime columns -> ffill/bfill
    dt_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    if len(dt_cols) > 0:
        df[dt_cols] = df[dt_cols].ffill().bfill()

    # Safety defaults to keep dashboards stable
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.Timestamp.utcnow()
    if "risk_level" not in df.columns:
        df["risk_level"] = "UNKNOWN"
    else:
        df["risk_level"] = df["risk_level"].fillna("UNKNOWN").astype(str)

    return df


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    path = config.KAGGLE_DATASET_PATH
    df: pd.DataFrame | None = None
    if path.exists():
        try:
            df = pd.read_csv(path)
        except Exception:
            df = None

    # Fallback to sample dataset if Kaggle file is missing/unreadable
    if df is None:
        sample_path = config.DATA_DIR / "transactions_sample.csv"
        if sample_path.exists():
            try:
                df = pd.read_csv(sample_path)
            except Exception:
                df = None

    # Last resort synthetic rows to keep dashboards alive
    if df is None:
        df = pd.DataFrame(
            [
                {
                    "transaction_id": "T-fallback-1",
                    "user_id": "U1",
                    "merchant_id": "M1",
                    "amount": 100.0,
                    "currency": "USD",
                    "category": "demo",
                    "transaction_type": "purchase",
                    "timestamp": pd.Timestamp.utcnow(),
                    "location": "US",
                    "device_id": "D1",
                    "fraud_label": 0,
                    "risk_level": "LOW",
                }
            ]
        )

    return _normalize_dataset(df)
