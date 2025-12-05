"""Main Streamlit dashboard for Online Fraud Detection – Enterprise Menu Version.

Run with:
    streamlit run fraud_lab/dashboard_app.py --server.port 8502
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import time

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

# Optional: scikit-learn for simple model experiments
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
except Exception:  # noqa: BLE001
    SKLEARN_AVAILABLE = False
else:
    SKLEARN_AVAILABLE = True

# ------------- GLOBAL CONFIG -------------

st.set_page_config(
    page_title="Online Fraud Detection – Enterprise Dashboard",
    layout="wide",
    page_icon="🛡️",
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# 👇 Prefer your Kaggle CSV here; edit this path if your file is elsewhere
DEFAULT_DATA_PATH = HERE / "data" / "AIML Dataset.csv"
# If you rename the file, use that exact name here instead
FALLBACK_DEMO_PATH = HERE / "data" / "transactions.csv"


def load_project_dataset() -> pd.DataFrame:
    """Single entry point for loading the main project dataset."""
    df = get_dataset_from_sidebar()
    return df


def configure_dataset_columns(df: pd.DataFrame) -> dict:
    """
    Sidebar config to map CSV columns to semantic roles
    (amount, label, time, etc.). Returns a dict with the mapping.
    """
    st.sidebar.subheader("Dataset column mapping")

    all_cols = list(df.columns)

    amount_default = 0
    if "amount" in all_cols:
        amount_default = all_cols.index("amount")
    elif "Amount" in all_cols:
        amount_default = all_cols.index("Amount")

    amount_col = st.sidebar.selectbox(
        "Transaction amount column",
        options=all_cols,
        index=amount_default,
        key="col_amount",
    )

    label_default = 0
    for cand in ["is_fraud", "isFraud", "Class", "class"]:
        if cand in all_cols:
            label_default = all_cols.index(cand)
            break

    label_col = st.sidebar.selectbox(
        "Fraud label column",
        options=all_cols,
        index=label_default,
        key="col_label",
    )

    time_options = ["<none>"] + all_cols
    time_default = 0
    for cand in ["timestamp", "time", "step"]:
        if cand in all_cols:
            time_default = time_options.index(cand)
            break

    time_col = st.sidebar.selectbox(
        "Timestamp / time column (optional)",
        options=time_options,
        index=time_default,
        key="col_time",
    )

    config = {
        "amount": amount_col,
        "label": label_col,
        "time": None if time_col == "<none>" else time_col,
    }

    st.sidebar.info(
        f"Using amount = `{amount_col}`, label = `{label_col}`"
        + (f", time = `{config['time']}`" if config["time"] else ", no time column")
    )
    return config


STREAM_BATCH_SIZE_DEFAULT = 8
STREAM_MAX_HISTORY = 120

CANDIDATE_TARGET_COLS: List[str] = [
    "is_fraud",
    "isFraud",
    "fraud",
    "fraud_flag",
    "Class",
    "class",
    "label",
    "target",
    "isFlaggedFraud",
]

COUNTRY_COORDS: Dict[str, Tuple[float, float]] = {
    "IN": (20.5937, 78.9629),
    "US": (37.0902, -95.7129),
    "UK": (55.3781, -3.4360),
    "SG": (1.3521, 103.8198),
    "BR": (-14.2350, -51.9253),
    "DE": (51.1657, 10.4515),
}

# ------------- DATA HELPERS -------------


def _synthetic_transactions(n: int = 800) -> pd.DataFrame:
    """Fallback synthetic data (for demo / viva)."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "tx_id": [f"TX-{i:06d}" for i in range(n)],
            "amount": rng.gamma(2.5, 60, size=n).round(2),
            "country": rng.choice(["IN", "US", "UK", "SG", "BR", "DE"], size=n),
            "device": rng.choice(["ANDROID", "IOS", "WEB"], size=n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="min"),
        }
    )
    return df


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise arbitrary CSV to the columns this app expects."""
    df = df.copy()
    lower_map = {c.lower(): c for c in df.columns}

    # --- tx_id ---
    if "tx_id" not in df.columns:
        tx_col = None
        for cand in ["txid", "transaction_id", "id"]:
            if cand in lower_map:
                tx_col = lower_map[cand]
                break
        if tx_col is not None:
            df["tx_id"] = df[tx_col].astype(str)
        else:
            df["tx_id"] = [f"TX-{i:06d}" for i in range(len(df))]

    # --- amount ---
    amount_col = None
    for cand in ["amount", "amt", "transactionamount", "value"]:
        if cand in lower_map:
            amount_col = lower_map[cand]
            break
    if amount_col is None:
        num_cols = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c.lower() != "step"
        ]
        amount_col = num_cols[0] if num_cols else None

    if amount_col is not None:
        df["amount"] = df[amount_col].astype(float)
    elif "amount" not in df.columns:
        df["amount"] = 0.0

    # --- timestamp ---
    if "timestamp" not in df.columns:
        if "step" in lower_map and pd.api.types.is_numeric_dtype(df[lower_map["step"]]):
            base = pd.Timestamp("2024-01-01")
            df["timestamp"] = base + pd.to_timedelta(df[lower_map["step"]], unit="h")
        else:
            df["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="min")

    # --- country ---
    if "country" not in df.columns:
        rng = np.random.default_rng(100)
        df["country"] = rng.choice(list(COUNTRY_COORDS.keys()), size=len(df))

    # --- device ---
    if "device" not in df.columns:
        rng = np.random.default_rng(101)
        df["device"] = rng.choice(["ANDROID", "IOS", "WEB"], size=len(df))

    return df


@st.cache_data
def _load_transactions_from_csv(path_str: str) -> pd.DataFrame:
    """Cached CSV loader; path passed as string for caching."""
    path = Path(path_str)
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = _standardise_columns(df)
    return df


def render_data_source_sidebar() -> None:
    """Compact Data Source panel in the sidebar."""
    if "dataset_local_path" not in st.session_state:
        st.session_state["dataset_local_path"] = str(DEFAULT_DATA_PATH)
    if "dataset_status_msg" not in st.session_state:
        st.session_state["dataset_status_msg"] = "Using default or synthetic demo data."

    with st.sidebar.expander("Data source / CSV", expanded=False):
        st.caption("Choose which CSV feeds the dashboard.")

        source = st.radio(
            "Dataset source",
            (
                "Default (AIML CSV or demo)",
                "Local CSV path",
                "Upload CSV",
            ),
            key="dataset_source",
            label_visibility="collapsed",
        )

        if source == "Local CSV path":
            st.text_input(
                "CSV path",
                key="dataset_local_path",
                placeholder=r"C:\path\to\your.csv",
            )
        elif source == "Upload CSV":
            st.file_uploader(
                "Upload CSV",
                type=["csv"],
                key="dataset_uploaded_file",
            )

        if st.button("Reload CSV from disk", use_container_width=True):
            _load_transactions_from_csv.clear()
            st.session_state["dataset_status_msg"] = "Reload triggered; loading fresh data."

        st.caption(
            st.session_state.get(
                "dataset_status_msg",
                "Using default or synthetic demo data.",
            )
        )


def get_dataset_from_sidebar() -> pd.DataFrame:
    """
    Sidebar controls for selecting the dataset source.
    Returns a pandas DataFrame.
    """
    default_local = DEFAULT_DATA_PATH
    fallback_local = FALLBACK_DEMO_PATH

    source_choice = st.sidebar.radio(
        "Choose dataset source",
        options=[
            "Default project dataset (AIML CSV or synthetic)",
            "Custom local CSV path",
        ],
        index=0,
    )

    existing_path = st.session_state.get("dataset_local_path_main") or st.session_state.get(
        "dataset_local_path", str(default_local)
    )
    path_str = st.sidebar.text_input(
        "Local CSV path (full path)",
        value=str(existing_path),  # initial value only
        key="dataset_local_path_main",
    )

    use_default = source_choice.startswith("Default")

    def _load_and_standardise(path: Path) -> pd.DataFrame:
        df_local = _load_transactions_from_csv(str(path))
        return df_local

    if use_default:
        candidates = [default_local, fallback_local]
        last_err: Optional[Exception] = None
        for cand in candidates:
            try:
                if cand.exists():
                    df = _load_and_standardise(cand)
                    st.sidebar.success(f"Loaded dataset from: {cand}")
                    st.session_state["dataset_status_msg"] = f"Loaded dataset from: {cand}"
                    return df
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue

        st.sidebar.warning(
            "Default CSV missing or unreadable; using synthetic demo data instead."
        )
        st.session_state["dataset_status_msg"] = (
            f"Fallback to synthetic data. Last error: {last_err}" if last_err else "Fallback to synthetic data."
        )
        return _synthetic_transactions()

    # Custom path
    data_path = Path(path_str).expanduser()
    if not data_path.exists() or data_path.is_dir():
        st.error(
            f"CSV file not found at: {data_path}\n"
            "Please enter a full path to a CSV file (e.g., C:\\Users\\you\\data.csv)."
        )
        st.stop()

    try:
        df = _load_and_standardise(data_path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load dataset from {data_path}:\n\n{exc}")
        st.stop()

    st.sidebar.success(f"Loaded dataset from: {data_path}")
    st.session_state["dataset_status_msg"] = f"Loaded dataset from: {data_path}"
    return df

# ------------- RISK ENGINE (ANOMALY + RULES) -------------


def simple_anomaly_score(row: pd.Series) -> float:
    """Demo anomaly score: amount + country risk + device risk → [0,1]."""
    amount = float(row.get("amount", 0.0))
    amt_score = 1.0 - np.exp(-amount / 500.0)

    country = str(row.get("country", "IN"))
    country_risk = {
        "IN": 0.2,
        "US": 0.25,
        "UK": 0.25,
        "SG": 0.3,
        "BR": 0.5,
        "DE": 0.2,
    }.get(country, 0.3)

    device = str(row.get("device", "WEB"))
    device_risk = 0.2 if device in {"ANDROID", "IOS"} else 0.35

    score = 0.6 * amt_score + 0.25 * country_risk + 0.15 * device_risk
    return float(np.clip(score, 0.0, 1.0))


def rule_score(row: pd.Series) -> Tuple[float, Dict[str, bool]]:
    """Simple rule engine with explanations."""
    rules: Dict[str, bool] = {}

    amount = float(row.get("amount", 0.0))
    country = str(row.get("country", "IN"))
    device = str(row.get("device", "WEB"))

    rules["high_amount"] = amount > 5000
    rules["very_high_amount"] = amount > 20000
    rules["high_risk_country"] = country in {"BR"}
    rules["web_device"] = device == "WEB"

    score = 0.0
    if rules["high_amount"]:
        score += 0.4
    if rules["very_high_amount"]:
        score += 0.3
    if rules["high_risk_country"]:
        score += 0.2
    if rules["web_device"]:
        score += 0.1

    return float(min(score, 1.0)), rules


def combined_risk(row: pd.Series) -> Dict[str, object]:
    """Combine anomaly + rules → risk_score & band."""
    a_score = simple_anomaly_score(row)
    r_score, rules = rule_score(row)
    risk = 0.7 * a_score + 0.3 * r_score

    if risk < 0.4:
        band = "LOW"
    elif risk < 0.7:
        band = "MEDIUM"
    else:
        band = "HIGH"

    return {
        "anomaly_score": a_score,
        "rule_score": r_score,
        "risk_score": risk,
        "risk_band": band,
        "rules": rules,
    }


def add_risk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk columns used across pages."""
    records = df.to_dict(orient="records")
    risks = [combined_risk(pd.Series(rec)) for rec in records]

    out = df.copy()
    out["anomaly_score"] = [r["anomaly_score"] for r in risks]
    out["rule_score"] = [r["rule_score"] for r in risks]
    out["risk_score"] = [r["risk_score"] for r in risks]
    out["risk_band"] = [r["risk_band"] for r in risks]
    out["rule_hits"] = [
        ", ".join(k for k, v in r["rules"].items() if v) if any(r["rules"].values()) else "-"
        for r in risks
    ]
    return out

# ------------- SESSION STATE -------------


def init_stream_state() -> None:
    # current position in the dataset
    if "stream_cursor" not in st.session_state:
        st.session_state.stream_cursor = 0

    # rolling window of streamed rows
    if "stream_history" not in st.session_state:
        st.session_state.stream_history = pd.DataFrame()

    # batch size for both manual + auto
    if "stream_batch_size" not in st.session_state:
        st.session_state.stream_batch_size = STREAM_BATCH_SIZE_DEFAULT

    # whether auto streaming is turned on
    if "auto_stream_enabled" not in st.session_state:
        st.session_state.auto_stream_enabled = False


def init_manual_review_state() -> None:
    if "mr_seen_ids" not in st.session_state:
        st.session_state.mr_seen_ids = set()
    if "mr_history" not in st.session_state:
        st.session_state.mr_history = pd.DataFrame(
            columns=[
                "reviewed_at",
                "tx_id",
                "amount",
                "country",
                "device",
                "risk_score",
                "risk_band",
                "rule_hits",
                "action",
            ]
        )


def init_model_lab_state() -> None:
    if "model_history" not in st.session_state:
        st.session_state.model_history = pd.DataFrame(
            columns=[
                "run_id",
                "model",
                "type",
                "auc",
                "precision",
                "recall",
                "f1",
                "test_size",
                "max_depth_rf",
                "n_estimators_rf",
            ]
        )
    if "model_run_id" not in st.session_state:
        st.session_state.model_run_id = 1

# ------------- STREAMING HELPERS -------------


def get_next_stream_batch(df: pd.DataFrame) -> pd.DataFrame:
    batch_size = int(st.session_state.get("stream_batch_size", STREAM_BATCH_SIZE_DEFAULT))
    start = st.session_state.stream_cursor
    end = min(start + batch_size, len(df))
    batch = df.iloc[start:end].copy()
    if end >= len(df):
        st.session_state.stream_cursor = 0
    else:
        st.session_state.stream_cursor = end
    return batch


def update_stream_history(processed: pd.DataFrame) -> None:
    hist = st.session_state.stream_history
    updated = (
        pd.concat([processed, hist], axis=0)
        .drop_duplicates(subset=["tx_id"], keep="first")
        .head(STREAM_MAX_HISTORY)
        .reset_index(drop=True)
    )
    st.session_state.stream_history = updated

    # make it available for Live Graphs
    st.session_state["live_graph_df"] = updated

# ------------- GEO HELPERS -------------


def ensure_geo_columns(df: pd.DataFrame) -> pd.DataFrame:
    for lat_name in ["lat", "latitude", "Lat", "Latitude"]:
        for lon_name in ["lon", "lng", "longitude", "Lon", "Longitude"]:
            if lat_name in df.columns and lon_name in df.columns:
                out = df.copy()
                out["lat"] = out[lat_name]
                out["lon"] = out[lon_name]
                return out

    rng = np.random.default_rng(123)
    out = df.copy()
    lats, lons = [], []
    for _, row in out.iterrows():
        country = str(row.get("country", "IN"))
        base_lat, base_lon = COUNTRY_COORDS.get(country, (20.0, 0.0))
        lats.append(base_lat + rng.normal(scale=1.5))
        lons.append(base_lon + rng.normal(scale=1.5))
    out["lat"] = lats
    out["lon"] = lons
    return out


def add_visual_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["radius"] = (out["risk_score"].clip(0, 1) * 70000) + 10000
    rs = out["risk_score"].clip(0, 1)
    out["color_r"] = (rs * 255).astype(int)
    out["color_g"] = ((1.0 - rs) * 180).astype(int)
    out["color_b"] = np.full(len(out), 60, dtype=int)
    return out

# ------------- MODEL LAB HELPERS -------------


def ensure_target_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    for col in CANDIDATE_TARGET_COLS:
        if col in df.columns:
            df[col] = df[col].astype(int)
            return df, col
    df = df.copy()
    if "amount" in df.columns:
        q = df["amount"].quantile(0.9)
        df["is_fraud_synth"] = (df["amount"] >= q).astype(int)
        return df, "is_fraud_synth"
    rng = np.random.default_rng(101)
    df["is_fraud_synth"] = rng.integers(0, 2, size=len(df))
    return df, "is_fraud_synth"


def make_feature_matrix(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = df.copy()
    drop_cols = {target_col}
    for col in ["tx_id", "timestamp", "TxID", "TX_ID"]:
        if col in df.columns:
            drop_cols.add(col)

    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in drop_cols]
    small_cat_cols = [c for c in cat_cols if c in {"country", "device", "card_type", "channel"}]
    df = pd.get_dummies(df, columns=small_cat_cols, drop_first=True)

    feature_cols = [
        c
        for c in df.columns
        if c not in drop_cols
        and pd.api.types.is_numeric_dtype(df[c])
        and not c.startswith("is_fraud_synth")
    ]

    X = df[feature_cols].astype(float)
    y = df[target_col].astype(int)
    return X, y, feature_cols


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m: Dict[str, float] = {}
    try:
        m["auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:  # noqa: BLE001
        m["auc"] = float("nan")
    try:
        m["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        m["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        m["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    except Exception:  # noqa: BLE001
        m["precision"] = m["recall"] = m["f1"] = float("nan")
    return m


def run_model_experiments(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    max_depth_rf: int,
    n_estimators_rf: int,
) -> pd.DataFrame:
    results: List[Dict[str, object]] = []

    lr = LogisticRegression(max_iter=200, solver="lbfgs")
    try:
        lr.fit(X_train, y_train)
        y_prob_lr = lr.predict_proba(X_test)[:, 1]
        y_pred_lr = lr.predict(X_test)
        m_lr = compute_metrics(y_test.values, y_prob_lr, y_pred_lr)
        results.append({"model": "Logistic Regression", "type": "Linear", **m_lr})
    except Exception as e:  # noqa: BLE001
        results.append(
            {
                "model": "Logistic Regression",
                "type": "Linear",
                "auc": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "error": str(e),
            }
        )

    rf = RandomForestClassifier(
        n_estimators=n_estimators_rf,
        max_depth=max_depth_rf or None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    try:
        rf.fit(X_train, y_train)
        y_prob_rf = rf.predict_proba(X_test)[:, 1]
        y_pred_rf = rf.predict(X_test)
        m_rf = compute_metrics(y_test.values, y_prob_rf, y_pred_rf)
        results.append({"model": "Random Forest", "type": "Ensemble", **m_rf})
    except Exception as e:  # noqa: BLE001
        results.append(
            {
                "model": "Random Forest",
                "type": "Ensemble",
                "auc": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "error": str(e),
            }
        )

    return pd.DataFrame(results)


def append_model_history(
    df_run: pd.DataFrame, test_size: float, max_depth_rf: int, n_estimators_rf: int
) -> None:
    hist = st.session_state.model_history
    run_id = st.session_state.model_run_id
    st.session_state.model_run_id += 1

    df_run = df_run.copy()
    df_run["run_id"] = run_id
    df_run["test_size"] = test_size
    df_run["max_depth_rf"] = max_depth_rf
    df_run["n_estimators_rf"] = n_estimators_rf

    st.session_state.model_history = (
        pd.concat([df_run, hist], axis=0).reset_index(drop=True)
    )

# ------------- PAGES -------------


def page_dashboard(df_scored: pd.DataFrame) -> None:
    cfg = st.session_state.get("col_cfg", {})
    amount_col = cfg.get("amount", "amount")
    label_col = cfg.get("label", "is_fraud")

    st.title("Dashboard")
    total = len(df_scored)
    high = int((df_scored["risk_band"] == "HIGH").sum())
    medium = int((df_scored["risk_band"] == "MEDIUM").sum())
    low = int((df_scored["risk_band"] == "LOW").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("High Risk", f"{high}")
    c3.metric("Medium Risk", f"{medium}")
    c4.metric("Low Risk", f"{low}")

    avg_amount_text = "n/a"
    if amount_col in df_scored.columns:
        avg_amount = float(df_scored[amount_col].mean())
        avg_amount_text = f"{avg_amount:,.2f}"

    fraud_rate_text = "n/a"
    if label_col in df_scored.columns and len(df_scored) > 0:
        fraud_rate = float((df_scored[label_col] == 1).mean())
        fraud_rate_text = f"{fraud_rate:.2%}"

    c5, c6 = st.columns(2)
    c5.metric("Avg amount (mapped)", avg_amount_text)
    c6.metric("Fraud rate (mapped label)", fraud_rate_text)

    st.markdown("---")
    col1, col2 = st.columns([1.4, 1.0])

    with col1:
        st.subheader("Risk Band Distribution")
        fig = px.histogram(
            df_scored,
            x="risk_band",
            color="risk_band",
            category_orders={"risk_band": ["LOW", "MEDIUM", "HIGH"]},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Amount vs. Risk Score")
        x_axis = amount_col if amount_col in df_scored.columns else "amount"
        fig2 = px.scatter(
            df_scored,
            x=x_axis,
            y="risk_score",
            color="risk_band",
            hover_data=["tx_id", "country", "device"],
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Recent Transactions Snapshot")
    if "timestamp" in df_scored.columns:
        snapshot = df_scored.sort_values("timestamp").tail(20)
    else:
        snapshot = df_scored.head(20)
    st.dataframe(snapshot, use_container_width=True, height=320)

# Real-time Stream


def page_real_time_stream(df_scored: pd.DataFrame) -> None:
    init_stream_state()
    st.title("Real-time Stream")

    left, right = st.columns([1.6, 1.0])

    with left:
        st.subheader("Streaming control")

        # batch size shared by manual + auto
        st.session_state.stream_batch_size = st.slider(
            "Batch size (events per tick)",
            min_value=1,
            max_value=256,
            value=int(st.session_state.stream_batch_size),
            step=1,
        )

        # manual one-shot
        if st.button("Process next batch (manual)"):
            batch = get_next_stream_batch(df_scored)
            update_stream_history(batch)

        # auto mode toggle
        auto_on = st.checkbox(
            "Enable auto streaming",
            key="auto_stream_enabled",
        )

        # interval only relevant when auto is on
        interval = st.slider(
            "Seconds between auto batches",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.5,
            key="auto_stream_interval",
        )

        # if auto is enabled, process one batch and rerun
        if auto_on:
            batch = get_next_stream_batch(df_scored)
            update_stream_history(batch)
            time.sleep(interval)
            st.rerun()

    with right:
        hist = st.session_state.stream_history
        total = len(hist)
        high = int((hist.get("risk_band", pd.Series(dtype=str)) == "HIGH").sum())
        rate = (high / total * 100.0) if total > 0 else 0.0
        avg = (
            float(hist.get("risk_score", pd.Series(dtype=float)).mean())
            if total > 0
            else 0.0
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Processed", f"{total}")
        c2.metric("High risk seen", f"{high}")
        c3.metric("High risk %", f"{rate:.1f}%")
        st.caption(f"Average risk score: **{avg:.2f}**")

    st.markdown("---")
    col3, col4 = st.columns([1.2, 1.0])
    hist = st.session_state.stream_history

    with col3:
        st.subheader("Streaming history")
        if hist.empty:
            st.info("No streaming data yet. Click manual button or enable auto.")
        else:
            cols = [
                "tx_id",
                "amount",
                "country",
                "device",
                "risk_score",
                "risk_band",
                "rule_hits",
            ]
            cols = [c for c in cols if c in hist.columns]
            st.dataframe(
                hist[cols],
                use_container_width=True,
                height=280,
            )

    with col4:
        st.subheader("Risk over time")
        if hist.empty:
            st.info("Stream some data to see the timeline.")
        else:
            tmp = hist.copy()
            tmp["index"] = range(len(tmp), 0, -1)
            fig = px.line(
                tmp.sort_values("index"),
                x="index",
                y="risk_score",
                color="risk_band",
            )
            fig.update_layout(xaxis_title="Event Index (latest → oldest)")
            st.plotly_chart(fig, use_container_width=True)

# Live Graphs


def page_live_graphs(df_scored: pd.DataFrame) -> None:
    cfg = st.session_state.get("col_cfg", {})
    amount_col = cfg.get("amount", "amount")
    time_col = cfg.get("time", "timestamp")

    st.title("Live Graphs")

    auto_refresh = st.sidebar.checkbox(
        "Auto-refresh live graphs (every 5s)",
        value=True,
        key="auto_refresh_live",
    )
    if auto_refresh:
        st.caption("🔄 Auto-refresh is ON (5s)")
        time.sleep(5)
        st.rerun()

    # Prefer live streamed data; fall back to full dataset
    df_source = st.session_state.get("live_graph_df")
    if df_source is None or df_source.empty:
        df_source = df_scored
        source_label = "full dataset (no live stream yet)"
    else:
        source_label = "latest streamed transactions"
    st.caption(f"Charts use **{source_label}**.")

    numeric_cols = [c for c in df_source.columns if pd.api.types.is_numeric_dtype(df_source[c])]
    y_col = amount_col if amount_col in df_source.columns else (numeric_cols[0] if numeric_cols else None)
    if y_col is None:
        st.info("No numeric column available for amount-based charts.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Amount Distribution by Country")
        fig = px.box(df_source, x="country", y=y_col, color="country")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Device Mix by Risk Band")
        fig2 = px.histogram(
            df_source,
            x="device",
            color="risk_band",
            barmode="group",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Timeline of Total Amount")
    time_key = time_col if time_col and time_col in df_source.columns else "timestamp"
    if time_key in df_source.columns:
        tmp = df_source.copy()
        tmp["day"] = pd.to_datetime(tmp[time_key]).dt.date
        daily = tmp.groupby("day")[y_col].sum().reset_index()
        fig3 = px.line(daily, x="day", y=y_col, markers=True)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Timestamp column not available; using synthetic timestamps only.")

# Advanced Transaction Analysis


def page_advanced_tx_analysis(df_scored: pd.DataFrame) -> None:
    st.title("Advanced Transaction Analysis")

    with st.sidebar.expander("Filters", expanded=True):
        min_amount, max_amount = float(df_scored["amount"].min()), float(
            df_scored["amount"].max()
        )
        amount_range = st.slider(
            "Amount range",
            min_value=float(min_amount),
            max_value=float(max_amount),
            value=(float(min_amount), float(max_amount)),
        )
        countries = sorted(df_scored["country"].astype(str).unique())
        country_sel = st.multiselect("Countries", countries, default=countries)
        bands = st.multiselect(
            "Risk bands", ["LOW", "MEDIUM", "HIGH"], default=["MEDIUM", "HIGH"]
        )

    filt = df_scored[
        (df_scored["amount"].between(amount_range[0], amount_range[1]))
        & (df_scored["country"].astype(str).isin(country_sel))
        & (df_scored["risk_band"].isin(bands))
    ]

    c1, c2 = st.columns(2)
    c1.metric("Filtered Count", f"{len(filt)}")
    if len(filt) > 0:
        high_frac = (filt["risk_band"] == "HIGH").mean() * 100.0
    else:
        high_frac = 0.0
    c2.metric("High-Risk % (filtered)", f"{high_frac:.1f}%")

    st.subheader("Filtered Transactions")
    st.dataframe(
        filt.sort_values("risk_score", ascending=False),
        use_container_width=True,
        height=320,
    )

# Threshold & Mode Tuning


def page_threshold_mode_tuning(df_scored: pd.DataFrame) -> None:
    st.title("Threshold & Mode Tuning")

    st.write(
        "Experiment with different **global risk thresholds** and see how many "
        "transactions would be flagged as high-risk."
    )

    threshold = st.slider(
        "Decision threshold on risk_score",
        0.0,
        1.0,
        0.7,
        0.05,
    )

    df = df_scored.copy()
    df["flagged"] = df["risk_score"] >= threshold
    total = len(df)
    flagged = int(df["flagged"].sum())
    flagged_pct = (flagged / total * 100.0) if total > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions", f"{total}")
    c2.metric("Flagged at this threshold", f"{flagged}")
    c3.metric("Flagged (%)", f"{flagged_pct:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Risk Score Histogram")
        fig = px.histogram(df, x="risk_score", nbins=40)
        fig.add_vline(x=threshold, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Flagged vs Not Flagged")
        pie_df = df["flagged"].value_counts().rename_axis("flagged").reset_index(
            name="count"
        )
        pie_df["flagged"] = pie_df["flagged"].map({True: "Flagged", False: "Not Flagged"})
        fig2 = px.pie(pie_df, values="count", names="flagged")
        st.plotly_chart(fig2, use_container_width=True)

# Model Health & Drift


def page_model_health_drift(df_scored: pd.DataFrame) -> None:
    st.title("Model Health & Drift")

    if "timestamp" not in df_scored.columns:
        st.info("Timestamp column not present – using synthetic split for drift demo.")
        df_scored = df_scored.copy()
        df_scored["timestamp"] = pd.date_range(
            "2024-01-01", periods=len(df_scored), freq="min"
        )

    df_sorted = df_scored.sort_values("timestamp")
    mid = len(df_sorted) // 2
    old = df_sorted.iloc[:mid]
    new = df_sorted.iloc[mid:]

    st.subheader("Basic Drift Indicators")
    col1, col2, col3 = st.columns(3)
    col1.metric("Old window size", f"{len(old)}")
    col2.metric("New window size", f"{len(new)}")
    col3.metric(
        "Δ mean amount",
        f"{float(new['amount'].mean() - old['amount'].mean()):.2f}",
    )

    st.markdown("---")
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Amount distribution – Old vs New")
        tmp = pd.concat(
            [
                old.assign(window="Old"),
                new.assign(window="New"),
            ]
        )
        fig = px.box(tmp, x="window", y="amount", color="window")
        st.plotly_chart(fig, use_container_width=True)
    with col5:
        st.subheader("Risk band share – Old vs New")
        tmp2 = (
            tmp.groupby(["window", "risk_band"])["tx_id"]
            .count()
            .reset_index(name="count")
        )
        fig2 = px.bar(
            tmp2,
            x="risk_band",
            y="count",
            color="window",
            barmode="group",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.info(
        "In a full production setup we would compute PSI / KL-divergence and track "
        "these metrics on a schedule; here we show a simplified visual drift check."
    )

# Data Explorer


def page_data_explorer(df_scored: pd.DataFrame) -> None:
    st.title("Data Explorer")

    cols = st.multiselect(
        "Columns to show", list(df_scored.columns), default=list(df_scored.columns)
    )
    st.dataframe(
        df_scored[cols],
        use_container_width=True,
        height=360,
    )

# Ensemble Status


def page_ensemble_status() -> None:
    st.title("Ensemble Status")

    st.write(
        "This section summarises the different models in the fraud ensemble "
        "(baseline, anomaly, supervised). For now we display a conceptual overview."
    )

    data = pd.DataFrame(
        [
            ["Logistic Regression", "Baseline classifier", "OK", "Low", "Fast"],
            ["Random Forest", "Main production model", "OK", "Medium", "Medium"],
            ["Anomaly Model", "Isolation Forest / Autoencoder", "Planned", "High", "Medium"],
        ],
        columns=["Model", "Role", "Status", "Risk Focus", "Latency"],
    )
    st.dataframe(data, use_container_width=True, height=220)

    st.markdown(
        """
- **Status**: whether the model is actively used.  
- **Risk Focus**: how aggressively the model targets rare / outlier behaviour.  
- **Latency**: relative cost per transaction.
"""
    )

# Enhanced Global Fraud Network (map)


def page_enhanced_global_fraud_network(df_scored: pd.DataFrame) -> None:
    st.title("Enhanced Global Fraud Network")

    df_geo = ensure_geo_columns(df_scored)
    df_vis = add_visual_columns(df_geo)

    st.sidebar.subheader("Map Filters")
    min_risk = st.sidebar.slider("Minimum risk score", 0.0, 1.0, 0.6, 0.05)
    bands = st.sidebar.multiselect(
        "Risk bands", ["LOW", "MEDIUM", "HIGH"], default=["MEDIUM", "HIGH"]
    )
    devices = sorted(df_vis["device"].astype(str).unique())
    sel_devices = st.sidebar.multiselect("Devices", devices, default=devices)

    filt = df_vis[
        (df_vis["risk_score"] >= min_risk)
        & (df_vis["risk_band"].isin(bands))
        & (df_vis["device"].astype(str).isin(sel_devices))
    ]

    col1, col2 = st.columns([1.6, 1.0])

    with col1:
        if filt.empty:
            st.warning("No transactions match current filters.")
        else:
            view_state = pdk.ViewState(
                latitude=20.0,
                longitude=10.0,
                zoom=1.2,
                pitch=45,
                bearing=0,
            )
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=filt,
                get_position="[lon, lat]",
                get_radius="radius",
                get_fill_color="[color_r, color_g, color_b]",
                pickable=True,
                opacity=0.8,
                stroked=True,
                get_line_color=[200, 200, 200],
                line_width_min_pixels=1,
            )
            tooltip = {
                "html": (
                    "<b>Tx ID:</b> {tx_id}<br/>"
                    "<b>Amount:</b> ₹{amount}<br/>"
                    "<b>Country:</b> {country}<br/>"
                    "<b>Device:</b> {device}<br/>"
                    "<b>Risk Score:</b> {risk_score}<br/>"
                    "<b>Risk Band:</b> {risk_band}<br/>"
                    "<b>Rules:</b> {rule_hits}"
                ),
                "style": {
                    "backgroundColor": "rgba(15,23,42,0.95)",
                    "color": "white",
                },
            }
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="mapbox://styles/mapbox/dark-v11",
            )
            st.pydeck_chart(deck, use_container_width=True, height=550)

    with col2:
        st.subheader("Visible Transactions")
        if filt.empty:
            st.info("No rows to show.")
        else:
            cols = [
                "tx_id",
                "amount",
                "country",
                "device",
                "timestamp",
                "risk_score",
                "risk_band",
                "rule_hits",
            ]
            cols = [c for c in cols if c in filt.columns]
            st.dataframe(
                filt[cols].sort_values("risk_score", ascending=False),
                use_container_width=True,
                height=260,
            )

# Fraud Rings (placeholder)


def page_fraud_rings() -> None:
    st.title("Fraud Rings")
    st.info(
        "This section is reserved for network-graph based fraud ring detection "
        "(card–device–IP relationships). For the current version, we describe the "
        "design conceptually."
    )
    st.markdown(
        """
- Build a graph with nodes = **cards, devices, IPs, merchants**.  
- Draw edges for every transaction.  
- Identify **clusters** where many confirmed frauds share the same node
  (e.g., same device being used across multiple cards).  
- These clusters form **fraud rings** and can be visualised and prioritised
  for investigation.
"""
    )

# Fraud Investigation (Manual Review)


def page_fraud_investigation(df_scored: pd.DataFrame) -> None:
    init_manual_review_state()
    st.title("Fraud Investigation")

    seen = st.session_state.mr_seen_ids
    mask_new = ~df_scored["tx_id"].isin(seen)
    mask_risk = df_scored["risk_band"].isin(["MEDIUM", "HIGH"])
    queue = df_scored[mask_new & mask_risk].sort_values("risk_score", ascending=False).head(
        10
    )

    total = len(df_scored)
    high_all = int((df_scored["risk_band"] == "HIGH").sum())
    medium_all = int((df_scored["risk_band"] == "MEDIUM").sum())
    reviewed = len(st.session_state.mr_history)
    fraud_marked = int((st.session_state.mr_history["action"] == "MARK_FRAUD").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total}")
    c2.metric("High-Risk (ALL)", f"{high_all}")
    c3.metric("Medium-Risk (ALL)", f"{medium_all}")
    c4.metric("Reviewed / Fraud Marked", f"{reviewed} / {fraud_marked}")

    st.markdown("---")
    st.subheader("Manual Review Queue")

    if queue.empty:
        st.success("No new MEDIUM/HIGH risk transactions waiting for review.")
    else:
        for _, row in queue.iterrows():
            with st.container():
                band = row["risk_band"]
                risk_color = {"LOW": "#22c55e", "MEDIUM": "#facc15", "HIGH": "#ef4444"}[
                    band
                ]

                st.markdown(
                    """
                    <div style="
                        border-radius: 16px;
                        padding: 0.75rem 1rem;
                        margin-bottom: 0.6rem;
                        border: 1px solid rgba(148,163,184,0.5);
                        background: radial-gradient(circle at top left, #020617, #020617);
                        box-shadow: 0 10px 25px rgba(15,23,42,0.5);
                    ">
                    """,
                    unsafe_allow_html=True,
                )

                c1, c2, c3, c4 = st.columns([2.6, 1.5, 2.5, 2.2])

                with c1:
                    st.markdown(f"**{row['tx_id']}**")
                    st.caption(
                        f"₹{row['amount']:.2f} • {row['country']} • {row['device']} • "
                        f"{row.get('timestamp', '')}"
                    )

                with c2:
                    st.markdown(
                        f"<p style='margin-bottom:0.1rem;'>Risk Score</p>"
                        f"<p style='font-size:1.3rem; font-weight:600; color:{risk_color};'>"
                        f"{row['risk_score']:.2f}</p>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Band: **{band}**")

                with c3:
                    st.markdown("**Rules Hit**")
                    hits = row.get("rule_hits", "-")
                    st.write(hits if hits else "-")
                    st.caption(
                        f"Anomaly: `{row['anomaly_score']:.2f}` | Rules: `{row['rule_score']:.2f}`"
                    )

                with c4:
                    st.markdown("**Action**")
                    col_a, col_b, col_c = st.columns(3)

                    def record_action(action: str) -> None:
                        st.session_state.mr_seen_ids.add(row["tx_id"])
                        entry = {
                            "reviewed_at": pd.Timestamp.utcnow().isoformat(),
                            "tx_id": row["tx_id"],
                            "amount": float(row["amount"]),
                            "country": row["country"],
                            "device": row["device"],
                            "risk_score": float(row["risk_score"]),
                            "risk_band": row["risk_band"],
                            "rule_hits": row["rule_hits"],
                            "action": action,
                        }
                        st.session_state.mr_history = pd.concat(
                            [pd.DataFrame([entry]), st.session_state.mr_history],
                            axis=0,
                        ).reset_index(drop=True)
                        st.rerun()

                    if col_a.button("Fraud", key=f"fraud_{row['tx_id']}"):
                        record_action("MARK_FRAUD")
                    if col_b.button("Legit", key=f"legit_{row['tx_id']}"):
                        record_action("MARK_LEGIT")
                    if col_c.button("Skip", key=f"skip_{row['tx_id']}"):
                        record_action("SKIP")

                st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Review History")
    hist = st.session_state.mr_history
    if hist.empty:
        st.info("No transactions reviewed yet.")
    else:
        cols = [
            "reviewed_at",
            "tx_id",
            "amount",
            "country",
            "device",
            "risk_score",
            "risk_band",
            "rule_hits",
            "action",
        ]
        st.dataframe(
            hist[cols],
            use_container_width=True,
            height=260,
        )

# Rule Engine page


def page_rule_engine() -> None:
    st.title("Rule Engine")

    st.write("Current demo rules (also used by the risk engine):")
    rules = [
        ("high_amount", "amount > 5000", "Adds 0.4 to rule_score"),
        ("very_high_amount", "amount > 20000", "Adds 0.3 to rule_score"),
        ("high_risk_country", "country in {BR}", "Adds 0.2 to rule_score"),
        ("web_device", "device == WEB", "Adds 0.1 to rule_score"),
    ]
    df_rules = pd.DataFrame(rules, columns=["Rule Name", "Condition", "Effect"])
    st.dataframe(df_rules, use_container_width=True, height=200)

    st.info(
        "In a full product, this page would allow risk analysts to add/edit rules "
        "through a safe configuration interface."
    )

# Audit Log page


def page_audit_log(df_scored: pd.DataFrame) -> None:  # df_scored unused but kept for symmetry
    st.title("Audit Log")

    hist_review = st.session_state.get("mr_history", pd.DataFrame())
    hist_stream = st.session_state.get("stream_history", pd.DataFrame())

    st.subheader("Manual Review Decisions")
    if hist_review.empty:
        st.info("No manual decisions recorded yet.")
    else:
        st.dataframe(
            hist_review,
            use_container_width=True,
            height=260,
        )

    st.subheader("Recent Streamed Transactions")
    if hist_stream.empty:
        st.info("No streaming history yet.")
    else:
        cols = [
            "tx_id",
            "amount",
            "country",
            "device",
            "risk_score",
            "risk_band",
            "rule_hits",
        ]
        cols = [c for c in cols if c in hist_stream.columns]
        st.dataframe(
            hist_stream[cols],
            use_container_width=True,
            height=260,
        )

# Case Management (placeholder)


def page_case_management() -> None:
    st.title("Case Management")
    st.info(
        "Future extension: grouping alerts and manual decisions into cases, assigning "
        "owners, and tracking SLA / resolution notes."
    )

# Batch Scoring (placeholder – can be upgraded later)


def page_batch_scoring(df_scored: pd.DataFrame) -> None:  # df_scored kept for future use
    st.title("Batch Scoring")
    st.info(
        "Future extension: upload a CSV file, score it using the model ensemble, and "
        "download results with risk scores and decisions."
    )

# Executive Report


def page_executive_report(df_scored: pd.DataFrame) -> None:
    st.title("Executive Report")

    total = len(df_scored)
    high = int((df_scored["risk_band"] == "HIGH").sum())
    fraud_rate_est = (high / total * 100.0) if total > 0 else 0.0
    avg_amount = float(df_scored["amount"].mean())

    st.subheader("Key KPIs (Demo)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Transactions Analysed", f"{total:,}")
    c2.metric("High-Risk Transactions", f"{high}")
    c3.metric("High-Risk Rate", f"{fraud_rate_est:.1f}%")

    st.markdown(
        f"""
**Summary (for slides / IEEE paper):**

- The platform processes approximately **{total:,}** transactions in the dataset.  
- Around **{high}** transactions (≈ **{fraud_rate_est:.1f}%**) are classified as high-risk.  
- The average transaction amount is **₹{avg_amount:,.2f}**, with heavy-tail behaviour
  captured by the anomaly model and rule engine.
"""
    )

# What-If Simulator


def page_what_if_simulator() -> None:
    st.title("What-If Simulator")

    st.write("Simulate a single transaction and see the risk score live.")

    col1, col2, col3 = st.columns(3)
    amount = col1.number_input("Amount (₹)", min_value=0.0, value=2500.0, step=100.0)
    country = col2.selectbox("Country", ["IN", "US", "UK", "SG", "BR", "DE"])
    device = col3.selectbox("Device", ["ANDROID", "IOS", "WEB"])

    dummy_row = pd.Series({"amount": amount, "country": country, "device": device})
    a_score = simple_anomaly_score(dummy_row)
    r_score, rules = rule_score(dummy_row)
    risk = 0.7 * a_score + 0.3 * r_score

    if risk < 0.4:
        band = "LOW"
    elif risk < 0.7:
        band = "MEDIUM"
    else:
        band = "HIGH"

    st.markdown("---")
    c4, c5, c6 = st.columns(3)
    c4.metric("Anomaly Score", f"{a_score:.2f}")
    c5.metric("Rule Score", f"{r_score:.2f}")
    c6.metric("Combined Risk", f"{risk:.2f}")

    st.write(f"**Risk Band:** {band}")
    st.write(
        "**Rules triggered:** "
        + (", ".join(k for k, v in rules.items() if v) if any(rules.values()) else "None")
    )

# Project Report (text)


def page_project_report() -> None:
    st.title("Project Report")

    st.markdown(
        """
**System Overview**

- Real-time streaming of card / online transactions.  
- Hybrid detection:
  - anomaly score from model  
  - business rule engine  
- Unified risk score used to drive:
  - real-time stream view  
  - fraud investigation queue  
  - 3D global fraud map  
  - executive metrics and model lab  

This page can be used as a base to write the full IEEE project report /
internal documentation.
"""
    )

# Phase 2 Plan (text)


def page_phase2_plan() -> None:
    st.title("Phase 2 Plan")

    st.markdown(
        """
Planned enhancements:

1. **True ensemble with multiple ML models**  
   - Isolation Forest, Autoencoder, Gradient Boosting.  
2. **Fraud ring detection**  
   - Graph analytics across cards, devices and IPs.  
3. **Production-grade persistence**  
   - PostgreSQL / cloud warehouse for audit log and alerts.  
4. **User and case management**  
   - Role-based access, case workflows, SLA tracking.  

These points can be directly reused in your viva / future work section.
"""
    )

# Real-Time Fraud Decision Pipeline (static diagram)


def page_real_time_pipeline_diagram() -> None:
    st.title("Real-Time Fraud Decision Pipeline")

    st.write(
        "High-level view of how transactions move through the system "
        "from ingestion to decision."
    )

    stages = [
        "Client / Merchant",
        "Gateway & API",
        "Preprocessing & Features",
        "Anomaly + Rules Engine",
        "Risk Score & Decision",
        "Alerts & Case Mgmt",
    ]

    boxes = []
    for name in stages:
        boxes.append(
            f"""
            <div style="
                flex: 1;
                margin: 0.35rem;
                padding: 0.9rem 0.5rem;
                text-align: center;
                border-radius: 12px;
                background: #0f172a;
                border: 1px solid #64748b;
                box-shadow: 0 0 14px rgba(15,23,42,0.9);
                color: white;
                font-size: 0.85rem;
                font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            ">
                {name}
            </div>
            """
        )

    row_html = '<div style="display:flex;flex-direction:row;align-items:stretch;justify-content:space-between;">'
    for i, box in enumerate(boxes):
        row_html += box
        if i < len(boxes) - 1:
            row_html += (
                '<div style="align-self:center;font-size:1.4rem;margin:0 0.1rem;">➡️</div>'
            )
    row_html += "</div>"

    st.markdown(
        f"""
        <div style="
            border-radius: 16px;
            padding: 0.9rem;
            background: radial-gradient(circle at top left, #1e293b, #020617);
            border: 1px solid #1f2937;
            margin-bottom: 0.75rem;
        ">
            {row_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
Below the boxes you can explain each step in your viva or report and
relate it back to the individual pages in this dashboard.
"""
    )

    st.markdown(
        """
### How this pipeline runs in your project

1. **Client / Merchant → CSV / Stream**  
   - Real transactions are captured by the payment gateway.  
   - For the project demo we simulate this with a CSV (Kaggle dataset).  

2. **Gateway & API**  
   - In production this would be a FastAPI / Kafka layer receiving live transactions.  
   - In the current version, the *Real-time Stream* page mimics this by reading the
     dataset batch by batch.

3. **Preprocessing & Features**  
   - Columns are normalised in `_standardise_columns()` (tx_id, amount, country, device, timestamp).  
   - The sidebar mapping lets you choose which CSV column is amount / label / time.

4. **Anomaly + Rules Engine**  
   - `simple_anomaly_score()` looks at amount, country and device and converts behaviour
     into a [0,1] anomaly score.  
   - `rule_score()` applies business rules (high amount, high-risk country, web device, etc.).  
   - `combined_risk()` merges both into a single `risk_score` and `risk_band`.

5. **Risk Score & Decision**  
   - The Dashboard, Threshold Tuning and What-If Simulator pages use `risk_score` to decide
     which transactions are HIGH / MEDIUM / LOW.  

6. **Alerts, Investigation & Audit**  
   - High-risk transactions flow into the *Fraud Investigation* queue for manual review.  
   - All actions and streamed events are stored in session state and shown in *Audit Log*.

Together, this behaves like a mini real-time fraud decision engine suitable for IEEE / viva demo.
"""
    )

# ------------- MAIN -------------


def main() -> None:
    render_data_source_sidebar()
    df_raw = load_project_dataset()
    col_cfg = configure_dataset_columns(df_raw)

    df_work = df_raw.copy()
    amount_col = col_cfg.get("amount")
    label_col = col_cfg.get("label")
    time_col = col_cfg.get("time")

    if amount_col and amount_col in df_work.columns and amount_col != "amount":
        if "amount" not in df_work.columns:
            df_work = df_work.rename(columns={amount_col: "amount"})
        col_cfg["amount"] = "amount" if "amount" in df_work.columns else amount_col

    if label_col and label_col in df_work.columns and label_col not in {"is_fraud", "isFraud"}:
        if "is_fraud" not in df_work.columns:
            df_work = df_work.rename(columns={label_col: "is_fraud"})
            col_cfg["label"] = "is_fraud"
        else:
            col_cfg["label"] = "is_fraud"

    if time_col and time_col in df_work.columns and time_col != "timestamp":
        if "timestamp" not in df_work.columns:
            df_work = df_work.rename(columns={time_col: "timestamp"})
            col_cfg["time"] = "timestamp"
        else:
            col_cfg["time"] = "timestamp"

    st.session_state["col_cfg"] = col_cfg

    if st.sidebar.checkbox("Show raw dataset preview", value=False, key="show_raw_preview"):
        st.write("✅ Dataset shape:", df_work.shape)
        st.dataframe(df_work.head())
    df_scored = add_risk_columns(df_work)

    st.sidebar.markdown("---")
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to",
        (
            "Dashboard",
            "Real-time Stream",
            "Live Graphs",
            "Advanced Transaction Analysis",
            "Threshold & Mode Tuning",
            "Model Health & Drift",
            "Data Explorer",
            "Ensemble Status",
            "Enhanced Global Fraud Network",
            "Fraud Rings",
            "Fraud Investigation",
            "Rule Engine",
            "Audit Log",
            "Case Management",
            "Batch Scoring",
            "Executive Report",
            "What-If Simulator",
            "Project Report",
            "Phase 2 Plan",
            "Real-Time Fraud Decision Pipeline",
        ),
    )

    if page == "Dashboard":
        page_dashboard(df_scored)
    elif page == "Real-time Stream":
        page_real_time_stream(df_scored)
    elif page == "Live Graphs":
        page_live_graphs(df_scored)
    elif page == "Advanced Transaction Analysis":
        page_advanced_tx_analysis(df_scored)
    elif page == "Threshold & Mode Tuning":
        page_threshold_mode_tuning(df_scored)
    elif page == "Model Health & Drift":
        page_model_health_drift(df_scored)
    elif page == "Data Explorer":
        page_data_explorer(df_scored)
    elif page == "Ensemble Status":
        page_ensemble_status()
    elif page == "Enhanced Global Fraud Network":
        page_enhanced_global_fraud_network(df_scored)
    elif page == "Fraud Rings":
        page_fraud_rings()
    elif page == "Fraud Investigation":
        page_fraud_investigation(df_scored)
    elif page == "Rule Engine":
        page_rule_engine()
    elif page == "Audit Log":
        page_audit_log(df_scored)
    elif page == "Case Management":
        page_case_management()
    elif page == "Batch Scoring":
        page_batch_scoring(df_scored)
    elif page == "Executive Report":
        page_executive_report(df_scored)
    elif page == "What-If Simulator":
        page_what_if_simulator()
    elif page == "Project Report":
        page_project_report()
    elif page == "Phase 2 Plan":
        page_phase2_plan()
    elif page == "Real-Time Fraud Decision Pipeline":
        page_real_time_pipeline_diagram()


if __name__ == "__main__":
    main()

