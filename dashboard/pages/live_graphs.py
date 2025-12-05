from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from shared.data_loader import load_dataset


def get_label_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("is_fraud", "isFraud", "fraud_label", "label"):
        if col in df.columns:
            return col
    return None


def get_amount_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("amount", "amt", "transaction_amount"):
        if col in df.columns:
            return col
    return None


def get_time_col(df: pd.DataFrame) -> Optional[str]:
    """Return a usable time-like column; try to parse to datetime if needed."""
    for col in ("tx_datetime", "timestamp", "event_time", "date", "step"):
        if col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
            return col
    return None


st.title("Live Graphs")
st.caption("Near real-time view of transactions and fraud behaviour (via periodic reload).")

# ---- Load data safely ----
try:
    df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – cannot render live graphs.")
    st.stop()

label_col = get_label_col(df)
amount_col = get_amount_col(df)
time_col = get_time_col(df)

# ---- Session state for refresh tracking ----
if "live_last_refresh" not in st.session_state:
    st.session_state["live_last_refresh"] = time.time()

with st.sidebar:
    st.subheader("Live Controls")

    max_sample = max(1, len(df))
    default_sample = min(2000, max_sample)
    min_sample = min(500, max_sample)

    sample_size = st.slider(
        "Sample size for live view",
        min_value=min_sample,
        max_value=max_sample,
        value=default_sample,
        step=max(100, max_sample // 20),
    )

    auto_refresh = st.checkbox("Auto-refresh (simulated)", value=False)
    refresh_interval = st.slider(
        "Refresh interval (seconds)",
        min_value=5,
        max_value=60,
        value=15,
    )

    manual_refresh = st.button("🔄 Refresh now")

# ---- Handle refresh logic ----
now = time.time()

if manual_refresh:
    st.session_state["live_last_refresh"] = now
    st.rerun()

if auto_refresh and (now - st.session_state["live_last_refresh"]) >= refresh_interval:
    st.session_state["live_last_refresh"] = now
    st.rerun()

# Use last_refresh as a seed so each "tick" reshuffles data consistently
seed = int(st.session_state["live_last_refresh"]) & 0xFFFFFFFF
rng = np.random.default_rng(seed)

if sample_size >= len(df):
    live_df = df.copy()
else:
    live_df = df.sample(sample_size, random_state=seed).copy()

# ---- KPIs for the current live window ----
total_live = len(live_df)
fraud_count = None
fraud_rate = None
total_amount = None

if label_col:
    fraud_count = int(live_df[label_col].sum())
    fraud_rate = 100 * fraud_count / max(1, total_live)

if amount_col:
    total_amount = float(live_df[amount_col].sum())

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Live Window Transactions", f"{total_live:,}")
with c2:
    st.metric("Live Fraud Count", f"{fraud_count:,}" if fraud_count is not None else "Unknown")
with c3:
    st.metric("Live Fraud Rate", f"{fraud_rate:.2f}%" if fraud_rate is not None else "Unknown")
with c4:
    st.metric("Live Amount Volume", f"{total_amount:,.2f}" if total_amount is not None else "N/A")

st.markdown(
    f"_Last refresh: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state['live_last_refresh']))}_"
)

st.markdown("---")

# ---- Time-series graph (if we have a time column) ----
if time_col and pd.api.types.is_datetime64_any_dtype(live_df[time_col]):
    st.subheader("Live Time-Series (count per minute)")

    recent_df = live_df.sort_values(time_col).tail(1000).set_index(time_col)
    try:
        ts_counts = recent_df.resample("1min").size().rename("transactions")
        fig_ts = px.line(
            ts_counts,
            labels={"value": "Transactions", "index": "Time"},
            title="Transactions per Minute (Live Window)",
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not build time-series view ({exc}); falling back to simple histogram.")
        st.bar_chart(ts_counts if "ts_counts" in locals() else recent_df.index.value_counts())

else:
    st.info(
        "No usable datetime column found for a proper time-series view. "
        "To enable it, ensure the dataset has a timestamp-like column."
    )

# ---- Fraud vs Non-Fraud bar ----
if label_col:
    st.subheader("Fraud vs Non-Fraud (Live Window)")
    class_counts = (
        live_df[label_col]
        .map({0: "Non-Fraud", 1: "Fraud"})
        .value_counts()
        .rename_axis("class")
        .reset_index(name="count")
    )
    fig_bar = px.bar(
        class_counts,
        x="class",
        y="count",
        title="Class Distribution in Live Sample",
        text="count",
    )
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)
else:
    st.info("No explicit fraud label column found (e.g., 'is_fraud'); class split not available.")

# ---- Amount distribution (optional) ----
if amount_col:
    st.subheader("Amount Distribution (Live Sample)")
    fig_hist = px.histogram(
        live_df,
        x=amount_col,
        nbins=50,
        title="Transaction Amount Histogram (Live Sample)",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ---- Raw view ----
with st.expander("Show raw live sample table"):
    st.dataframe(live_df.head(500), use_container_width=True)
