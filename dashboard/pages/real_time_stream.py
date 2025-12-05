from __future__ import annotations

from typing import Optional

import time

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from shared.data_loader import load_dataset


# --------- helpers ---------
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
    for col in ("tx_datetime", "timestamp", "event_time", "date", "step"):
        if col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
            return col
    return None


# --------- page title ---------
st.title("Real Time Stream")
st.caption(
    "Simulated real-time transaction stream over the dataset with start/stop controls, "
    "rolling window KPIs and live charts."
)

# --------- load dataset ---------
try:
    df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – cannot simulate real-time stream.")
    st.stop()

label_col = get_label_col(df)
amount_col = get_amount_col(df)
time_col = get_time_col(df)

n_rows = len(df)

# --------- session state init ---------
if "rt_running" not in st.session_state:
    st.session_state["rt_running"] = False
if "rt_offset" not in st.session_state:
    st.session_state["rt_offset"] = 0  # how many rows have been 'seen' so far
if "rt_last_batch_size" not in st.session_state:
    st.session_state["rt_last_batch_size"] = 100
if "rt_last_window_size" not in st.session_state:
    st.session_state["rt_last_window_size"] = 500
if "rt_last_refresh" not in st.session_state:
    st.session_state["rt_last_refresh"] = 2.0

# --------- sidebar controls ---------
with st.sidebar:
    st.subheader("Streaming Controls")

    batch_size = st.number_input(
        "Batch size per tick",
        min_value=10,
        max_value=5000,
        value=int(st.session_state["rt_last_batch_size"]),
        step=10,
        help="How many new transactions to reveal on each step / auto-tick.",
    )
    window_size = st.number_input(
        "Rolling window size",
        min_value=100,
        max_value=20000,
        value=int(st.session_state["rt_last_window_size"]),
        step=100,
        help="How many most-recent transactions to keep in the live window.",
    )
    refresh_secs = st.number_input(
        "Auto-refresh interval (seconds)",
        min_value=0.5,
        max_value=10.0,
        value=float(st.session_state["rt_last_refresh"]),
        step=0.5,
        help="Only used when streaming is running.",
    )

    st.session_state["rt_last_batch_size"] = batch_size
    st.session_state["rt_last_window_size"] = window_size
    st.session_state["rt_last_refresh"] = refresh_secs

    st.markdown("---")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("▶ Start"):
            st.session_state["rt_running"] = True
    with col_btn2:
        if st.button("⏸ Stop"):
            st.session_state["rt_running"] = False

    if st.button("⏭ Step once"):
        st.session_state["rt_running"] = False  # manual step
        st.session_state["rt_offset"] = int(
            min(n_rows, st.session_state["rt_offset"] + batch_size)
        )

    if st.button("⏹ Reset"):
        st.session_state["rt_running"] = False
        st.session_state["rt_offset"] = 0

# --------- compute current window ---------
offset = int(st.session_state["rt_offset"])
offset = max(0, min(offset, n_rows))
st.session_state["rt_offset"] = offset

if offset == 0:
    window_df = df.iloc[0:0].copy()
else:
    start_idx = max(0, offset - window_size)
    window_df = df.iloc[start_idx:offset].copy()

st.write(
    f"Stream position: **{offset:,} / {n_rows:,}** rows revealed "
    f"(window shows up to last **{window_size:,}** rows)."
)

if window_df.empty:
    st.info("No events in window yet – click **Step once** or **Start** to begin streaming.")
else:
    # --------- KPIs for current window ---------
    st.subheader("Window KPIs")

    tx_window = len(window_df)
    fraud_window = int(window_df[label_col].sum()) if label_col else None
    fraud_rate_window = (
        fraud_window / tx_window * 100.0 if (label_col and tx_window > 0) else None
    )
    total_amount_window = float(window_df[amount_col].sum()) if amount_col else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Events in Window", f"{tx_window:,}")
    with c2:
        st.metric(
            "Fraud in Window",
            f"{fraud_window:,}" if fraud_window is not None else "Unknown",
        )
    with c3:
        st.metric(
            "Fraud Rate (Window)",
            f"{fraud_rate_window:.2f}%"
            if fraud_rate_window is not None
            else "Unknown",
        )
    with c4:
        st.metric(
            "Total Amount (Window)",
            f"{total_amount_window:,.2f}" if total_amount_window is not None else "N/A",
        )

    st.markdown("---")

    # --------- charts ---------
    tab_vol, tab_amount, tab_class = st.tabs(
        ["Volume over Time", "Amount over Time", "Fraud vs Non-Fraud"]
    )

    # Volume over time (within window)
    with tab_vol:
        st.subheader("Volume over Time (window)")
        if time_col and pd.api.types.is_datetime64_any_dtype(window_df[time_col]):
            tmp = window_df.set_index(time_col)
            # choose resampling frequency based on span
            span = tmp.index.max() - tmp.index.min()
            span_days = span.days + span.seconds / 86400.0
            freq = "1T" if span_days <= 1 else "1H"
            vol = tmp.resample(freq).size().to_frame("tx_count").reset_index()
            vol = vol.rename(columns={time_col: "time"})
            fig_vol = px.line(
                vol,
                x="time",
                y="tx_count",
                title=f"Transactions per {freq} (window)",
                labels={"time": "Time", "tx_count": "Transactions"},
            )
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info(
                "No usable datetime column found in dataset (e.g., 'timestamp'). "
                "Time-based volume chart is disabled."
            )

    # Amount over time
    with tab_amount:
        st.subheader("Amount over Time (window)")
        if amount_col and time_col and pd.api.types.is_datetime64_any_dtype(
            window_df[time_col]
        ):
            tmp = window_df.set_index(time_col)
            span = tmp.index.max() - tmp.index.min()
            span_days = span.days + span.seconds / 86400.0
            freq = "1T" if span_days <= 1 else "1H"
            amt = (
                tmp[amount_col]
                .resample(freq)
                .sum()
                .to_frame("amount_sum")
                .reset_index()
            )
            amt = amt.rename(columns={time_col: "time"})
            fig_amt = px.line(
                amt,
                x="time",
                y="amount_sum",
                title=f"Total {amount_col} per {freq} (window)",
                labels={"time": "Time", "amount_sum": "Amount"},
            )
            st.plotly_chart(fig_amt, use_container_width=True)
        elif amount_col:
            st.info(
                "Amount column found but no usable datetime column – cannot plot amount over time."
            )
        else:
            st.info("No amount-like column found (e.g., 'amount'); skipping amount chart.")

    # Fraud vs Non-Fraud
    with tab_class:
        st.subheader("Fraud vs Non-Fraud in Window")
        if label_col:
            cls_counts = (
                window_df[label_col]
                .map({0: "Non-Fraud", 1: "Fraud"})
                .value_counts()
                .rename_axis("class")
                .reset_index(name="count")
            )
            fig_cls = px.bar(
                cls_counts,
                x="class",
                y="count",
                title="Fraud vs Non-Fraud (window)",
                text="count",
                labels={"class": "Class", "count": "Count"},
            )
            fig_cls.update_traces(textposition="outside")
            st.plotly_chart(fig_cls, use_container_width=True)
        else:
            st.info(
                "No fraud label column detected (e.g., 'is_fraud'); cannot show class split."
            )

    st.markdown("---")
    st.subheader("Window Sample")
    st.dataframe(window_df.tail(100), use_container_width=True)

# --------- auto streaming logic (bottom) ---------
# If streaming is running and we haven't reached the end, sleep then advance and rerun.
if st.session_state["rt_running"]:
    if st.session_state["rt_offset"] >= n_rows:
        st.session_state["rt_running"] = False
        st.info("End of dataset reached – streaming stopped.")
    else:
        # simple auto-tick
        time.sleep(float(refresh_secs))
        st.session_state["rt_offset"] = int(
            min(n_rows, st.session_state["rt_offset"] + batch_size)
        )
        st.rerun()
