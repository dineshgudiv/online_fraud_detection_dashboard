from __future__ import annotations

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
    for col in ("tx_datetime", "timestamp", "event_time", "date", "step"):
        if col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
            return col
    return None


def get_type_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("type", "category", "txn_type", "channel"):
        if col in df.columns:
            return col
    return None


st.title("Advanced Transaction Analysis")
st.caption("Deep-dive transaction analytics with flexible filters and rich charts.")

# ---- Load data ----
try:
    df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – cannot analyse transactions.")
    st.stop()

label_col = get_label_col(df)
amount_col = get_amount_col(df)
time_col = get_time_col(df)
type_col = get_type_col(df)

cust_col = next((c for c in ["customer_id", "cust_id", "nameOrig", "client_id"] if c in df.columns), None)
merch_col = next((c for c in ["merchant_id", "nameDest", "terminal_id"] if c in df.columns), None)

# ---- Sidebar filters ----
with st.sidebar:
    st.subheader("Filters")

    working_df = df.copy()

    # Time filter
    if time_col and pd.api.types.is_datetime64_any_dtype(working_df[time_col]):
        min_dt = working_df[time_col].min()
        max_dt = working_df[time_col].max()
        start, end = st.date_input(
            "Date range",
            value=(min_dt.date(), max_dt.date()),
        )
        working_df = working_df[
            (working_df[time_col].dt.date >= start)
            & (working_df[time_col].dt.date <= end)
        ]

    # Amount filter
    if amount_col:
        min_amt = float(working_df[amount_col].min())
        max_amt = float(working_df[amount_col].max())
        low, high = st.slider(
            "Amount range",
            min_value=min_amt,
            max_value=max_amt,
            value=(min_amt, max_amt),
        )
        working_df = working_df[
            (working_df[amount_col] >= low)
            & (working_df[amount_col] <= high)
        ]

    # Type filter
    if type_col:
        types = (
            working_df[type_col]
            .astype(str)
            .dropna()
            .value_counts()
            .head(30)
            .index.tolist()
        )
        selected_types = st.multiselect(
            "Transaction types",
            options=types,
            default=types,
        )
        working_df = working_df[working_df[type_col].astype(str).isin(selected_types)]

    max_rows = st.slider(
        "Rows to show in table",
        min_value=100,
        max_value=min(5000, len(working_df)),
        value=min(1000, len(working_df)),
        step=100,
    )

st.write(f"Filtered transactions: **{len(working_df):,}**")

if working_df.empty:
    st.warning("No data after applying filters. Adjust filters in the sidebar.")
    st.stop()

# ---- KPIs for filtered set ----
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Transactions (filtered)", f"{len(working_df):,}")
with c2:
    st.metric(
        "Unique Customers",
        f"{working_df[cust_col].nunique():,}" if cust_col else "N/A",
    )
with c3:
    st.metric(
        "Unique Merchants",
        f"{working_df[merch_col].nunique():,}" if merch_col else "N/A",
    )
with c4:
    if label_col:
        st.metric(
            "Fraud Rate (filtered)",
            f"{100 * working_df[label_col].mean():.2f}%",
        )
    else:
        st.metric("Fraud Rate (filtered)", "Unknown")

st.markdown("---")

# ---- Charts ----
tab_dist, tab_time, tab_type = st.tabs(["Amount Distribution", "Amount over Time", "Amount vs Type"])

# Amount Distribution
with tab_dist:
    if amount_col:
        st.subheader("Amount Distribution (filtered sample)")
        fig_hist = px.histogram(
            working_df,
            x=amount_col,
            nbins=50,
            title=f"Histogram of {amount_col}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        if label_col:
            st.subheader("Amount by Fraud Label")
            fig_box = px.box(
                working_df,
                x=label_col,
                y=amount_col,
                points="outliers",
                labels={label_col: "Is Fraud"},
                title=f"{amount_col} vs Fraud Label",
            )
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No numeric amount-like column found (e.g., 'amount'); cannot plot amount distribution.")

# Amount over Time
with tab_time:
    st.subheader("Amount over Time")
    if time_col and amount_col and pd.api.types.is_datetime64_any_dtype(working_df[time_col]):
        ts_df = working_df[[time_col, amount_col]].copy().set_index(time_col)
        # resample by hour/day depending on span
        span_days = (ts_df.index.max() - ts_df.index.min()).days
        freq = "1H" if span_days <= 3 else "1D"
        ts = ts_df.resample(freq)[amount_col].sum().rename("amount_sum").reset_index()

        fig_ts = px.line(
            ts,
            x=time_col,
            y="amount_sum",
            title=f"Total {amount_col} over time (freq={freq})",
            labels={time_col: "Time", "amount_sum": "Total Amount"},
        )
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info(
            "No usable datetime column and amount column combination found; "
            "cannot build time-based amount chart."
        )

# Amount vs Type
with tab_type:
    st.subheader("Amount by Transaction Type")
    if type_col and amount_col:
        fig_type = px.box(
            working_df,
            x=type_col,
            y=amount_col,
            points="outliers",
            title=f"{amount_col} by {type_col}",
        )
        st.plotly_chart(fig_type, use_container_width=True)
    else:
        st.info(
            "Cannot build amount vs type chart – missing either a type/category column or an amount column."
        )

# ---- Raw table ----
st.subheader("Sample Transactions (filtered)")
st.dataframe(working_df.head(max_rows), use_container_width=True)
