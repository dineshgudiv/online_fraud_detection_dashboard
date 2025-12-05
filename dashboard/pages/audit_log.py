from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from shared.data_loader import load_dataset


def get_label_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("is_fraud", "isFraud", "fraud_label", "label"):
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


def get_txid_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("transaction_id", "tx_id", "id", "row_id"):
        if col in df.columns:
            return col
    return None


def get_actor_cols(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    user_col = None
    for c in ("analyst", "user_id", "created_by", "updated_by"):
        if c in df.columns:
            user_col = c
            break
    action_col = None
    for c in ("action", "event_type", "decision", "status"):
        if c in df.columns:
            action_col = c
            break
    return user_col, action_col


st.title("Audit Log")
st.caption("Chronological view of events, decisions, and suspicious activity.")

# ---- Load dataset ----
try:
    df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – nothing to show in the audit log.")
    st.stop()

label_col = get_label_col(df)
time_col = get_time_col(df)
txid_col = get_txid_col(df)
user_col, action_col = get_actor_cols(df)

# Sort newest first if we have a time column
if time_col and pd.api.types.is_datetime64_any_dtype(df[time_col]):
    df = df.sort_values(time_col, ascending=False).reset_index(drop=True)
else:
    df = df.reset_index(drop=True)

# ---- Sidebar filters ----
with st.sidebar:
    st.subheader("Filters")

    working_df = df.copy()

    # Date range filter
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

    # Fraud / non-fraud filter
    if label_col:
        cls_choice = st.selectbox(
            "Class filter",
            options=["All", "Fraud only", "Non-fraud only"],
        )
        if cls_choice == "Fraud only":
            working_df = working_df[working_df[label_col] == 1]
        elif cls_choice == "Non-fraud only":
            working_df = working_df[working_df[label_col] == 0]

    max_rows = st.slider(
        "Rows to display",
        min_value=100,
        max_value=min(5000, len(working_df)),
        value=min(1000, len(working_df)),
        step=100,
    )

# ---- Search box ----
search = st.text_input(
    "Search text (transaction ID, user, action, device, etc.)",
    "",
    help="Performs a simple text search across visible columns.",
)

filtered_df = working_df

if search:
    query = search.lower().strip()

    def row_matches(row: pd.Series) -> bool:
        return query in " ".join(map(str, row.values)).lower()

    # Use apply row-wise search (OK for typical dashboard sizes)
    filtered_df = working_df[working_df.apply(row_matches, axis=1)]

st.write(
    f"Filtered events: **{len(filtered_df):,}** "
    f"(from total **{len(df):,}** rows)"
)

if filtered_df.empty:
    st.warning("No events match the current filters/search.")
    st.stop()

# ---- Top KPIs ----
fraud_count = None
if label_col:
    fraud_count = int(filtered_df[label_col].sum())

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Events in view", f"{len(filtered_df):,}")
with c2:
    st.metric("Distinct Transactions", f"{filtered_df[txid_col].nunique():,}" if txid_col else "N/A")
with c3:
    st.metric("Distinct Analysts/Users", f"{filtered_df[user_col].nunique():,}" if user_col else "N/A")
with c4:
    if fraud_count is not None:
        st.metric("Fraud Events", f"{fraud_count:,}")
    else:
        st.metric("Fraud Events", "Unknown")

st.markdown("---")

# ---- Quick timeline chart ----
if time_col and pd.api.types.is_datetime64_any_dtype(filtered_df[time_col]):
    st.subheader("Event Timeline")

    df_ts = filtered_df.set_index(time_col)
    # group by hour or day depending on span
    span_days = (df_ts.index.max() - df_ts.index.min()).days
    freq = "1H" if span_days <= 3 else "1D"
    counts = df_ts.resample(freq).size().rename("events").reset_index()

    fig = px.line(
        counts,
        x=time_col,
        y="events",
        labels={"events": "Events", time_col: "Time"},
        title=f"Events over time (freq={freq})",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Audit Log Table")

st.dataframe(
    filtered_df.head(max_rows),
    use_container_width=True,
)

# ---- Download button ----
csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered audit log as CSV",
    data=csv_bytes,
    file_name="audit_log_filtered.csv",
    mime="text/csv",
)
