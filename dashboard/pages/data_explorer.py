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


st.title("Data Explorer")
st.caption("Explore the raw transaction dataset with flexible filters and quick summaries.")

# ---- Load data ----
try:
    df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – nothing to explore.")
    st.stop()

label_col = get_label_col(df)

st.write(f"Total rows: **{len(df):,}**, columns: **{len(df.columns)}**")

# ---- Sidebar: filters ----
with st.sidebar:
    st.subheader("Filters")

    # Column selector for quick filtering
    filter_col = st.selectbox(
        "Filter by column",
        options=["(none)"] + df.columns.tolist(),
        index=0,
    )

    filtered_df = df

    if filter_col != "(none)":
        col_data = df[filter_col]
        if pd.api.types.is_numeric_dtype(col_data):
            min_val = float(col_data.min())
            max_val = float(col_data.max())
            vmin, vmax = st.slider(
                f"Range for {filter_col}",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
            )
            filtered_df = filtered_df[
                (filtered_df[filter_col] >= vmin)
                & (filtered_df[filter_col] <= vmax)
            ]
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            try:
                min_dt = col_data.min().date()
                max_dt = col_data.max().date()
                start, end = st.date_input(
                    f"Date range for {filter_col}",
                    value=(min_dt, max_dt),
                )
                filtered_df = filtered_df[
                    (filtered_df[filter_col].dt.date >= start)
                    & (filtered_df[filter_col].dt.date <= end)
                ]
            except Exception:
                st.info(f"Could not parse {filter_col} as dates correctly; no date filter applied.")
        else:
            # Categorical / string filter
            unique_vals = (
                filtered_df[filter_col]
                .astype(str)
                .dropna()
                .value_counts()
                .head(50)
                .index.tolist()
            )
            selected_vals = st.multiselect(
                f"Values for {filter_col}",
                options=unique_vals,
                default=unique_vals,
            )
            filtered_df = filtered_df[filtered_df[filter_col].astype(str).isin(selected_vals)]

    st.markdown("---")
    max_rows = st.slider(
        "Rows to display",
        min_value=100,
        max_value=min(5000, len(filtered_df)),
        value=min(1000, len(filtered_df)),
        step=100,
    )

st.write(f"Filtered rows: **{len(filtered_df):,}**")

# ---- Column summary & distribution ----
st.subheader("Column Summary")

summary_col = st.selectbox(
    "Column to summarise",
    options=filtered_df.columns.tolist(),
)

col_series = filtered_df[summary_col]

if pd.api.types.is_numeric_dtype(col_series):
    st.write("Numeric summary:")
    desc = col_series.describe()
    st.table(desc.to_frame("value"))

    fig = px.histogram(
        filtered_df,
        x=summary_col,
        nbins=40,
        title=f"Distribution of {summary_col}",
    )
    st.plotly_chart(fig, use_container_width=True)

elif pd.api.types.is_datetime64_any_dtype(col_series):
    st.write("Datetime summary:")
    st.write(
        {
            "min": col_series.min(),
            "max": col_series.max(),
            "non-null count": col_series.notna().sum(),
        }
    )

    # Histogram by day
    try:
        dt_df = filtered_df.copy()
        dt_df["_date_tmp"] = col_series.dt.date
        daily_counts = (
            dt_df["_date_tmp"].value_counts().sort_index().rename_axis("date").reset_index(name="count")
        )
        fig = px.bar(
            daily_counts,
            x="date",
            y="count",
            title=f"Counts by day for {summary_col}",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Could not build date-based histogram for this column.")
else:
    st.write("Categorical / text summary:")
    value_counts = (
        col_series.astype(str)
        .value_counts()
        .head(30)
        .rename_axis("value")
        .reset_index(name="count")
    )
    st.dataframe(value_counts, use_container_width=True)

    fig = px.bar(
        value_counts,
        x="value",
        y="count",
        title=f"Top {len(value_counts)} values for {summary_col}",
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Raw table ----
st.subheader("Data Table")

st.dataframe(
    filtered_df.head(max_rows),
    use_container_width=True,
)

# ---- Download filtered data ----
csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered data as CSV",
    data=csv_bytes,
    file_name="filtered_transactions.csv",
    mime="text/csv",
)
