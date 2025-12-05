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


def get_country_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("country", "country_code", "merchant_country", "location", "region"):
        if col in df.columns:
            return col
    return None


def get_type_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("type", "category", "txn_type", "channel"):
        if col in df.columns:
            return col
    return None


def render_page(client=None, health=None):
    st.title("Dashboard Home")
    st.caption("High-level overview of online fraud activity and key trends.")

    try:
        df = load_dataset()
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        return

    if df is None or df.empty:
        st.warning("Dataset is empty – nothing to show on the dashboard.")
        return

    label_col = get_label_col(df)
    amount_col = get_amount_col(df)
    time_col = get_time_col(df)
    country_col = get_country_col(df)
    type_col = get_type_col(df)

    total_tx = len(df)
    fraud_count = int(df[label_col].sum()) if label_col else None
    fraud_rate = (fraud_count / total_tx * 100.0) if (label_col and total_tx > 0) else None
    total_amount = float(df[amount_col].sum()) if amount_col else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total_tx:,}")
    c2.metric("Fraud Count", f"{fraud_count:,}" if fraud_count is not None else "Unknown")
    c3.metric("Fraud Rate", f"{fraud_rate:.2f}%" if fraud_rate is not None else "Unknown")
    c4.metric("Total Volume", f"{total_amount:,.2f}" if total_amount is not None else "N/A")

    st.markdown("---")

    tab_overview, tab_trends, tab_segments = st.tabs(["Overview", "Trends", "Segments"])

    with tab_overview:
        st.subheader("Snapshot")
        if label_col:
            class_counts = (
                df[label_col]
                .map({0: "Non-Fraud", 1: "Fraud"})
                .value_counts()
                .rename_axis("class")
                .reset_index(name="count")
            )
            fig_cls = px.pie(class_counts, values="count", names="class", title="Fraud vs Non-Fraud Share", hole=0.4)
            st.plotly_chart(fig_cls, use_container_width=True)
        else:
            st.info("No fraud label column found (e.g., 'is_fraud'); cannot show fraud split pie chart.")

        if type_col:
            st.subheader("Top Transaction Types by Volume")
            type_counts = df[type_col].astype(str).value_counts().head(10).rename_axis(type_col).reset_index(name="count")
            fig_types = px.bar(
                type_counts,
                x=type_col,
                y="count",
                title=f"Top {len(type_counts)} {type_col} by Transaction Count",
                text="count",
            )
            fig_types.update_traces(textposition="outside")
            st.plotly_chart(fig_types, use_container_width=True)
        else:
            st.info("No transaction type/category column found; skipping type breakdown.")

    with tab_trends:
        st.subheader("Temporal Trends")
        if time_col and pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df_ts = df.copy().set_index(time_col)
            daily = df_ts.resample("1D").size().to_frame("tx_count")
            if label_col:
                fraud_daily = df_ts[df_ts[label_col] == 1].resample("1D").size().to_frame("fraud_count")
                daily = daily.join(fraud_daily, how="left").fillna(0)
                daily["fraud_rate"] = daily["fraud_count"] / daily["tx_count"].replace(0, np.nan)
            daily = daily.reset_index().rename(columns={time_col: "date"})
            fig_tx = px.line(daily, x="date", y="tx_count", title="Daily Transaction Volume", labels={"tx_count": "Transactions", "date": "Date"})
            st.plotly_chart(fig_tx, use_container_width=True)
            if label_col and "fraud_rate" in daily.columns:
                fig_fr = px.line(daily, x="date", y="fraud_rate", title="Daily Fraud Rate", labels={"fraud_rate": "Fraud Rate", "date": "Date"})
                st.plotly_chart(fig_fr, use_container_width=True)
        else:
            st.info("No usable datetime column found (e.g., 'timestamp', 'tx_datetime') – cannot build temporal trends chart.")

    with tab_segments:
        st.subheader("Segment View")
        options = []
        if country_col:
            options.append(("Country/Region", country_col))
        if type_col:
            options.append(("Transaction Type", type_col))
        if not options:
            st.info("No obvious segment columns found (e.g., country, type).")
        else:
            label_map = {pretty: col for pretty, col in options}
            segment_label = st.selectbox("Segment by", options=[pretty for pretty, _ in options])
            seg_col = label_map[segment_label]
            seg_df = df.copy()
            seg_df[seg_col] = seg_df[seg_col].astype(str)
            agg = seg_df.groupby(seg_col).size().to_frame("tx_count")
            if label_col:
                agg["fraud_rate"] = seg_df.groupby(seg_col)[label_col].mean()
            if amount_col:
                agg["total_amount"] = seg_df.groupby(seg_col)[amount_col].sum()
            agg = agg.sort_values("tx_count", ascending=False).head(15)
            agg_reset = agg.reset_index().rename(columns={seg_col: "segment"})
            st.dataframe(agg_reset, use_container_width=True)
            metric_choice = st.selectbox("Bar chart metric", [m for m in ["tx_count", "fraud_rate", "total_amount"] if m in agg.columns])
            fig_seg = px.bar(agg_reset, x="segment", y=metric_choice, title=f"{metric_choice.replace('_', ' ').title()} by {segment_label}")
            st.plotly_chart(fig_seg, use_container_width=True)


if __name__ == "__main__":
    render_page()
