from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from shared.data_loader import load_dataset


# ---- Helpers ----
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


def build_daily_view(
    df: pd.DataFrame,
    time_col: Optional[str],
    label_col: Optional[str],
) -> Optional[pd.DataFrame]:
    if not time_col or not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        return None

    ts_df = df.copy().set_index(time_col).sort_index()

    # Daily transaction counts
    daily = ts_df.resample("1D").size().to_frame("tx_count")

    if label_col:
        fraud_daily = ts_df[ts_df[label_col] == 1].resample("1D").size().to_frame(
            "fraud_count"
        )
        daily = daily.join(fraud_daily, how="left").fillna(0)
        daily["fraud_rate"] = daily["fraud_count"] / daily["tx_count"].replace(0, np.nan)

    daily = daily.reset_index().rename(columns={time_col: "date"})
    return daily


def build_segment_summary(
    df: pd.DataFrame,
    seg_col: str,
    label_col: Optional[str],
    amount_col: Optional[str],
    min_tx: int = 20,
    top_n: int = 10,
) -> pd.DataFrame:
    work = df.copy()
    work[seg_col] = work[seg_col].astype(str)

    agg = work.groupby(seg_col).size().to_frame("tx_count")

    if label_col:
        agg["fraud_count"] = work.groupby(seg_col)[label_col].sum()
        agg["fraud_rate"] = agg["fraud_count"] / agg["tx_count"].replace(0, np.nan)

    if amount_col:
        agg["total_amount"] = work.groupby(seg_col)[amount_col].sum()

    agg = agg[agg["tx_count"] >= min_tx]
    if "fraud_rate" in agg.columns:
        agg = agg.sort_values("fraud_rate", ascending=False)
    else:
        agg = agg.sort_values("tx_count", ascending=False)

    return agg.head(top_n).reset_index().rename(columns={seg_col: "segment"})


# ---- Page body ----
st.title("Executive Report")
st.caption(
    "Single-page summary for leadership: key volumes, fraud rates, trends and high-risk segments."
)

# Load data
try:
    df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – cannot build executive report.")
    st.stop()

label_col = get_label_col(df)
amount_col = get_amount_col(df)
time_col = get_time_col(df)
country_col = get_country_col(df)
type_col = get_type_col(df)

total_tx = len(df)
fraud_count = int(df[label_col].sum()) if label_col else None
fraud_rate = (
    fraud_count / total_tx * 100.0 if (label_col and total_tx > 0) else None
)
total_amount = float(df[amount_col].sum()) if amount_col else None

if time_col and pd.api.types.is_datetime64_any_dtype(df[time_col]):
    date_min = df[time_col].min()
    date_max = df[time_col].max()
else:
    date_min = None
    date_max = None

# ---- Sidebar controls ----
with st.sidebar:
    st.subheader("Executive Report Settings")

    seg_options: List[Tuple[str, str]] = []
    if country_col:
        seg_options.append(("Country / Region", country_col))
    if type_col:
        seg_options.append(("Transaction Type", type_col))

    if seg_options:
        seg_label = st.selectbox(
            "Primary risk segment view",
            options=[pretty for pretty, _ in seg_options],
        )
        seg_col = dict(seg_options)[seg_label]
    else:
        seg_label = None
        seg_col = None

    min_tx_seg = st.slider(
        "Min transactions per segment",
        min_value=10,
        max_value=500,
        value=50,
        step=10,
    )

# ---- KPI Row ----
st.subheader("Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Transactions", f"{total_tx:,}")
with c2:
    st.metric(
        "Fraud Count",
        f"{fraud_count:,}" if fraud_count is not None else "Unknown",
    )
with c3:
    st.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%" if fraud_rate is not None else "Unknown",
    )
with c4:
    if date_min is not None and date_max is not None:
        st.metric(
            "Data Window",
            f"{date_min.date()} → {date_max.date()}",
        )
    else:
        st.metric("Data Window", "Not time-indexed")

c5, c6 = st.columns(2)
with c5:
    st.metric(
        "Total Volume",
        f"{total_amount:,.2f}" if total_amount is not None else "N/A",
    )
with c6:
    if amount_col and label_col:
        avg_fraud_amt = float(
            df[df[label_col] == 1][amount_col].mean()
        ) if (df[label_col] == 1).any() else 0.0
        st.metric("Avg Fraud Transaction Amount", f"{avg_fraud_amt:,.2f}")
    else:
        st.metric("Avg Fraud Transaction Amount", "N/A")

st.markdown("---")

# ---- Narrative summary ----
st.subheader("Narrative Summary")

summary_lines: List[str] = []

summary_lines.append(
    f"- The platform processed **{total_tx:,}** transactions"
    + (f" between **{date_min.date()}** and **{date_max.date()}**" if date_min and date_max else "")
    + "."
)

if fraud_count is not None and fraud_rate is not None:
    summary_lines.append(
        f"- **{fraud_count:,}** transactions were labelled as fraud, corresponding to a fraud rate of "
        f"**{fraud_rate:.2f}%**."
    )

if total_amount is not None:
    summary_lines.append(
        f"- The total processed volume in the period is approximately **{total_amount:,.2f}** units."
    )

if seg_col and label_col:
    seg_df_tmp = build_segment_summary(
        df,
        seg_col=seg_col,
        label_col=label_col,
        amount_col=amount_col,
        min_tx=min_tx_seg,
        top_n=3,
    )
    if not seg_df_tmp.empty and "fraud_rate" in seg_df_tmp.columns:
        top_seg = seg_df_tmp.iloc[0]
        summary_lines.append(
            f"- The highest-risk segment by **{seg_label}** is **{top_seg['segment']}**, "
            f"with a fraud rate of **{top_seg['fraud_rate'] * 100:.2f}%** over "
            f"**{int(top_seg['tx_count']):,}** transactions."
        )

if not summary_lines:
    summary_lines.append(
        "- Dataset loaded successfully. Add labels / segment columns to enable richer executive summaries."
    )

st.markdown("\n".join(summary_lines))

st.markdown("---")

# ---- Trend view ----
st.subheader("Volume & Fraud Trend")

daily = build_daily_view(df, time_col, label_col)
if daily is None or daily.empty:
    st.info(
        "No usable datetime column – cannot show daily volume and fraud trend. "
        "Add a timestamp-like column to enable this chart."
    )
else:
    fig_tx = px.line(
        daily,
        x="date",
        y="tx_count",
        title="Daily Transaction Volume",
        labels={"tx_count": "Transactions", "date": "Date"},
    )
    st.plotly_chart(fig_tx, use_container_width=True)

    if "fraud_rate" in daily.columns:
        fig_fr = px.line(
            daily,
            x="date",
            y="fraud_rate",
            title="Daily Fraud Rate",
            labels={"fraud_rate": "Fraud Rate", "date": "Date"},
        )
        st.plotly_chart(fig_fr, use_container_width=True)

# ---- High-risk segments ----
st.subheader("High-Risk Segments")

if seg_col is None:
    st.info(
        "No obvious segment columns found (country / type). "
        "Add such columns to enable high-risk segment analysis."
    )
else:
    seg_df = build_segment_summary(
        df,
        seg_col=seg_col,
        label_col=label_col,
        amount_col=amount_col,
        min_tx=min_tx_seg,
        top_n=10,
    )

    if seg_df.empty:
        st.info(
            f"No segments of **{seg_label}** exceed the minimum volume threshold "
            f"({min_tx_seg} transactions)."
        )
    else:
        st.dataframe(seg_df, use_container_width=True)

        metric_choice = "fraud_rate" if label_col and "fraud_rate" in seg_df.columns else "tx_count"

        fig_seg = px.bar(
            seg_df,
            x="segment",
            y=metric_choice,
            title=f"Top Segments by {metric_choice.replace('_', ' ').title()} "
            f"({seg_label})",
        )
        st.plotly_chart(fig_seg, use_container_width=True)

# ---- Optional: quick export ----
st.subheader("Export Snapshot Data")

export_cols = [c for c in df.columns if c in [time_col, label_col, amount_col, country_col, type_col] and c]
snapshot_df = df[export_cols].copy() if export_cols else df.copy()

csv_bytes = snapshot_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download snapshot data as CSV",
    data=csv_bytes,
    file_name="executive_report_snapshot.csv",
    mime="text/csv",
)
