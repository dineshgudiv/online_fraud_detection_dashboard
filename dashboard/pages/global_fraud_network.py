from __future__ import annotations

from typing import Optional

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


def get_country_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("country", "country_code", "merchant_country", "location", "region"):
        if col in df.columns:
            return col
    return None


st.title("Global Fraud Network")
st.caption("Geographical view of transaction and fraud activity by region/country.")

# ---- Load data safely ----
try:
    df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – cannot build global view.")
    st.stop()

label_col = get_label_col(df)
amount_col = get_amount_col(df)
country_col = get_country_col(df)

if country_col is None:
    st.warning(
        "No country/location column found in the dataset "
        "(expected something like 'country', 'country_code', 'merchant_country', or 'location').\n\n"
        "Showing a generic distribution instead."
    )

# ---- Top-level KPIs ----
total_tx = len(df)
total_amount = float(df[amount_col].sum()) if amount_col else None

if label_col:
    fraud_tx = df[df[label_col] == 1]
    num_fraud = int(fraud_tx.shape[0])
    fraud_rate = 100 * num_fraud / max(1, total_tx)
    fraud_amount = float(fraud_tx[amount_col].sum()) if amount_col else None
else:
    fraud_tx = pd.DataFrame()
    num_fraud = 0
    fraud_rate = 0.0
    fraud_amount = None

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Transactions", f"{total_tx:,}")
with c2:
    st.metric("Total Amount", f"{total_amount:,.2f}" if total_amount is not None else "N/A")
with c3:
    st.metric("Fraud Count", f"{num_fraud:,}" if label_col else "Unknown")
with c4:
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%" if label_col else "Unknown")

st.markdown("---")

# ---- If we have a country/location column, build regional aggregation ----
if country_col is not None:
    st.subheader("Regional Aggregation")

    # Base aggregation
    agg = df.groupby(country_col).size().to_frame("tx_count")

    if label_col:
        agg["fraud_rate"] = df.groupby(country_col)[label_col].mean()
    if amount_col:
        agg["total_amount"] = df.groupby(country_col)[amount_col].sum()

    agg = agg.sort_values("tx_count", ascending=False)

    min_tx = st.slider(
        "Minimum transactions per region to display",
        min_value=1,
        max_value=int(agg["tx_count"].max()),
        value=min(100, int(agg["tx_count"].max())),
    )
    agg_filtered = agg[agg["tx_count"] >= min_tx].copy()

    st.write(f"Regions shown: **{len(agg_filtered):,}**")

    st.dataframe(
        agg_filtered.reset_index().rename(columns={country_col: "region"}),
        use_container_width=True,
    )

    # ---- Choropleth map (fraud rate or tx_count) ----
    st.subheader("Global Heatmap")

    metric_choice = st.selectbox(
        "Color regions by",
        [m for m in ["fraud_rate", "tx_count", "total_amount"] if m in agg_filtered.columns],
    )

    if metric_choice:
        map_df = agg_filtered.reset_index()

        try:
            fig_map = px.choropleth(
                map_df,
                locations=country_col,
                locationmode="country names",
                color=metric_choice,
                hover_name=country_col,
                color_continuous_scale="Reds",
                title=f"{metric_choice.replace('_', ' ').title()} by Region",
            )
            fig_map.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not render world map ({exc}). Falling back to bar chart.")

            fig_fallback = px.bar(
                map_df.sort_values(metric_choice, ascending=False).head(20),
                x=country_col,
                y=metric_choice,
                title=f"Top 20 Regions by {metric_choice.replace('_', ' ').title()}",
            )
            st.plotly_chart(fig_fallback, use_container_width=True)

    # ---- Top risky regions (table & bar plot) ----
    if label_col and "fraud_rate" in agg_filtered.columns:
        st.subheader("Top Risky Regions (by Fraud Rate)")

        risky = agg_filtered[agg_filtered["tx_count"] >= min_tx].copy()
        risky = risky.sort_values("fraud_rate", ascending=False).head(10)

        st.dataframe(
            risky.reset_index().rename(columns={country_col: "region"}),
            use_container_width=True,
        )

        fig_risk = px.bar(
            risky.reset_index(),
            x=country_col,
            y="fraud_rate",
            title="Top 10 Regions by Fraud Rate",
        )
        st.plotly_chart(fig_risk, use_container_width=True)

else:
    # ---- Fallback: no country column, show a generic distribution ----
    st.subheader("Fallback View (No Country Column)")

    fallback_col = None
    for c in ("type", "category", "txn_type", "channel"):
        if c in df.columns:
            fallback_col = c
            break

    if fallback_col is None:
        st.info(
            "No obvious geography or category columns found. "
            "Showing a simple count of transactions."
        )
        st.bar_chart(pd.Series({"transactions": total_tx}))
    else:
        st.info(
            f"Using `{fallback_col}` as a proxy for 'region' since no "
            "country/location column is present."
        )
        counts = df[fallback_col].value_counts().head(20)
        st.bar_chart(counts)
