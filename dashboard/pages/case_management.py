from __future__ import annotations

from typing import Optional, List, Tuple

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


def discover_case_dimensions(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Find reasonable columns to build 'cases' on.
    Returns list of (pretty_label, column_name).
    """
    candidates: List[Tuple[str, List[str]]] = [
        ("Customer", ["customer_id", "cust_id", "nameOrig", "client_id"]),
        ("Merchant", ["merchant_id", "nameDest", "terminal_id"]),
        ("Device", ["device_id", "device", "ip_address"]),
        ("Card / Account", ["card_id", "card_number", "account_id", "pan"]),
        ("Country / Region", ["country", "country_code", "merchant_country", "location", "region"]),
    ]

    found: List[Tuple[str, str]] = []
    for pretty, cols in candidates:
        for c in cols:
            if c in df.columns:
                found.append((f"{pretty} ({c})", c))
                break  # only keep the first matching column for each group

    # As a last resort, allow grouping by any categorical column
    if not found:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        for c in cat_cols[:5]:
            found.append((f"Generic ({c})", c))

    return found


st.title("Case Management")
st.caption(
    "Group suspicious activity by customer, merchant, device or region to create analyst-friendly 'cases'."
)

# ---- Load dataset ----
try:
    df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – cannot build cases.")
    st.stop()

label_col = get_label_col(df)
amount_col = get_amount_col(df)
time_col = get_time_col(df)

# ---- Discover possible case keys ----
dimensions = discover_case_dimensions(df)
if not dimensions:
    st.warning(
        "Could not find suitable columns for case grouping "
        "(customer, merchant, device, etc.). "
        "Add such columns to the dataset to enable case management."
    )
    st.stop()

pretty_to_col = {pretty: col for pretty, col in dimensions}

with st.sidebar:
    st.subheader("Case Settings")

    dim_pretty = st.selectbox(
        "Group cases by",
        options=list(pretty_to_col.keys()),
    )
    case_col = pretty_to_col[dim_pretty]

    min_tx_per_case = st.slider(
        "Minimum transactions per case",
        min_value=2,
        max_value=100,
        value=5,
        step=1,
    )

    only_suspicious = False
    if label_col:
        only_suspicious = st.checkbox(
            "Show only cases with ≥1 fraud transaction",
            value=True,
        )

    max_cases_display = st.slider(
        "Maximum cases to display",
        min_value=20,
        max_value=500,
        value=100,
        step=10,
    )

# ---- Build case summary ----
work_df = df.copy()
work_df[case_col] = work_df[case_col].astype(str).fillna("(unknown)")

grouped = work_df.groupby(case_col)

summary_rows = []
for case_id, grp in grouped:
    tx_count = len(grp)
    if tx_count < min_tx_per_case:
        continue

    fraud_count = int(grp[label_col].sum()) if label_col else None
    fraud_rate = (
        fraud_count / tx_count if (fraud_count is not None and tx_count > 0) else None
    )

    total_amount = float(grp[amount_col].sum()) if amount_col else None
    last_time = (
        grp[time_col].max() if (time_col and pd.api.types.is_datetime64_any_dtype(grp[time_col])) else None
    )

    summary_rows.append(
        {
            "case_id": case_id,
            "tx_count": tx_count,
            "fraud_count": fraud_count if fraud_count is not None else 0,
            "fraud_rate": fraud_rate if fraud_rate is not None else np.nan,
            "total_amount": total_amount if total_amount is not None else np.nan,
            "last_event_time": last_time,
        }
    )

summary_df = pd.DataFrame(summary_rows)

if summary_df.empty:
    st.warning(
        "No cases meet the minimum transaction threshold. "
        "Try lowering 'Minimum transactions per case' in the sidebar."
    )
    st.stop()

if label_col and only_suspicious:
    summary_df = summary_df[summary_df["fraud_count"] > 0]

if summary_df.empty:
    st.warning(
        "No cases remain after applying the 'only suspicious' filter. "
        "Try disabling it or lowering thresholds."
    )
    st.stop()

# Sorting: suspicious first (more fraud_count), then by tx_count or total_amount
summary_df = summary_df.sort_values(
    ["fraud_count", "tx_count"],
    ascending=[False, False],
).head(max_cases_display)

num_cases = len(summary_df)
num_suspicious_cases = int((summary_df["fraud_count"] > 0).sum()) if label_col else None
max_case_size = int(summary_df["tx_count"].max())

# ---- KPIs ----
st.subheader("Case Overview")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Grouping Column", case_col)
with c2:
    st.metric("Cases in View", f"{num_cases:,}")
with c3:
    if num_suspicious_cases is not None:
        st.metric("Suspicious Cases (≥1 fraud)", f"{num_suspicious_cases:,}")
    else:
        st.metric("Suspicious Cases", "Unknown (no labels)")
with c4:
    st.metric("Max Case Size (tx)", f"{max_case_size:,}")

st.markdown("---")

# ---- Cases table + chart ----
st.subheader("Cases Summary")

# Prepare display frame
display_df = summary_df.copy()
if not label_col:
    # Hide fraud-rate column if labels don't exist
    if "fraud_rate" in display_df.columns:
        display_df = display_df.drop(columns=["fraud_rate"])

st.dataframe(display_df, use_container_width=True)

# Bar chart: top cases by fraud_count or tx_count
metric_for_chart = "fraud_count" if label_col else "tx_count"
fig_cases = px.bar(
    summary_df.head(20),
    x="case_id",
    y=metric_for_chart,
    title=f"Top Cases by {metric_for_chart.replace('_', ' ').title()}",
)
st.plotly_chart(fig_cases, use_container_width=True)

# ---- Case detail drill-down ----
st.subheader("Case Detail")

case_choices = summary_df["case_id"].astype(str).tolist()
selected_case_id_str = st.selectbox(
    "Select a case to inspect",
    options=case_choices,
)

# Use string comparison to be safe
detail_df = work_df[work_df[case_col].astype(str) == selected_case_id_str].copy()

st.write(
    f"Transactions for case **{selected_case_id_str}** "
    f"(rows: **{len(detail_df):,}**)."
)

if label_col:
    fraud_in_case = int(detail_df[label_col].sum())
    st.write(f"Fraud-labelled transactions in this case: **{fraud_in_case:,}**")

# Sort by time if available
if time_col and pd.api.types.is_datetime64_any_dtype(detail_df[time_col]):
    detail_df = detail_df.sort_values(time_col, ascending=False)

# Small chart: amount over time for this case (if possible)
if amount_col and time_col and pd.api.types.is_datetime64_any_dtype(detail_df[time_col]):
    fig_case_ts = px.line(
        detail_df,
        x=time_col,
        y=amount_col,
        title=f"Amount over Time for Case {selected_case_id_str}",
        labels={time_col: "Time", amount_col: "Amount"},
    )
    st.plotly_chart(fig_case_ts, use_container_width=True)

st.dataframe(
    detail_df.head(500),
    use_container_width=True,
)

# ---- Download buttons ----
st.subheader("Export")

csv_cases = summary_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download cases summary as CSV",
    data=csv_cases,
    file_name="cases_summary.csv",
    mime="text/csv",
)

csv_detail = detail_df.to_csv(index=False).encode("utf-8")
st.download_button(
    f"Download transactions for case {selected_case_id_str}",
    data=csv_detail,
    file_name=f"case_{selected_case_id_str}_transactions.csv",
    mime="text/csv",
)
