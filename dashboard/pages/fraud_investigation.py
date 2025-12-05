from __future__ import annotations

from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from shared.data_loader import load_dataset


# ---------- Helper functions ----------

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


def find_id_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Return best-guess columns for key entities."""
    tx_id = None
    for c in ("transaction_id", "tx_id", "id", "row_id"):
        if c in df.columns:
            tx_id = c
            break

    cust = None
    for c in ("customer_id", "cust_id", "nameOrig", "client_id", "account_id"):
        if c in df.columns:
            cust = c
            break

    merch = None
    for c in ("merchant_id", "nameDest", "terminal_id", "acquirer_id"):
        if c in df.columns:
            merch = c
            break

    device = None
    for c in ("device_id", "device", "ip_address"):
        if c in df.columns:
            device = c
            break

    return {
        "tx_id": tx_id,
        "customer": cust,
        "merchant": merch,
        "device": device,
    }


# ---------- Page layout ----------

st.title("Fraud Investigation")
st.caption(
    "Drill down into a single transaction or entity and explore its surrounding activity, "
    "timeline and related fraud signals."
)

# Load dataset
try:
    df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – cannot perform investigations.")
    st.stop()

label_col = get_label_col(df)
amount_col = get_amount_col(df)
time_col = get_time_col(df)
ids = find_id_columns(df)

tx_id_col = ids["tx_id"]
cust_col = ids["customer"]
merch_col = ids["merchant"]
device_col = ids["device"]

# Build available search modes
search_modes: List[str] = []
if tx_id_col:
    search_modes.append("Transaction ID")
if cust_col:
    search_modes.append("Customer")
if merch_col:
    search_modes.append("Merchant")
if device_col:
    search_modes.append("Device")

if not search_modes:
    search_modes.append("Free Text")

with st.sidebar:
    st.subheader("Search")

    mode = st.selectbox("Search by", options=search_modes)

    search_value = st.text_input(
        "Search value",
        help="Enter ID / name / substring depending on mode.",
    )

    max_candidates = st.slider(
        "Max candidates to list",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

if not search_value:
    st.info("Enter a search value in the sidebar to begin an investigation.")
    st.stop()

# ---------- Candidate transactions ----------

work_df = df.copy()

if mode == "Transaction ID" and tx_id_col:
    # Prefer exact match first
    mask_exact = work_df[tx_id_col].astype(str) == search_value
    if not mask_exact.any():
        # fallback to contains
        mask = work_df[tx_id_col].astype(str).str.contains(search_value, case=False, na=False)
    else:
        mask = mask_exact

elif mode == "Customer" and cust_col:
    mask = work_df[cust_col].astype(str).str.contains(search_value, case=False, na=False)
elif mode == "Merchant" and merch_col:
    mask = work_df[merch_col].astype(str).str.contains(search_value, case=False, na=False)
elif mode == "Device" and device_col:
    mask = work_df[device_col].astype(str).str.contains(search_value, case=False, na=False)
else:
    # Free text search across all columns
    s = search_value.lower().strip()

    def _row_match(row: pd.Series) -> bool:
        return s in " ".join(map(str, row.values)).lower()

    mask = work_df.apply(_row_match, axis=1)

candidates = work_df[mask].copy()

if candidates.empty:
    st.warning("No transactions matched the search criteria.")
    st.stop()

# Sort candidates by time (newest first) if possible
if time_col and pd.api.types.is_datetime64_any_dtype(candidates[time_col]):
    candidates = candidates.sort_values(time_col, ascending=False)

candidates = candidates.head(max_candidates).reset_index(drop=True)

st.subheader("Matched Transactions")
st.write(f"Found **{len(candidates):,}** candidate transactions.")

# Build a display label for each candidate
def build_display_label(row: pd.Series) -> str:
    parts = []
    if tx_id_col:
        parts.append(f"TX={row[tx_id_col]}")
    if amount_col:
        parts.append(f"amt={row[amount_col]}")
    if time_col and pd.notna(row[time_col]):
        ts = row[time_col]
        parts.append(f"t={ts}")
    if label_col:
        parts.append(f"fraud={int(row[label_col])}")
    return " | ".join(map(str, parts)) if parts else str(row.name)


candidates["_display"] = candidates.apply(build_display_label, axis=1)

selected_label = st.selectbox(
    "Select anchor transaction",
    options=candidates["_display"].tolist(),
)

anchor_row = candidates.loc[candidates["_display"] == selected_label].iloc[0:1]
anchor_idx = anchor_row.index[0]
anchor = anchor_row.iloc[0]

st.markdown("---")

# ---------- Anchor transaction details ----------

st.subheader("Anchor Transaction Details")

c1, c2, c3, c4 = st.columns(4)
with c1:
    if tx_id_col:
        st.metric("Transaction ID", str(anchor[tx_id_col]))
with c2:
    if cust_col:
        st.metric("Customer", str(anchor[cust_col]))
with c3:
    if merch_col:
        st.metric("Merchant", str(anchor[merch_col]))
with c4:
    if amount_col:
        st.metric("Amount", f"{float(anchor[amount_col]):,.2f}")

c5, c6, c7, c8 = st.columns(4)
with c5:
    if device_col:
        st.metric("Device", str(anchor[device_col]))
with c6:
    if time_col and pd.notna(anchor[time_col]):
        st.metric("Time", str(anchor[time_col]))
with c7:
    if label_col:
        st.metric("Fraud Label", "FRAUD" if int(anchor[label_col]) == 1 else "LEGIT")
with c8:
    st.metric("Row Index", str(anchor_idx))

with st.expander("Show full anchor row"):
    st.json(anchor.to_dict())

# ---------- Related transactions / “ego network” ----------

st.markdown("---")
st.subheader("Related Transactions (Same Entity)")

# Build masks for related activity
rel_mask = pd.Series(False, index=df.index)
tags = []

if cust_col and not pd.isna(anchor.get(cust_col, None)):
    m = df[cust_col].astype(str) == str(anchor[cust_col])
    rel_mask |= m
    tags.append("customer")

if merch_col and not pd.isna(anchor.get(merch_col, None)):
    m = df[merch_col].astype(str) == str(anchor[merch_col])
    rel_mask |= m
    tags.append("merchant")

if device_col and not pd.isna(anchor.get(device_col, None)):
    m = df[device_col].astype(str) == str(anchor[device_col])
    rel_mask |= m
    tags.append("device")

# Always ensure anchor itself is included
if tx_id_col:
    rel_mask |= df[tx_id_col].astype(str) == str(anchor[tx_id_col])
else:
    rel_mask.iloc[anchor_idx] = True

related_df = df[rel_mask].copy()

if related_df.empty:
    st.info("No related transactions found for this anchor.")
    st.stop()

# Sort by time if possible
if time_col and pd.api.types.is_datetime64_any_dtype(related_df[time_col]):
    related_df = related_df.sort_values(time_col, ascending=True)

st.write(
    f"Related transactions based on: **{', '.join(tags) if tags else 'anchor row only'}** "
    f"→ rows: **{len(related_df):,}**"
)

if label_col:
    fraud_cnt = int(related_df[label_col].sum())
    st.write(f"Fraud-labelled transactions in this neighborhood: **{fraud_cnt:,}**")

# Timeline chart
if time_col and amount_col and pd.api.types.is_datetime64_any_dtype(related_df[time_col]):
    st.subheader("Timeline of Related Activity")

    plot_df = related_df.copy()
    if label_col:
        plot_df["fraud_flag"] = plot_df[label_col].map({0: "Non-Fraud", 1: "Fraud"})
    else:
        plot_df["fraud_flag"] = "Unknown"

    fig_timeline = px.scatter(
        plot_df,
        x=time_col,
        y=amount_col,
        color="fraud_flag",
        hover_data=[tx_id_col] if tx_id_col else None,
        title="Amount over Time for Related Transactions",
        labels={time_col: "Time", amount_col: "Amount"},
    )
    # Highlight anchor as a different marker size
    if tx_id_col:
        anchor_mask = plot_df[tx_id_col].astype(str) == str(anchor[tx_id_col])
        fig_timeline.add_scatter(
            x=plot_df.loc[anchor_mask, time_col],
            y=plot_df.loc[anchor_mask, amount_col],
            mode="markers",
            marker=dict(size=14, symbol="star"),
            name="Anchor TX",
        )

    st.plotly_chart(fig_timeline, use_container_width=True)

# Summary by dimension
st.subheader("Summary in Neighborhood")

cols = []
if cust_col:
    cols.append(("Customer", cust_col))
if merch_col:
    cols.append(("Merchant", merch_col))
if device_col:
    cols.append(("Device", device_col))

if cols:
    seg_pretty = st.selectbox(
        "Summarise by",
        options=[p for p, _ in cols],
    )
    seg_col = dict(cols)[seg_pretty]
    tmp = related_df.copy()
    tmp[seg_col] = tmp[seg_col].astype(str)

    agg = tmp.groupby(seg_col).size().to_frame("tx_count")
    if label_col:
        agg["fraud_count"] = tmp.groupby(seg_col)[label_col].sum()
        agg["fraud_rate"] = agg["fraud_count"] / agg["tx_count"].replace(0, np.nan)

    if amount_col:
        agg["total_amount"] = tmp.groupby(seg_col)[amount_col].sum()

    agg = agg.sort_values("tx_count", ascending=False).head(20)
    st.dataframe(agg.reset_index(), use_container_width=True)

# ---------- Raw table of related transactions ----------

st.subheader("Related Transactions Table")

st.dataframe(
    related_df.head(500),
    use_container_width=True,
)

csv_bytes = related_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download related transactions as CSV",
    data=csv_bytes,
    file_name="related_transactions_anchor.csv",
    mime="text/csv",
)
