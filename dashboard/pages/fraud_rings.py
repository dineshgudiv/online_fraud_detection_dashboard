from __future__ import annotations

from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from shared.data_loader import load_dataset


# ---------- Helpers ----------

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


def discover_edge_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to discover origin -> destination columns (customer->merchant).
    First try known pairs, then generic fallbacks.
    """
    # 1) Typical PaySim style: nameOrig -> nameDest
    if "nameOrig" in df.columns and "nameDest" in df.columns:
        return "nameOrig", "nameDest"

    # 2) Generic customer/merchant combos
    cust_candidates = ["customer_id", "cust_id", "client_id", "account_id", "nameOrig"]
    merch_candidates = ["merchant_id", "nameDest", "terminal_id", "acquirer_id"]

    src_col = None
    dst_col = None

    for c in cust_candidates:
        if c in df.columns:
            src_col = c
            break

    for c in merch_candidates:
        if c in df.columns:
            dst_col = c
            break

    # 3) Last resort: use first two string columns as endpoints
    if src_col is None or dst_col is None:
        obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if len(obj_cols) >= 2:
            src_col = src_col or obj_cols[0]
            dst_col = dst_col or obj_cols[1]

    return src_col, dst_col


def build_ring_summary(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    label_col: Optional[str],
    amount_col: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build node + edge summaries for potential fraud rings.
    Returns (src_nodes, dst_nodes, edges).
    """
    work = df.copy()
    work[src_col] = work[src_col].astype(str)
    work[dst_col] = work[dst_col].astype(str)

    # Edge aggregation
    edge_group_cols = [src_col, dst_col]
    edges = work.groupby(edge_group_cols).size().to_frame("tx_count")

    if label_col:
        edges["fraud_count"] = work.groupby(edge_group_cols)[label_col].sum()
        edges["fraud_rate"] = edges["fraud_count"] / edges["tx_count"].replace(0, np.nan)

    if amount_col:
        edges["total_amount"] = work.groupby(edge_group_cols)[amount_col].sum()

    edges = edges.reset_index()

    # Node metrics for src (customers) and dst (merchants)
    src_nodes = work.groupby(src_col).size().to_frame("tx_count")
    dst_nodes = work.groupby(dst_col).size().to_frame("tx_count")

    if label_col:
        src_nodes["fraud_count"] = work.groupby(src_col)[label_col].sum()
        dst_nodes["fraud_count"] = work.groupby(dst_col)[label_col].sum()
        src_nodes["fraud_rate"] = src_nodes["fraud_count"] / src_nodes["tx_count"].replace(0, np.nan)
        dst_nodes["fraud_rate"] = dst_nodes["fraud_count"] / dst_nodes["tx_count"].replace(0, np.nan)

    if amount_col:
        src_nodes["total_amount"] = work.groupby(src_col)[amount_col].sum()
        dst_nodes["total_amount"] = work.groupby(dst_col)[amount_col].sum()

    src_nodes = src_nodes.reset_index().rename(columns={src_col: "node"})
    dst_nodes = dst_nodes.reset_index().rename(columns={dst_col: "node"})

    src_nodes["type"] = "source"
    dst_nodes["type"] = "target"

    return src_nodes, dst_nodes, edges


def make_star_graph_figure(
    hub: str,
    neighbors: List[Dict[str, object]],
    hub_type: str,
) -> go.Figure:
    """
    Simple star layout: hub at center, neighbors arranged vertically.
    neighbors: list of dicts with 'node' and optional 'weight'.
    """
    # Hub position
    hub_x, hub_y = 0.0, 0.0

    # Neighbor positions
    n = len(neighbors)
    ys = np.linspace(-1.0, 1.0, num=max(n, 2))
    xs = np.full_like(ys, 2.0, dtype=float)

    node_x = [hub_x]
    node_y = [hub_y]
    node_text = [f"{hub} ({hub_type})"]
    node_size = [18]

    edge_x = []
    edge_y = []

    for i, neigh in enumerate(neighbors):
        node_x.append(xs[i])
        node_y.append(ys[i])
        label = str(neigh["node"])
        node_text.append(label)
        node_size.append(12)

        # Edge from hub to neighbor
        edge_x.extend([hub_x, xs[i], None])
        edge_y.extend([hub_y, ys[i], None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=2),
        hoverinfo="none",
        showlegend=False,
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="middle right",
        marker=dict(size=node_size),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Ring View (star layout)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ---------- Page body ----------

st.title("Fraud Rings")
st.caption(
    "Identify clusters of suspicious customers and merchants (rings) based on repeated fraud patterns."
)

# Load dataset
try:
    df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – cannot analyse fraud rings.")
    st.stop()

label_col = get_label_col(df)
amount_col = get_amount_col(df)
time_col = get_time_col(df)

src_col, dst_col = discover_edge_columns(df)

if src_col is None or dst_col is None:
    st.warning(
        "Could not reliably detect origin/destination columns (customer→merchant). "
        "Expected something like (nameOrig, nameDest) or (customer_id, merchant_id)."
    )
    st.stop()

with st.sidebar:
    st.subheader("Ring Detection Settings")

    mode = st.selectbox(
        "Focus rings on",
        options=[f"Target side ({dst_col})", f"Source side ({src_col})"],
    )

    min_tx = st.slider(
        "Min transactions per node",
        min_value=2,
        max_value=200,
        value=10,
        step=1,
    )

    min_fraud = st.slider(
        "Min fraud count per node",
        min_value=1,
        max_value=50,
        value=3,
        step=1,
    )

    min_neighbors = st.slider(
        "Min distinct counterparties (neighbors)",
        min_value=2,
        max_value=50,
        value=3,
        step=1,
    )

    max_nodes_display = st.slider(
        "Max ring hubs to display",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
    )

# Build node/edge metrics
src_nodes, dst_nodes, edges = build_ring_summary(df, src_col, dst_col, label_col, amount_col)

if src_nodes.empty or dst_nodes.empty:
    st.warning("No node statistics could be computed; check the dataset columns.")
    st.stop()

# Determine which side acts as hub
if mode.startswith("Target"):
    hubs_df = dst_nodes.copy()
    hubs_label = dst_col
    neighbor_side = "source"
    hub_type = "target"
else:
    hubs_df = src_nodes.copy()
    hubs_label = src_col
    neighbor_side = "target"
    hub_type = "source"

# Count distinct neighbors and apply thresholds
if neighbor_side == "source":
    neighbor_counts = edges.groupby(dst_col)[src_col].nunique().to_frame("neighbor_count")
    hubs_df = hubs_df.merge(neighbor_counts, left_on="node", right_index=True, how="left")
else:
    neighbor_counts = edges.groupby(src_col)[dst_col].nunique().to_frame("neighbor_count")
    hubs_df = hubs_df.merge(neighbor_counts, left_on="node", right_index=True, how="left")

hubs_df["neighbor_count"] = hubs_df["neighbor_count"].fillna(0).astype(int)

# Apply thresholds
hubs_filtered = hubs_df[hubs_df["tx_count"] >= min_tx].copy()
if label_col and "fraud_count" in hubs_filtered.columns:
    hubs_filtered = hubs_filtered[hubs_filtered["fraud_count"] >= min_fraud]
if "neighbor_count" in hubs_filtered.columns:
    hubs_filtered = hubs_filtered[hubs_filtered["neighbor_count"] >= min_neighbors]

hubs_filtered = hubs_filtered.sort_values(
    ["fraud_count" if "fraud_count" in hubs_filtered.columns else "tx_count", "neighbor_count"],
    ascending=[False, False],
).head(max_nodes_display)

if hubs_filtered.empty:
    st.warning(
        "No ring hubs found with the current thresholds. "
        "Try lowering the minimum transactions / fraud count / neighbors."
    )
    st.stop()

num_hubs = len(hubs_filtered)
max_deg = int(hubs_filtered["neighbor_count"].max())
max_tx_hub = int(hubs_filtered["tx_count"].max())

# ---------- KPIs ----------

st.subheader("Fraud Ring Overview")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Hub side column", hubs_label)
with c2:
    st.metric("Ring hubs found", f"{num_hubs:,}")
with c3:
    st.metric("Max neighbors per hub", f"{max_deg:,}")
with c4:
    st.metric("Max tx per hub", f"{max_tx_hub:,}")

st.markdown("---")

# ---------- Hubs table ----------

st.subheader("Suspicious Ring Hubs")

display_cols = ["node", "tx_count", "neighbor_count"]
if "fraud_count" in hubs_filtered.columns:
    display_cols.append("fraud_count")
if "fraud_rate" in hubs_filtered.columns:
    display_cols.append("fraud_rate")
if "total_amount" in hubs_filtered.columns:
    display_cols.append("total_amount")

st.dataframe(
    hubs_filtered[display_cols].reset_index(drop=True),
    use_container_width=True,
)

# ---------- Select a hub for detail ----------

st.subheader("Ring Detail")

hub_choices = hubs_filtered["node"].astype(str).tolist()
selected_hub = st.selectbox(
    "Select a ring hub",
    options=hub_choices,
)

# Get edges touching the selected hub
if mode.startswith("Target"):  # hub is dst_col
    hub_edges = edges[edges[dst_col].astype(str) == selected_hub].copy()
    neighbor_col = src_col
else:  # hub is src_col
    hub_edges = edges[edges[src_col].astype(str) == selected_hub].copy()
    neighbor_col = dst_col

if hub_edges.empty:
    st.info("No edges found for this hub (possible after filters).")
else:
    # Sort neighbors by fraud_count then tx_count
    sort_cols = []
    if "fraud_count" in hub_edges.columns:
        sort_cols.append("fraud_count")
    sort_cols.append("tx_count")
    hub_edges = hub_edges.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    neighbor_rows: List[Dict[str, object]] = []
    for _, row in hub_edges.iterrows():
        neighbor_rows.append(
            {
                "node": str(row[neighbor_col]),
                "tx_count": int(row["tx_count"]),
                "fraud_count": int(row["fraud_count"]) if "fraud_count" in row else None,
                "total_amount": float(row["total_amount"]) if "total_amount" in row else None,
            }
        )

    neighbors_df = pd.DataFrame(neighbor_rows)
    st.write(
        f"Hub **{selected_hub}** has **{len(neighbors_df):,}** distinct counterparties "
        f"({neighbor_col})."
    )

    st.dataframe(neighbors_df.head(100), use_container_width=True)

    # Simple ring/star visualization
    st.subheader("Ring Visualization (Star Layout)")

    fig_star = make_star_graph_figure(
        hub=selected_hub,
        neighbors=neighbor_rows[:20],  # limit to avoid clutter
        hub_type=hub_type,
    )
    st.plotly_chart(fig_star, use_container_width=True)

# ---------- Download buttons ----------

st.subheader("Export")

csv_hubs = hubs_filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download ring hubs as CSV",
    data=csv_hubs,
    file_name="fraud_ring_hubs.csv",
    mime="text/csv",
)

# All edges for hubs in view
edges_for_hubs = edges[
    edges[dst_col].astype(str).isin(hubs_filtered["node"].astype(str))
    | edges[src_col].astype(str).isin(hubs_filtered["node"].astype(str))
].copy()

csv_edges = edges_for_hubs.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download ring edges (for hubs in view)",
    data=csv_edges,
    file_name="fraud_ring_edges.csv",
    mime="text/csv",
)
