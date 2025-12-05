import pandas as pd
import plotly.express as px
import streamlit as st

from shared.data_loader import load_dataset


def _build_connections(df: pd.DataFrame) -> pd.DataFrame:
    """Construct simple connections between entities (user, merchant, device)."""
    cols = [c for c in ["user_id", "merchant_id", "device_id"] if c in df.columns]
    if len(cols) < 2 or df.empty:
        return pd.DataFrame(columns=["source", "target"])
    edges = []
    for _, row in df[cols].dropna().head(5000).iterrows():
        vals = [row[c] for c in cols]
        for i in range(len(vals) - 1):
            edges.append({"source": vals[i], "target": vals[i + 1]})
    return pd.DataFrame(edges)


def render_page(client=None):
    st.title("Global Network")
    st.caption("Entity graph (users, merchants, devices). Uses simple co-occurrence to show connectivity.")

    df = load_dataset()
    if df.empty:
        st.warning("Dataset empty; cannot build network.")
        return

    edges = _build_connections(df)
    if edges.empty:
        st.info("Insufficient columns to build network (need user_id, merchant_id, or device_id).")
        return

    st.metric("Nodes", f"{len(pd.unique(edges[['source', 'target']].values.ravel())):,}")
    st.metric("Edges", f"{len(edges):,}")

    top_nodes = pd.concat([edges["source"], edges["target"]]).value_counts().head(15).reset_index()
    top_nodes.columns = ["node", "degree"]
    fig = px.bar(top_nodes, x="node", y="degree", title="Top Connected Entities")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Edge Sample")
    st.dataframe(edges.head(200), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    render_page()
