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


def build_demo_score(df: pd.DataFrame, amount_col: Optional[str]) -> pd.Series:
    """
    Build a synthetic risk score in [0, 1] when there is no real model score.
    Uses amount (if available) plus random noise so higher amounts tend to have higher scores.
    """
    n = len(df)
    rng = np.random.default_rng(42)

    base_noise = rng.beta(2.0, 5.0, size=n)  # skewed towards low risk by default

    if amount_col and amount_col in df.columns and df[amount_col].max() > df[amount_col].min():
        amt = df[amount_col].astype(float)
        amt_norm = (amt - amt.min()) / (amt.max() - amt.min() + 1e-9)
        score = 0.7 * amt_norm + 0.3 * base_noise
    else:
        score = base_noise

    score = np.clip(score, 0.0, 1.0)
    return pd.Series(score, index=df.index, name="risk_score")


st.title("Batch Scoring")
st.caption(
    "Upload a batch of transactions to generate risk scores and fraud alerts. "
    "If no real model is wired in yet, this page uses a synthetic risk score based on amount."
)

# ---- Load base dataset for fallback/sample ----
try:
    base_df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load base dataset: {exc}")
    base_df = None

label_col_base = get_label_col(base_df) if base_df is not None else None
amount_col_base = get_amount_col(base_df) if base_df is not None else None

# ---- File upload / sample selection ----
st.subheader("1. Input Batch")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded batch with {len(batch_df):,} rows and {len(batch_df.columns)} columns.")
    except Exception as exc:
        st.error(f"Could not read uploaded CSV: {exc}")
        st.stop()
else:
    if base_df is None or base_df.empty:
        st.warning(
            "No uploaded file and base dataset is not available/empty. "
            "Upload a CSV to proceed with batch scoring."
        )
        st.stop()

    st.info(
        "No file uploaded – using a sampled subset of the main dataset "
        "as a demo batch."
    )
    batch_df = base_df.sample(min(5000, len(base_df)), random_state=42).reset_index(drop=True)

st.dataframe(batch_df.head(50), use_container_width=True)

st.markdown("---")

# ---- Threshold and scoring ----
st.subheader("2. Scoring Configuration")

# Try to reuse an existing score if the uploaded batch already has one
score_col_existing = None
for cand in ("risk_score", "fraud_score", "score", "probability", "pred_proba"):
    if cand in batch_df.columns:
        score_col_existing = cand
        break

amount_col_batch = get_amount_col(batch_df)

if score_col_existing:
    st.info(
        f"Existing score-like column detected in batch: **{score_col_existing}**. "
        "Using it as the risk score."
    )
    scores = batch_df[score_col_existing].astype(float)
    score_name = score_col_existing
else:
    st.info(
        "No existing score column detected in the batch; generating a synthetic risk score "
        "based on amount (if available) and noise."
    )
    scores = build_demo_score(batch_df, amount_col_batch)
    score_name = scores.name
    batch_df[score_name] = scores

threshold = st.slider(
    "Alert threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.01,
    help="Transactions with score ≥ threshold will be flagged as alerts.",
)

batch_df["predicted_fraud"] = (scores >= threshold).astype(int)

# ---- Metrics ----
st.subheader("3. Batch Metrics")

label_col_batch = get_label_col(batch_df)
total_rows = len(batch_df)
alerts = int(batch_df["predicted_fraud"].sum())

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Batch Size", f"{total_rows:,}")
with c2:
    st.metric("Alerts", f"{alerts:,}")
with c3:
    st.metric("Alert Rate", f"{alerts / max(1, total_rows) * 100:.2f}%")
with c4:
    if label_col_batch:
        actual_fraud = int(batch_df[label_col_batch].sum())
        tp = int(
            batch_df[
                (batch_df["predicted_fraud"] == 1)
                & (batch_df[label_col_batch] == 1)
            ].shape[0]
        )
        precision = tp / alerts if alerts > 0 else 0.0
        st.metric("Precision (wrt labels)", f"{precision:.3f}")
    else:
        st.metric("Precision (wrt labels)", "N/A (no label column)")

# ---- Score distribution ----
st.subheader("4. Score Distribution")

fig_scores = px.histogram(
    batch_df,
    x=score_name,
    nbins=40,
    title=f"Distribution of {score_name}",
)
fig_scores.add_vline(
    x=threshold,
    line_dash="dash",
    annotation_text=f"Threshold = {threshold:.2f}",
    annotation_position="top right",
)
st.plotly_chart(fig_scores, use_container_width=True)

# ---- Sample alerts table ----
st.subheader("5. Flagged Transactions (sample)")

alerts_df = batch_df[batch_df["predicted_fraud"] == 1].copy()
st.write(f"Flagged alerts: **{len(alerts_df):,}**")

if not alerts_df.empty:
    st.dataframe(alerts_df.head(500), use_container_width=True)
else:
    st.info("No rows exceed the current threshold.")

# ---- Download scored batch ----
st.subheader("6. Download Scored Batch")

csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download scored batch as CSV",
    data=csv_bytes,
    file_name="scored_batch.csv",
    mime="text/csv",
)
