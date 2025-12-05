from __future__ import annotations

from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from shared.data_loader import load_dataset


# ---------- helpers ----------

def get_label_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("is_fraud", "isFraud", "fraud_label", "label", "y"):
        if col in df.columns:
            return col
    return None


def get_amount_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("amount", "amt", "transaction_amount", "value"):
        if col in df.columns:
            return col
    return None


def get_score_col(df: pd.DataFrame) -> Optional[str]:
    preferred = ["risk_score", "fraud_score", "score", "probability", "pred_proba"]
    for c in preferred:
        if c in df.columns and np.issubdtype(df[c].dtype, np.number):
            return c

    for c in df.columns:
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        name = c.lower()
        if "score" in name or "prob" in name or "risk" in name:
            return c

    return None


def build_synthetic_score(
    df: pd.DataFrame,
    label_col: Optional[str],
    amount_col: Optional[str],
) -> pd.Series:
    """
    Fallback: build a fake risk score in [0,1] so the Threshold Tuning
    page can still work even without a trained model.
    """
    n = len(df)
    base = np.zeros(n, dtype=float)

    if amount_col:
        amt = df[amount_col].astype(float).to_numpy()
        if amt.max() > amt.min():
            amt_scaled = (amt - amt.min()) / (amt.max() - amt.min())
        else:
            amt_scaled = np.zeros_like(amt)
        base += 0.6 * amt_scaled  # higher amount → higher risk

    if label_col:
        y = df[label_col].astype(int).to_numpy()
        base += 0.3 * y  # confirmed frauds get a bump

    # small noise so we don't get all identical scores
    rng = np.random.default_rng(42)
    base += 0.05 * rng.random(n)

    base = np.clip(base, 0.0, 1.0)
    return pd.Series(base, index=df.index, name="synthetic_score")


def compute_confusion_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    total = tp + tn + fp + fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
        "precision": precision,
        "recall": recall,
        "tnr": tnr,
        "f1": f1,
    }


# ---------- page ----------

st.title("Threshold Tuning")
st.caption(
    "Interactively adjust the decision threshold on the model score (or a synthetic risk "
    "score) and observe precision/recall trade-offs for fraud detection."
)

# Load dataset
try:
    df = load_dataset()
except Exception as exc:
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – cannot run threshold tuning.")
    st.stop()

label_col = get_label_col(df)
amount_col = get_amount_col(df)
score_col = get_score_col(df)

# Build or locate score
score_is_synthetic = False
if score_col is None:
    # Create synthetic score
    df["_synthetic_score"] = build_synthetic_score(df, label_col, amount_col)
    score_col = "_synthetic_score"
    score_is_synthetic = True

# Sanity: ensure numeric and within [0,1] (if not, rescale)
score = df[score_col].astype(float)
if score.min() < 0.0 or score.max() > 1.0:
    # Min-max rescale into [0,1]
    if score.max() > score.min():
        score = (score - score.min()) / (score.max() - score.min())
    else:
        score = pd.Series(np.zeros(len(score)), index=score.index)
    df[score_col] = score

# Sidebar controls
with st.sidebar:
    st.subheader("Threshold Settings")

    threshold = st.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )

    st.caption(
        f"Using score column: **{score_col}**"
        + (" (synthetic)" if score_is_synthetic else "")
    )
    if score_is_synthetic:
        st.info(
            "No model score column found; created a synthetic risk score from amount/labels "
            "for demo purposes."
        )

    st.markdown("---")
    top_n = st.number_input(
        "Top N highest-score transactions to show",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
    )

# Predictions at current threshold
df["_score"] = df[score_col]
df["_pred"] = (df["_score"] >= threshold).astype(int)

# ---------- KPIs at current threshold ----------

st.subheader("Metrics at Current Threshold")

if label_col:
    metrics = compute_confusion_metrics(df[label_col].astype(int), df["_pred"])
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Precision", f"{metrics['precision'] * 100:,.2f}%")
    with c2:
        st.metric("Recall (TPR)", f"{metrics['recall'] * 100:,.2f}%")
    with c3:
        st.metric("Specificity (TNR)", f"{metrics['tnr'] * 100:,.2f}%")
    with c4:
        st.metric("F1 Score", f"{metrics['f1'] * 100:,.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("TP", str(metrics["tp"]))
    with c6:
        st.metric("FP", str(metrics["fp"]))
    with c7:
        st.metric("FN", str(metrics["fn"]))
    with c8:
        st.metric("TN", str(metrics["tn"]))

    # Confusion matrix heatmap
    cm = np.array(
        [
            [metrics["tn"], metrics["fp"]],
            [metrics["fn"], metrics["tp"]],
        ]
    )
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred 0", "Pred 1"],
            y=["True 0", "True 1"],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig_cm.update_layout(
        title=f"Confusion Matrix @ threshold = {threshold:.2f}",
        xaxis_title="Predicted label",
        yaxis_title="True label",
        height=350,
    )
    st.plotly_chart(fig_cm, use_container_width=True)
else:
    st.info(
        "No label column found (e.g., 'is_fraud'). Metrics such as precision/recall cannot "
        "be computed, but you can still explore score distributions."
    )

st.markdown("---")

# ---------- Metrics vs threshold curves ----------

st.subheader("Threshold Sweep (Precision / Recall / F1)")

if label_col:
    y_true = df[label_col].astype(int).to_numpy()
    s = df["_score"].to_numpy()

    thresholds = np.linspace(0.0, 1.0, num=41)
    rows: List[Dict[str, float]] = []
    for t in thresholds:
        y_pred = (s >= t).astype(int)
        m = compute_confusion_metrics(pd.Series(y_true), pd.Series(y_pred))
        rows.append(
            {
                "threshold": t,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        )
    sweep_df = pd.DataFrame(rows)

    fig_thr = px.line(
        sweep_df,
        x="threshold",
        y=["precision", "recall", "f1"],
        labels={"value": "Metric value", "threshold": "Threshold", "variable": "Metric"},
        title="Precision / Recall / F1 vs Threshold",
    )
    st.plotly_chart(fig_thr, use_container_width=True)
else:
    st.info(
        "Cannot generate threshold curves without ground-truth labels. "
        "Add a label column to enable this section."
    )

st.markdown("---")

# ---------- Score distribution ----------

st.subheader("Score Distribution")

if label_col:
    plot_df = df[[score_col, label_col]].copy()
    plot_df["class"] = plot_df[label_col].map({0: "Non-Fraud", 1: "Fraud"})
    fig_dist = px.histogram(
        plot_df,
        x=score_col,
        color="class",
        barmode="overlay",
        nbins=40,
        title="Score Distribution by Class",
        labels={score_col: "Score"},
        opacity=0.6,
    )
else:
    plot_df = df[[score_col]].copy()
    fig_dist = px.histogram(
        plot_df,
        x=score_col,
        nbins=40,
        title="Score Distribution",
        labels={score_col: "Score"},
    )

# Add vertical line for current threshold
fig_dist.add_vline(
    x=threshold,
    line_width=2,
    line_dash="dash",
)

st.plotly_chart(fig_dist, use_container_width=True)

st.markdown("---")

# ---------- Top flagged transactions ----------

st.subheader("Top High-Score Transactions")

flagged = df[df["_pred"] == 1].copy()
flagged = flagged.sort_values("_score", ascending=False)

if flagged.empty:
    st.info("No transactions are above the current threshold.")
else:
    view_cols: List[str] = []
    # Important columns first
    view_cols.append("_score")
    if label_col:
        view_cols.append(label_col)
    if amount_col:
        view_cols.append(amount_col)
    # Then some ID-like columns if present
    for candidate in ("transaction_id", "tx_id", "id", "customer_id", "nameOrig", "merchant_id", "nameDest"):
        if candidate in flagged.columns and candidate not in view_cols:
            view_cols.append(candidate)
    # Fill the rest
    for c in flagged.columns:
        if c not in view_cols:
            view_cols.append(c)

    st.dataframe(
        flagged[view_cols].head(int(top_n)),
        use_container_width=True,
    )

    csv_bytes = flagged[view_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download all flagged transactions as CSV",
        data=csv_bytes,
        file_name="threshold_tuning_flagged_transactions.csv",
        mime="text/csv",
    )
