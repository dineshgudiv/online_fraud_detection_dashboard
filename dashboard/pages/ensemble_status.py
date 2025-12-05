from __future__ import annotations

from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from shared.data_loader import load_dataset

# Optional sklearn import for AUC; page must not crash if missing
try:  # noqa: SIM105
    from sklearn.metrics import roc_auc_score
except Exception:  # noqa: BLE001
    roc_auc_score = None  # type: ignore[assignment]


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


def find_score_columns(df: pd.DataFrame) -> List[str]:
    """
    Try to automatically discover model/score columns.
    Looks for typical names and also anything ending with '_score' or '_prob'.
    """
    candidates: List[str] = []
    preferred = ["risk_score", "fraud_score", "score", "probability", "pred_proba"]

    for c in preferred:
        if c in df.columns and np.issubdtype(df[c].dtype, np.number):
            candidates.append(c)

    for c in df.columns:
        if c in candidates:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        name = c.lower()
        if name.endswith("_score") or name.endswith("_prob") or name.endswith("_probability"):
            candidates.append(c)

    return candidates


def build_demo_score(df: pd.DataFrame, amount_col: Optional[str]) -> pd.Series:
    """Create a synthetic score in [0,1] to show ensemble metrics when no scores are present."""
    n = len(df)
    rng = np.random.default_rng(7)
    noise = rng.beta(2, 5, size=n)
    if amount_col and df[amount_col].max() > df[amount_col].min():
        amt = df[amount_col].astype(float)
        amt_norm = (amt - amt.min()) / (amt.max() - amt.min() + 1e-9)
        score = 0.7 * amt_norm + 0.3 * noise
    else:
        score = noise
    return pd.Series(np.clip(score, 0, 1), index=df.index, name="risk_score_demo")


def compute_model_metrics(
    scores: pd.Series,
    y_true: Optional[pd.Series],
    threshold: float = 0.8,
) -> Dict[str, object]:
    """
    Compute metrics for a single model score column.
    """
    scores_clean = scores.astype(float).clip(0.0, 1.0)
    n = len(scores_clean)
    mean_score = float(scores_clean.mean())
    high_risk_share = float((scores_clean >= threshold).mean())

    auc_val: Optional[float] = None
    recall_at_thr: Optional[float] = None
    precision_at_thr: Optional[float] = None

    if y_true is not None and roc_auc_score is not None:
        try:
            y_bin = (y_true == 1).astype(int)
            if y_bin.nunique() == 2:
                auc_val = float(roc_auc_score(y_bin, scores_clean))
        except Exception:
            auc_val = None

        # classification at threshold
        y_pred = (scores_clean >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_bin == 1)).sum())
        fp = int(((y_pred == 1) & (y_bin == 0)).sum())
        fn = int(((y_pred == 0) & (y_bin == 1)).sum())

        recall_at_thr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_at_thr = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "n": n,
        "mean_score": mean_score,
        "high_risk_share": high_risk_share,
        "auc": auc_val,
        "recall_at_thr": recall_at_thr,
        "precision_at_thr": precision_at_thr,
    }


st.title("Ensemble Status")
st.caption(
    "Overview of fraud model and score health across the ensemble. "
    "Each score column is treated as one model in the ensemble."
)

# ---- Load dataset ----
try:
    df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – cannot evaluate ensemble status.")
    st.stop()

label_col = get_label_col(df)
amount_col = get_amount_col(df)
score_cols = find_score_columns(df)

if not score_cols:
    st.info(
        "No model/score columns detected. "
        "Expected columns like 'risk_score', 'fraud_score', 'score', "
        "or any numeric column ending with '_score' / '_prob'. Using a synthetic demo score."
    )
    demo_score = build_demo_score(df, amount_col)
    df[demo_score.name] = demo_score
    score_cols = [demo_score.name]

with st.sidebar:
    st.subheader("Ensemble Settings")

    threshold = st.slider(
        "High-risk threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.01,
    )

    st.caption(
        "Scores ≥ threshold are counted as 'high-risk'. "
        "This is used for high_risk_share and recall/precision metrics."
    )

# ---- Compute metrics per model ----
y_true = df[label_col] if label_col else None
rows: List[Dict[str, object]] = []

for col in score_cols:
    metrics = compute_model_metrics(df[col], y_true, threshold=threshold)
    rows.append(
        {
            "model": col,
            "n": metrics["n"],
            "mean_score": metrics["mean_score"],
            "high_risk_share": metrics["high_risk_share"],
            "auc": metrics["auc"],
            "recall_at_thr": metrics["recall_at_thr"],
            "precision_at_thr": metrics["precision_at_thr"],
        }
    )

metrics_df = pd.DataFrame(rows)

# ---- KPIs ----
st.subheader("Ensemble Overview")

num_models = len(metrics_df)
max_auc = (
    metrics_df["auc"].dropna().max()
    if ("auc" in metrics_df.columns and metrics_df["auc"].notna().any())
    else None
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Detected Models", f"{num_models:,}")
with c2:
    if max_auc is not None:
        st.metric("Best AUC", f"{max_auc:.3f}")
    else:
        st.metric("Best AUC", "N/A")
with c3:
    st.metric("Threshold", f"{threshold:.2f}")
with c4:
    st.metric("Has Labels", "Yes" if label_col else "No")

if roc_auc_score is None and label_col:
    st.info(
        "scikit-learn not available; AUC metrics are shown as N/A. "
        "Install scikit-learn in your environment to enable AUC calculation."
    )

st.markdown("---")

# ---- Summary table ----
st.subheader("Per-Model Metrics")

display_df = metrics_df.copy()

# Pretty formatting for high_risk_share / auc / recall / precision
for col in ["high_risk_share", "auc", "recall_at_thr", "precision_at_thr"]:
    if col in display_df.columns:
        display_df[col] = display_df[col].apply(
            lambda v: f"{v:.3f}" if isinstance(v, (float, int)) and not pd.isna(v) else "N/A"
        )

st.dataframe(display_df, use_container_width=True)

# ---- Bar chart: AUC or mean_score ----
st.subheader("Model Comparison")

metric_choice_options = ["mean_score"]
if label_col and metrics_df["auc"].notna().any():
    metric_choice_options.insert(0, "auc")

metric_choice = st.selectbox(
    "Select comparison metric",
    options=metric_choice_options,
)

plot_df = metrics_df.copy()
fig_comp = px.bar(
    plot_df,
    x="model",
    y=metric_choice,
    title=f"Models by {metric_choice}",
    labels={metric_choice: metric_choice.replace("_", " ").title(), "model": "Model"},
)
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

# ---- Detailed view for a selected model ----
st.subheader("Model Detail")

selected_model = st.selectbox(
    "Choose a model / score column",
    options=score_cols,
)

scores_sel = df[selected_model].astype(float).clip(0.0, 1.0)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Mean Score", f"{scores_sel.mean():.3f}")
with c2:
    st.metric("High-Risk Share", f"{(scores_sel >= threshold).mean():.3f}")
with c3:
    st.metric("Std Dev", f"{scores_sel.std():.3f}")

fig_hist = px.histogram(
    scores_sel,
    nbins=40,
    title=f"Score Distribution – {selected_model}",
    labels={"value": "Score", "count": "Count"},
)
fig_hist.add_vline(
    x=threshold,
    line_dash="dash",
    annotation_text=f"Threshold = {threshold:.2f}",
    annotation_position="top right",
)
st.plotly_chart(fig_hist, use_container_width=True)

# If labels exist, show score vs label boxplot
if label_col:
    st.subheader("Score vs Label")

    tmp = pd.DataFrame(
        {
            "score": scores_sel,
            "label": df[label_col].map({0: "Non-Fraud", 1: "Fraud"}),
        }
    )
    fig_box = px.box(
        tmp,
        x="label",
        y="score",
        title=f"{selected_model} by Label",
    )
    st.plotly_chart(fig_box, use_container_width=True)
else:
    st.info(
        "No fraud label column (e.g., 'is_fraud') detected – cannot show score vs label. "
        "You can still compare score distributions across models."
    )
