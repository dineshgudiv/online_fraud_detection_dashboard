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


def get_score_col(df: pd.DataFrame) -> Optional[str]:
    for col in ("risk_score", "fraud_score", "score", "probability", "pred_proba"):
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


def split_reference_current(
    df: pd.DataFrame,
    frac_current: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into (reference, current).
    If a usable time column exists, split by time. Otherwise, split by row index.
    """
    time_col = get_time_col(df)
    if time_col and pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)

    n = len(df_sorted)
    if n < 10:
        return df_sorted.iloc[:0], df_sorted

    frac_current = float(np.clip(frac_current, 0.05, 0.5))
    cut_idx = int(n * (1.0 - frac_current))
    if cut_idx <= 0 or cut_idx >= n:
        cut_idx = max(1, n // 2)

    reference = df_sorted.iloc[:cut_idx].copy()
    current = df_sorted.iloc[cut_idx:].copy()
    return reference, current


def compute_drift_for_numeric(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    For each numeric column, compute mean/std in reference vs current
    and a simple drift score based on mean shift in units of reference std.
    """
    if exclude_cols is None:
        exclude_cols = []

    numeric_cols = ref.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    rows: List[dict] = []

    for col in numeric_cols:
        if col not in cur.columns:
            continue

        ref_col = ref[col].dropna()
        cur_col = cur[col].dropna()
        if len(ref_col) == 0 or len(cur_col) == 0:
            continue

        mean_ref = float(ref_col.mean())
        mean_cur = float(cur_col.mean())
        std_ref = float(ref_col.std(ddof=1) if len(ref_col) > 1 else 0.0)
        std_cur = float(cur_col.std(ddof=1) if len(cur_col) > 1 else 0.0)

        if std_ref > 0:
            mean_shift_std = abs(mean_cur - mean_ref) / std_ref
        else:
            mean_shift_std = abs(mean_cur - mean_ref)

        rows.append(
            {
                "feature": col,
                "mean_reference": mean_ref,
                "mean_current": mean_cur,
                "std_reference": std_ref,
                "std_current": std_cur,
                "mean_shift_std": mean_shift_std,
            }
        )

    drift_df = pd.DataFrame(rows)
    if not drift_df.empty:
        drift_df = drift_df.sort_values("mean_shift_std", ascending=False)
    return drift_df


st.title("Model Health & Data Drift")
st.caption(
    "Compare feature distributions between a historical reference window and a recent current window. "
    "Helps detect when your fraud model is seeing data that differs from training."
)

# ---- Load data ----
try:
    df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – cannot compute drift.")
    st.stop()

label_col = get_label_col(df)
score_col = get_score_col(df)
time_col = get_time_col(df)

# ---- Sidebar controls ----
with st.sidebar:
    st.subheader("Drift Settings")

    frac_current = st.slider(
        "Current window fraction (tail of data)",
        min_value=0.05,
        max_value=0.5,
        value=0.3,
        step=0.05,
    )

    st.caption(
        "Example: 0.3 means last 30% of the dataset is 'current', "
        "first 70% is 'reference'. If you have timestamps, the split is chronological."
    )

# ---- Split reference vs current ----
ref_df, cur_df = split_reference_current(df, frac_current)

if ref_df.empty or cur_df.empty:
    st.warning("Not enough data to create reference and current windows.")
    st.stop()

st.markdown(
    f"Reference window size: **{len(ref_df):,}** rows – Current window size: **{len(cur_df):,}** rows."
)

# ---- Drift computation ----
exclude_cols = [c for c in [label_col, score_col] if c is not None]
drift_df = compute_drift_for_numeric(ref_df, cur_df, exclude_cols=exclude_cols)

num_features = len(drift_df)
top_feature = drift_df.iloc[0]["feature"] if num_features > 0 else None
top_drift = drift_df.iloc[0]["mean_shift_std"] if num_features > 0 else 0.0

# ---- KPIs ----
st.subheader("Drift Summary")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Numeric Features Monitored", f"{num_features:,}")
with c2:
    st.metric("Current Window Fraction", f"{frac_current * 100:.0f}%")
with c3:
    if top_feature is not None:
        st.metric("Top Drift Feature", str(top_feature))
    else:
        st.metric("Top Drift Feature", "N/A")
with c4:
    if top_feature is not None:
        st.metric("Top Drift (std units)", f"{top_drift:.2f}")
    else:
        st.metric("Top Drift (std units)", "0.00")

st.markdown("---")

if drift_df.empty:
    st.info(
        "No numeric features found to compute drift, or all numeric columns are excluded. "
        "Ensure your dataset has numeric columns (e.g., amount, scores, counts)."
    )
    st.stop()

# ---- Table of drift metrics ----
st.subheader("Per-Feature Drift Metrics")

st.dataframe(drift_df, use_container_width=True)

# ---- Bar plot of top drifted features ----
st.subheader("Top Drifted Features")

top_n = st.slider(
    "Number of features to display",
    min_value=1,
    max_value=max(1, min(20, num_features)),
    value=min(10, max(1, num_features)),
)

fig_bar = px.bar(
    drift_df.head(top_n),
    x="feature",
    y="mean_shift_std",
    title="Top Drifted Features (Mean Shift in Reference Std Units)",
    labels={"mean_shift_std": "Mean Shift (std units)", "feature": "Feature"},
    text="mean_shift_std",
)
fig_bar.update_traces(texttemplate="%{text:.2f}", textposition="outside")
st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ---- Distribution comparison for a selected feature ----
st.subheader("Distribution Comparison: Reference vs Current")

feature_choice = st.selectbox(
    "Select feature to inspect",
    options=drift_df["feature"].tolist(),
    index=0 if top_feature is None else drift_df["feature"].tolist().index(top_feature),
)

ref_vals = ref_df[feature_choice].dropna()
cur_vals = cur_df[feature_choice].dropna()

if ref_vals.empty or cur_vals.empty:
    st.info(f"No data for feature '{feature_choice}' in one of the windows.")
else:
    # Build a small combined DataFrame with a 'window' column
    combined = pd.DataFrame(
        {
            feature_choice: pd.concat([ref_vals, cur_vals], ignore_index=True),
            "window": ["reference"] * len(ref_vals) + ["current"] * len(cur_vals),
        }
    )

    fig_hist = px.histogram(
        combined,
        x=feature_choice,
        color="window",
        barmode="overlay",
        nbins=40,
        title=f"Distribution of '{feature_choice}' – Reference vs Current",
        opacity=0.6,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ---- Optional: label/score drift context ----
if label_col or score_col:
    st.markdown("---")
    st.subheader("Label / Score Drift Context")

    cols = st.columns(2)

    if label_col:
        with cols[0]:
            ref_rate = ref_df[label_col].mean()
            cur_rate = cur_df[label_col].mean()
            st.metric(
                "Fraud Rate Drift",
                f"{cur_rate * 100:.2f}%",
                delta=f"{(cur_rate - ref_rate) * 100:.2f} pp",
            )

    if score_col:
        with cols[1]:
            ref_score_mean = float(ref_df[score_col].mean())
            cur_score_mean = float(cur_df[score_col].mean())
            st.metric(
                "Score Mean Drift",
                f"{cur_score_mean:.3f}",
                delta=f"{(cur_score_mean - ref_score_mean):+.3f}",
            )
else:
    st.info(
        "No explicit label or score column detected (e.g., 'is_fraud', 'risk_score'). "
        "You can still use numeric feature drift as a proxy for model health."
    )
