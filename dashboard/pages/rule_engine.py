from __future__ import annotations

from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from shared.data_loader import load_dataset


# ------------ helpers ------------

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


def get_customer_col(df: pd.DataFrame) -> Optional[str]:
    # 1) Known customer-style column names
    for col in ("customer_id", "cust_id", "nameOrig", "client_id", "account_id"):
        if col in df.columns:
            return col

    # 2) Any column whose name looks like a customer/user/account id
    for col in df.columns:
        name = col.lower()
        if "cust" in name or "user" in name or "account" in name:
            return col

    # 3) Last resort: first string-like column
    str_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(str_cols) > 0:
        return str_cols[0]

    return None


def get_score_col(df: pd.DataFrame) -> Optional[str]:
    # Try to find a model score if present
    preferred = ["risk_score", "fraud_score", "score", "probability", "pred_proba"]
    for c in preferred:
        if c in df.columns and np.issubdtype(df[c].dtype, np.number):
            return c
    for c in df.columns:
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        name = c.lower()
        if name.endswith("_score") or name.endswith("_prob"):
            return c
    return None


# ------------ page header ------------

st.title("Rule Engine")
st.caption(
    "Configure simple heuristic rules (high amount, risky country, high activity) and "
    "see which transactions would be flagged for review."
)

# ------------ load data ------------

try:
    df = load_dataset()
except Exception as exc:  # pragma: no cover – safety
    st.error(f"Failed to load dataset: {exc}")
    st.stop()

if df is None or df.empty:
    st.warning("Dataset is empty – cannot run rule engine.")
    st.stop()

label_col = get_label_col(df)
amount_col = get_amount_col(df)
country_col = get_country_col(df)
customer_col = get_customer_col(df)
score_col = get_score_col(df)

n_rows = len(df)

# ------------ sidebar: rule configuration ------------

with st.sidebar:
    st.subheader("Rules Configuration")

    # Rule 1: High amount
    enable_amount = st.checkbox(
        "Enable: High Amount Rule",
        value=True if amount_col else False,
        disabled=amount_col is None,
    )
    if amount_col:
        amt_min = float(df[amount_col].min())
        amt_max = float(df[amount_col].max())
        high_amt_threshold = st.number_input(
            "Amount ≥ (High Amount Rule)",
            min_value=0.0,
            max_value=max(amt_max, 1.0),
            value=float(np.percentile(df[amount_col], 95)) if amt_max > 0 else max(1000.0, amt_max),
            step=(amt_max - amt_min) / 50 if amt_max > amt_min else 100.0,
        )
    else:
        high_amt_threshold = None
        st.info("No numeric amount column found; High Amount rule is disabled.")

    st.markdown("---")

    # Rule 2: Risky country / region
    enable_country = st.checkbox(
        "Enable: Risky Country Rule",
        value=True if country_col else False,
        disabled=country_col is None,
    )
    risky_countries: List[str] = []
    if country_col:
        all_countries = (
            df[country_col].astype(str).dropna().value_counts().head(30).index.tolist()
        )
        risky_countries = st.multiselect(
            "Risky countries / regions",
            options=all_countries,
            default=all_countries[:3] if all_countries else [],
        )
    else:
        st.info("No country/region column found; Risky Country rule is disabled.")

    st.markdown("---")

    # Rule 3: High activity customer
    enable_activity = st.checkbox(
        "Enable: High Activity Customer Rule",
        value=True if customer_col else False,
        disabled=customer_col is None,
    )
    if customer_col:
        min_tx_per_customer = st.number_input(
            "Min transactions per customer (High Activity)",
            min_value=2,
            max_value=1000,
            value=10,
            step=1,
        )
        min_total_amount_per_customer = st.number_input(
            "Min total amount per customer",
            min_value=0.0,
            max_value=1e9,
            value=10000.0,
            step=1000.0,
        )
    else:
        min_tx_per_customer = None
        min_total_amount_per_customer = None
        st.info("No customer column found; High Activity rule is disabled.")

    st.markdown("---")

    if score_col:
        st.caption(f"Detected score column: **{score_col}** (optional, used for analysis only).")
    else:
        st.caption("Score column optional; using rules only (amount/country/activity).")

    st.caption("Rules auto-apply whenever you change these settings.")

# ------------ apply rules (always) ------------

work = df.copy()
rule_flags: Dict[str, pd.Series] = {}

# Rule 1: High Amount
if enable_amount and amount_col and high_amt_threshold is not None:
    rule_name = "high_amount"
    rule_flags[rule_name] = work[amount_col] >= high_amt_threshold
else:
    rule_flags["high_amount"] = pd.Series(False, index=work.index)

# Rule 2: Risky Country
if enable_country and country_col and risky_countries:
    rule_name = "risky_country"
    rule_flags[rule_name] = work[country_col].astype(str).isin(risky_countries)
else:
    rule_flags["risky_country"] = pd.Series(False, index=work.index)

# Rule 3: High Activity Customer
if enable_activity and customer_col:
    rule_name = "high_activity"
    cust_stats = work.groupby(customer_col).size().to_frame("tx_count")
    if amount_col:
        cust_stats["total_amount"] = work.groupby(customer_col)[amount_col].sum()
    else:
        cust_stats["total_amount"] = 0.0

    cust_stats["flag"] = (
        (cust_stats["tx_count"] >= (min_tx_per_customer or 0))
        & (cust_stats["total_amount"] >= (min_total_amount_per_customer or 0.0))
    )

    # Map back to rows
    cust_flag = work[customer_col].map(
        cust_stats["flag"].to_dict()
    ).fillna(False)
    rule_flags[rule_name] = cust_flag.astype(bool)
else:
    rule_flags["high_activity"] = pd.Series(False, index=work.index)

# Combine into dataframe
for rn, ser in rule_flags.items():
    work[f"rule_{rn}"] = ser.astype(bool)

rule_cols = [c for c in work.columns if c.startswith("rule_")]
if not rule_cols:
    st.warning("No active rules; nothing to evaluate.")
    st.stop()

work["rule_any"] = work[rule_cols].any(axis=1)

flagged_df = work[work["rule_any"]].copy()
n_flagged = len(flagged_df)

# ------------ KPIs ------------

st.subheader("Rule Engine Summary")

fraud_flagged = None
fraud_rate_flagged = None
fraud_rate_overall = None

if label_col:
    fraud_flagged = int(flagged_df[label_col].sum())
    fraud_rate_flagged = fraud_flagged / max(1, n_flagged) * 100.0 if n_flagged else 0.0
    fraud_total = int(df[label_col].sum())
    fraud_rate_overall = fraud_total / max(1, n_rows) * 100.0

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Transactions", f"{n_rows:,}")
with c2:
    st.metric("Flagged by Rules", f"{n_flagged:,}")
with c3:
    st.metric(
        "Flag Rate",
        f"{(n_flagged / max(1, n_rows) * 100.0):.2f}%",
    )
with c4:
    if fraud_rate_flagged is not None and fraud_rate_overall is not None:
        st.metric(
            "Fraud Rate (Flagged vs Overall)",
            f"{fraud_rate_flagged:.2f}% vs {fraud_rate_overall:.2f}%",
        )
    else:
        st.metric("Fraud Rate (Flagged)", "Unknown (no labels)")

st.markdown("---")

# ------------ per-rule breakdown ------------

st.subheader("Per-Rule Breakdown")

rule_counts = []
for rc in rule_cols:
    name = rc.replace("rule_", "")
    mask = work[rc]
    count = int(mask.sum())
    entry: Dict[str, object] = {
        "rule": name,
        "flagged": count,
        "flag_rate": count / max(1, n_rows),
    }
    if label_col:
        fraud_c = int(work.loc[mask, label_col].sum())
        entry["fraud_flagged"] = fraud_c
        entry["fraud_rate_in_rule"] = fraud_c / max(1, count)
    rule_counts.append(entry)

rules_df = pd.DataFrame(rule_counts)

if not rules_df.empty:
    display_df = rules_df.copy()
    for col in ["flag_rate", "fraud_rate_in_rule"]:
        if col in display_df.columns:
            display_df[col] = col and display_df[col].apply(
                lambda v: f"{v * 100:.2f}%" if isinstance(v, (float, int)) else "N/A"
            )
    st.dataframe(display_df, use_container_width=True)

    # Simple bar chart: flagged by rule
    fig_rules = px.bar(
        rules_df,
        x="rule",
        y="flagged",
        title="Transactions Flagged per Rule",
        labels={"flagged": "Flagged Count", "rule": "Rule"},
    )
    st.plotly_chart(fig_rules, use_container_width=True)
else:
    st.info("No rules flagged any transactions with the current configuration.")

st.markdown("---")

# ------------ flagged sample table ------------

st.subheader("Flagged Transactions (Sample)")

if n_flagged == 0:
    st.info("No transactions were flagged by the current rule configuration.")
else:
    # Order columns: rules + label + score + important features first
    important_cols: List[str] = []
    important_cols.extend(rule_cols)
    if "rule_any" in flagged_df.columns:
        important_cols.append("rule_any")
    if label_col:
        important_cols.append(label_col)
    if score_col:
        important_cols.append(score_col)
    if amount_col:
        important_cols.append(amount_col)
    if country_col:
        important_cols.append(country_col)
    if customer_col:
        important_cols.append(customer_col)

    # Ensure uniqueness / valid cols
    important_cols = [c for c in dict.fromkeys(important_cols) if c in flagged_df.columns]
    other_cols = [c for c in flagged_df.columns if c not in important_cols]
    ordered_cols = important_cols + other_cols

    st.dataframe(
        flagged_df[ordered_cols].head(500),
        use_container_width=True,
    )

    csv_bytes = flagged_df[ordered_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download all flagged transactions as CSV",
        data=csv_bytes,
        file_name="rule_engine_flagged_transactions.csv",
        mime="text/csv",
    )
