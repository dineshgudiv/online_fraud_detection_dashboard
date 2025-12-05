from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import plotly.express as px
import streamlit as st


def _decision_from_score(score: float) -> Dict[str, str]:
    """
    Map a numeric fraud_score in [0,1] to decision + risk_level.
    """
    if score >= 0.8:
        return {"decision": "decline", "risk_level": "high"}
    if score >= 0.5:
        return {"decision": "review", "risk_level": "medium"}
    return {"decision": "approve", "risk_level": "low"}


def _synthetic_score(amount: float, risk_boost: float = 0.0) -> float:
    """
    Simple deterministic score using amount + optional boost.
    Keeps value in [0, 1].
    """
    # Non-linear squash of amount
    score = amount / (amount + 500.0)
    score = min(max(score + risk_boost, 0.0), 1.0)
    return float(score)


def _build_synthetic_result(
    base_payload: Dict[str, Any],
    variations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Local fallback result if backend /simulate/what_if is unavailable.
    Uses only amount + simple heuristics so the page always works.
    """
    base_amount = float(base_payload.get("amount", 0.0))
    base_score = _synthetic_score(base_amount)
    base_meta = _decision_from_score(base_score)

    scenarios: List[Dict[str, Any]] = []

    # Variation 1 – amount + location
    if len(variations) >= 1:
        v1 = variations[0]
        v1_amount = float(v1.get("amount", base_amount))
        v1_location = str(v1.get("location", base_payload.get("location", ""))).upper()

        boost = 0.0
        if v1_amount > base_amount:
            boost += 0.1
        # Treat non-home countries as slightly riskier
        home_loc = str(base_payload.get("location", "")).upper()
        if home_loc and v1_location and v1_location != home_loc:
            boost += 0.1

        v1_score = _synthetic_score(v1_amount, boost)
        v1_meta = _decision_from_score(v1_score)
        scenarios.append(
            {
                "label": "Variation 1",
                "fraud_score": v1_score,
                **v1_meta,
            }
        )

    # Variation 2 – category + device
    if len(variations) >= 2:
        v2 = variations[1]
        v2_category = str(v2.get("category", base_payload.get("category", ""))).lower()
        v2_device = str(v2.get("device_id", base_payload.get("device_id", "")))

        boost = 0.0
        if any(k in v2_category for k in ("crypto", "gaming")):
            boost += 0.15
        if v2_device and v2_device != base_payload.get("device_id"):
            boost += 0.1

        v2_score = _synthetic_score(base_amount, boost)
        v2_meta = _decision_from_score(v2_score)
        scenarios.append(
            {
                "label": "Variation 2",
                "fraud_score": v2_score,
                **v2_meta,
            }
        )

    return {
        "mode": "synthetic",
        "base_result": {
            "fraud_score": base_score,
            **base_meta,
        },
        "scenarios": scenarios,
    }


def _call_backend_or_fallback(
    client: Any,
    base_payload: Dict[str, Any],
    variations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Try backend /simulate/what_if; fall back to local synthetic simulation
    if backend is unavailable or returns an unexpected shape.
    """
    if client is None:
        return _build_synthetic_result(base_payload, variations)

    try:
        resp = client.post(
            "/simulate/what_if",
            json={"base": base_payload, "variations": variations},
        )

        # Support both raw dict and Response-like objects
        if hasattr(resp, "json"):
            data = resp.json()
        else:
            data = resp

        if not isinstance(data, dict):
            raise ValueError("Unexpected response type")

        # Ensure keys exist, otherwise treat as bad response and fall back
        if "base_result" not in data or "scenarios" not in data:
            raise ValueError("Missing keys in response")

        return {"mode": "backend", **data}
    except Exception as exc:
        st.warning(
            f"Backend what-if simulation failed, using local demo instead: {exc}"
        )
        return _build_synthetic_result(base_payload, variations)


def render_page(client):
    """
    What-If Simulator page.

    Parameters
    ----------
    client : object
        API client with a .post(path, json=...) method, or None.
    """
    st.title("What-If Simulator")
    st.caption("Experiment with variations of a transaction to see how risk shifts.")

    # Ensure session state key exists
    if "what_if_result" not in st.session_state:
        st.session_state["what_if_result"] = None

    col_base, col_var = st.columns(2)

    with col_base:
        st.subheader("Base Transaction")
        user_id = st.text_input("User ID", "U1234")
        merchant_id = st.text_input("Merchant ID", "M5678")
        amount = st.number_input("Amount", min_value=0.0, value=250.0, step=10.0)
        currency = st.selectbox(
            "Currency", ["USD", "EUR", "GBP", "JPY", "INR"], index=0
        )
        transaction_type = st.selectbox(
            "Transaction Type", ["purchase", "withdrawal", "transfer", "refund"]
        )
        category = st.selectbox(
            "Category",
            ["electronics", "fashion", "groceries", "gaming", "travel", "crypto"],
        )
        location = st.selectbox(
            "Location",
            ["US", "GB", "DE", "FR", "IN", "CN", "SG", "BR", "ZA", "AU"],
        )
        device_id = st.text_input("Device ID", "D9001")

    with col_var:
        st.subheader("Variations")
        var1_amount = st.number_input(
            "Variation 1 Amount", min_value=0.0, value=500.0, step=10.0
        )
        var1_location = st.text_input("Variation 1 Location", "GB")
        var2_category = st.text_input("Variation 2 Category", "crypto")
        var2_device = st.text_input("Variation 2 Device", "D7777")

    run = st.button("Run What-If Analysis", use_container_width=True)

    if run:
        base_payload = {
            "user_id": user_id,
            "merchant_id": merchant_id,
            "amount": amount,
            "currency": currency,
            "transaction_type": transaction_type,
            "category": category,
            "location": location,
            "device_id": device_id,
        }
        variations = [
            {"amount": var1_amount, "location": var1_location},
            {"category": var2_category, "device_id": var2_device},
        ]

        result = _call_backend_or_fallback(client, base_payload, variations)
        st.session_state["what_if_result"] = result

    result = st.session_state.get("what_if_result")

    if not result:
        return

    base = result.get("base_result", {}) or {}
    scenarios = result.get("scenarios", []) or []
    mode = result.get("mode", "unknown")

    st.subheader("Results")

    cols = st.columns(max(1, len(scenarios) + 1))
    with cols[0]:
        st.markdown(
            f"**Base**  \n"
            f"Fraud Score: `{base.get('fraud_score', 0):.3f}`  \n"
            f"Decision: **{base.get('decision', 'n/a')}**  \n"
            f"Risk: **{base.get('risk_level', 'n/a')}**"
        )

    for idx, sc in enumerate(scenarios, start=1):
        with cols[idx]:
            st.markdown(
                f"**{sc.get('label', f'S{idx}') }**  \n"
                f"Fraud Score: `{sc.get('fraud_score', 0):.3f}`  \n"
                f"Decision: **{sc.get('decision', 'n/a')}**  \n"
                f"Risk: **{sc.get('risk_level', 'n/a')}**"
            )

    # Bar chart comparison
    labels = ["Base"] + [s.get("label", f"S{i+1}") for i, s in enumerate(scenarios)]
    scores = [base.get("fraud_score", 0.0)] + [
        float(s.get("fraud_score", 0.0)) for s in scenarios
    ]

    fig = px.bar(
        x=labels,
        y=scores,
        labels={"x": "Scenario", "y": "Fraud Score"},
        title="Fraud Score Comparison Across Scenarios",
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(scores) > 1:
        deltas = [f"{labels[i]}: {scores[i] - scores[0]:+.3f}" for i in range(1, len(scores))]
        st.info("Score deltas vs. base: " + "; ".join(deltas))

    st.caption(
        "Source: "
        + ("backend /simulate/what_if" if mode == "backend" else "local synthetic demo")
    )


if __name__ == "__main__":
    # Standalone testing: run this page by itself.
    try:
        from dashboard.app import ApiClient  # type: ignore[attr-defined]
    except Exception:
        api_client = None
    else:
        api_client = ApiClient()

    try:
        render_page(api_client)
    finally:
        if api_client is not None and hasattr(api_client, "close"):
            api_client.close()
