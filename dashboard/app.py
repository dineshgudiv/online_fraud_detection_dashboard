import os
import sys
from pathlib import Path

import httpx

# Raise Streamlit upload limit to ~7 GB for large CSV batch scoring
os.environ.setdefault("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", "7000")
import streamlit as st

# Ensure project root (parent of dashboard/) is on sys.path for shared modules
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.styling import inject_global_css
from dashboard.pages import (
    dashboard_home,
    real_time_stream,
    live_graphs,
    advanced_tx_analysis,
    threshold_tuning,
    model_health_drift,
    data_explorer,
    ensemble_status,
    global_fraud_network,
    global_network,
    fraud_pipeline_3d,
    fraud_rings,
    fraud_investigation,
    rule_engine,
    audit_log,
    what_if_simulator,
    phase2_plan,
    case_management,
    batch_scoring,
    executive_report,
)


API_BASE = os.getenv("FRAUD_API_BASE", "http://127.0.0.1:8000")

PAGES = {
    # Core monitoring
    "Dashboard": dashboard_home,
    "Real-time Stream": real_time_stream,
    "Live Graphs": live_graphs,
    # Analysis & scoring
    "Advanced Transaction Analysis": advanced_tx_analysis,
    "Batch Scoring": batch_scoring,
    "Data Explorer": data_explorer,
    # Investigation
    "Fraud Investigation": fraud_investigation,
    "Fraud Detection Pipeline": fraud_pipeline_3d,
    "Fraud Rings": fraud_rings,
    # Global views
    "Enhanced Global Fraud Network": global_fraud_network,
    "Global Network": global_network,
    # Models & rules
    "Ensemble Status": ensemble_status,
    "Model Health & Drift": model_health_drift,
    "Threshold & Mode Tuning": threshold_tuning,
    "Rule Engine": rule_engine,
    # Reporting & ops
    "Audit Log": audit_log,
    "Executive Report": executive_report,
    "What-If Simulator": what_if_simulator,
    "Case Management": case_management,
    "Phase 2 Plan": phase2_plan,
}


class ApiClient:
    """Synchronous API client for FastAPI backend."""

    def __init__(self, base_url: str = API_BASE, timeout_seconds: float = 120.0) -> None:
        # Increased timeout to handle large uploads/batch scoring
        self._client = httpx.Client(base_url=base_url, timeout=timeout_seconds)

    def get(self, path: str, params: dict | None = None) -> dict:
        resp = self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    def post(self, path: str, json: dict | None = None, files: dict | None = None) -> dict:
        resp = self._client.post(path, json=json, files=files)
        resp.raise_for_status()
        return resp.json()

    def patch(self, path: str, json: dict | None = None) -> dict:
        resp = self._client.patch(path, json=json)
        resp.raise_for_status()
        return resp.json()

    def get_bytes(self, path: str) -> bytes:
        resp = self._client.get(path)
        resp.raise_for_status()
        return resp.content

    def raw_get(self, path: str):
        """Return raw response for streaming downloads."""
        resp = self._client.get(path)
        resp.raise_for_status()
        return resp

    def close(self) -> None:
        self._client.close()


def run_dashboard():
    st.set_page_config(page_title="Online Fraud Detection Dashboard", layout="wide", initial_sidebar_state="expanded")
    inject_global_css()

    client = ApiClient()

    with st.sidebar:
        st.title("Control Panel")
        st.markdown("### Navigation")
        page_name = st.radio(
            "Navigation",
            options=list(PAGES.keys()),
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### System Status")
        try:
            health = client.get("/health")
        except Exception:
            health = None
        if health and isinstance(health, dict) and health.get("status") == "ok":
            st.success("🟢 API Online")
        else:
            st.error("🔴 API Offline")

        st.markdown("### Quick Actions")
        if st.button("Refresh Data"):
            st.toast("Data refresh triggered (simulated)", icon="🔄")
        if st.button("Generate Report"):
            st.info("Report generation queued (demo).")

    selected_module = PAGES.get(page_name)
    if selected_module is not None:
        selected_module.render_page(client, health) if page_name in {"Dashboard", "Real-time Stream"} else selected_module.render_page(client)


if __name__ == "__main__":
    run_dashboard()
