"""Expose page renderers."""

from pathlib import Path
import sys

# Ensure project root is importable when pages are imported directly
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dashboard.pages import dashboard_home
from dashboard.pages import real_time_stream
from dashboard.pages import live_graphs
from dashboard.pages import advanced_tx_analysis
from dashboard.pages import threshold_tuning
from dashboard.pages import model_health_drift
from dashboard.pages import data_explorer
from dashboard.pages import ensemble_status
from dashboard.pages import global_network
from dashboard.pages import fraud_rings
from dashboard.pages import fraud_investigation
from dashboard.pages import rule_engine
from dashboard.pages import audit_log
from dashboard.pages import what_if_simulator
from dashboard.pages import phase2_plan

__all__ = [
    "dashboard_home",
    "real_time_stream",
    "live_graphs",
    "advanced_tx_analysis",
    "threshold_tuning",
    "model_health_drift",
    "data_explorer",
    "ensemble_status",
    "global_network",
    "fraud_rings",
    "fraud_investigation",
    "rule_engine",
    "audit_log",
    "what_if_simulator",
    "phase2_plan",
]
