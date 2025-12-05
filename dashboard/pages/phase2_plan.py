from __future__ import annotations

from typing import List, Dict

import pandas as pd
import streamlit as st


st.title("Phase-2 Upgrade Plan")
st.caption(
    "Side-by-side view of Phase-1 vs Phase-2, plus a tracker for all 22 pages."
)

# ---------------------------------------------------------------------------
# 1) Page-by-page tracker (you can edit this list later if you rename pages)
# ---------------------------------------------------------------------------

# Status values used:
# - "Phase 2 ready"  → code upgraded with shared.data_loader, robust layout, charts, etc.
# - "In progress"    → partially upgraded or needs a second pass.
# - "Planned"        → still Phase-1 style or not started.

pages_plan: List[Dict[str, str]] = [
    {
        "order": 1,
        "page": "App Shell / Menu",
        "phase1": "Single script feeling; manual commands and ports",
        "phase2": "run_all.ps1 + multi-page sidebar, backend + dashboard launched together",
        "status": "In progress",
    },
    {
        "order": 2,
        "page": "Advanced Tx Analysis",
        "phase1": "Basic filters / table only",
        "phase2": "Rich filters + amount/time/type charts",
        "status": "Phase 2 ready",
    },
    {
        "order": 3,
        "page": "Audit Log",
        "phase1": "Raw table of events",
        "phase2": "Time filters, class filters, search box, CSV export",
        "status": "Phase 2 ready",
    },
    {
        "order": 4,
        "page": "Batch Scoring",
        "phase1": "No clear UI for scoring CSVs",
        "phase2": "Upload CSV or sample, synthetic risk_score, threshold + scored download",
        "status": "Phase 2 ready",
    },
    {
        "order": 5,
        "page": "Case Management",
        "phase1": "No case grouping",
        "phase2": "Group by customer/merchant/device, suspicious cases, drill-down view",
        "status": "Phase 2 ready",
    },
    {
        "order": 6,
        "page": "Dashboard Home",
        "phase1": "Simple charts / counts",
        "phase2": "Executive KPIs + Overview / Trends / Segments tabs",
        "status": "Phase 2 ready",
    },
    {
        "order": 7,
        "page": "Data Explorer",
        "phase1": "Static CSV view",
        "phase2": "Generic explorer with filters, histograms and CSV export",
        "status": "Phase 2 ready",
    },
    {
        "order": 8,
        "page": "Ensemble Status",
        "phase1": "Scattered model metrics (if any)",
        "phase2": "Per-model summary, AUC (if sklearn), threshold metrics, score distributions",
        "status": "Phase 2 ready",
    },
    {
        "order": 9,
        "page": "Executive Report",
        "phase1": "No CxO-level summary",
        "phase2": "KPI row, narrative text, trends and high-risk segments",
        "status": "Phase 2 ready",
    },
    {
        "order": 10,
        "page": "Fraud Investigation",
        "phase1": "Manual search in tables",
        "phase2": "Search by TX / customer / merchant / device + related activity timeline",
        "status": "Phase 2 ready",
    },
    {
        "order": 11,
        "page": "Fraud Pipeline 3D",
        "phase1": "Static / unclear view",
        "phase2": "Dedicated pipeline page (Phase-2 layout ready)",
        "status": "Phase 2 ready",
    },
    {
        "order": 12,
        "page": "Fraud Rings",
        "phase1": "No ring view",
        "phase2": "Ring hubs, neighbors, star graph, CSV exports",
        "status": "Phase 2 ready",
    },
    {
        "order": 13,
        "page": "Global Fraud Network",
        "phase1": "Basic map / graph",
        "phase2": "Upgraded Phase-2 style network (separate page)",
        "status": "Phase 2 ready",
    },
    {
        "order": 14,
        "page": "Global Network",
        "phase1": "Legacy code (Series.append etc.)",
        "phase2": "Uses concat + Phase-2 layout",
        "status": "Phase 2 ready",
    },
    {
        "order": 15,
        "page": "Live Graphs",
        "phase1": "Tight coupling to main app, brittle",
        "phase2": "Decoupled via shared.data_loader + fixes for pandas changes",
        "status": "Phase 2 ready",
    },
    {
        "order": 16,
        "page": "Model Health Drift",
        "phase1": "No drift metrics",
        "phase2": "Reference vs current split, per-feature drift, distributions",
        "status": "Phase 2 ready",
    },
    {
        "order": 17,
        "page": "Phase-2 Plan",
        "phase1": "Not available",
        "phase2": "This meta page tracking upgrades + roadmap",
        "status": "Phase 2 ready",
    },
    {
        "order": 18,
        "page": "Project Report",
        "phase1": "Outside dashboard",
        "phase2": "TODO: embed short project summary / links here",
        "status": "Planned",
    },
    {
        "order": 19,
        "page": "Real Time Stream",
        "phase1": "Static charts only",
        "phase2": "Rolling window, batch ticks, auto-refresh, streaming KPIs",
        "status": "Phase 2 ready",
    },
    {
        "order": 20,
        "page": "Rule Engine",
        "phase1": "Hard-coded or missing",
        "phase2": "Configurable high-amount and high-risk country rules + CSV export",
        "status": "Phase 2 ready",
    },
    {
        "order": 21,
        "page": "Threshold Tuning",
        "phase1": "No threshold UI",
        "phase2": "Threshold slider, precision/recall curves and score distributions",
        "status": "Phase 2 ready",
    },
    {
        "order": 22,
        "page": "What-If Simulator",
        "phase1": "No what-if playground",
        "phase2": "Scenario comparison page with fraud score deltas",
        "status": "Phase 2 ready",
    },
]

plan_df = pd.DataFrame(pages_plan).sort_values("order").reset_index(drop=True)

total_pages = len(plan_df)
phase2_ready = int((plan_df["status"] == "Phase 2 ready").sum())
in_progress = int((plan_df["status"] == "In progress").sum())
planned = int((plan_df["status"] == "Planned").sum())

# ---------------------------------------------------------------------------
# 2) Top summary metrics
# ---------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Pages", str(total_pages))
with c2:
    st.metric("Phase-2 Ready", f"{phase2_ready} / {total_pages}")
with c3:
    st.metric("In Progress", str(in_progress))
with c4:
    st.metric("Planned / TODO", str(planned))

st.markdown("---")

# ---------------------------------------------------------------------------
# 3) Tabs: Architecture vs Tracker vs Checklist
# ---------------------------------------------------------------------------

tab_arch, tab_tracker, tab_notes = st.tabs(
    ["Phase-1 vs Phase-2 Architecture", "Page Tracker", "Checklist / Next Steps"]
)

# ---- Tab 1: Architecture comparison ----
with tab_arch:
    st.subheader("Architecture: Phase-1 vs Phase-2")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Phase-1 (Baseline)")
        st.markdown(
            """
- Single or few Streamlit scripts tied together manually.
- CSVs loaded directly inside pages; no shared loader.
- Backend (FastAPI) and dashboard started with separate manual commands.
- Limited error handling (missing files, backend down → app crashes).
- Pages not clearly separated by role (analyst vs exec vs ML).
- Minimal navigation and no clear enterprise story.
            """
        )

    with col_right:
        st.markdown("### Phase-2 (Enterprise Upgrade)")
        st.markdown(
            """
- **Multi-page Streamlit app** with a clear enterprise menu (22 pages).
- Shared **`shared.data_loader.load_dataset()`** so all pages load data in one consistent way.
- **`run_all.ps1`** script:
  - Creates/activates `.venv`
  - Installs `requirements.txt` on first run
  - Starts **FastAPI backend (port 8000)** and **dashboard (port 8502)** together.
- Pages are split by responsibility:
  - Ops: *Real Time Stream, Live Graphs, Model Health Drift*
  - Analysts: *Fraud Investigation, Case Management, Fraud Rings, Rule Engine*
  - Exec: *Dashboard Home, Executive Report, Phase-2 Plan*
- More robust behaviour:
  - Graceful fallbacks when dataset is missing or empty.
  - No `st.set_page_config` conflicts across pages.
  - Modern `pandas` patterns (e.g. `pd.concat` instead of deprecated `Series.append`).
            """
        )

    st.markdown("### Visual Status")

    status_counts = (
        plan_df["status"]
        .value_counts()
        .reindex(["Phase 2 ready", "In progress", "Planned"])
        .fillna(0)
        .rename_axis("status")
        .reset_index(name="count")
    )
    st.bar_chart(
        status_counts.set_index("status")["count"],
        use_container_width=True,
    )

# ---- Tab 2: Page tracker table ----
with tab_tracker:
    st.subheader("Page-by-Page Status")

    st.caption(
        "You can adjust this list directly in `phase2_plan.py` later if you rename pages "
        "or change their status."
    )

    st.dataframe(
        plan_df[["order", "page", "phase1", "phase2", "status"]],
        use_container_width=True,
        hide_index=True,
    )

# ---- Tab 3: Checklist / next steps ----
with tab_notes:
    st.subheader("Checklist for Completing Phase-2")

    st.markdown(
        """
**Short-term TODOs**

- [ ] Finalise *App Shell / Menu* (ensure ordering of 22 pages matches your plan).
- [ ] Implement **Project Report** page:
  - High-level project write-up
  - Architecture diagram screenshot
  - Links to GitHub repo and IEEE paper draft (optional)
- [ ] Confirm every page:
  - Uses `shared.data_loader.load_dataset()` (or a shared API client)  
  - Has helpful error messages instead of crashing
  - Avoids duplicate `st.set_page_config` calls

**Optional Phase-3 ideas**

- [ ] Add authentication / roles (analyst vs admin vs exec).
- [ ] Connect to a real message queue / streaming source instead of simulated batches.
- [ ] Wire real model endpoints (`/score/transaction`, `/simulate/what_if`, etc.) in backend.
- [ ] Add PDF/PNG screenshots of the pipeline and dashboards into “Project Report”.
        """
    )

    st.info(
        "This page is just a planning tool inside the dashboard. "
        "You can keep editing the lists/statuses as your project evolves."
    )
