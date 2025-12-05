"""Styling helpers for Streamlit UI."""

import streamlit as st


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        .main { background: #ffffff; }
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
        .kpi-card {
            border-radius: 14px;
            padding: 18px 18px 14px 18px;
            color: #0f172a;
            background: linear-gradient(135deg, #eef2ff, #e0f2fe);
            border: 1px solid #e2e8f0;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
        }
        .kpi-card h3 { margin: 0; font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.5px; }
        .kpi-card .value { font-size: 1.8rem; font-weight: 700; margin-top: 6px; }
        .status-pill { padding: 4px 10px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; color: #0f172a; display: inline-block; }
        .status-ok { background: #dcfce7; color: #14532d; }
        .status-warning { background: #fef9c3; color: #92400e; }
        .status-drifted { background: #fee2e2; color: #991b1b; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, help_text: str | None = None, emoji: str | None = None) -> None:
    icon = f"{emoji} " if emoji else ""
    help_html = f'<p class="kpi-help">{help_text}</p>' if help_text else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <h3>{icon}{label}</h3>
            <div class="value">{value}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def two_column_kpi_row(left: tuple[str, str], right: tuple[str, str]) -> None:
    col1, col2 = st.columns(2)
    with col1:
        kpi_card(left[0], left[1])
    with col2:
        kpi_card(right[0], right[1])

