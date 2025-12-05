import numpy as np
import plotly.graph_objects as go
import streamlit as st


def _pipeline_nodes() -> dict[int, tuple[str, float, float]]:
    return {
        0: ("Ingestion", 0.0, 0.0),
        1: ("Validation & Cleaning", 1.5, 0.0),
        2: ("Feature Store", 3.0, 0.0),
        3: ("Models (IForest/RF/LR)", 4.5, 0.0),
        4: ("Rules", 6.0, 0.0),
        5: ("Decision", 7.5, 0.0),
        6: ("Queue / Approve", 9.0, 0.0),
    }


def _build_3d_pipeline_figure() -> go.Figure:
    nodes = _pipeline_nodes()
    xs, ys, zs, labels = [], [], [], []
    for _, (label, x, y) in nodes.items():
        xs.append(x)
        ys.append(y)
        zs.append(0.2 + 0.05 * np.random.rand())
        labels.append(label)

    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            marker=dict(size=10, color="#4f46e5"),
            text=[str(i + 1) for i in range(len(xs))],
            textposition="top center",
            hovertext=labels,
            hovertemplate="%{hovertext}<extra></extra>",
        )
    )
    for i, j in edges:
        fig.add_trace(
            go.Scatter3d(
                x=[xs[i], xs[j]],
                y=[ys[i], ys[j]],
                z=[zs[i], zs[j]],
                mode="lines",
                line=dict(width=4, color="rgba(100,116,139,0.8)"),
                hoverinfo="skip",
            )
        )

    frames = []
    for angle in range(0, 360, 18):
        rad = np.radians(angle)
        eye = dict(x=3.5 * np.cos(rad), y=3.5 * np.sin(rad), z=1.5)
        frames.append(go.Frame(layout=dict(scene_camera=dict(eye=eye))))
    fig.frames = frames

    fig.update_layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        scene_aspectmode="data",
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Rotate",
                        method="animate",
                        args=[None, {"frame": {"duration": 80, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}],
                    )
                ],
                x=0.0,
                y=0.0,
            )
        ],
    )
    fig.update_layout(scene_camera=dict(eye=dict(x=3.0, y=0.6, z=1.3)), height=520)
    return fig


def render_page(client=None, health=None):
    st.title("Fraud Detection Pipeline (3D)")
    st.caption("End-to-end flow from ingestion to decision. Demo visualization with rotating 3D nodes.")

    st.plotly_chart(_build_3d_pipeline_figure(), use_container_width=True)

    st.subheader("Stage Details")
    st.markdown(
        """
- **Ingestion**: APIs/Kafka ingest transactions.
- **Validation**: Schema/type checks; drop malformed records.
- **Feature Store**: Velocity, geo, device, behavioral signals.
- **Models**: IsolationForest, RandomForest, LogisticRegression (demo ensemble).
- **Rules**: Blacklists, geo/velocity thresholds, device/IP reputation.
- **Decision**: Combine model + rules → APPROVE / REVIEW / REJECT.
- **Queue/Approve**: Manual review for high-risk; feedback loop.
"""
    )


if __name__ == "__main__":
    render_page()
