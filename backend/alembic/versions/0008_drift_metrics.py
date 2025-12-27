"""Add drift metrics table.

Revision ID: 0008_drift_metrics
Revises: 0007_retrain_jobs
Create Date: 2025-01-05 00:00:02.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_drift_metrics"
down_revision = "0007_retrain_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "drift_metrics",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("feature", sa.String(), nullable=False),
        sa.Column("psi", sa.Float(), nullable=False),
        sa.Column("ks_pvalue", sa.Float(), nullable=True),
        sa.Column("overall_score", sa.Float(), nullable=False),
    )
    op.create_index("ix_drift_metrics_timestamp", "drift_metrics", ["timestamp"])
    op.create_index("ix_drift_metrics_model_version", "drift_metrics", ["model_version"])
    op.create_index("ix_drift_metrics_feature", "drift_metrics", ["feature"])


def downgrade() -> None:
    op.drop_index("ix_drift_metrics_feature", table_name="drift_metrics")
    op.drop_index("ix_drift_metrics_model_version", table_name="drift_metrics")
    op.drop_index("ix_drift_metrics_timestamp", table_name="drift_metrics")
    op.drop_table("drift_metrics")
