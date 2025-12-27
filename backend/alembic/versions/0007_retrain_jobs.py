"""Add retrain jobs table.

Revision ID: 0007_retrain_jobs
Revises: 0006_audit_log_compliance
Create Date: 2025-01-05 00:00:01.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_retrain_jobs"
down_revision = "0006_audit_log_compliance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "retrain_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("requested_by", sa.String(), nullable=True),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("model_name", sa.String(), nullable=True),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("rq_job_id", sa.String(), nullable=True),
    )
    op.create_index("ix_retrain_jobs_status", "retrain_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_retrain_jobs_status", table_name="retrain_jobs")
    op.drop_table("retrain_jobs")
