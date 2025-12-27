"""Add fraud_rows_written and last_updated_at to scoring_jobs."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_scoring_job_partial_fraud"
down_revision = "0004_cases_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("scoring_jobs", sa.Column("fraud_rows_written", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("scoring_jobs", sa.Column("last_updated_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("scoring_jobs", "last_updated_at")
    op.drop_column("scoring_jobs", "fraud_rows_written")
