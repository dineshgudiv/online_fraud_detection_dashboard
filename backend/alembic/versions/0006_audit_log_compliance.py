"""Add compliance audit fields.

Revision ID: 0006_audit_log_compliance
Revises: 0005_scoring_job_partial_fraud
Create Date: 2025-01-05 00:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006_audit_log_compliance"
down_revision = "0005_scoring_job_partial_fraud"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("audit_log", sa.Column("user_id", sa.String(), nullable=True))
    op.add_column("audit_log", sa.Column("role", sa.String(), nullable=True))
    op.add_column("audit_log", sa.Column("resource_type", sa.String(), nullable=True))
    op.add_column("audit_log", sa.Column("resource_id", sa.String(), nullable=True))
    op.add_column("audit_log", sa.Column("ip", sa.String(), nullable=True))
    op.add_column("audit_log", sa.Column("user_agent", sa.String(), nullable=True))
    op.add_column("audit_log", sa.Column("correlation_id", sa.String(), nullable=True))
    op.add_column(
        "audit_log",
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )

    op.alter_column("audit_log", "transaction_id", existing_type=sa.String(), nullable=True)
    op.alter_column("audit_log", "model_name", existing_type=sa.String(), nullable=True)
    op.alter_column("audit_log", "model_version", existing_type=sa.String(), nullable=True)
    op.alter_column("audit_log", "score", existing_type=sa.Float(), nullable=True)
    op.alter_column("audit_log", "decision", existing_type=sa.String(), nullable=True)

    op.create_index("ix_audit_log_action", "audit_log", ["action"])
    op.create_index("ix_audit_log_resource_type", "audit_log", ["resource_type"])
    op.create_index("ix_audit_log_correlation_id", "audit_log", ["correlation_id"])


def downgrade() -> None:
    op.drop_index("ix_audit_log_correlation_id", table_name="audit_log")
    op.drop_index("ix_audit_log_resource_type", table_name="audit_log")
    op.drop_index("ix_audit_log_action", table_name="audit_log")

    op.alter_column("audit_log", "decision", existing_type=sa.String(), nullable=False)
    op.alter_column("audit_log", "score", existing_type=sa.Float(), nullable=False)
    op.alter_column("audit_log", "model_version", existing_type=sa.String(), nullable=False)
    op.alter_column("audit_log", "model_name", existing_type=sa.String(), nullable=False)
    op.alter_column("audit_log", "transaction_id", existing_type=sa.String(), nullable=False)

    op.drop_column("audit_log", "metadata_json")
    op.drop_column("audit_log", "correlation_id")
    op.drop_column("audit_log", "user_agent")
    op.drop_column("audit_log", "ip")
    op.drop_column("audit_log", "resource_id")
    op.drop_column("audit_log", "resource_type")
    op.drop_column("audit_log", "role")
    op.drop_column("audit_log", "user_id")
