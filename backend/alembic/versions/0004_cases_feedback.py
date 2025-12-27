"""Add case_items, feedback_labels, and created_by on cases."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0004_cases_feedback"
down_revision = "0003_dataset_schema_mapping"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("cases", sa.Column("created_by", sa.String(), nullable=True))

    op.create_table(
        "case_items",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("case_id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=True),
        sa.Column("tx_id", sa.String(), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=True),
        sa.Column("payload_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["case_id"], ["cases.id"]),
    )
    op.create_index("ix_case_items_case_id", "case_items", ["case_id"])
    op.create_index("ix_case_items_job_id", "case_items", ["job_id"])
    op.create_index("ix_case_items_tx_id", "case_items", ["tx_id"])

    op.create_table(
        "feedback_labels",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("tx_id", sa.String(), nullable=False),
        sa.Column("job_id", sa.String(), nullable=True),
        sa.Column("dataset_version_id", sa.String(), nullable=True),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_feedback_labels_tx_id", "feedback_labels", ["tx_id"])
    op.create_index("ix_feedback_labels_job_id", "feedback_labels", ["job_id"])
    op.create_index("ix_feedback_labels_dataset_version_id", "feedback_labels", ["dataset_version_id"])


def downgrade() -> None:
    op.drop_index("ix_feedback_labels_dataset_version_id", table_name="feedback_labels")
    op.drop_index("ix_feedback_labels_job_id", table_name="feedback_labels")
    op.drop_index("ix_feedback_labels_tx_id", table_name="feedback_labels")
    op.drop_table("feedback_labels")

    op.drop_index("ix_case_items_tx_id", table_name="case_items")
    op.drop_index("ix_case_items_job_id", table_name="case_items")
    op.drop_index("ix_case_items_case_id", table_name="case_items")
    op.drop_table("case_items")

    op.drop_column("cases", "created_by")
