"""Add dataset_versions and scoring_jobs tables."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002_datasets_jobs"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_versions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("original_filename", sa.String(), nullable=False),
        sa.Column("stored_path", sa.String(), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("schema_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.create_index("ix_dataset_versions_is_active", "dataset_versions", ["is_active"])
    op.create_index("ix_dataset_versions_stored_path", "dataset_versions", ["stored_path"])

    op.create_table(
        "scoring_jobs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("dataset_version_id", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=True),
        sa.Column("rows_total", sa.Integer(), nullable=True),
        sa.Column("rows_done", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("output_path", sa.String(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["dataset_version_id"], ["dataset_versions.id"]),
    )
    op.create_index("ix_scoring_jobs_dataset_version_id", "scoring_jobs", ["dataset_version_id"])
    op.create_index("ix_scoring_jobs_status", "scoring_jobs", ["status"])


def downgrade() -> None:
    op.drop_index("ix_scoring_jobs_status", table_name="scoring_jobs")
    op.drop_index("ix_scoring_jobs_dataset_version_id", table_name="scoring_jobs")
    op.drop_table("scoring_jobs")

    op.drop_index("ix_dataset_versions_stored_path", table_name="dataset_versions")
    op.drop_index("ix_dataset_versions_is_active", table_name="dataset_versions")
    op.drop_table("dataset_versions")
