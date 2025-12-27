"""Add dataset_schema_mappings table."""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_dataset_schema_mapping"
down_revision = "0002_datasets_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_schema_mappings",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("dataset_version_id", sa.String(), nullable=False),
        sa.Column("amount_col", sa.String(), nullable=True),
        sa.Column("timestamp_col", sa.String(), nullable=True),
        sa.Column("user_id_col", sa.String(), nullable=True),
        sa.Column("merchant_col", sa.String(), nullable=True),
        sa.Column("device_id_col", sa.String(), nullable=True),
        sa.Column("country_col", sa.String(), nullable=True),
        sa.Column("label_col", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["dataset_version_id"], ["dataset_versions.id"]),
        sa.UniqueConstraint("dataset_version_id"),
    )
    op.create_index(
        "ix_dataset_schema_mappings_dataset_version_id",
        "dataset_schema_mappings",
        ["dataset_version_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_dataset_schema_mappings_dataset_version_id", table_name="dataset_schema_mappings")
    op.drop_table("dataset_schema_mappings")
