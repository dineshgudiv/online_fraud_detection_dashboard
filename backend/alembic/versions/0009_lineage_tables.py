"""Add lineage tables and links.

Revision ID: 0009_lineage_tables
Revises: 0008_drift_metrics
Create Date: 2025-01-05 00:00:03.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from alembic import op

revision = "0009_lineage_tables"
down_revision = "0008_drift_metrics"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(), nullable=True),
    )
    op.create_index("ix_datasets_version", "datasets", ["version"])

    op.create_table(
        "feature_sets",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("schema_hash", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_feature_sets_version", "feature_sets", ["version"])

    op.create_table(
        "model_versions",
        sa.Column("version", sa.String(), primary_key=True),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dataset_id", sa.String(), sa.ForeignKey("datasets.id"), nullable=True),
        sa.Column("feature_set_id", sa.String(), sa.ForeignKey("feature_sets.id"), nullable=True),
        sa.Column("metrics_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index("ix_model_versions_trained_at", "model_versions", ["trained_at"])
    op.create_index("ix_model_versions_dataset_id", "model_versions", ["dataset_id"])
    op.create_index("ix_model_versions_feature_set_id", "model_versions", ["feature_set_id"])

    op.add_column("retrain_jobs", sa.Column("dataset_id", sa.String(), nullable=True))
    op.add_column("retrain_jobs", sa.Column("feature_set_id", sa.String(), nullable=True))
    op.create_foreign_key(
        "fk_retrain_jobs_dataset_id",
        "retrain_jobs",
        "datasets",
        ["dataset_id"],
        ["id"],
    )
    op.create_foreign_key(
        "fk_retrain_jobs_feature_set_id",
        "retrain_jobs",
        "feature_sets",
        ["feature_set_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_retrain_jobs_feature_set_id", "retrain_jobs", type_="foreignkey")
    op.drop_constraint("fk_retrain_jobs_dataset_id", "retrain_jobs", type_="foreignkey")
    op.drop_column("retrain_jobs", "feature_set_id")
    op.drop_column("retrain_jobs", "dataset_id")

    op.drop_index("ix_model_versions_feature_set_id", table_name="model_versions")
    op.drop_index("ix_model_versions_dataset_id", table_name="model_versions")
    op.drop_index("ix_model_versions_trained_at", table_name="model_versions")
    op.drop_table("model_versions")

    op.drop_index("ix_feature_sets_version", table_name="feature_sets")
    op.drop_table("feature_sets")

    op.drop_index("ix_datasets_version", table_name="datasets")
    op.drop_table("datasets")
