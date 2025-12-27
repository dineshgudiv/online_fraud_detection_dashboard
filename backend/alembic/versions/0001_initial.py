"""Initial schema.

Revision ID: 0001_initial
Revises:
Create Date: 2025-12-25
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("email", sa.String(), nullable=False),
        sa.Column("password_hash", sa.String(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_role", "users", ["role"], unique=False)

    op.create_table(
        "alerts",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("transaction_id", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("risk_score", sa.Float(), nullable=False),
        sa.Column("risk_level", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("merchant_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("amount", sa.Float(), nullable=True),
        sa.Column("currency", sa.String(), nullable=True),
        sa.Column("decision", sa.String(), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("features_json", postgresql.JSONB(), nullable=True),
        sa.Column("case_id", sa.String(), nullable=True),
    )
    op.create_index("ix_alerts_status", "alerts", ["status"], unique=False)
    op.create_index("ix_alerts_risk_level", "alerts", ["risk_level"], unique=False)
    op.create_index("ix_alerts_created_at", "alerts", ["created_at"], unique=False)
    op.create_index("ix_alerts_transaction_id", "alerts", ["transaction_id"], unique=False)

    op.create_table(
        "cases",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("assigned_to", sa.String(), nullable=True),
        sa.Column("alert_id", sa.String(), nullable=True),
        sa.Column("transaction_id", sa.String(), nullable=True),
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("risk_level", sa.String(), nullable=True),
        sa.Column("risk_score", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("notes_history", postgresql.JSONB(), nullable=True),
    )
    op.create_index("ix_cases_status", "cases", ["status"], unique=False)
    op.create_index("ix_cases_updated_at", "cases", ["updated_at"], unique=False)

    op.create_table(
        "audit_log",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor", sa.String(), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("transaction_id", sa.String(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("model_version", sa.String(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("decision", sa.String(), nullable=False),
        sa.Column("risk_factors_json", postgresql.JSONB(), nullable=True),
        sa.Column("alert_id", sa.String(), nullable=True),
        sa.Column("case_id", sa.String(), nullable=True),
    )
    op.create_index("ix_audit_log_timestamp", "audit_log", ["timestamp"], unique=False)
    op.create_index("ix_audit_log_transaction_id", "audit_log", ["transaction_id"], unique=False)

    op.create_table(
        "feedback",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("alert_id", sa.String(), nullable=True),
        sa.Column("case_id", sa.String(), nullable=True),
        sa.Column("reviewer", sa.String(), nullable=False),
        sa.Column("label", sa.String(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
    )

    op.create_table(
        "rules",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("enabled", sa.Integer(), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("rules")
    op.drop_table("feedback")
    op.drop_index("ix_audit_log_transaction_id", table_name="audit_log")
    op.drop_index("ix_audit_log_timestamp", table_name="audit_log")
    op.drop_table("audit_log")
    op.drop_index("ix_cases_updated_at", table_name="cases")
    op.drop_index("ix_cases_status", table_name="cases")
    op.drop_table("cases")
    op.drop_index("ix_alerts_transaction_id", table_name="alerts")
    op.drop_index("ix_alerts_created_at", table_name="alerts")
    op.drop_index("ix_alerts_risk_level", table_name="alerts")
    op.drop_index("ix_alerts_status", table_name="alerts")
    op.drop_table("alerts")
    op.drop_index("ix_users_role", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")

