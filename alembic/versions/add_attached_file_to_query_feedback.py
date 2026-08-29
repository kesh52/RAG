"""add attached file columns to query_feedback

Revision ID: add_attached_file_to_query_feedback
Revises: add_query_feedback
Create Date: 2026-08-29 14:48:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_feedback_attachment'
down_revision = 'add_query_feedback'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE query_feedback ADD COLUMN IF NOT EXISTS attached_filename TEXT;")
    op.execute("ALTER TABLE query_feedback ADD COLUMN IF NOT EXISTS attached_file_data BYTEA;")
    op.execute("ALTER TABLE query_feedback ADD COLUMN IF NOT EXISTS attached_file_mime TEXT;")


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE query_feedback
        DROP COLUMN IF EXISTS attached_file_mime,
        DROP COLUMN IF EXISTS attached_file_data,
        DROP COLUMN IF EXISTS attached_filename;
        """
    )

