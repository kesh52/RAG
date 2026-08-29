"""add documentation_gaps to query_feedback

Revision ID: add_documentation_gaps_to_query_feedback
Revises: add_attached_file_to_query_feedback
Create Date: 2026-08-29 15:06:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_documentation_gaps'
down_revision = 'add_feedback_attachment'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE query_feedback
        ADD COLUMN IF NOT EXISTS documentation_gaps TEXT;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE query_feedback
        DROP COLUMN IF EXISTS documentation_gaps;
        """
    )

