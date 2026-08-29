"""add query_feedback table

Revision ID: add_query_feedback
Revises: add_evaluation_runs
Create Date: 2026-08-29 12:54:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_query_feedback'
down_revision = 'add_evaluation_runs'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS query_feedback (
            id BIGSERIAL PRIMARY KEY,
            created_at TIMESTAMPTZ DEFAULT NOW(),

            -- Query and output
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            retrieved_contexts JSONB NOT NULL,
            sources JSONB NOT NULL,

            -- Configuration snapshot
            use_hybrid BOOLEAN NOT NULL,
            use_reranker BOOLEAN NOT NULL,
            pool_size INTEGER NOT NULL,
            final_top_k INTEGER NOT NULL,
            generation_model TEXT NOT NULL,
            latency_ms INTEGER,

            -- Feedback data
            rating INTEGER NOT NULL,
            issue_tags JSONB DEFAULT '[]'::jsonb,
            corrected_reference TEXT,
            user_comment TEXT,
            is_promoted_to_benchmark BOOLEAN DEFAULT FALSE
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_query_feedback_rating ON query_feedback(rating);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_query_feedback_created ON query_feedback(created_at DESC);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS query_feedback;")

