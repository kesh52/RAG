"""add evaluation_runs table

Revision ID: add_evaluation_runs
Revises: initial_schema
Create Date: 2026-08-29 12:17:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_evaluation_runs'
down_revision = 'initial_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id BIGSERIAL PRIMARY KEY,
            run_name TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),

            -- Pipeline parameters used
            use_hybrid BOOLEAN NOT NULL,
            use_reranker BOOLEAN NOT NULL,
            pool_size INTEGER NOT NULL,
            final_top_k INTEGER NOT NULL,
            chunk_size INTEGER NOT NULL,
            chunk_overlap INTEGER NOT NULL,
            embedding_model TEXT NOT NULL,
            rerank_model TEXT NOT NULL,
            generation_model TEXT NOT NULL,

            -- Dataset used
            dataset_name TEXT NOT NULL,
            dataset_cases JSONB NOT NULL,

            -- Aggregated metrics
            avg_faithfulness FLOAT,
            avg_answer_relevancy FLOAT,
            avg_context_precision FLOAT,
            avg_context_recall FLOAT,

            -- Per-question detailed results
            detailed_results JSONB NOT NULL,

            notes TEXT
        );
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS evaluation_runs;")

