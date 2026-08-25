"""initial schema

Revision ID: initial_schema
Revises: 
Create Date: 2026-08-25 19:21:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'initial_schema'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Create documents table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id BIGSERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            metadata JSONB,
            embedding vector(768)
        );
        """
    )

    # 3. Add generated tsvector column (idempotently)
    op.execute(
        """
        ALTER TABLE documents 
        ADD COLUMN IF NOT EXISTS text_search_tsv tsvector 
        GENERATED ALWAYS AS (to_tsvector('english', COALESCE(content, ''))) STORED;
        """
    )

    # 4. Create GIN index for FTS tsvector
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_fts 
        ON documents USING gin (text_search_tsv);
        """
    )

    # 5. Create HNSW index for Vector search
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw 
        ON documents USING hnsw (embedding vector_cosine_ops);
        """
    )


def downgrade() -> None:
    # Drop indexes
    op.execute("DROP INDEX IF EXISTS idx_documents_embedding_hnsw;")
    op.execute("DROP INDEX IF EXISTS idx_documents_fts;")
    
    # Drop column
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS text_search_tsv;")
    
    # Drop table
    op.execute("DROP TABLE IF EXISTS documents;")
    
    # Drop extension
    op.execute("DROP EXTENSION IF EXISTS vector;")

