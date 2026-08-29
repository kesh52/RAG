"""add chat_ui tables (users, chat_sessions, chat_messages, chat_attachments, message_feedback)

Revision ID: add_chat_ui_tables
Revises: add_documentation_gaps
Create Date: 2026-08-29 19:35:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_chat_ui_tables'
down_revision = 'add_documentation_gaps'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Users Table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            email VARCHAR(255) UNIQUE NOT NULL,
            full_name VARCHAR(255),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_active_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")

    # 2. Chat Sessions Table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            title VARCHAR(255) NOT NULL DEFAULT 'New Conversation',
            is_archived BOOLEAN DEFAULT FALSE,
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated ON chat_sessions(user_id, updated_at DESC);")

    # 3. Chat Messages Table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
            role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            condensed_query TEXT,
            retrieved_contexts JSONB DEFAULT '[]'::jsonb,
            sources JSONB DEFAULT '[]'::jsonb,
            documentation_gaps TEXT,
            generation_model VARCHAR(100),
            latency_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created ON chat_messages(session_id, created_at ASC);")

    # 4. Chat Attachments Table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_attachments (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            message_id UUID NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
            filename VARCHAR(255) NOT NULL,
            mime_type VARCHAR(100) NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            file_data BYTEA,
            storage_path TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_attachments_message ON chat_attachments(message_id);")

    # 5. Message Feedback Table
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS message_feedback (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            message_id UUID UNIQUE NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
            rating SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
            issue_tags JSONB DEFAULT '[]'::jsonb,
            user_comment TEXT,
            corrected_reference TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_message_feedback_rating ON message_feedback(rating);")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS message_feedback;")
    op.execute("DROP TABLE IF EXISTS chat_attachments;")
    op.execute("DROP TABLE IF EXISTS chat_messages;")
    op.execute("DROP TABLE IF EXISTS chat_sessions;")
    op.execute("DROP TABLE IF EXISTS users;")

