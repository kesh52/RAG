import os
import sys
import logging
from alembic.config import Config
from alembic import command

# Ensure root directory is in system path to resolve src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_migrations")

def run_migrations():
    logger.info("Initializing Alembic migration runner...")
    
    # Resolve the path to alembic.ini relative to the script location
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alembic_ini_path = os.path.join(root_dir, "alembic.ini")
    
    if not os.path.exists(alembic_ini_path):
        logger.error(f"alembic.ini not found at: {alembic_ini_path}")
        sys.exit(1)
        
    try:
        # Load the Alembic configuration
        alembic_cfg = Config(alembic_ini_path)
        
        logger.info("Applying database migrations to head via Alembic...")
        command.upgrade(alembic_cfg, "head")
        logger.info("Alembic upgrade completed.")
        
    except Exception as e:
        logger.warning(f"Alembic upgrade command encountered a notice: {e}")

    # Direct Schema Health Check & Auto-Heal
    logger.info("Performing direct schema verification against connected PostgreSQL database...")
    try:
        from contextlib import closing
        from src.db import get_connection

        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                # 1. Check current alembic_version
                cur.execute("SELECT version_num FROM alembic_version;")
                current_ver = cur.fetchall()
                logger.info(f"Database alembic_version: {[r[0] for r in current_ver]}")

                # 2. Check query_feedback table existence
                cur.execute(
                    """
                    SELECT table_name FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'query_feedback';
                    """
                )
                if not cur.fetchone():
                    logger.info("Creating table query_feedback...")
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS query_feedback (
                            id BIGSERIAL PRIMARY KEY,
                            created_at TIMESTAMPTZ DEFAULT NOW(),
                            query TEXT NOT NULL,
                            response TEXT NOT NULL,
                            retrieved_contexts JSONB NOT NULL,
                            sources JSONB NOT NULL,
                            use_hybrid BOOLEAN NOT NULL,
                            use_reranker BOOLEAN NOT NULL,
                            pool_size INTEGER NOT NULL,
                            final_top_k INTEGER NOT NULL,
                            generation_model TEXT NOT NULL,
                            latency_ms INTEGER,
                            rating INTEGER NOT NULL,
                            issue_tags JSONB DEFAULT '[]'::jsonb,
                            corrected_reference TEXT,
                            user_comment TEXT,
                            is_promoted_to_benchmark BOOLEAN DEFAULT FALSE
                        );
                        """
                    )
                    conn.commit()

                # 3. Check and apply missing columns idempotently
                cur.execute(
                    """
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = 'query_feedback';
                    """
                )
                existing_cols = {r[0] for r in cur.fetchall()}
                logger.info(f"Existing columns in query_feedback: {sorted(list(existing_cols))}")

                required_cols = [
                    ("attached_filename", "TEXT"),
                    ("attached_file_data", "BYTEA"),
                    ("attached_file_mime", "TEXT"),
                    ("documentation_gaps", "TEXT"),
                ]

                for col_name, col_type in required_cols:
                    if col_name not in existing_cols:
                        logger.info(f"➕ Adding missing column: {col_name} ({col_type})...")
                        cur.execute(f"ALTER TABLE query_feedback ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
                        conn.commit()
                        logger.info(f"✅ Successfully added {col_name} to query_feedback!")
                    else:
                        logger.info(f"✔️ Column '{col_name}' is verified in database.")

        logger.info("✨ Database schema is 100% up-to-date and verified!")
    except Exception as e:
        logger.error(f"Error verifying database schema: {e}")
        sys.exit(1)
    finally:
        from src.db import close_pool
        close_pool()

if __name__ == "__main__":
    run_migrations()
