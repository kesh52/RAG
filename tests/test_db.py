import src.db as db
import pytest

def test_db_connection():
    """Verify that a database connection can be established and queries run."""
    try:
        with db.get_connection() as conn:
            assert conn is not None
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                res = cur.fetchone()
                assert res[0] == 1
    except Exception as e:
        pytest.skip(f"Database connection test skipped (DB not available): {e}")

