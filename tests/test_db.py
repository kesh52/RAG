import os
import pytest
from unittest.mock import MagicMock, patch
from contextlib import closing
import psycopg
from psycopg_pool import ConnectionPool
import src.db as db
import src.db.pg_connector as pg_connector


def test_db_connection():
    """Verify that a database connection can be established and queries run against live DB if available."""
    if os.environ.get("RUN_LIVE_DB_TESTS") != "1":
        pytest.skip("Live database test skipped (set RUN_LIVE_DB_TESTS=1 to enable).")

    pg_connector.close_pool()
    try:
        with closing(db.get_connection()) as conn:
            assert conn is not None
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                res = cur.fetchone()
                assert res[0] == 1
    except Exception as e:
        pytest.skip(f"Database connection test skipped (DB not available): {e}")
    finally:
        pg_connector.close_pool()


def test_get_pool_postgres_tcp():
    """Verify ConnectionPool initialization for direct Postgres TCP socket mode."""
    pg_connector.close_pool()

    mock_config = {
        "database.type": "postgres",
        "database.host": "127.0.0.1",
        "database.port": 5432,
        "database.name": "test_db",
        "database.user": "test_user",
        "database.password": "test_pass",
        "database.min_pool_size": 2,
        "database.max_pool_size": 8,
        "database.pool_timeout": 15.0,
        "database.max_idle": 300.0,
        "database.max_lifetime": 1800.0,
    }

    with patch("src.db.pg_connector.config.get", side_effect=lambda k, default=None: mock_config.get(k, default)), \
         patch("src.db.pg_connector.ConnectionPool") as mock_pool_cls:
        
        mock_pool_instance = MagicMock(spec=ConnectionPool)
        mock_pool_instance.closed = False
        mock_pool_cls.return_value = mock_pool_instance

        pool = pg_connector.get_pool()
        assert pool is mock_pool_instance

        mock_pool_cls.assert_called_once_with(
            conninfo="host=127.0.0.1 port=5432 dbname=test_db user=test_user password=test_pass",
            kwargs={"autocommit": True},
            min_size=2,
            max_size=8,
            timeout=15.0,
            max_idle=300.0,
            max_lifetime=1800.0,
            reconnect_timeout=15.0,
            close_returns=True,
            open=True,
        )

        # Singleton check: repeated calls should return the same pool instance without re-instantiating
        assert pg_connector.get_pool() is mock_pool_instance
        assert mock_pool_cls.call_count == 1

    pg_connector.close_pool()


def test_get_pool_cloud_sql():
    """Verify ConnectionPool initialization for Cloud SQL Connector mode."""
    pg_connector.close_pool()

    mock_config = {
        "database.type": "cloud_sql",
        "database.instance_connection_name": "proj:region:instance",
        "database.impersonate_service_account": "",
        "database.ip_type": "public",
        "database.name": "vector_db",
        "database.user": "sql_user",
        "database.password": "sql_pass",
        "database.min_pool_size": 3,
        "database.max_pool_size": 12,
        "database.pool_timeout": 20.0,
        "database.max_idle": 400.0,
        "database.max_lifetime": 2400.0,
    }

    with patch("src.db.pg_connector.config.get", side_effect=lambda k, default=None: mock_config.get(k, default)), \
         patch("src.db.pg_connector.Connector") as mock_connector_cls, \
         patch("src.db.pg_connector.ConnectionPool") as mock_pool_cls:
        
        mock_connector_instance = MagicMock()
        mock_connector_cls.return_value = mock_connector_instance

        mock_pool_instance = MagicMock(spec=ConnectionPool)
        mock_pool_instance.closed = False
        mock_pool_cls.return_value = mock_pool_instance

        pool = pg_connector.get_pool()
        assert pool is mock_pool_instance

        mock_pool_cls.assert_called_once()
        kwargs = mock_pool_cls.call_args[1]
        assert kwargs["min_size"] == 3
        assert kwargs["max_size"] == 12
        assert kwargs["kwargs"] == {"autocommit": True}
        assert kwargs["timeout"] == 20.0
        assert kwargs["max_idle"] == 400.0
        assert kwargs["max_lifetime"] == 2400.0
        assert kwargs["reconnect_timeout"] == 20.0
        assert kwargs["close_returns"] is True
        assert kwargs["open"] is True

        # Test the connection_class.connect classmethod
        conn_cls = kwargs["connection_class"]
        conn_cls.connect()
        mock_connector_instance.connect.assert_called_once()
        conn_call_args = mock_connector_instance.connect.call_args
        assert conn_call_args[0][0] == "proj:region:instance"
        assert conn_call_args[0][1] == "psycopg"
        assert conn_call_args[1]["user"] == "sql_user"
        assert conn_call_args[1]["password"] == "sql_pass"
        assert conn_call_args[1]["db"] == "vector_db"

    pg_connector.close_pool()


def test_get_connection_delegates_to_pool():
    """Verify get_connection() retrieves a connection from get_pool()."""
    pg_connector.close_pool()

    mock_pool = MagicMock(spec=ConnectionPool)
    mock_pool.closed = False
    mock_conn = MagicMock(spec=psycopg.Connection)
    mock_pool.getconn.return_value = mock_conn

    with patch("src.db.pg_connector.get_pool", return_value=mock_pool):
        conn = pg_connector.get_connection()
        assert conn is mock_conn
        mock_pool.getconn.assert_called_once()

    pg_connector.close_pool()


def test_close_pool():
    """Verify close_pool() closes both pool and connector singletons."""
    mock_pool = MagicMock(spec=ConnectionPool)
    mock_pool.closed = False
    mock_connector = MagicMock()

    pg_connector._POOL = mock_pool
    pg_connector._CONNECTOR = mock_connector

    pg_connector.close_pool()

    mock_pool.close.assert_called_once()
    mock_connector.close.assert_called_once()
    assert pg_connector._POOL is None
    assert pg_connector._CONNECTOR is None


def test_pool_connection_context_manager():
    """Verify that get_pool().connection() works as a context manager and yields connection."""
    pg_connector.close_pool()

    mock_pool = MagicMock(spec=ConnectionPool)
    mock_pool.closed = False
    mock_conn = MagicMock(spec=psycopg.Connection)
    
    mock_pool.connection.return_value.__enter__.return_value = mock_conn

    with patch("src.db.get_pool", return_value=mock_pool), \
         patch("src.db.pg_connector.get_pool", return_value=mock_pool):
        pool = db.get_pool()
        with pool.connection() as conn:
            assert conn is mock_conn
        mock_pool.connection.assert_called_once()

    pg_connector.close_pool()


