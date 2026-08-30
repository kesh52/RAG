import logging
import psycopg
from psycopg_pool import ConnectionPool
import google.auth
from google.auth import impersonated_credentials
from src.utils.config import config

try:
    from google.cloud.sql.connector import Connector, IPTypes
    _HAS_CONNECTOR = True
except ImportError:
    _HAS_CONNECTOR = False

logger = logging.getLogger(__name__)

# Singletons for connection pool and Cloud SQL connector
_CONNECTOR = None
_POOL: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """Initialize (if necessary) and return the singleton psycopg_pool.ConnectionPool."""
    global _CONNECTOR, _POOL

    if _POOL is not None and not _POOL.closed:
        return _POOL

    db_type = config.get("database.type", "cloud_sql")
    min_pool_size = int(config.get("database.min_pool_size", 1))
    max_pool_size = int(config.get("database.max_pool_size", 10))
    pool_timeout = float(config.get("database.pool_timeout", 30.0))
    max_idle = float(config.get("database.max_idle", 600.0))
    max_lifetime = float(config.get("database.max_lifetime", 3600.0))

    if db_type == "cloud_sql":
        instance_connection_name = config.get("database.instance_connection_name")
        if not instance_connection_name:
            raise ValueError(
                "database.type is set to 'cloud_sql', but database.instance_connection_name is not configured in config.yaml or environment."
            )

        if not _HAS_CONNECTOR:
            raise ImportError(
                "google-cloud-sql-connector is not installed in the current environment. "
                "Please run `pip install \"cloud-sql-python-connector[psycopg]\"` to use native Cloud SQL connections."
            )

        if _CONNECTOR is None:
            # Resolve custom credentials if service account impersonation is configured
            creds = None
            impersonate_sa = config.get("database.impersonate_service_account")
            if impersonate_sa:
                logger.info(f"Setting up credentials impersonating service account: {impersonate_sa}")
                base_creds, _ = google.auth.default()
                creds = impersonated_credentials.Credentials(
                    source_credentials=base_creds,
                    target_principal=impersonate_sa,
                    target_scopes=["https://www.googleapis.com/auth/sqlservice.admin"]
                )

            logger.info("Initializing Cloud SQL Python Connector...")
            _CONNECTOR = Connector(credentials=creds)

        # Map configured IP type to IPTypes enum
        ip_type_str = config.get("database.ip_type", "public").lower()
        if ip_type_str == "private":
            ip_type = IPTypes.PRIVATE
            logger.debug("Connecting using Private IP configuration...")
        elif ip_type_str == "psc":
            ip_type = IPTypes.PSC
            logger.debug("Connecting using Private Service Connect (PSC) configuration...")
        else:
            ip_type = IPTypes.PUBLIC
            logger.debug("Connecting using Public IP configuration...")

        db_user = config.get("database.user", "postgres")
        db_password = config.get("database.password")
        db_name = config.get("database.name", "vector")

        class CloudSQLConnection(psycopg.Connection):
            @classmethod
            def connect(cls, conninfo="", **kwargs):
                logger.debug(f"Establishing pooled connection to Cloud SQL: {instance_connection_name}")
                return _CONNECTOR.connect(
                    instance_connection_name,
                    "psycopg",
                    user=db_user,
                    password=db_password,
                    db=db_name,
                    ip_type=ip_type,
                    **kwargs
                )

        logger.info(
            f"Initializing Cloud SQL ConnectionPool (min={min_pool_size}, max={max_pool_size}) for {instance_connection_name}"
        )
        _POOL = ConnectionPool(
            min_size=min_pool_size,
            max_size=max_pool_size,
            kwargs={"autocommit": True},
            timeout=pool_timeout,
            max_idle=max_idle,
            max_lifetime=max_lifetime,
            reconnect_timeout=pool_timeout,
            connection_class=CloudSQLConnection,
            close_returns=True,
            open=True
        )
    else:
        db_host = config.get("database.host", "127.0.0.1")
        db_port = config.get("database.port", 5432)
        db_name = config.get("database.name", "vector")
        db_user = config.get("database.user", "postgres")
        db_password = config.get("database.password")

        logger.info(
            f"Initializing Postgres TCP ConnectionPool (min={min_pool_size}, max={max_pool_size}) for '{db_name}' at {db_host}:{db_port}"
        )
        db_conn = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password}"
        _POOL = ConnectionPool(
            conninfo=db_conn,
            kwargs={"autocommit": True},
            min_size=min_pool_size,
            max_size=max_pool_size,
            timeout=pool_timeout,
            max_idle=max_idle,
            max_lifetime=max_lifetime,
            reconnect_timeout=pool_timeout,
            close_returns=True,
            open=True
        )

    return _POOL


def get_connection():
    """Acquire and return a connection from the connection pool.

    With `close_returns=True`, calling `conn.close()` or using `with closing(get_connection()) as conn:`
    will automatically return the connection back to the pool.
    """
    pool = get_pool()
    return pool.getconn()


def close_pool():
    """Close the active connection pool and clean up connector resources."""
    global _POOL, _CONNECTOR
    if _POOL is not None:
        try:
            if not _POOL.closed:
                logger.info("Closing database connection pool...")
                _POOL.close(timeout=2.0)
        except Exception as e:
            logger.warning(f"Error while closing database connection pool: {e}")
        finally:
            _POOL = None
    if _CONNECTOR is not None:
        try:
            logger.info("Closing Cloud SQL connector...")
            _CONNECTOR.close()
        except Exception as e:
            logger.warning(f"Error while closing Cloud SQL connector: {e}")
        finally:
            _CONNECTOR = None


# Automatically close connection pool before Python interpreter shutdown/finalization
import atexit
atexit.register(close_pool)
