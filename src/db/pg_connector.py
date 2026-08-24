import logging
import psycopg
from src.utils.config import config

try:
    from google.cloud.sql.connector import Connector
    _HAS_CONNECTOR = True
except ImportError:
    _HAS_CONNECTOR = False

logger = logging.getLogger(__name__)

DB_HOST = config.get("database.host", "127.0.0.1")
DB_PORT = config.get("database.port", 5432)
DB_NAME = config.get("database.name", "vector")
DB_USER = config.get("database.user", "postgres")
DB_PASSWORD = config.get("database.password")
INSTANCE_CONNECTION_NAME = config.get("database.instance_connection_name")

# Create a singleton connector instance
_CONNECTOR = None

def get_connection():
    """Acquire and return a new connection to the database."""
    global _CONNECTOR
    
    if INSTANCE_CONNECTION_NAME:
        if not _HAS_CONNECTOR:
            raise ImportError(
                "google-cloud-sql-connector is not installed in the current environment. "
                "Please run `pip install \"cloud-sql-python-connector[psycopg]\"` to use native Cloud SQL connections."
            )
            
        if _CONNECTOR is None:
            logger.info("Initializing Cloud SQL Python Connector...")
            _CONNECTOR = Connector()
            
        logger.info(f"Establishing secure database connection to Cloud SQL: {INSTANCE_CONNECTION_NAME}")
        return _CONNECTOR.connect(
            INSTANCE_CONNECTION_NAME,
            "psycopg",
            user=DB_USER,
            password=DB_PASSWORD,
            db=DB_NAME
        )
    else:
        logger.debug(f"Connecting to Postgres database '{DB_NAME}' via TCP socket at {DB_HOST}:{DB_PORT}")
        db_conn = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
        return psycopg.connect(db_conn)

