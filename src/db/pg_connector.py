import logging
import psycopg
import google.auth
from google.auth import impersonated_credentials
from src.utils.config import config

try:
    from google.cloud.sql.connector import Connector, IPTypes
    _HAS_CONNECTOR = True
except ImportError:
    _HAS_CONNECTOR = False

logger = logging.getLogger(__name__)

DB_TYPE = config.get("database.type", "cloud_sql")
DB_HOST = config.get("database.host", "127.0.0.1")
DB_PORT = config.get("database.port", 5432)
DB_NAME = config.get("database.name", "vector")
DB_USER = config.get("database.user", "postgres")
DB_PASSWORD = config.get("database.password")
INSTANCE_CONNECTION_NAME = config.get("database.instance_connection_name")
IMPERSONATE_SA = config.get("database.impersonate_service_account")
IP_TYPE_STR = config.get("database.ip_type", "public").lower()

# Create a singleton connector instance
_CONNECTOR = None

def get_connection():
    """Acquire and return a new connection to the database."""
    global _CONNECTOR
    
    if DB_TYPE == "cloud_sql":
        if not INSTANCE_CONNECTION_NAME:
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
            if IMPERSONATE_SA:
                logger.info(f"Setting up credentials impersonating service account: {IMPERSONATE_SA}")
                base_creds, _ = google.auth.default()
                creds = impersonated_credentials.Credentials(
                    source_credentials=base_creds,
                    target_principal=IMPERSONATE_SA,
                    target_scopes=["https://www.googleapis.com/auth/sqlservice.admin"]
                )
            
            logger.info("Initializing Cloud SQL Python Connector...")
            _CONNECTOR = Connector(credentials=creds)
            
        # Map configured IP type to IPTypes enum
        if IP_TYPE_STR == "private":
            ip_type = IPTypes.PRIVATE
            logger.debug("Connecting using Private IP configuration...")
        elif IP_TYPE_STR == "psc":
            ip_type = IPTypes.PSC
            logger.debug("Connecting using Private Service Connect (PSC) configuration...")
        else:
            ip_type = IPTypes.PUBLIC
            logger.debug("Connecting using Public IP configuration...")
            
        logger.info(f"Establishing secure database connection to Cloud SQL: {INSTANCE_CONNECTION_NAME}")
        return _CONNECTOR.connect(
            INSTANCE_CONNECTION_NAME,
            "psycopg",
            user=DB_USER,
            password=DB_PASSWORD,
            db=DB_NAME,
            ip_type=ip_type
        )
    else:
        logger.debug(f"Connecting to Postgres database '{DB_NAME}' via TCP socket at {DB_HOST}:{DB_PORT}")
        db_conn = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
        return psycopg.connect(db_conn)
