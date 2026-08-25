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
        
        logger.info("Applying database migrations to head...")
        # Programmatically run 'alembic upgrade head'
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations applied successfully!")
        
    except Exception as e:
        logger.error(f"Error applying database migrations: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_migrations()
