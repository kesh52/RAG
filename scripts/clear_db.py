import sys
import os
import argparse
import logging
from contextlib import closing

# Ensure root directory is in system path to resolve src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import get_connection

# Set up logging for console output
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("clear_db")

def clear_database():
    parser = argparse.ArgumentParser(description="Drop all tables in the public schema of the connected database.")
    parser.add_argument("--yes", action="store_true", help="Bypass interactive confirmation prompt.")
    args = parser.parse_args()

    logger.info("Connecting to the database to scan tables...")
    try:
        with closing(get_connection()) as conn:
            with conn.cursor() as cur:
                # Query all user tables in the public schema
                cur.execute(
                    """
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
                    """
                )
                tables = [row[0] for row in cur.fetchall()]

                if not tables:
                    logger.info("No tables found in the database. Nothing to drop.")
                    return

                print("=" * 60)
                print(f"FOUND {len(tables)} TABLES IN THE DATABASE SCHEMA:")
                for table in tables:
                    print(f"  - {table}")
                print("=" * 60)

                # Interactive confirmation prompt
                if not args.yes:
                    confirm = input("WARNING: This will drop ALL listed tables and delete all data! Proceed? (yes/no): ")
                    if confirm.strip().lower() != "yes":
                        logger.info("Execution aborted by the user.")
                        return

                logger.info("Dropping all database tables...")
                for table in tables:
                    logger.info(f"Dropping table '{table}' (CASCADE)...")
                    # Use double quotes to handle case-sensitive or reserved table names safely
                    cur.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE;')
                
            conn.commit()
            logger.info("All tables dropped successfully! Database is now empty.")
            
    except Exception as e:
        logger.error(f"Error occurred while clearing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    clear_database()

