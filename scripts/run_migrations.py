import os
import sys
import tempfile
from dotenv import load_dotenv
from pyliquibase import Pyliquibase

def run_migrations():
    # Load environment variables
    load_dotenv()
    
    db_host = os.getenv("DB_HOST", "127.0.0.1")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "vector")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")
    
    jdbc_url = f"jdbc:postgresql://{db_host}:{db_port}/{db_name}"
    
    print(f"JDBC URL: {jdbc_url}")
    print(f"User:     {db_user}")
    
    # Create a temporary liquibase.properties file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".properties") as f:
        f.write(f"url: {jdbc_url}\n")
        f.write(f"username: {db_user}\n")
        f.write(f"password: {db_password}\n")
        f.write("changeLogFile: db/changelog.yaml\n")
        properties_path = f.name

    try:
        liquibase = Pyliquibase(defaultsFile=properties_path)
        print("Running database update...")
        liquibase.update()
        print("Database migrations applied successfully!")
    except Exception as e:
        print(f"Error executing migrations: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up the temporary file
        if os.path.exists(properties_path):
            os.remove(properties_path)

if __name__ == "__main__":
    run_migrations()

