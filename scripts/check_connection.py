import sys
import os
import time
import logging
from contextlib import closing

# Ensure root directory is in system path to resolve src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import get_connection
from src.utils.config import config

# Set up clean basic config for screen output
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("check_connection")

def check_connectivity():
    db_type = config.get("database.type", "cloud_sql")
    db_host = config.get("database.host", "127.0.0.1")
    db_port = config.get("database.port", 5432)
    db_name = config.get("database.name", "vector")
    db_user = config.get("database.user", "postgres")
    instance_name = config.get("database.instance_connection_name")
    impersonate_sa = config.get("database.impersonate_service_account")
    ip_type = config.get("database.ip_type", "public")

    print("=" * 60)
    print("           DATABASE CONNECTION DIAGNOSTICS")
    print("=" * 60)
    print(f"Connection Strategy : {db_type.upper()}")
    print(f"Target Database     : {db_name}")
    print(f"Database User       : {db_user}")

    if db_type == "cloud_sql":
        print(f"GCP Instance Name   : {instance_name}")
        print(f"IP Routing Type     : {ip_type.upper()}")
        print(f"Impersonating SA    : {impersonate_sa if impersonate_sa else 'None (Uses default ADC)'}")
    else:
        print(f"TCP Host / Port     : {db_host}:{db_port}")
    
    print("-" * 60)
    print("Attempting connection to database...")
    
    start_time = time.perf_counter()
    try:
        with closing(get_connection()) as conn:
            latency = (time.perf_counter() - start_time) * 1000
            print(f"SUCCESS: Connected in {latency:.2f} ms!")
            print("-" * 60)
            
            with conn.cursor() as cur:
                # 1. Fetch Postgres Engine Version
                cur.execute("SELECT version();")
                pg_version = cur.fetchone()[0]
                print(f"PostgreSQL Version  : {pg_version.split(',')[0]}")
                
                # 2. Fetch Active Database and Connected User
                cur.execute("SELECT current_database(), current_user;")
                db_name_act, user_act = cur.fetchone()
                print(f"Active Database     : {db_name_act}")
                print(f"Connected User      : {user_act}")
                
                # 3. Check pgvector extension version
                cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
                vec_ext = cur.fetchone()
                if vec_ext:
                    print(f"pgvector Extension  : Active (v{vec_ext[0]})")
                else:
                    print("pgvector Extension  : NOT FOUND / NOT ENABLED")
                    
                # 4. Fetch Database Hardware & Memory Metrics
                print("-" * 60)
                print("           DATABASE RESOURCE & MEMORY METRICS")
                print("-" * 60)
                cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()));")
                db_size = cur.fetchone()[0]
                print(f"Total Database Size : {db_size}")
                
                cur.execute("SELECT current_setting('max_connections');")
                max_conn = cur.fetchone()[0]
                print(f"Max Connections     : {max_conn}")
                
                cur.execute("SELECT current_setting('shared_buffers');")
                shared_buf = cur.fetchone()[0]
                print(f"Shared Buffers (RAM): {shared_buf}")
                
                cur.execute("SELECT current_setting('effective_cache_size');")
                eff_cache = cur.fetchone()[0]
                print(f"Effective Cache Size: {eff_cache}")
                
                cur.execute("SELECT current_setting('work_mem');")
                work_mem = cur.fetchone()[0]
                print(f"Work Memory (Sort)  : {work_mem}")
                
                # 5. Check document counts and table disk size
                try:
                    cur.execute("SELECT count(*) FROM documents;")
                    doc_count = cur.fetchone()[0]
                    print(f"Documents Ingested  : {doc_count} rows")
                    
                    cur.execute("SELECT pg_size_pretty(pg_total_relation_size('documents'));")
                    table_size = cur.fetchone()[0]
                    print(f"Documents Table Size: {table_size}")
                except Exception:
                    print("Documents Table     : NOT FOUND (Run migrations to create the schema)")
                    
            print("=" * 60)
            
    except Exception as e:
        latency = (time.perf_counter() - start_time) * 1000
        print(f"FAILED: Connection attempt failed after {latency:.2f} ms")
        print("-" * 60)
        print(f"Error Details:\n{e}")
        print("-" * 60)
        print("Troubleshooting Suggestions:")
        if db_type == "cloud_sql":
            print("1. Confirm that Cloud SQL Admin API is enabled in your GCP project.")
            print("2. Verify your active credentials via 'gcloud auth application-default print-access-token'.")
            print("3. If using Private IP / PSC, ensure you are running within the corresponding VPC network.")
            print("4. Verify if your Service Account impersonation principal has 'Service Account Token Creator' role.")
        else:
            print("1. If using Cloud SQL Auth Proxy, ensure 'python3 scripts/start_proxy.py' is running in another tab.")
            print("2. Check that the port (5432) is not blocked or used by another local PostgreSQL server.")
            print("3. Check that host, database name, user, and password are correct in .env.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    check_connectivity()

