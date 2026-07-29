import psycopg2
import time
from app.config_loader import get_config

def setup_database():
    config = get_config()
    db_config = config["database"]
    
    # Wait for PostgreSQL to be ready
    max_retries = 15
    retry_interval = 3
    conn = None
    
    for attempt in range(1, max_retries + 1):
        try:
            conn = psycopg2.connect(
                host=db_config["host"],
                port=db_config["port"],
                user=db_config["user"],
                password=db_config["password"],
                dbname="postgres"  # first connect to default postgres db
            )
            conn.autocommit = True
            break
        except Exception as e:
            print(f"[{attempt}/{max_retries}] Waiting for database server: {e}")
            time.sleep(retry_interval)
    
    if not conn:
        raise Exception("Could not connect to PostgreSQL server.")
        
    cur = conn.cursor()
    
    # Check if target database exists, if not create it
    target_db = db_config["dbname"]
    cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{target_db}'")
    exists = cur.fetchone()
    if not exists:
        cur.execute(f"CREATE DATABASE {target_db}")
        print(f"Created database: {target_db}")
    else:
        print(f"Database {target_db} already exists.")
        
    cur.close()
    conn.close()
    
    # Now connect to target database and enable vector extension
    try:
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            dbname=target_db
        )
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("PGVector extension checked/enabled.")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Failed to enable PGVector extension: {e}")
        return False

if __name__ == "__main__":
    setup_database()
