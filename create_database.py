import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Database connection details (replace with your actual details)
db_name = "ai_job_db"
db_user = "your_username"
db_password = "your_password"
db_host = "localhost"
db_port = "5432"

def create_database(db_name, db_user, db_password, db_host, db_port):
    """Creates a new PostgreSQL database if it doesn't exist."""
    conn = None
    try:
        # Connect to the default database to create a new one
        conn = psycopg2.connect(user=db_user, password=db_password, host=db_host, port=db_port)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        # Check if the database exists
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cur.fetchone()

        if not exists:
            # Create the database
            cur.execute(f'CREATE DATABASE {db_name}')
            print(f"Database '{db_name}' created successfully.")
        else:
            print(f"Database '{db_name}' already exists.")

    except psycopg2.Error as e:
        print(f"Error during database creation: {e}")
    finally:
        if conn is not None:
            cur.close()
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    create_database(db_name, db_user, db_password, db_host, db_port) 