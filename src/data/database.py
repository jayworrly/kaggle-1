import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import sql
import pandas as pd
from ..utils.config import get_config

class DatabaseManager:
    """Class to handle database operations."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.db_config = self.config['database']
    
    def create_database(self):
        """Create a new PostgreSQL database if it doesn't exist."""
        conn = None
        try:
            # Connect to the default database to create a new one
            conn = psycopg2.connect(
                user=self.db_config['user'],
                password=self.db_config['password'],
                host=self.db_config['host'],
                port=self.db_config['port']
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cur = conn.cursor()

            # Check if the database exists
            cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{self.db_config['name']}'")
            exists = cur.fetchone()

            if not exists:
                # Create the database
                cur.execute(f'CREATE DATABASE {self.db_config["name"]}')
                print(f"Database '{self.db_config['name']}' created successfully.")
            else:
                print(f"Database '{self.db_config['name']}' already exists.")

        except psycopg2.Error as e:
            print(f"Error during database creation: {e}")
            return False
        finally:
            if conn is not None:
                cur.close()
                conn.close()
                print("Database connection closed.")
        return True
    
    def import_csv_to_postgres(self, csv_file):
        """Import data from a CSV file into PostgreSQL database."""
        conn = None
        try:
            # Connect to the PostgreSQL database
            conn = psycopg2.connect(
                dbname=self.db_config['name'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                host=self.db_config['host'],
                port=self.db_config['port']
            )
            cur = conn.cursor()

            # Read the CSV file
            df = pd.read_csv(csv_file)
            print(f"Loaded {len(df)} rows from CSV file")

            # Create table query
            create_table_query = sql.SQL("""
                CREATE TABLE IF NOT EXISTS ai_jobs (
                    job_id VARCHAR(255) PRIMARY KEY,
                    job_title VARCHAR(255),
                    salary_usd DECIMAL,
                    salary_currency VARCHAR(50),
                    experience_level VARCHAR(50),
                    employment_type VARCHAR(50),
                    company_location VARCHAR(255),
                    company_size VARCHAR(50),
                    employee_residence VARCHAR(255),
                    remote_ratio INTEGER,
                    required_skills TEXT,
                    education_required VARCHAR(50),
                    years_experience INTEGER,
                    industry VARCHAR(255),
                    posting_date DATE,
                    application_deadline DATE,
                    job_description_length INTEGER,
                    benefits_score DECIMAL,
                    company_name VARCHAR(255)
                );
            """).format()
            
            cur.execute(create_table_query)
            conn.commit()
            print("Table 'ai_jobs' checked/created successfully.")

            # Clear existing data if any
            cur.execute("DELETE FROM ai_jobs")
            conn.commit()
            print("Cleared existing data from table")

            # Prepare data for insertion with proper type handling
            data_rows = []
            for index, row in df.iterrows():
                # Convert each value to appropriate type
                row_data = []
                for col in df.columns:
                    value = row[col]
                    if pd.isna(value):
                        row_data.append(None)
                    elif col in ['salary_usd', 'benefits_score']:
                        # Handle decimal values
                        try:
                            row_data.append(float(value) if value else None)
                        except (ValueError, TypeError):
                            row_data.append(None)
                    elif col in ['remote_ratio', 'years_experience', 'job_description_length']:
                        # Handle integer values
                        try:
                            row_data.append(int(value) if value else None)
                        except (ValueError, TypeError):
                            row_data.append(None)
                    elif col in ['posting_date', 'application_deadline']:
                        # Handle date values
                        try:
                            if pd.notna(value) and value:
                                row_data.append(pd.to_datetime(value).date())
                            else:
                                row_data.append(None)
                        except (ValueError, TypeError):
                            row_data.append(None)
                    else:
                        # Handle string values
                        row_data.append(str(value) if value else None)
                data_rows.append(tuple(row_data))

            # Construct the INSERT statement
            columns = df.columns.tolist()
            placeholders = ','.join(['%s'] * len(columns))
            insert_query = f"INSERT INTO ai_jobs ({','.join(columns)}) VALUES ({placeholders})"

            # Execute the INSERT statement for each row
            cur.executemany(insert_query, data_rows)

            # Commit the transaction
            conn.commit()
            print(f"Successfully imported {len(df)} rows from {csv_file} into 'ai_jobs' table.")
            return True

        except (psycopg2.Error, FileNotFoundError) as e:
            print(f"Error during database import: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn is not None:
                cur.close()
                conn.close()
                print("Database connection closed.")

def create_database():
    """Convenience function to create the database."""
    db_manager = DatabaseManager()
    return db_manager.create_database()

def import_to_postgres(csv_file=None):
    """Convenience function to import CSV to PostgreSQL."""
    db_manager = DatabaseManager()
    if csv_file is None:
        csv_file = db_manager.config['dataset']['raw_file']
    return db_manager.import_csv_to_postgres(csv_file)

if __name__ == "__main__":
    create_database()
    import_to_postgres() 