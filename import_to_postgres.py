import pandas as pd
import psycopg2
from psycopg2 import sql
import os

# Database connection details (replace with your actual details)
db_name = "ai_job_db"
db_user = "your_username"
db_password = "your_password"
db_host = "localhost"
db_port = "5432"

# CSV file path
csv_file = 'data/ai_job_dataset.csv'

def import_csv_to_postgres(db_name, db_user, db_password, db_host, db_port, csv_file):
    """Imports data from a CSV file into a PostgreSQL database."""
    conn = None
    try:
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(dbname=db_name, user=db_user, password=db_password, host=db_host, port=db_port)
        cur = conn.cursor()

        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Get column names from the DataFrame
        columns = df.columns.tolist()

        # Create table query (adjust data types as needed)
        # This is a basic example, you might need to refine data types based on your data
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

        # Prepare data for insertion
        # Handle potential NaN values by converting to None for SQL NULL
        data = [tuple(None if pd.isna(x) else x for x in row) for row in df.values]

        # Construct the INSERT statement
        insert_query = sql.SQL("INSERT INTO ai_jobs ({}) VALUES ({})").format(
            sql.Identifier(*columns),
            sql.Placeholder()*len(columns)
        )

        # Execute the INSERT statement for each row
        cur.executemany(insert_query, data)

        # Commit the transaction
        conn.commit()
        print(f"Successfully imported {len(df)} rows from {csv_file} into 'ai_jobs' table.")

    except (psycopg2.Error, FileNotFoundError) as e:
        print(f"Error during database import: {e}")
    finally:
        if conn is not None:
            cur.close()
            conn.close()
            print("Database connection closed.")

if __name__ == "__main__":
    # Ensure the CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found at {csv_file}")
    else:
        import_csv_to_postgres(db_name, db_user, db_password, db_host, db_port, csv_file) 