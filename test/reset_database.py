import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Database connection parameters from docker-compose.yml
DB_PARAMS = {
    "dbname": "mydb",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

def reset_database():
    """Drop all tables and constraints to start fresh"""
    
    # Connect to the database
    conn = psycopg2.connect(**DB_PARAMS)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Drop all tables in the correct order to handle dependencies
    tables = [
        "quoted_tweet_user_description_urls",
        "quoted_tweet_urls",
        "quoted_tweet_hashtags",
        "quoted_tweet_media",
        "user_description_urls",
        "media",
        "urls",
        "tweet_hashtags",
        "hashtags",
        "tweets",
        "users"
    ]
    
    print("Dropping all tables...")
    for table in tables:
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
            print(f"Dropped {table}")
        except Exception as e:
            print(f"Error dropping {table}: {e}")
    
    # Drop the vector extension
    try:
        cursor.execute("DROP EXTENSION IF EXISTS vector;")
        print("Dropped vector extension")
    except Exception as e:
        print(f"Error dropping vector extension: {e}")
    
    # Close the connection
    cursor.close()
    conn.close()
    
    print("Database reset complete")

if __name__ == "__main__":
    reset_database() 