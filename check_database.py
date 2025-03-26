import psycopg2

# Database connection parameters from docker-compose.yml
DB_PARAMS = {
    "dbname": "mydb",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

def check_database():
    """Check counts of records in each table"""
    
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    
    tables = [
        "users", "tweets", "hashtags", "tweet_hashtags", 
        "urls", "media", "user_description_urls"
    ]
    
    print("Database record counts:")
    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} records")
    
    # Check a random tweet's data
    cursor.execute("""
    SELECT t.id, t.text, u.screen_name, t.created_at
    FROM tweets t
    JOIN users u ON t.user_id = u.id
    LIMIT 1
    """)
    
    tweet = cursor.fetchone()
    if tweet:
        print("\nSample tweet:")
        print(f"  ID: {tweet[0]}")
        print(f"  Text: {tweet[1][:100]}...")
        print(f"  User: @{tweet[2]}")
        print(f"  Date: {tweet[3]}")
    
    cursor.close()
    conn.close()

if __name__ == "__main__":
    check_database() 