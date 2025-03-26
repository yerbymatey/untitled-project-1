import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
import glob

# Database connection parameters from docker-compose.yml
DB_PARAMS = {
    "dbname": "mydb",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

def column_exists(cursor, table_name, column_name):
    """Check if a column exists in a table"""
    cursor.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = %s AND column_name = %s
    """, (table_name, column_name))
    return cursor.fetchone() is not None

def create_tables():
    """Create database tables for storing bookmarks data with vector support"""
    
    # Connect to the database
    conn = psycopg2.connect(**DB_PARAMS)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Enable vector extension if needed
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # Create tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        name TEXT,
        verified BOOLEAN,
        followers_count INTEGER,
        following_count INTEGER,
        description TEXT
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tweets (
        id TEXT PRIMARY KEY,
        user_id TEXT REFERENCES users(id),
        text TEXT,
        created_at TIMESTAMP,
        retweet_count INTEGER,
        favorite_count INTEGER,
        reply_count INTEGER,
        quote_count INTEGER,
        is_quote_status BOOLEAN,
        quoted_tweet_id TEXT,
        url TEXT,
        has_media BOOLEAN,
        embedding vector(1536)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS hashtags (
        id SERIAL PRIMARY KEY,
        tag TEXT UNIQUE
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS tweet_hashtags (
        tweet_id TEXT,
        hashtag_id INTEGER,
        PRIMARY KEY (tweet_id, hashtag_id)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS urls (
        id SERIAL PRIMARY KEY,
        tweet_id TEXT,
        url TEXT,
        expanded_url TEXT,
        display_url TEXT,
        UNIQUE(tweet_id, url)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS media (
        id SERIAL PRIMARY KEY,
        tweet_id TEXT,
        media_url TEXT,
        type TEXT,
        alt_text TEXT,
        UNIQUE(tweet_id, media_url)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_description_urls (
        id SERIAL PRIMARY KEY,
        user_id TEXT REFERENCES users(id),
        url TEXT,
        expanded_url TEXT,
        display_url TEXT,
        UNIQUE(user_id, url)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quoted_tweet_media (
        id SERIAL PRIMARY KEY,
        quoted_tweet_id TEXT,
        media_url TEXT,
        type TEXT,
        alt_text TEXT,
        video_info JSONB,
        UNIQUE(quoted_tweet_id, media_url)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quoted_tweet_hashtags (
        quoted_tweet_id TEXT,
        hashtag_id INTEGER,
        PRIMARY KEY (quoted_tweet_id, hashtag_id)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quoted_tweet_urls (
        id SERIAL PRIMARY KEY,
        quoted_tweet_id TEXT,
        url TEXT,
        expanded_url TEXT,
        display_url TEXT,
        UNIQUE(quoted_tweet_id, url)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS quoted_tweet_user_description_urls (
        id SERIAL PRIMARY KEY,
        quoted_tweet_id TEXT,
        url TEXT,
        expanded_url TEXT,
        display_url TEXT,
        UNIQUE(quoted_tweet_id, url)
    );
    """)
    
    # Create indexes for better performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tweet_user_id ON tweets(user_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_hashtag_tag ON hashtags(tag);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tweet_quoted_tweet_id ON tweets(quoted_tweet_id);")
    
    # Close the connection
    cursor.close()
    conn.close()
    
    print("Database tables created successfully")

if __name__ == "__main__":
    create_tables() 