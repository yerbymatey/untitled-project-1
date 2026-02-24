import os


def get_schema_statements(dimension: int = int(os.getenv("EMBEDDING_DIM", "1024"))) -> list[str]:
    """Return all schema statements in execution order."""
    return [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            name TEXT,
            verified BOOLEAN DEFAULT FALSE,
            followers_count INTEGER DEFAULT 0,
            following_count INTEGER DEFAULT 0,
            description TEXT
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS tweets (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id),
            text TEXT,
            created_at TIMESTAMP WITH TIME ZONE,
            retweet_count INTEGER DEFAULT 0,
            favorite_count INTEGER DEFAULT 0,
            reply_count INTEGER DEFAULT 0,
            quote_count INTEGER DEFAULT 0,
            is_quote_status BOOLEAN DEFAULT FALSE,
            quoted_tweet_id TEXT,
            url TEXT,
            has_media BOOLEAN DEFAULT FALSE,
            embedding vector({dimension})
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS hashtags (
            id SERIAL PRIMARY KEY,
            tag VARCHAR(255) UNIQUE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS tweet_hashtags (
            tweet_id TEXT REFERENCES tweets(id),
            hashtag_id INTEGER REFERENCES hashtags(id),
            PRIMARY KEY (tweet_id, hashtag_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS urls (
            tweet_id TEXT REFERENCES tweets(id),
            url TEXT,
            expanded_url TEXT,
            display_url TEXT,
            PRIMARY KEY (tweet_id, url)
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS media (
            tweet_id TEXT REFERENCES tweets(id),
            media_url TEXT,
            type TEXT,
            alt_text TEXT,
            image_desc TEXT,
            joint_embedding vector({dimension}),
            image_embedding vector({dimension}),
            extr_text TEXT,
            PRIMARY KEY (tweet_id, media_url)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS user_description_urls (
            user_id TEXT REFERENCES users(id),
            url TEXT,
            expanded_url TEXT,
            display_url TEXT,
            PRIMARY KEY (user_id, url)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS tweet_interpretations (
            tweet_id TEXT PRIMARY KEY REFERENCES tweets(id),
            interpretation TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """,
    ]


def setup_database(dimension: int = int(os.getenv("EMBEDDING_DIM", "1024"))):
    """Set up the complete database schema including vector support."""
    from db.session import get_db_session

    with get_db_session() as session:
        for statement in get_schema_statements(dimension):
            session.execute(statement)
        session.commit()
        print("✓ Database schema is set up")

def main():
    """Main function to run schema checks and setup"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configure PostgreSQL database schema")
    parser.add_argument(
        "--dimension",
        type=int,
        default=int(os.getenv("EMBEDDING_DIM", "1024")),
        help="Vector dimension to use (default: EMBEDDING_DIM or 1024)",
    )
    args = parser.parse_args()
    
    setup_database(args.dimension)
    print("\n✓ Database is correctly configured")

if __name__ == "__main__":
    main() 
