from db.session import get_db_session

def setup_database(dimension: int = 768):
    """Set up the complete database schema including vector support"""
    with get_db_session() as session:
        # First ensure vector extension exists
        session.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create users table
        session.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                verified BOOLEAN DEFAULT FALSE,
                followers_count INTEGER DEFAULT 0,
                following_count INTEGER DEFAULT 0,
                description TEXT
            );
        """)
        
        # Create tweets table
        session.execute(f"""
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
        """)
        
        # Create media table
        session.execute(f"""
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
        """)
        
        # Create tweet_interpretations table
        session.execute("""
            CREATE TABLE IF NOT EXISTS tweet_interpretations (
                tweet_id TEXT PRIMARY KEY REFERENCES tweets(id),
                interpretation TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        session.commit()
        print("✓ Database schema is set up")

def main():
    """Main function to run schema checks and setup"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Configure PostgreSQL database schema")
    parser.add_argument("--dimension", type=int, default=768, 
                        help="Vector dimension to use (default: 768)")
    args = parser.parse_args()
    
    setup_database(args.dimension)
    print("\n✓ Database is correctly configured")

if __name__ == "__main__":
    main() 