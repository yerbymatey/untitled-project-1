import psycopg2
import json
import glob
import os
from datetime import datetime
from psycopg2.extras import execute_values

# Database connection parameters from docker-compose.yml
DB_PARAMS = {
    "dbname": "mydb",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": 5432
}

def parse_twitter_date(date_str):
    """Convert Twitter date format to PostgreSQL timestamp format"""
    try:
        # Twitter date format: "Wed Oct 10 20:19:24 +0000 2018"
        dt = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        return dt
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None

# Add a function to generate a fake ID for users with missing IDs
def generate_fake_user_id(screen_name):
    """Generate a fake user ID for users with missing IDs based on screen name"""
    return f"fake_id_{screen_name}_{hash(screen_name) % 10000000}"

def import_bookmarks_file(file_path):
    """Import bookmarks from a JSON file into the database"""
    
    print(f"Importing data from {file_path}...")
    
    # Load JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    bookmarks = data.get('bookmarks', [])
    if not bookmarks:
        print("No bookmarks found in the file")
        return 0
    
    # Connect to the database
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    
    # Process users and tweets
    users_dict = {}  # Use dictionary to de-duplicate users by ID for this batch only
    tweets_data = []
    hashtags_data = []
    tweet_hashtags_data = []
    urls_data = []
    media_data = []
    user_urls_data = []
    
    try:
        # Begin transaction
        conn.autocommit = False
        
        # Process each bookmark
        for bookmark in bookmarks:
            # Extract user data
            user = bookmark['user']
            
            # Generate fake ID if original is missing
            if not user.get('id'):
                user['id'] = generate_fake_user_id(user.get('screen_name', 'unknown'))
                print(f"Generated fake ID for user {user.get('screen_name')}: {user['id']}")
            
            # Store user data with most recent info (in case of duplicates)
            users_dict[user['id']] = (
                user['id'],
                user.get('name', ''),
                user.get('screen_name', ''),
                user.get('verified', False),
                user.get('followers_count', 0),
                user.get('following_count', 0),
                user.get('description', '')
            )
            
            # Extract tweet data
            created_at = parse_twitter_date(bookmark['created_at']) if bookmark.get('created_at') else None
            
            # Skip tweets missing IDs
            if not bookmark.get('id'):
                print(f"Skipping tweet with missing ID from user {user.get('screen_name')}")
                continue
                
            tweet_data = (
                bookmark['id'],
                user['id'],
                bookmark.get('text', ''),
                created_at,
                bookmark.get('retweet_count', 0),
                bookmark.get('favorite_count', 0),
                bookmark.get('reply_count', 0),
                bookmark.get('quote_count', 0),
                bookmark.get('is_quote_status', False),
                bookmark.get('url', ''),
                bookmark.get('media', {}).get('has_media', False),
            )
            tweets_data.append(tweet_data)
            
            # Process hashtags
            for tag in bookmark.get('hashtags', []):
                if tag:  # Skip empty hashtags
                    hashtags_data.append((tag,))
                    tweet_hashtags_data.append((bookmark['id'], tag))
            
            # Process URLs
            for url_data in bookmark.get('urls', []):
                if url_data:  # Skip empty URL objects
                    urls_data.append((
                        bookmark['id'],
                        url_data.get('url', ''),
                        url_data.get('expanded_url', ''),
                        url_data.get('display_url', '')
                    ))
            
            # Process media
            for media_item in bookmark.get('media', {}).get('items', []):
                if media_item:  # Skip empty media objects
                    media_data.append((
                        bookmark['id'],
                        media_item.get('media_url', ''),
                        media_item.get('type', ''),
                        media_item.get('alt_text', '')
                    ))
            
            # Process user description URLs
            for url_data in user.get('description_urls', []):
                if url_data:  # Skip empty URL objects
                    user_urls_data.append((
                        user['id'],
                        url_data.get('url', ''),
                        url_data.get('expanded_url', ''),
                        url_data.get('display_url', '')
                    ))
        
        # Insert users - now using the de-duplicated dictionary values
        users_data = list(users_dict.values())
        print(f"Inserting {len(users_data)} unique users...")
        execute_values(
            cursor,
            """
            INSERT INTO users (id, name, screen_name, verified, followers_count, following_count, description)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                screen_name = EXCLUDED.screen_name,
                verified = EXCLUDED.verified,
                followers_count = EXCLUDED.followers_count,
                following_count = EXCLUDED.following_count,
                description = EXCLUDED.description
            """,
            users_data
        )
        
        # Insert tweets with ON CONFLICT DO NOTHING
        print(f"Inserting {len(tweets_data)} tweets...")
        execute_values(
            cursor,
            """
            INSERT INTO tweets (id, user_id, text, created_at, retweet_count, favorite_count, 
                              reply_count, quote_count, is_quote_status, url, has_media)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                user_id = EXCLUDED.user_id,
                text = EXCLUDED.text,
                created_at = EXCLUDED.created_at,
                retweet_count = EXCLUDED.retweet_count,
                favorite_count = EXCLUDED.favorite_count,
                reply_count = EXCLUDED.reply_count,
                quote_count = EXCLUDED.quote_count,
                is_quote_status = EXCLUDED.is_quote_status,
                url = EXCLUDED.url,
                has_media = EXCLUDED.has_media
            """,
            tweets_data
        )
        
        # Insert hashtags
        if hashtags_data:
            print(f"Inserting {len(hashtags_data)} hashtags...")
            execute_values(
                cursor,
                """
                INSERT INTO hashtags (tag)
                VALUES %s
                ON CONFLICT (tag) DO NOTHING
                """,
                hashtags_data
            )
            
            # Create map of hashtag to id
            cursor.execute("SELECT id, tag FROM hashtags")
            hashtag_map = {tag: id for id, tag in cursor.fetchall()}
            
            # Insert tweet_hashtags connections
            print(f"Creating {len(tweet_hashtags_data)} tweet-hashtag connections...")
            tweet_hashtag_values = [(tweet_id, hashtag_map.get(tag)) 
                                   for tweet_id, tag in tweet_hashtags_data
                                   if hashtag_map.get(tag)]
            
            if tweet_hashtag_values:
                execute_values(
                    cursor,
                    """
                    INSERT INTO tweet_hashtags (tweet_id, hashtag_id)
                    VALUES %s
                    ON CONFLICT (tweet_id, hashtag_id) DO NOTHING
                    """,
                    tweet_hashtag_values
                )
        
        # Insert URLs
        if urls_data:
            print(f"Inserting {len(urls_data)} URLs...")
            execute_values(
                cursor,
                """
                INSERT INTO urls (tweet_id, url, expanded_url, display_url)
                VALUES %s
                """,
                urls_data
            )
        
        # Insert media
        if media_data:
            print(f"Inserting {len(media_data)} media items...")
            execute_values(
                cursor,
                """
                INSERT INTO media (tweet_id, media_url, type, alt_text)
                VALUES %s
                """,
                media_data
            )
        
        # Insert user description URLs
        if user_urls_data:
            print(f"Inserting {len(user_urls_data)} user description URLs...")
            execute_values(
                cursor,
                """
                INSERT INTO user_description_urls (user_id, url, expanded_url, display_url)
                VALUES %s
                """,
                user_urls_data
            )
        
        # Commit the transaction
        conn.commit()
        print(f"Successfully imported {len(bookmarks)} bookmarks from {file_path}")
        return len(bookmarks)
    
    except Exception as e:
        conn.rollback()
        print(f"Error importing data: {e}")
        return 0
    
    finally:
        cursor.close()
        conn.close()

def import_all_bookmarks():
    """Import all bookmark JSON files in the current directory"""
    
    # Find all bookmark JSON files
    bookmark_files = glob.glob("bookmarks_*.json")
    
    if not bookmark_files:
        print("No bookmark files found in the current directory")
        return
    
    total_imported = 0
    for file_path in bookmark_files:
        count = import_bookmarks_file(file_path)
        total_imported += count
    
    print(f"Total imported bookmarks: {total_imported}")

if __name__ == "__main__":
    import_all_bookmarks() 