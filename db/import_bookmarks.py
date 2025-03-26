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

BATCH_SIZE = 1000

def parse_twitter_date(date_str):
    """Convert Twitter date format to PostgreSQL timestamp format"""
    try:
        # Twitter date format: "Wed Oct 10 20:19:24 +0000 2018"
        dt = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        return dt
    except Exception as e:
        print(f"Error parsing date '{date_str}': {e}")
        return None

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
    
    try:
        # Begin transaction
        conn.autocommit = False
        processed_count = 0
        
        # Prepare batch data structures
        users_data = []
        tweets_data = []
        hashtags_data = []
        tweet_hashtags_data = []
        urls_data = []
        media_data = []
        user_urls_data = []
        
        # Process each bookmark
        for bookmark in bookmarks:
            try:
                # Extract user data
                user = bookmark.get('user', {})
                if not user:
                    print(f"Skipping bookmark with missing user data: {bookmark.get('id', 'unknown')}")
                    continue
                
                # Use screen_name as the user ID
                screen_name = user.get('screen_name')
                if not screen_name:
                    print(f"Skipping bookmark with missing screen name from user {user.get('name', 'unknown')}")
                    continue
                
                # Store user data
                user_data = (
                    screen_name,  # Use screen_name as ID
                    user.get('name', ''),
                    user.get('verified', False),
                    user.get('followers_count', 0),
                    user.get('following_count', 0),
                    user.get('description', '')
                )
                users_data.append(user_data)
                
                # Skip tweets missing IDs
                if not bookmark.get('id'):
                    print(f"Skipping tweet with missing ID from user {screen_name}")
                    continue
                
                # Extract tweet data with defensive checks
                created_at = parse_twitter_date(bookmark.get('created_at'))
                quoted_status = bookmark.get('quoted_status', {})
                media = bookmark.get('media', {})
                
                # Store tweet data
                tweet_data = (
                    bookmark['id'],
                    screen_name,  # Use screen_name as user_id
                    bookmark.get('text', ''),
                    created_at,
                    bookmark.get('retweet_count', 0),
                    bookmark.get('favorite_count', 0),
                    bookmark.get('reply_count', 0),
                    bookmark.get('quote_count', 0),
                    bookmark.get('is_quote_status', False),
                    quoted_status.get('id') if quoted_status else None,
                    bookmark.get('url', ''),
                    media.get('has_media', False) if media else False
                )
                tweets_data.append(tweet_data)
                
                # Process hashtags with defensive checks
                hashtags = bookmark.get('hashtags', [])
                if isinstance(hashtags, list):
                    for tag in hashtags:
                        if tag:
                            hashtags_data.append((tag,))
                            tweet_hashtags_data.append((bookmark['id'], tag))
                
                # Process URLs with defensive checks
                urls = bookmark.get('urls', [])
                if isinstance(urls, list):
                    for url_data in urls:
                        if url_data:
                            urls_data.append((
                                bookmark['id'],
                                url_data.get('url', ''),
                                url_data.get('expanded_url', ''),
                                url_data.get('display_url', '')
                            ))
                
                # Process media with defensive checks
                media_items = media.get('items', []) if media else []
                if isinstance(media_items, list):
                    for media_item in media_items:
                        if media_item:
                            media_data.append((
                                bookmark['id'],
                                media_item.get('media_url', ''),
                                media_item.get('type', ''),
                                media_item.get('alt_text', '')
                            ))
                
                # Process user description URLs with defensive checks
                description_urls = user.get('description_urls', [])
                if isinstance(description_urls, list):
                    for url_data in description_urls:
                        if url_data:
                            user_urls_data.append((
                                screen_name,  # Use screen_name as user_id
                                url_data.get('url', ''),
                                url_data.get('expanded_url', ''),
                                url_data.get('display_url', '')
                            ))
                
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} bookmarks...")
                
            except Exception as e:
                print(f"Error processing bookmark {bookmark.get('id', 'unknown')}: {e}")
                continue
        
        # Deduplicate data before batch inserts using dictionaries
        users_dict = {data[0]: data for data in users_data}  # Use screen_name as key
        tweets_dict = {data[0]: data for data in tweets_data}  # Use tweet ID as key
        hashtags_dict = {data[0]: data for data in hashtags_data}  # Use tag as key
        urls_dict = {(data[0], data[1]): data for data in urls_data}  # Use (tweet_id, url) as key
        media_dict = {(data[0], data[1]): data for data in media_data}  # Use (tweet_id, media_url) as key
        user_urls_dict = {(data[0], data[1]): data for data in user_urls_data}  # Use (user_id, url) as key
        
        # Convert back to lists
        users_data = list(users_dict.values())
        tweets_data = list(tweets_dict.values())
        hashtags_data = list(hashtags_dict.values())
        urls_data = list(urls_dict.values())
        media_data = list(media_dict.values())
        user_urls_data = list(user_urls_dict.values())
        
        # Batch insert users
        if users_data:
            print(f"Inserting {len(users_data)} unique users...")
            execute_values(
                cursor,
                """
                INSERT INTO users (id, name, verified, followers_count, following_count, description)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    verified = EXCLUDED.verified,
                    followers_count = EXCLUDED.followers_count,
                    following_count = EXCLUDED.following_count,
                    description = EXCLUDED.description
                """,
                users_data
            )
        
        # Batch insert tweets
        if tweets_data:
            print(f"Inserting {len(tweets_data)} unique tweets...")
            execute_values(
                cursor,
                """
                INSERT INTO tweets (id, user_id, text, created_at, retweet_count, favorite_count, 
                                  reply_count, quote_count, is_quote_status, quoted_tweet_id, url, has_media)
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
                    quoted_tweet_id = EXCLUDED.quoted_tweet_id,
                    url = EXCLUDED.url,
                    has_media = EXCLUDED.has_media
                """,
                tweets_data
            )
        
        # Batch insert hashtags
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
            
            # Get hashtag IDs for tweet_hashtags
            cursor.execute("SELECT id, tag FROM hashtags")
            hashtag_map = {tag: id for id, tag in cursor.fetchall()}
            
            # Create tweet_hashtags connections
            tweet_hashtag_values = [(tweet_id, hashtag_map.get(tag)) 
                                   for tweet_id, tag in tweet_hashtags_data
                                   if hashtag_map.get(tag)]
            
            if tweet_hashtag_values:
                print(f"Creating {len(tweet_hashtag_values)} tweet-hashtag connections...")
                execute_values(
                    cursor,
                    """
                    INSERT INTO tweet_hashtags (tweet_id, hashtag_id)
                    VALUES %s
                    ON CONFLICT (tweet_id, hashtag_id) DO NOTHING
                    """,
                    tweet_hashtag_values
                )
        
        # Batch insert URLs
        if urls_data:
            print(f"Inserting {len(urls_data)} URLs...")
            execute_values(
                cursor,
                """
                INSERT INTO urls (tweet_id, url, expanded_url, display_url)
                VALUES %s
                ON CONFLICT (tweet_id, url) DO UPDATE SET
                    expanded_url = EXCLUDED.expanded_url,
                    display_url = EXCLUDED.display_url
                """,
                urls_data
            )
        
        # Batch insert media
        if media_data:
            print(f"Inserting {len(media_data)} media items...")
            execute_values(
                cursor,
                """
                INSERT INTO media (tweet_id, media_url, type, alt_text)
                VALUES %s
                ON CONFLICT (tweet_id, media_url) DO UPDATE SET
                    type = EXCLUDED.type,
                    alt_text = EXCLUDED.alt_text
                """,
                media_data
            )
        
        # Batch insert user description URLs
        if user_urls_data:
            print(f"Inserting {len(user_urls_data)} user description URLs...")
            execute_values(
                cursor,
                """
                INSERT INTO user_description_urls (user_id, url, expanded_url, display_url)
                VALUES %s
                ON CONFLICT (user_id, url) DO UPDATE SET
                    expanded_url = EXCLUDED.expanded_url,
                    display_url = EXCLUDED.display_url
                """,
                user_urls_data
            )
        
        # Commit the transaction
        conn.commit()
        print(f"Successfully imported {processed_count} bookmarks from {file_path}")
        return processed_count
    
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