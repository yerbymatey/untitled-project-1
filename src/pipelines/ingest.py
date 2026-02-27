import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.db.session import get_db_session
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime objects"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class BookmarkIngester:
    def __init__(self):
        self.hashtag_map = {}  # Cache for hashtag IDs
    
    def _get_or_create_hashtags(self, session, tags: Set[str]) -> Dict[str, int]:
        """Get or create hashtag IDs in batch"""
        # First try to get existing hashtags
        if tags:
            placeholders = ','.join(['%s'] * len(tags))
            session.execute(
                f"SELECT id, tag FROM hashtags WHERE tag IN ({placeholders})",
                tuple(tags)
            )
            existing = {row['tag']: row['id'] for row in session.fetchall()}
            
            # Update cache with existing hashtags
            self.hashtag_map.update(existing)
            
            # Find missing hashtags
            missing = tags - set(existing.keys())
            
            if missing:
                # Insert missing hashtags
                values = [(tag,) for tag in missing]
                session.execute_values(
                    "INSERT INTO hashtags (tag) VALUES %s RETURNING id, tag",
                    values
                )
                new_hashtags = {row['tag']: row['id'] for row in session.fetchall()}
                self.hashtag_map.update(new_hashtags)
        
        return self.hashtag_map
    
    def _insert_users(self, session, users: List[Dict]) -> None:
        """Insert or update users in batch"""
        if not users:
            return
            
        # Prepare user data with validation
        user_values = []
        for user in users:
            # Use screen_name as the primary ID
            user_id = user.get('screen_name')
            if not user_id:
                logger.warning(f"Skipping user with missing screen_name: {user.get('name', 'unknown')}")
                continue
                
            user_values.append((
                user_id,  # Use screen_name as the ID
                user.get('name', ''),
                user.get('verified', False),
                user.get('followers_count', 0),
                user.get('following_count', 0),
                user.get('description', '')
            ))
        
        if not user_values:
            logger.warning("No valid users to insert")
            return
        
        # Insert/update users
        session.execute_values(
            """
            INSERT INTO users (
                id, name, verified, followers_count, following_count, description
            )
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                verified = EXCLUDED.verified,
                followers_count = EXCLUDED.followers_count,
                following_count = EXCLUDED.following_count,
                description = EXCLUDED.description
            """,
            user_values
        )
        
        # Handle user description URLs
        url_values = []
        for user in users:
            user_id = user.get('screen_name')
            if not user_id:
                continue
                
            if user.get('description_urls'):
                url_values.extend([
                    (user_id, url['url'])
                    for url in user['description_urls']
                ])
        
        if url_values:
            session.execute_values(
                """
                INSERT INTO user_description_urls (user_id, url)
                VALUES %s
                ON CONFLICT (user_id, url) DO NOTHING
                """,
                url_values
            )
    
    def _extract_user_id(self, user: Dict) -> Optional[str]:
        """Extract and validate user ID from user data"""
        user_id = user.get('screen_name')
        if not user_id:
            logger.warning(f"Skipping user with missing screen_name: {user.get('name', 'unknown')}")
        return user_id

    def _collect_users(self, bookmarks: List[Dict]) -> List[Dict]:
        """Collect and deduplicate users from bookmarks"""
        user_map = {}
        for bookmark in bookmarks:
            # Add main tweet user
            user = bookmark['user']
            user_id = self._extract_user_id(user)
            if user_id:
                user_map[user_id] = user
            
            # Add quoted tweet user if present
            if bookmark.get('quoted_status'):
                quoted_user = bookmark['quoted_status']['user']
                quoted_user_id = self._extract_user_id(quoted_user)
                if quoted_user_id:
                    user_map[quoted_user_id] = quoted_user
        
        return list(user_map.values())

    def _collect_tweet_data(self, session, tweets: List[Dict]) -> Dict:
        """Collect all tweet-related data in a single pass"""
        data = {
            'tweet_values': [],
            'hashtags': set(),
            'url_values': [],
            'media_values': []
        }
        
        for tweet in tweets:
            # Get user_id from screen_name
            user_id = tweet['user'].get('screen_name')
            if not user_id:
                logger.warning(f"Skipping tweet {tweet.get('id', 'unknown')} with missing user screen_name")
                continue
            
            # Collect tweet data
            data['tweet_values'].append((
                tweet['id'],
                user_id,
                tweet['text'],
                tweet['created_at'],
                tweet['retweet_count'],
                tweet['favorite_count'],
                tweet['reply_count'],
                tweet['quote_count'],
                tweet['is_quote_status'],
                tweet['quoted_status']['id'] if tweet.get('quoted_status') else None,
                tweet['url'],
                tweet['media']['has_media']
            ))
            
            # Collect hashtags
            if tweet.get('hashtags'):
                data['hashtags'].update(tweet['hashtags'])
            
            # Collect URLs
            if tweet.get('urls'):
                data['url_values'].extend([
                    (tweet['id'], url['url'])
                    for url in tweet['urls']
                ])
            
            # Collect media
            if tweet.get('media', {}).get('items'):
                data['media_values'].extend([
                    (
                        tweet['id'],
                        media['media_url'],
                        media['type'],
                        media.get('alt_text')
                    ) for media in tweet['media']['items']
                ])
        
        return data

    def _insert_tweets(self, session, tweets: List[Dict]) -> None:
        """Insert or update tweets and related entities in batch"""
        if not tweets:
            return
        
        # Collect all tweet-related data in a single pass
        data = self._collect_tweet_data(session, tweets)
        
        if not data['tweet_values']:
            logger.warning("No valid tweets to insert")
            return
        
        # Insert/update tweets
        session.execute_values(
            """
            INSERT INTO tweets (
                id, user_id, text, created_at, retweet_count, favorite_count,
                reply_count, quote_count, is_quote_status, quoted_tweet_id, url, has_media
            )
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
            data['tweet_values']
        )
        
        # Get or create hashtag IDs
        hashtag_map = self._get_or_create_hashtags(session, data['hashtags'])
        
        # Prepare tweet-hashtag relationships
        tweet_hashtag_values = [
            (tweet['id'], hashtag_map[tag])
            for tweet in tweets
            if tweet.get('hashtags')
            for tag in tweet['hashtags']
        ]
        
        if tweet_hashtag_values:
            session.execute_values(
                """
                INSERT INTO tweet_hashtags (tweet_id, hashtag_id)
                VALUES %s
                ON CONFLICT (tweet_id, hashtag_id) DO NOTHING
                """,
                tweet_hashtag_values
            )
        
        # Handle URLs
        if data['url_values']:
            session.execute_values(
                """
                INSERT INTO urls (tweet_id, url)
                VALUES %s
                ON CONFLICT (tweet_id, url) DO NOTHING
                """,
                data['url_values']
            )
        
        # Handle media
        if data['media_values']:
            session.execute_values(
                """
                INSERT INTO media (tweet_id, media_url, type, alt_text)
                VALUES %s
                ON CONFLICT (tweet_id, media_url) DO UPDATE SET
                    type = EXCLUDED.type,
                    alt_text = EXCLUDED.alt_text
                """,
                data['media_values']
            )

    def save_to_file(self, bookmarks: List[Dict], output_dir: str = ".") -> str:
        """Save bookmarks to a JSON file with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bookmarks_{timestamp}.json"
        output_path = Path(output_dir) / filename
        
        data = {
            "total_bookmarks": len(bookmarks),
            "fetch_date": datetime.now().isoformat(),
            "bookmarks": bookmarks
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
        
        return str(output_path)
    
    def ingest_bookmarks(self, bookmarks: List[Dict], save_to_file: bool = True) -> Optional[str]:
        """Ingest bookmarks into the database and optionally save to file"""
        logger.info(f"Starting ingestion of {len(bookmarks)} bookmarks...")
        
        # Save to file if requested
        file_path = None
        if save_to_file:
            file_path = self.save_to_file(bookmarks)
            logger.info(f"Saved bookmarks to: {file_path}")
        
        # Ingest into database
        with get_db_session() as session:
            try:
                # Collect and deduplicate users
                users = self._collect_users(bookmarks)
                
                # Insert users first
                self._insert_users(session, users)
                
                # Insert tweets and related entities
                self._insert_tweets(session, bookmarks)
                
                logger.info("Successfully ingested all bookmarks into database")
                return file_path
                
            except Exception as e:
                logger.error(f"Error during bookmark ingestion: {e}")
                raise 