import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from db.session import get_db_session
from utils.logger import setup_logger

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
    
    def _resolve_screen_name(self, user: Optional[Dict], fallback_url: Optional[str] = None) -> Optional[str]:
        """Resolve a user's handle/screen_name from available metadata."""
        if not user:
            return None

        handle = (
            user.get('screen_name')
            or user.get('handle')
            or user.get('legacy', {}).get('screen_name')
        )

        if not handle:
            # Try to parse from URLs like https://x.com/{handle}/status/...
            candidate_sources = [fallback_url, user.get('url')]
            for source in candidate_sources:
                if not source:
                    continue
                match = re.search(r"https?://(?:x|twitter)\.com/([^/?]+)/", source)
                if match:
                    handle = match.group(1)
                    break

        if handle:
            # Persist the resolved value back on the user dict so downstream consumers can rely on it
            user.setdefault('screen_name', handle)
        else:
            logger.warning(f"Unable to resolve screen_name for user payload: {user.get('name', 'unknown user')}")

        return handle

    def _insert_users(self, session, users: List[Dict]) -> None:
        """Insert or update users in batch"""
        if not users:
            return
            
        # Prepare user data with validation
        user_values = []
        for user in users:
            handle = self._resolve_screen_name(user)
            user_id = user.get('id') or handle
            if not user_id:
                continue

            user_values.append((
                user_id,
                user.get('name', ''),
                bool(user.get('verified', False)),
                user.get('followers_count', 0) or 0,
                user.get('following_count', 0) or 0,
                user.get('description', '') or ''
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
            handle = self._resolve_screen_name(user)
            user_id = user.get('id') or handle
            if not user_id:
                continue

            if user.get('description_urls'):
                url_values.extend([
                    (user_id, url['url'])
                    for url in user['description_urls']
                    if url.get('url')
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
        return self._resolve_screen_name(user)

    def _collect_users(self, bookmarks: List[Dict]) -> List[Dict]:
        """Collect and deduplicate users from bookmarks"""
        user_map = {}
        for bookmark in bookmarks:
            # Add main tweet user
            user = bookmark['user']
            handle = self._resolve_screen_name(user, bookmark.get('url'))
            user_id = user.get('id') or handle
            if user_id:
                user_map[user_id] = user
            
            # Add quoted tweet user if present
            if bookmark.get('quoted_status'):
                quoted_user = bookmark['quoted_status']['user']
                quoted_handle = self._resolve_screen_name(quoted_user, bookmark['quoted_status'].get('url'))
                quoted_user_id = quoted_user.get('id') or quoted_handle
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
            # Get user_id from provided id or fallback handle
            handle = self._resolve_screen_name(tweet['user'], tweet.get('url'))
            user_id = tweet['user'].get('id') or handle
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
                    (tweet['id'], url['url'], url.get('expanded_url'), url.get('display_url'))
                    for url in tweet['urls']
                ])
            
            # Collect media
            if tweet.get('media', {}).get('items'):
                data['media_values'].extend([
                    (
                        tweet['id'],
                        media['media_url'],
                        media['type'],
                        media.get('alt_text'),
                        None  # Placeholder for extr_text
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
                INSERT INTO urls (tweet_id, url, expanded_url, display_url)
                VALUES %s
                ON CONFLICT (tweet_id, url) DO UPDATE SET
                    expanded_url = COALESCE(EXCLUDED.expanded_url, urls.expanded_url),
                    display_url = COALESCE(EXCLUDED.display_url, urls.display_url)
                """,
                data['url_values']
            )
        
        # Handle media
        if data['media_values']:
            session.execute_values(
                """
                INSERT INTO media (tweet_id, media_url, type, alt_text, extr_text)
                VALUES %s
                ON CONFLICT (tweet_id, media_url) DO UPDATE SET
                    type = EXCLUDED.type,
                    alt_text = EXCLUDED.alt_text
                    -- extr_text is only inserted initially, not updated here
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
    
    def _extract_og_tweets(self, bookmarks: List[Dict]) -> List[Dict]:
        """Extract original tweets from quote tweet bookmarks as independent tweet dicts.

        Returns a list of tweet dicts in the same shape as regular bookmarks so they
        can be fed through _insert_users / _insert_tweets unchanged.
        Skips OG tweets with no resolvable user (FK constraint requires valid user_id).
        """
        og_tweets = {}
        skipped = 0
        for bm in bookmarks:
            qs = bm.get('quoted_status')
            if not qs or not qs.get('id'):
                continue
            og_id = qs['id']
            if og_id in og_tweets:
                continue  # dedup

            user = qs.get('user', {})
            # Resolve user_id — must have one for FK constraint
            user_id = (
                user.get('id')
                or user.get('screen_name')
                or user.get('handle')
                or user.get('legacy', {}).get('screen_name')
            )
            if not user_id:
                # Try to parse from URL
                og_url = qs.get('url', '')
                import re as _re
                match = _re.search(r"https?://(?:x|twitter)\.com/([^/?]+)/", og_url)
                if match:
                    user_id = match.group(1)
                    user['screen_name'] = user_id

            if not user_id:
                skipped += 1
                continue

            # Normalize the quoted_status into the same shape as a top-level bookmark
            og_tweet = {
                'id': og_id,
                'text': qs.get('text', ''),
                'created_at': qs.get('created_at'),
                'retweet_count': qs.get('retweet_count', 0),
                'favorite_count': qs.get('favorite_count', 0),
                'reply_count': qs.get('reply_count', 0),
                'quote_count': qs.get('quote_count', 0),
                'is_quote_status': False,
                'quoted_status': None,
                'url': qs.get('url', ''),
                'user': user,
                'hashtags': qs.get('hashtags', []),
                'urls': qs.get('urls', []),
                'media': qs.get('media', {'has_media': False, 'items': []}),
            }
            og_tweets[og_id] = og_tweet

        if skipped:
            logger.warning(f"Skipped {skipped} OG tweets with no resolvable user")
        return list(og_tweets.values())

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
                # Extract OG tweets from quoted statuses
                og_tweets = self._extract_og_tweets(bookmarks)
                if og_tweets:
                    logger.info(f"Extracted {len(og_tweets)} original tweets from quote tweets")

                # Combine all tweets for user collection
                all_tweets = bookmarks + og_tweets

                # Collect and deduplicate users
                users = self._collect_users(all_tweets)
                
                # Insert users first
                self._insert_users(session, users)

                # Insert OG tweets first (so FK from quoted_tweet_id resolves)
                if og_tweets:
                    self._insert_tweets(session, og_tweets)
                    logger.info(f"Inserted {len(og_tweets)} original tweets")
                
                # Insert bookmarked tweets and related entities
                self._insert_tweets(session, bookmarks)
                
                logger.info("Successfully ingested all bookmarks into database")
                return file_path
                
            except Exception as e:
                logger.error(f"Error during bookmark ingestion: {e}")
                raise 
