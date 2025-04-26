import json
import time
import re
from typing import Dict, List, Optional

import requests
from utils.config import API_URL, API_HEADERS, API_FEATURES, BATCH_SIZE, RATE_LIMIT_DELAY, MAX_EMPTY_PAGES
from utils.logger import setup_logger
from db.session import get_db_session

logger = setup_logger(__name__)

class TwitterBookmarkScraper:
    def __init__(self):
        self.features = API_FEATURES.copy()
    
    def fetch_bookmarks(self, cursor: Optional[str] = None) -> Optional[Dict]:
        """Fetch a page of bookmarks from the X/Twitter API"""
        variables = {
            "count": BATCH_SIZE,
            "cursor": cursor,
            "includePromotedContent": False
        }

        params = {
            "features": json.dumps(self.features),
            "variables": json.dumps(variables),
        }

        response = requests.get(API_URL, headers=API_HEADERS, params=params)
        
        if response.status_code == 429:
            logger.warning("Rate limited. Waiting 60 seconds...")
            time.sleep(60)
            return self.fetch_bookmarks(cursor)
        
        if response.status_code == 400:
            try:
                data = response.json()
                if data.get("errors") and data["errors"][0].get("message"):
                    error_msg = data["errors"][0]["message"]
                    if "features cannot be null" in error_msg:
                        missing_features = error_msg.split("cannot be null: ")[1].split(", ")
                        logger.info(f"Adding missing features: {missing_features}")
                        
                        for feature in missing_features:
                            self.features[feature.strip()] = True
                        
                        logger.info("Retrying with updated features...")
                        return self.fetch_bookmarks(cursor)
            except Exception as e:
                logger.error(f"Error processing response: {e}")
        
        if response.status_code != 200:
            logger.error(f"Error {response.status_code}: {response.text}")
            return None
        
        return response.json()

    def extract_tweet_data(self, entry: Dict) -> Dict:
        """Extract relevant tweet data from the API response"""
        tweet_result = entry["content"]["itemContent"]["tweet_results"]["result"]
        legacy = tweet_result["legacy"]
        user = tweet_result["core"]["user_results"]["result"]["legacy"]
        
        # Get full text without modifying newlines
        text = legacy.get("full_text", "")
        
        # Strip out image links (https://t.co/...)
        cleaned_text = re.sub(r'https://t\.co/\w+', '', text)
        
        # Extract media information if available
        media_items = []
        hashtags = []
        urls = []
        
        # Check for media and extract details
        if "extended_entities" in legacy and "media" in legacy["extended_entities"]:
            for media in legacy["extended_entities"]["media"]:
                media_item = {
                    "media_url": media.get("media_url_https"),
                    "type": media.get("type"),
                    "alt_text": media.get("ext_alt_text")
                }
                # Add video info if available
                if media.get("type") == "video" or media.get("type") == "animated_gif":
                    if "video_info" in media:
                        media_item["video_info"] = {
                            "duration_millis": media["video_info"].get("duration_millis"),
                            "variants": media["video_info"].get("variants")
                        }
                media_items.append(media_item)
        
        # Extract hashtags if available
        if "entities" in legacy and "hashtags" in legacy["entities"]:
            hashtags = [hashtag["text"] for hashtag in legacy["entities"]["hashtags"]]
        
        # Extract URLs if available
        if "entities" in legacy and "urls" in legacy["entities"]:
            for url_entity in legacy["entities"]["urls"]:
                url_item = {
                    "url": url_entity.get("url")
                }
                urls.append(url_item)
        
        # Extract user description and related URLs if available
        user_description = user.get("description", "")
        user_description_urls = []
        
        if "entities" in user and "description" in user["entities"] and "urls" in user["entities"]["description"]:
            for url_entity in user["entities"]["description"]["urls"]:
                url_item = {
                    "url": url_entity.get("url")
                }
                user_description_urls.append(url_item)

        # Extract quoted tweet data if available
        quoted_status = None
        if tweet_result.get("quoted_status_result", {}).get("result"):
            quoted_result = tweet_result["quoted_status_result"]["result"]
            if quoted_result.get("__typename") == "Tweet":
                quoted_legacy = quoted_result.get("legacy", {})
                quoted_user = quoted_result.get("core", {}).get("user_results", {}).get("result", {}).get("legacy", {})
                
                # Extract quoted tweet media
                quoted_media_items = []
                if "extended_entities" in quoted_legacy and "media" in quoted_legacy["extended_entities"]:
                    for media in quoted_legacy["extended_entities"]["media"]:
                        media_item = {
                            "media_url": media.get("media_url_https"),
                            "type": media.get("type"),
                            "alt_text": media.get("ext_alt_text")
                        }
                        if media.get("type") == "video" or media.get("type") == "animated_gif":
                            if "video_info" in media:
                                media_item["video_info"] = {
                                    "duration_millis": media["video_info"].get("duration_millis"),
                                    "variants": media["video_info"].get("variants")
                                }
                        quoted_media_items.append(media_item)
                
                # Extract quoted tweet hashtags
                quoted_hashtags = []
                if "entities" in quoted_legacy and "hashtags" in quoted_legacy["entities"]:
                    quoted_hashtags = [hashtag["text"] for hashtag in quoted_legacy["entities"]["hashtags"]]
                
                # Extract quoted tweet URLs
                quoted_urls = []
                if "entities" in quoted_legacy and "urls" in quoted_legacy["entities"]:
                    for url_entity in quoted_legacy["entities"]["urls"]:
                        url_item = {
                            "url": url_entity.get("url")
                        }
                        quoted_urls.append(url_item)
                
                quoted_status = {
                    "id": quoted_legacy.get("id_str"),
                    "text": quoted_legacy.get("full_text", ""),
                    "created_at": quoted_legacy.get("created_at"),
                    "retweet_count": quoted_legacy.get("retweet_count"),
                    "favorite_count": quoted_legacy.get("favorite_count"),
                    "reply_count": quoted_legacy.get("reply_count"),
                    "quote_count": quoted_legacy.get("quote_count"),
                    "hashtags": quoted_hashtags,
                    "urls": quoted_urls,
                    "media": {
                        "items": quoted_media_items,
                        "has_media": len(quoted_media_items) > 0
                    },
                    "user": {
                        "id": quoted_user.get("id_str"),
                        "name": quoted_user.get("name"),
                        "screen_name": quoted_user.get("screen_name"),
                        "verified": quoted_user.get("verified", False),
                        "followers_count": quoted_user.get("followers_count"),
                        "following_count": quoted_user.get("friends_count"),
                        "description": quoted_user.get("description")
                    },
                    "url": f"https://x.com/{quoted_user.get('screen_name')}/status/{quoted_legacy.get('id_str')}"
                }
        
        return {
            "id": entry["entryId"].split("-")[1],
            "text": cleaned_text,
            "created_at": legacy.get("created_at"),
            "retweet_count": legacy.get("retweet_count"),
            "favorite_count": legacy.get("favorite_count"),
            "reply_count": legacy.get("reply_count"),
            "quote_count": legacy.get("quote_count"),
            "is_quote_status": legacy.get("is_quote_status", False),
            "hashtags": hashtags,
            "urls": urls,
            "media": {
                "items": media_items,
                "has_media": len(media_items) > 0
            },
            "user": {
                "id": user.get("id_str"),
                "name": user.get("name"),
                "screen_name": user.get("screen_name"),
                "verified": user.get("verified", False),
                "followers_count": user.get("followers_count"),
                "following_count": user.get("friends_count"),
                "description": user_description,
                "description_urls": user_description_urls
            },
            "quoted_status": quoted_status,
            "url": f"https://x.com/{user.get('screen_name')}/status/{entry['entryId'].split('-')[1]}"
        }

    def _check_if_tweet_exists(self, tweet_id: str) -> bool:
        """Check if a tweet with the given ID exists in the database."""
        try:
            with get_db_session() as session:
                session.execute("SELECT 1 FROM tweets WHERE id = %s LIMIT 1", (tweet_id,))
                exists = session.fetchone() is not None
                if exists:
                    logger.info(f"Tweet ID {tweet_id} already exists in the database.")
                return exists
        except Exception as e:
            logger.error(f"Database error checking tweet {tweet_id}: {e}")
            return False # Assume not exists on error to avoid stopping prematurely

    def scrape_all_bookmarks(self) -> List[Dict]:
        """Fetch all bookmarks and return as a list of tweet data"""
        cursor = None
        all_tweets = []
        total_fetched = 0
        empty_pages_count = 0

        logger.info("Starting bookmark fetch...")
        while True:
            result = self.fetch_bookmarks(cursor)
            if not result:
                logger.warning("No result returned, ending fetch...")
                break

            try:
                entries = result.get("data", {}).get("bookmark_timeline_v2", {}).get("timeline", {}).get("instructions", [])[0].get("entries", [])
                logger.info(f"Found {len(entries)} entries")
                
                tweet_entries = [e for e in entries if e["entryId"].startswith("tweet-")]
                logger.info(f"Found {len(tweet_entries)} tweet entries")

                # Handle empty pages
                if len(tweet_entries) == 0:
                    empty_pages_count += 1
                    logger.info(f"No tweets on this page (empty page #{empty_pages_count})")
                    
                    if empty_pages_count >= MAX_EMPTY_PAGES:
                        logger.info("Multiple empty pages in a row - reached end of bookmarks")
                        break
                else:
                    empty_pages_count = 0

                    # --- Check if first tweet on page already exists ---
                    first_entry_id_raw = tweet_entries[0].get("entryId")
                    if first_entry_id_raw and first_entry_id_raw.startswith("tweet-"):
                        first_tweet_id = first_entry_id_raw.split("-")[1]
                        if self._check_if_tweet_exists(first_tweet_id):
                            logger.info("First tweet on page already exists. Stopping bookmark fetch.")
                            break # Stop pagination
                    # -------------------------------------------------

                # Process tweets from this page
                for entry in tweet_entries:
                    try:
                        tweet_data = self.extract_tweet_data(entry)
                        all_tweets.append(tweet_data)
                        total_fetched += 1
                        logger.info(f"Fetched tweet {total_fetched}: {tweet_data['url']}")
                    except Exception as e:
                        logger.error(f"Error processing tweet: {e}")

                # Check for next page cursor
                cursor_entries = [e for e in entries if e["entryId"].startswith("cursor-bottom-")]
                if cursor_entries:
                    cursor = cursor_entries[0]["content"]["value"]
                    logger.info("Found next cursor, continuing...")
                else:
                    logger.info("No more cursors - reached end of bookmarks!")
                    break

                time.sleep(RATE_LIMIT_DELAY)

            except Exception as e:
                logger.error(f"Error processing result: {e}")
                break

        logger.info(f"Bookmark fetch complete! Total tweets fetched: {len(all_tweets)}")
        return all_tweets