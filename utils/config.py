import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
BOOKMARKS_API_ID = os.getenv("BOOKMARKS_API_ID")
API_URL = f"https://x.com/i/api/graphql/{BOOKMARKS_API_ID}/Bookmarks"
AUTH_TOKEN = f"Bearer {os.getenv('X_AUTH_TOKEN')}"
COOKIE = os.getenv("X_COOKIE_STRING")
CSRF_TOKEN = os.getenv("X_CSRF_TOKEN")
USER_AGENT = os.getenv("USER_AGENT")

# API Headers
API_HEADERS = {
    "Authorization": AUTH_TOKEN,
    "Cookie": COOKIE,
    "x-csrf-token": CSRF_TOKEN,
    "User-Agent": USER_AGENT,
    "x-twitter-client-language": "en",
    "x-twitter-active-user": "yes",
    "Content-Type": "application/json",
}

# API Features
API_FEATURES = {
    "responsive_web_graphql_exclude_directive_enabled": True,
    "graphql_timeline_v2_bookmark_timeline": True,
    "rweb_tipjar_consumption_enabled": True,
    "verified_phone_label_enabled": False,
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "responsive_web_graphql_timeline_navigation_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "communities_web_enable_tweet_community_results_fetch": True,
    "c9s_tweet_anatomy_moderator_badge_enabled": True,
    "articles_preview_enabled": True,
    "tweetypie_unmention_optimization_enabled": True,
    "responsive_web_edit_tweet_api_enabled": True,
    "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
    "view_counts_everywhere_api_enabled": True,
    "longform_notetweets_consumption_enabled": True,
    "responsive_web_twitter_article_tweet_consumption_enabled": True,
    "tweet_awards_web_tipping_enabled": False,
    "creator_subscriptions_quote_tweet_preview_enabled": False,
    "freedom_of_speech_not_reach_fetch_enabled": True,
    "standardized_nudges_misinfo": True,
    "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
    "rweb_video_timestamps_enabled": True,
    "longform_notetweets_rich_text_read_enabled": True,
    "longform_notetweets_inline_media_enabled": True,
    "responsive_web_enhance_cards_enabled": False,
    "profile_label_improvements_pcf_label_in_post_enabled": False,
    "premium_content_api_read_enabled": True,
    "responsive_web_grok_analyze_button_fetch_trends_enabled": True,
    "responsive_web_grok_analyze_post_followups_enabled": True,
    "responsive_web_grok_share_attachment_enabled": True,
}

# Scraping Configuration
BATCH_SIZE = 100
RATE_LIMIT_DELAY = 2  # seconds
MAX_EMPTY_PAGES = 3

# Database Configuration
DB_CONFIG = {
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
}

# Database Schema
DB_SCHEMA = """
-- Create the vector extension if it doesn't exist
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255),
    verified BOOLEAN DEFAULT FALSE,
    followers_count INTEGER DEFAULT 0,
    following_count INTEGER DEFAULT 0,
    description TEXT
);

CREATE TABLE IF NOT EXISTS tweets (
    id VARCHAR(20) PRIMARY KEY,
    user_id VARCHAR(255) REFERENCES users(id),
    text TEXT,
    created_at TIMESTAMP WITH TIME ZONE,
    retweet_count INTEGER DEFAULT 0,
    favorite_count INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,
    quote_count INTEGER DEFAULT 0,
    is_quote_status BOOLEAN DEFAULT FALSE,
    quoted_tweet_id VARCHAR(20),
    url TEXT,
    has_media BOOLEAN DEFAULT FALSE,
    embedding vector(1536)
);

CREATE TABLE IF NOT EXISTS hashtags (
    id SERIAL PRIMARY KEY,
    tag VARCHAR(255) UNIQUE
);

CREATE TABLE IF NOT EXISTS tweet_hashtags (
    tweet_id VARCHAR(20) REFERENCES tweets(id),
    hashtag_id INTEGER REFERENCES hashtags(id),
    PRIMARY KEY (tweet_id, hashtag_id)
);

CREATE TABLE IF NOT EXISTS urls (
    tweet_id VARCHAR(20) REFERENCES tweets(id),
    url TEXT,
    expanded_url TEXT,
    display_url TEXT,
    PRIMARY KEY (tweet_id, url)
);

CREATE TABLE IF NOT EXISTS media (
    tweet_id VARCHAR(20) REFERENCES tweets(id),
    media_url TEXT,
    type VARCHAR(50),
    alt_text TEXT,
    PRIMARY KEY (tweet_id, media_url)
);

CREATE TABLE IF NOT EXISTS user_description_urls (
    user_id VARCHAR(255) REFERENCES users(id),
    url TEXT,
    expanded_url TEXT,
    display_url TEXT,
    PRIMARY KEY (user_id, url)
);

CREATE TABLE IF NOT EXISTS tweet_interpretations (
    tweet_id VARCHAR(20) REFERENCES tweets(id) PRIMARY KEY,
    interpretation TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""