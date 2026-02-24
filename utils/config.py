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
    "x-twitter-auth-type": "OAuth2Session",
}

# API Features
API_FEATURES = {
    "rweb_video_screen_enabled": False,
    "profile_label_improvements_pcf_label_in_post_enabled": True,
    "responsive_web_profile_redirect_enabled": False,
    "rweb_tipjar_consumption_enabled": False,
    "verified_phone_label_enabled": False,
    "creator_subscriptions_tweet_preview_api_enabled": True,
    "responsive_web_graphql_timeline_navigation_enabled": True,
    "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
    "premium_content_api_read_enabled": False,
    "communities_web_enable_tweet_community_results_fetch": True,
    "c9s_tweet_anatomy_moderator_badge_enabled": True,
    "responsive_web_grok_analyze_button_fetch_trends_enabled": False,
    "responsive_web_grok_analyze_post_followups_enabled": True,
    "responsive_web_jetfuel_frame": True,
    "responsive_web_grok_share_attachment_enabled": True,
    "responsive_web_grok_annotations_enabled": True,
    "articles_preview_enabled": True,
    "responsive_web_edit_tweet_api_enabled": True,
    "graphql_is_translatable_rweb_tweet_is_translatable_enabled": True,
    "view_counts_everywhere_api_enabled": True,
    "longform_notetweets_consumption_enabled": True,
    "responsive_web_twitter_article_tweet_consumption_enabled": True,
    "tweet_awards_web_tipping_enabled": False,
    "responsive_web_grok_show_grok_translated_post": True,
    "responsive_web_grok_analysis_button_from_backend": True,
    "post_ctas_fetch_enabled": True,
    "freedom_of_speech_not_reach_fetch_enabled": True,
    "standardized_nudges_misinfo": True,
    "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": True,
    "longform_notetweets_rich_text_read_enabled": True,
    "longform_notetweets_inline_media_enabled": True,
    "responsive_web_grok_image_annotation_enabled": True,
    "responsive_web_grok_imagine_annotation_enabled": True,
    "responsive_web_grok_community_note_auto_translation_is_enabled": False,
    "responsive_web_enhance_cards_enabled": False,
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
