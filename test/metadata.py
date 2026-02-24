import json
import os
from dotenv import load_dotenv
import requests
import time

load_dotenv()

# Reuse authentication details from the main script
BOOKMARKS_API_ID = os.getenv("BOOKMARKS_API_ID")
API_URL = f"https://x.com/i/api/graphql/{BOOKMARKS_API_ID}/Bookmarks"
AUTH_TOKEN = f"Bearer {os.getenv('X_AUTH_TOKEN')}"
COOKIE = os.getenv("X_COOKIE_STRING")
CSRF_TOKEN = os.getenv("X_CSRF_TOKEN")
USER_AGENT = os.getenv("USER_AGENT")

headers = {
    "Authorization": AUTH_TOKEN,
    "Cookie": COOKIE,
    "x-csrf-token": CSRF_TOKEN,
    "User-Agent": USER_AGENT,
    "x-twitter-client-language": "en",
    "x-twitter-active-user": "yes",
    "Content-Type": "application/json",
}

# Copy features dictionary from the main script
features = {
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

def fetch_bookmarks():
    """Fetch 10 bookmarks to examine their metadata structure"""
    global features
    
    variables = {
        "count": 10,  # Get 10 tweets
        "cursor": None,
        "includePromotedContent": False
    }

    params = {
        "features": json.dumps(features),
        "variables": json.dumps(variables),
    }

    response = requests.get(API_URL, headers=headers, params=params)
    
    if response.status_code == 429:
        print("Rate limited. Waiting 60 seconds...")
        time.sleep(60)
        return fetch_bookmarks()
    
    if response.status_code == 400:
        try:
            data = response.json()
            if data.get("errors") and data["errors"][0].get("message"):
                error_msg = data["errors"][0]["message"]
                if "features cannot be null" in error_msg:
                    # Extract missing features from error message
                    missing_features = error_msg.split("cannot be null: ")[1].split(", ")
                    print(f"Adding missing features: {missing_features}")
                    
                    # Update our features dictionary with the missing features
                    for feature in missing_features:
                        features[feature.strip()] = True
                    
                    print("Retrying with updated features...")
                    # Retry the request with updated features
                    return fetch_bookmarks()
        except Exception as e:
            print(f"Error processing response: {e}")
    
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None
    
    return response.json()

def save_full_metadata():
    """Fetch bookmarks and save their full metadata structure"""
    result = fetch_bookmarks()
    
    if not result:
        print("No result returned")
        return
    
    try:
        entries = result.get("data", {}).get("bookmark_timeline_v2", {}).get("timeline", {}).get("instructions", [])[0].get("entries", [])
        
        tweet_entries = [e for e in entries if e["entryId"].startswith("tweet-")]
        
        if not tweet_entries:
            print("No tweet entries found")
            return
        
        # Save the full metadata of all tweets
        with open("tweet_full_metadata.json", "w", encoding="utf-8") as f:
            json.dump(tweet_entries, f, ensure_ascii=False, indent=2)
            
        print("Full metadata saved to tweet_full_metadata.json")
        
        # Extract and save just the tweet_results portion for easier analysis
        tweet_results = [entry["content"]["itemContent"]["tweet_results"]["result"] for entry in tweet_entries]
        with open("tweet_results_metadata.json", "w", encoding="utf-8") as f:
            json.dump(tweet_results, f, ensure_ascii=False, indent=2)
            
        print("Tweet results metadata saved to tweet_results_metadata.json")
        
    except Exception as e:
        print(f"Error processing result: {e}")

if __name__ == "__main__":
    save_full_metadata()