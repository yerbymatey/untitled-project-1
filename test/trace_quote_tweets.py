import json
import os
import requests
import time
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Use the same auth and headers from your bookmark fetcher
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

# Base features needed - copied from your working script
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
    # Adding the new required features
    "responsive_web_jetfuel_frame": True,
    "responsive_web_grok_image_annotation_enabled": True,
    "responsive_web_grok_analysis_button_from_backend": True,
    "rweb_video_screen_enabled": True
}

def fetch_tweet_by_id(tweet_id):
    """Fetch a tweet by its ID using the Status endpoint"""
    # Using the Status endpoint which is what the web UI uses to fetch individual tweets
    status_api_id = os.getenv("STATUS_API_ID", "2ICDjqPd81tulZcYQgKuJw")
    API_URL = f"https://x.com/i/api/graphql/{status_api_id}/TweetResultByRestId"
    
    variables = {
        "tweetId": tweet_id,
        "withCommunity": True,
        "includePromotedContent": False,
        "withVoice": True
    }

    params = {
        "features": json.dumps(features),
        "variables": json.dumps(variables),
    }

    response = requests.get(API_URL, headers=headers, params=params)
    
    if response.status_code == 429:
        print("Rate limited. Waiting 60 seconds...")
        time.sleep(60)
        return fetch_tweet_by_id(tweet_id)
    
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
                    return fetch_tweet_by_id(tweet_id)
        except Exception as e:
            print(f"Error processing response: {e}")
    
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None
    
    return response.json()

def extract_tweet_data(tweet_result):
    """Extract relevant tweet data from the API response"""
    try:
        if tweet_result.get("__typename") != "Tweet":
            print(f"Not a standard tweet: {tweet_result.get('__typename')}")
            return None
        
        legacy = tweet_result.get("legacy", {})
        user = tweet_result.get("core", {}).get("user_results", {}).get("result", {}).get("legacy", {})
        
        # Extract quoted status if available
        quoted_status_result = None
        if tweet_result.get("quoted_status_result", {}).get("result"):
            quoted_result = tweet_result["quoted_status_result"]["result"]
            if quoted_result.get("__typename") == "Tweet":
                quoted_legacy = quoted_result.get("legacy", {})
                quoted_user = quoted_result.get("core", {}).get("user_results", {}).get("result", {}).get("legacy", {})
                
                quoted_status_result = {
                    "id": quoted_legacy.get("id_str"),
                    "text": quoted_legacy.get("full_text", ""),
                    "created_at": quoted_legacy.get("created_at"),
                    "user": {
                        "name": quoted_user.get("name"),
                        "screen_name": quoted_user.get("screen_name"),
                        "description": quoted_user.get("description"),
                    },
                    "url": f"https://x.com/{quoted_user.get('screen_name')}/status/{quoted_legacy.get('id_str')}"
                }
        
        return {
            "id": legacy.get("id_str"),
            "text": legacy.get("full_text", ""),
            "created_at": legacy.get("created_at"),
            "user": {
                "name": user.get("name"),
                "screen_name": user.get("screen_name"),
                "description": user.get("description"),
            },
            "is_quote_status": legacy.get("is_quote_status", False),
            "quoted_status": quoted_status_result,
            "url": f"https://x.com/{user.get('screen_name')}/status/{legacy.get('id_str')}"
        }
    except Exception as e:
        print(f"Error extracting tweet data: {e}")
        return None

def trace_quote_tweet_lineage(tweet_id):
    """Trace the lineage of quote tweets starting from the given tweet ID"""
    lineage = []
    current_id = tweet_id
    
    print(f"Starting lineage trace for tweet ID: {tweet_id}")
    
    # Limit to prevent infinite loops (in case of circular references or API issues)
    max_depth = 10
    depth = 0
    
    while current_id and depth < max_depth:
        print(f"Fetching details for tweet ID: {current_id}")
        tweet_data = fetch_tweet_by_id(current_id)
        
        if not tweet_data:
            print(f"Could not fetch data for tweet ID: {current_id}")
            break
        
        tweet_info = extract_tweet_data(tweet_data)
        if tweet_info:
            lineage.append(tweet_info)
            print(f"Added tweet to lineage: {tweet_info['url']}")
            
            # If this tweet is not a quote tweet, we've reached the end of the chain
            if not tweet_info.get("is_quote_status"):
                print("Reached end of quote tweet chain (not a quote tweet)")
                break
            
            # Get the ID of the quoted tweet to continue the chain
            quoted_id = tweet_info.get("quoted_status_id")
            if not quoted_id:
                print("No quoted status ID found, but tweet is marked as a quote tweet.")
                break
                
            if quoted_id == current_id:
                print("Quoted tweet ID is the same as current ID. Breaking to avoid loop.")
                break
                
            current_id = quoted_id
        else:
            print("Could not extract tweet info from response")
            break
        
        depth += 1
        time.sleep(2)  # Prevent rate limiting
    
    return lineage

def fetch_bookmarks(count=10):
    """Fetch a specified number of bookmarks to examine quote tweets"""
    bookmarks_api_id = os.getenv("BOOKMARKS_API_ID")
    API_URL = f"https://x.com/i/api/graphql/{bookmarks_api_id}/Bookmarks"
    
    variables = {
        "count": count,
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
        return fetch_bookmarks(count)
    
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
                    return fetch_bookmarks(count)
        except Exception as e:
            print(f"Error processing response: {e}")
    
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None
    
    return response.json()

def find_quote_tweets_in_recent_bookmarks(count=10):
    """Find quote tweets in recent bookmarks"""
    result = fetch_bookmarks(count)
    if not result:
        print("No bookmarks returned")
        return []
    
    try:
        entries = result.get("data", {}).get("bookmark_timeline_v2", {}).get("timeline", {}).get("instructions", [])[0].get("entries", [])
        tweet_entries = [e for e in entries if e["entryId"].startswith("tweet-")]
        
        quote_tweets = []
        for entry in tweet_entries:
            tweet_result = entry["content"]["itemContent"]["tweet_results"]["result"]
            if tweet_result["legacy"].get("is_quote_status"):
                tweet_data = extract_tweet_data(tweet_result)
                if tweet_data:
                    quote_tweets.append(tweet_data)
        
        print(f"Found {len(quote_tweets)} quote tweets in recent {len(tweet_entries)} bookmarks")
        return quote_tweets
    
    except Exception as e:
        print(f"Error finding quote tweets: {e}")
        return []

def main():
    # Find quote tweets in recent bookmarks
    quote_tweets = find_quote_tweets_in_recent_bookmarks(10)
    
    if not quote_tweets:
        print("No quote tweets found in recent bookmarks")
        return
    
    # Save the quote tweets and their quoted content
    output_file = "quote_tweet_lineages.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_quote_tweets": len(quote_tweets),
            "quote_tweets": quote_tweets
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(quote_tweets)} quote tweets to {output_file}")

if __name__ == "__main__":
    main() 