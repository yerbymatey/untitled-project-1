import requests, json, time
import os
from dotenv import load_dotenv
from datetime import datetime
import re

load_dotenv()

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

# Initial features set - will be expanded as needed based on API responses
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

def fetch_bookmarks(cursor=None):
    """Fetch a page of bookmarks from the X/Twitter API"""
    global features
    
    variables = {
        "count": 100,
        "cursor": cursor,
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
        return fetch_bookmarks(cursor)
    
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
                    return fetch_bookmarks(cursor)
        except Exception as e:
            print(f"Error processing response: {e}")
    
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return None
    
    return response.json()

def extract_tweet_data(entry):
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
            media_items.append(media_item)
    
    # Extract hashtags if available
    if "entities" in legacy and "hashtags" in legacy["entities"]:
        hashtags = [hashtag["text"] for hashtag in legacy["entities"]["hashtags"]]
    
    # Extract URLs if available
    if "entities" in legacy and "urls" in legacy["entities"]:
        for url_entity in legacy["entities"]["urls"]:
            url_item = {
                "url": url_entity.get("url"),
                "expanded_url": url_entity.get("expanded_url"),
                "display_url": url_entity.get("display_url")
            }
            urls.append(url_item)
    
    # Extract user description and related URLs if available
    user_description = user.get("description", "")
    user_description_urls = []
    
    if "entities" in user and "description" in user["entities"] and "urls" in user["entities"]["description"]:
        for url_entity in user["entities"]["description"]["urls"]:
            url_item = {
                "url": url_entity.get("url"),
                "expanded_url": url_entity.get("expanded_url"),
                "display_url": url_entity.get("display_url")
            }
            user_description_urls.append(url_item)
    
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
        "url": f"https://x.com/{user.get('screen_name')}/status/{entry['entryId'].split('-')[1]}"
    }

def main():
    """Main function to fetch all bookmarks and save to JSON"""
    cursor = None
    all_tweets = []
    total_fetched = 0
    empty_pages_count = 0  # Track consecutive pages with no tweets

    print("Fetching bookmarks...")
    while True:
        result = fetch_bookmarks(cursor)
        if not result:
            print("No result returned, ending fetch...")
            break

        try:
            entries = result.get("data", {}).get("bookmark_timeline_v2", {}).get("timeline", {}).get("instructions", [])[0].get("entries", [])
            print(f"Found {len(entries)} entries")
            
            tweet_entries = [e for e in entries if e["entryId"].startswith("tweet-")]
            print(f"Found {len(tweet_entries)} tweet entries")

            # Handle empty pages (no tweets but other entries)
            if len(tweet_entries) == 0:
                empty_pages_count += 1
                print(f"No tweets on this page (empty page #{empty_pages_count})")
                
                # If we've seen 3 consecutive empty pages, assume we're done
                if empty_pages_count >= 3:
                    print("Multiple empty pages in a row - reached end of bookmarks")
                    break
            else:
                # Reset counter if we found tweets
                empty_pages_count = 0

            # Process tweets from this page
            for entry in tweet_entries:
                try:
                    tweet_data = extract_tweet_data(entry)
                    all_tweets.append(tweet_data)
                    total_fetched += 1
                    print(f"Fetched tweet {total_fetched}: {tweet_data['url']}")
                except Exception as e:
                    print(f"Error processing tweet: {e}")

            # Check for next page cursor
            cursor_entries = [e for e in entries if e["entryId"].startswith("cursor-bottom-")]
            if cursor_entries:
                cursor = cursor_entries[0]["content"]["value"]
                print(f"Found next cursor, continuing...")
            else:
                print("\nNo more cursors - reached end of bookmarks!")
                break

            time.sleep(2)  # avoid aggressive rate limits

        except Exception as e:
            print(f"Error processing result: {e}")
            break

    if all_tweets:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bookmarks_{timestamp}.json"
        
        print(f"\nSaving {len(all_tweets)} tweets to {filename}...")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "total_bookmarks": len(all_tweets),
                "fetch_date": datetime.now().isoformat(),
                "bookmarks": all_tweets
            }, f, ensure_ascii=False, indent=2)

        print(f"Successfully saved to: {filename}")
    else:
        print("\nNo tweets were fetched!")

    print("\nBookmark fetch complete!")

if __name__ == "__main__":
    main()