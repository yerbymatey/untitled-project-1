from typing import Dict, List
from datetime import datetime
import re

def parse_twitter_date(date_str: str) -> datetime:
    """Convert Twitter date format to datetime object"""
    try:
        # Twitter date format: "Wed Oct 10 20:19:24 +0000 2018"
        return datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
    except Exception as e:
        raise ValueError(f"Error parsing date '{date_str}': {e}")

def clean_tweet_text(text: str) -> str:
    """Clean tweet text by removing t.co URLs and normalizing whitespace"""
    # Remove t.co URLs
    text = re.sub(r'https://t\.co/\w+', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_tweet_data(tweet: Dict) -> Dict:
    """Parse and clean tweet data"""
    parsed_tweet = tweet.copy()
    
    # Parse dates
    if 'created_at' in parsed_tweet:
        parsed_tweet['created_at'] = parse_twitter_date(parsed_tweet['created_at'])
    
    # Clean text
    if 'text' in parsed_tweet:
        parsed_tweet['text'] = clean_tweet_text(parsed_tweet['text'])
    
    # Handle quoted tweet if present
    if parsed_tweet.get('quoted_status'):
        parsed_tweet['quoted_status'] = parse_tweet_data(parsed_tweet['quoted_status'])
    
    return parsed_tweet 