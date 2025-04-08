import os
import tempfile
import requests
import base64
from openai import OpenAI
from dotenv import load_dotenv
from db.session import get_db_session

load_dotenv()

def get_all_tweets(limit=0):
    """Get all tweets or a limited number, ordered by creation date"""
    with get_db_session() as session:
        query = """
        SELECT t.id, t.text, t.created_at, t.quoted_tweet_id
        FROM tweets t
        ORDER BY t.created_at DESC
        """
        if limit > 0:
            query += " LIMIT %s"
            session.execute(query, (limit,))
        else:
            session.execute(query)
        return session.fetchall()

def get_new_tweets(limit=0):
    """Get tweets that don't have interpretations yet"""
    with get_db_session() as session:
        query = """
        SELECT t.id, t.text, t.created_at, t.quoted_tweet_id
        FROM tweets t
        LEFT JOIN tweet_interpretations ti ON t.id = ti.tweet_id
        WHERE ti.tweet_id IS NULL
        ORDER BY t.created_at DESC
        """
        if limit > 0:
            query += " LIMIT %s"
            session.execute(query, (limit,))
        else:
            session.execute(query)
        return session.fetchall()

def get_quoted_tweet(quoted_tweet_id):
    with get_db_session() as session:
        query = """
        SELECT id, text
        FROM tweets 
        WHERE id = %s
        """
        session.execute(query, (quoted_tweet_id,))
        return session.fetchone()

def get_media_for_tweet(tweet_id):
    with get_db_session() as session:
        query = """
        SELECT media_url, type as media_type
        FROM media
        WHERE tweet_id = %s
        AND type = 'photo'  -- Ignoring videos for now
        """
        session.execute(query, (tweet_id,))
        return session.fetchall()

def get_existing_interpretation(tweet_id):
    with get_db_session() as session:
        query = """
        SELECT interpretation 
        FROM tweet_interpretations
        WHERE tweet_id = %s
        """
        session.execute(query, (tweet_id,))
        result = session.fetchone()
        return result['interpretation'] if result else None

def analyze_with_gpt4o(tweet_id, tweet_data, prompt_type='default'):
    # Check if there's already an interpretation
    existing_interpretation = get_existing_interpretation(tweet_id)
    if existing_interpretation:
        print(f"Found existing interpretation: {existing_interpretation}")
        return existing_interpretation
    
    # Extract tweet data
    tweet_text = tweet_data['tweet_text']
    quoted_tweet_text = tweet_data.get('quoted_tweet_text', '')
    media_list = tweet_data['media_list']
    quoted_media_list = tweet_data.get('quoted_media_list', [])
    
    # Configure API client
    client = OpenAI(api_key=os.environ.get("OAI_API_KEY"))
    
    # Select prompt based on prompt_type
    if prompt_type == 'default':
        system_prompt = """You are a high-context digital media interpreter trained to extract latent semantic structures and thematic payloads from complex multimodal objects. Given a digital object (e.g., tweet, image, meme, or quote tweet), return a conceptually precise and structurally coherent distillation in 1–2 dense sentences.

Do not interpret from a viewer's perspective or describe superficial visual elements unless they directly reinforce the object's rhetorical function. If the object is a quote tweet, treat the quoted tweet as the source artifact and the main tweet as framing commentary—analyze their interplay as a single compositional unit.

Focus on the function of each component within the object's internal logic. Disregard purely decorative or non-semantic features. Your goal is to encode the object's essential mechanics, generative intention, and structural affordances with high compression and minimal redundancy."""
    elif prompt_type == 'simplified':
        system_prompt = """Analyze this tweet and any attached media. Provide a concise 1-2 sentence interpretation that captures the core meaning and context. Focus on what makes this content significant or interesting rather than merely describing what's visible."""
    elif prompt_type == 'detailed':
        system_prompt = """Perform a detailed analysis of this tweet, including any media, quoted content, and textual elements. Consider:
1. The rhetorical strategies employed
2. The cultural references or context required to understand it
3. Any subtext or implied meaning
4. The overall communicative function of the tweet
Provide your analysis in 2-3 dense, information-rich sentences."""
    else:
        # Fallback to default prompt
        system_prompt = """You are a high-context digital media interpreter trained to extract latent semantic structures and thematic payloads from complex multimodal objects. Given a digital object (e.g., tweet, image, meme, or quote tweet), return a conceptually precise and structurally coherent distillation in 1–2 dense sentences."""

    messages = [{"role": "system", "content": system_prompt}]
    
    # Handle different tweet scenarios
    has_main_media = len(media_list) > 0
    has_quoted_media = len(quoted_media_list) > 0
    is_quote_tweet = quoted_tweet_text != ''
    
    if has_main_media or has_quoted_media:
        # Process tweets with media (photos)
        message_content = []
        
        # Construct appropriate prompt text
        prompt_text = system_prompt + "\n\n"
        
        if is_quote_tweet:
            prompt_text += "This is a quote tweet. "
            if has_main_media and has_quoted_media:
                prompt_text += "Both the main tweet and quoted tweet contain images. "
            elif has_main_media:
                prompt_text += "The main tweet contains images. "
            elif has_quoted_media:
                prompt_text += "The quoted tweet contains images. "
            
            prompt_text += f"Main tweet: {tweet_text}\n"
            prompt_text += f"Quoted tweet: {quoted_tweet_text}"
        else:
            prompt_text += f"Here is the accompanied text from the tweet: {tweet_text}"
        
        # Add text part
        message_content.append({"type": "text", "text": prompt_text})
        
        # Add main tweet images
        for media in media_list:
            # Add image to content
            message_content.append(
                process_image(media['media_url'])
            )
        
        # Add quoted tweet images if present
        for media in quoted_media_list:
            # Add image to content
            message_content.append(
                process_image(media['media_url'])
            )
        
        # Create the message with text and images
        messages.append({"role": "user", "content": message_content})
    else:
        # Process text-only tweets
        prompt_text = system_prompt + "\n\n"
        
        if is_quote_tweet:
            prompt_text += "This is a text-only quote tweet.\n"
            prompt_text += f"Main tweet: {tweet_text}\n"
            prompt_text += f"Quoted tweet: {quoted_tweet_text}"
        else:
            prompt_text += f"Here is the tweet: {tweet_text}"
        
        messages.append({"role": "user", "content": prompt_text})
    
    # Generate response
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=150
    )
    
    interpretation = response.choices[0].message.content
    print(f"Interpretation: {interpretation}")
    
    # Store interpretation in database
    with get_db_session() as session:
        query = """
        INSERT INTO tweet_interpretations (tweet_id, interpretation)
        VALUES (%s, %s)
        ON CONFLICT (tweet_id) 
        DO UPDATE SET interpretation = %s
        """
        session.execute(query, (tweet_id, interpretation, interpretation))
        session.commit()
    
    return interpretation

def process_image(url):
    """Download and process an image for API submission"""
    # Download the image
    temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    response = requests.get(url)
    temp_file.write(response.content)
    temp_file.close()
    
    # Read image file as base64
    with open(temp_file.name, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode('utf-8')
    
    # Clean up temporary file
    os.unlink(temp_file.name)
    
    # Return formatted image data
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }

def view_tweet_with_interpretation(tweet_data, interpretation):
    tweet_id = tweet_data['tweet_id']
    tweet_text = tweet_data['tweet_text']
    quoted_tweet_text = tweet_data.get('quoted_tweet_text', '')
    media_list = tweet_data['media_list']
    quoted_media_list = tweet_data.get('quoted_media_list', [])
    
    print("\n=== TWEET WITH INTERPRETATION ===")
    print(f"Tweet ID: {tweet_id}")
    print(f"Tweet Text: {tweet_text}")
    
    if quoted_tweet_text:
        print(f"Quoted Tweet: {quoted_tweet_text}")
    
    if media_list:
        for i, media in enumerate(media_list):
            print(f"Media {i+1} URL: {media['media_url']}")
    
    if quoted_media_list:
        for i, media in enumerate(quoted_media_list):
            print(f"Quoted Tweet Media {i+1} URL: {media['media_url']}")
            
    if not media_list and not quoted_media_list:
        print("No media (text-only tweet)")
        
    print(f"Interpretation: {interpretation}")
    print("==================================\n")

def process_tweets(tweets, prompt_type='default', skip_existing=False):
    if not tweets:
        print("No tweets found to process")
        return
    
    print(f"Processing {len(tweets)} tweets with prompt type: {prompt_type}")
    
    for tweet in tweets:
        tweet_id = tweet['id']
        tweet_text = tweet['text']
        quoted_tweet_id = tweet.get('quoted_tweet_id')
        
        print(f"\nProcessing tweet: {tweet_id}")
        print(f"Text: {tweet_text}")
        
        # Check if we should skip this tweet
        if skip_existing:
            existing = get_existing_interpretation(tweet_id)
            if existing:
                print(f"Skipping tweet {tweet_id} (already has interpretation)")
                continue
        
        # Prepare tweet data structure
        tweet_data = {
            'tweet_id': tweet_id,
            'tweet_text': tweet_text,
            'media_list': get_media_for_tweet(tweet_id)
        }
        
        # Handle quoted tweet if present
        if quoted_tweet_id:
            quoted_tweet = get_quoted_tweet(quoted_tweet_id)
            if quoted_tweet:
                print(f"Found quoted tweet: {quoted_tweet['id']}")
                tweet_data['quoted_tweet_text'] = quoted_tweet['text']
                tweet_data['quoted_media_list'] = get_media_for_tweet(quoted_tweet['id'])
                
                if tweet_data['quoted_media_list']:
                    print(f"Found {len(tweet_data['quoted_media_list'])} media items in quoted tweet")
        
        # Get media for this tweet
        if tweet_data['media_list']:
            print(f"Found {len(tweet_data['media_list'])} media items in main tweet")
        
        if not tweet_data['media_list'] and not tweet_data.get('quoted_media_list', []):
            print("No media (text-only tweet/quote)")
        
        # Analyze tweet
        print("\nAnalyzing with GPT-4o...\n")
        interpretation = analyze_with_gpt4o(tweet_id, tweet_data, prompt_type)
        
        # Display results
        view_tweet_with_interpretation(tweet_data, interpretation)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze tweets with GPT-4o')
    parser.add_argument('--mode', choices=['all', 'new'], default='all', 
                        help='Process all tweets or only new ones without interpretations')
    parser.add_argument('--limit', type=int, default=0, 
                        help='Limit number of tweets to process (0 for no limit)')
    parser.add_argument('--prompt', choices=['default', 'simplified', 'detailed'], default='default',
                        help='Type of prompt to use for analysis')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip tweets that already have interpretations')
    
    args = parser.parse_args()
    
    # Get tweets based on mode
    if args.mode == 'new':
        tweets = get_new_tweets(args.limit)
        print(f"Found {len(tweets)} new tweets to analyze")
    else:
        tweets = get_all_tweets(args.limit)
        print(f"Found {len(tweets)} total tweets to analyze")
    
    # Process the tweets
    process_tweets(tweets, args.prompt, args.skip_existing)

if __name__ == "__main__":
    main()