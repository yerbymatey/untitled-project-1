import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import argparse
import os

from db.session import get_db_session
from utils.embedding_utils import (
    mean_pooling,
    DEVICE,
    TEXT_MODEL_NAME,
    VISION_MODEL_NAME,
    get_text_embedding,
    get_image_embedding,
    download_image
)

print(f"Using device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True).to(DEVICE)
text_model.eval()

vision_processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)
vision_model = AutoModel.from_pretrained(VISION_MODEL_NAME, trust_remote_code=True).to(DEVICE)
vision_model.eval()

def get_total_counts():
    """Get total counts of items to process"""
    with get_db_session() as session:
        session.execute("""
            SELECT COUNT(*) as count 
            FROM tweets 
            WHERE embedding IS NULL
        """)
        result = session.fetchone()
        tweets_total = result['count'] if result else 0
        
        session.execute("""
            SELECT COUNT(*) as count 
            FROM media 
            WHERE embedding IS NULL
            AND image_desc IS NOT NULL
            AND image_desc != ''
            AND type = 'photo'
        """)
        result = session.fetchone()
        media_total = result['count'] if result else 0
        
        return tweets_total, media_total

def get_unprocessed_items(offset=0, batch_size=100):
    """Get tweets and media items that need embeddings.
    
    For tweets with media:
    - The tweet text gets its own text embedding in the tweets table
    - The image + image description get a separate multimodal embedding in the media table
    """
    with get_db_session() as session:
        # Get tweets without embeddings (both with and without media)
        tweet_query = """
        SELECT DISTINCT t.id, t.text, NULL as media_url, NULL as image_desc
        FROM tweets t
        WHERE t.embedding IS NULL
        ORDER BY t.id
        LIMIT %s OFFSET %s
        """
        session.execute(tweet_query, (batch_size, offset))
        tweets = session.fetchall()
        
        # Get media items without embeddings (for image + description pairs)
        media_query = """
        SELECT DISTINCT m.tweet_id, t.text, m.media_url, m.image_desc
        FROM media m
        JOIN tweets t ON t.id = m.tweet_id
        WHERE m.type = 'photo'
        AND m.embedding IS NULL
        AND m.image_desc IS NOT NULL
        AND m.image_desc != ''
        ORDER BY m.tweet_id
        LIMIT %s OFFSET %s
        """
        session.execute(media_query, (batch_size, offset))
        media = session.fetchall()
        
        # Combine results
        all_items = []
        for tweet in tweets:
            all_items.append({
                'id': tweet['id'],
                'text': tweet['text'],
                'media_url': tweet['media_url'],
                'image_desc': tweet['image_desc'],
                'type': 'tweet'  # Will get text-only embedding
            })
        
        for item in media:
            all_items.append({
                'id': item['tweet_id'],
                'text': item['text'],
                'media_url': item['media_url'],
                'image_desc': item['image_desc'],
                'type': 'media'  # Will get multimodal embedding (image + description)
            })
        
        return all_items

def format_vector_for_postgres(vector):
    """Format vector for Postgres storage - convert to list and format as string"""
    # Convert tensor to numpy array and get the first vector if batched
    vector_array = vector.cpu().detach().numpy()
    if len(vector_array.shape) > 1:
        vector_array = vector_array[0]  # Take first vector if batched
    # Format as Postgres array string
    return f"[{','.join(str(x) for x in vector_array)}]"

def update_tweet_embeddings(tweet_ids, embeddings):
    """Update tweet embeddings in database"""
    with get_db_session() as session:
        for tweet_id, embedding in zip(tweet_ids, embeddings):
            vector_str = format_vector_for_postgres(embedding)
            query = """
                UPDATE tweets 
                SET embedding = %s::vector
                WHERE id = %s
            """
            session.execute(query, (vector_str, tweet_id))
        session.commit()

def update_media_embeddings(tweet_ids, media_urls, embeddings):
    """Update media embeddings in database"""
    with get_db_session() as session:
        for tweet_id, media_url, embedding in zip(tweet_ids, media_urls, embeddings):
            vector_str = format_vector_for_postgres(embedding)
            query = """
                UPDATE media 
                SET embedding = %s::vector
                WHERE tweet_id = %s AND media_url = %s
            """
            session.execute(query, (vector_str, tweet_id, media_url))
        session.commit()

def process_batch(batch_size=100, offset=0):
    """Process a batch of items for embeddings"""
    # Get items to process
    items = get_unprocessed_items(offset, batch_size)
    if not items:
        print("No items found without embeddings")
        return
    
    print(f"Processing {len(items)} items for embeddings...")
    
    # Separate items by type
    tweet_items = [item for item in items if item['type'] == 'tweet']
    media_items = [item for item in items if item['type'] == 'media']
    
    # Process tweets
    if tweet_items:
        tweet_ids = []
        tweet_embeddings = []
        
        for item in tqdm(tweet_items, desc="Processing tweets"):
            try:
                embedding = get_text_embedding(item['text'])
                tweet_embeddings.append(embedding)
                tweet_ids.append(item['id'])
            except Exception as e:
                print(f"Error processing tweet {item['id']}: {e}")
                continue
        
        if tweet_embeddings:
            update_tweet_embeddings(tweet_ids, tweet_embeddings)
            print(f"Updated {len(tweet_embeddings)} tweet embeddings")
    
    # Process media
    if media_items:
        tweet_ids = []
        media_urls = []
        media_embeddings = []
        
        for item in tqdm(media_items, desc="Processing media"):
            try:
                embedding = get_image_embedding(item['media_url'], item['image_desc'])
                media_embeddings.append(embedding)
                tweet_ids.append(item['id'])
                media_urls.append(item['media_url'])
            except Exception as e:
                print(f"Error processing media for tweet {item['id']}: {e}")
                continue
        
        if media_embeddings:
            update_media_embeddings(tweet_ids, media_urls, media_embeddings)
            print(f"Updated {len(media_embeddings)} media embeddings")

def main():
    parser = argparse.ArgumentParser(description='Process tweets and media with embeddings')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of items to process per batch (default: 100)')
    parser.add_argument('--small-batch', action='store_true',
                        help='Process just 5 items (for testing)')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force using CPU even if GPU is available')
    
    args = parser.parse_args()
    
    if args.force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Forcing CPU usage")
    
    print(f"Using device: {DEVICE}")
    
    batch_size = 5 if args.small_batch else args.batch_size
    if args.small_batch:
        print(f"Processing small test batch of {batch_size} items")
    
    # Get total counts
    tweets_total, media_total = get_total_counts()
    print(f"Total items to process: {tweets_total} tweets, {media_total} media items")
    
    if tweets_total == 0 and media_total == 0:
        print("No items to process!")
        return
    
    offset = 0
    while True:
        process_batch(batch_size, offset)
        offset += batch_size
        
        # Check if there are any remaining items without embeddings
        tweets_remaining, media_remaining = get_total_counts()
        
        if tweets_remaining == 0 and media_remaining == 0:
            print("All items have been processed!")
            break
        else:
            print(f"Remaining items to process: {tweets_remaining} tweets, {media_remaining} media items")
            
            # Calculate progress percentages safely
            tweet_progress = 100.0 if tweets_total == 0 else ((tweets_total - tweets_remaining) / tweets_total * 100)
            media_progress = 100.0 if media_total == 0 else ((media_total - media_remaining) / media_total * 100)
            
            print(f"Progress: {tweet_progress:.1f}% tweets, {media_progress:.1f}% media")

if __name__ == "__main__":
    main() 