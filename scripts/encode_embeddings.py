import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import argparse
import os
import logging
from collections import defaultdict

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

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TODO: Implement a more robust retry mechanism, potentially persisting retry state across runs.
MAX_RETRIES = 5

def process_items_with_retries(items, item_type, process_func, max_retries=MAX_RETRIES):
    """Processes items using process_func, with retries on failure."""
    if not items:
        return [], []

    items_to_process = list(items) # Make a mutable copy
    successful_items = []
    permanently_failed_items = []
    retry_counts = defaultdict(int)
    current_pass = 0

    while items_to_process and current_pass <= max_retries:
        next_retry_items = []
        logger.info(f"Processing pass {current_pass + 1}/{max_retries + 1} for {len(items_to_process)} {item_type} items...")
        
        progress_desc = f"Pass {current_pass + 1} - {item_type.capitalize()}"
        for item in tqdm(items_to_process, desc=progress_desc):
            item_id = item['id'] if item_type == 'tweet' else (item['tweet_id'], item['media_url'])
            
            try:
                embedding = process_func(item)
                item_copy = item.copy()
                item_copy['embedding'] = embedding
                successful_items.append(item_copy)
                # logger.debug(f"Successfully processed {item_type} {item_id}")
            except Exception as e:
                retry_counts[item_id] += 1
                logger.warning(f"Attempt {retry_counts[item_id]}/{max_retries} failed for {item_type} {item_id}: {e}")
                if retry_counts[item_id] < max_retries:
                    next_retry_items.append(item)
                else:
                    logger.error(f"{item_type.capitalize()} {item_id} failed after {max_retries} retries.")
                    permanently_failed_items.append(item)

        items_to_process = next_retry_items
        current_pass += 1

    # Any items left in items_to_process failed the last attempt but didn't reach max retries
    # if current_pass reached max_retries + 1. Treat them as failed for now.
    if items_to_process:
         for item in items_to_process:
            item_id = item['id'] if item_type == 'tweet' else (item['tweet_id'], item['media_url'])
            logger.error(f"{item_type.capitalize()} {item_id} failed after {retry_counts[item_id]} attempts (max was {max_retries}).")
            permanently_failed_items.append(item)

    return successful_items, permanently_failed_items

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

def update_tweet_embeddings(successful_tweet_items):
    """Update tweet embeddings in database from successfully processed items."""
    if not successful_tweet_items:
        return 0
        
    with get_db_session() as session:
        update_values = []
        for item in successful_tweet_items:
            vector_str = format_vector_for_postgres(item['embedding'])
            update_values.append((vector_str, item['id']))
            
        if update_values:
            from psycopg2.extras import execute_values
            execute_values(
                session.cursor, # Pass cursor directly
                """UPDATE tweets SET embedding = v.embedding::vector
                   FROM (VALUES %s) AS v(embedding, id)
                   WHERE tweets.id = v.id""",
                update_values
            )
            session.commit()
            logger.info(f"Bulk updated {len(update_values)} tweet embeddings.")
            return len(update_values)
    return 0

def update_media_embeddings(successful_media_items):
    """Update media embeddings in database from successfully processed items."""
    if not successful_media_items:
        return 0

    with get_db_session() as session:
        update_values = []
        for item in successful_media_items:
            vector_str = format_vector_for_postgres(item['embedding'])
            update_values.append((vector_str, item['tweet_id'], item['media_url']))

        if update_values:
            from psycopg2.extras import execute_values
            execute_values(
                session.cursor, # Pass cursor directly
                """UPDATE media SET embedding = v.embedding::vector
                   FROM (VALUES %s) AS v(embedding, tweet_id, media_url)
                   WHERE media.tweet_id = v.tweet_id AND media.media_url = v.media_url""",
                update_values
            )
            session.commit()
            logger.info(f"Bulk updated {len(update_values)} media embeddings.")
            return len(update_values)
    return 0

def fetch_item_data(tweet_ids=None, media_identifiers=None):
    """Fetch full item data for given tweet IDs or media identifiers."""
    items = []
    with get_db_session() as session:
        if tweet_ids:
            # Fetch tweet data
            tweet_query = """
            SELECT id, text
            FROM tweets
            WHERE id = ANY(%s)
            """
            session.execute(tweet_query, (list(tweet_ids),))
            tweets_result = session.fetchall()
            for tweet in tweets_result:
                items.append({
                    'id': tweet['id'],
                    'text': tweet['text'],
                    'type': 'tweet'
                })
        
        if media_identifiers:
            # Fetch media data
            # media_identifiers is a list of tuples: [(tweet_id, media_url), ...]
            media_query = """
            SELECT m.tweet_id, t.text, m.media_url, m.image_desc
            FROM media m
            JOIN tweets t ON t.id = m.tweet_id
            WHERE (m.tweet_id, m.media_url) IN %s
            """
            # psycopg2 expects a tuple of tuples for IN operator with multiple columns
            session.execute(media_query, (tuple(media_identifiers),))
            media_result = session.fetchall()
            for item in media_result:
                 items.append({
                    'tweet_id': item['tweet_id'],
                    'text': item['text'], # Include tweet text for context if needed by embedding func
                    'media_url': item['media_url'],
                    'image_desc': item['image_desc'],
                    'type': 'media'
                })
    return items

def main():
    parser = argparse.ArgumentParser(description='Process tweets and media with embeddings')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of items to process per batch (default: 100)')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force using CPU even if GPU is available')
    
    args = parser.parse_args()
    
    if args.force_cpu:
        global DEVICE
        DEVICE = 'cpu'
        # Reload models onto CPU if needed (adjust based on your setup)
        # text_model.to(DEVICE)
        # vision_model.to(DEVICE)
        logger.info("Forcing CPU usage")
    else:
        logger.info(f"Using device: {DEVICE}")

    batch_size = args.batch_size
    total_processed_tweets = 0
    total_failed_tweets = 0
    total_processed_media = 0
    total_failed_media = 0

    # --- Fetch all IDs/Identifiers needing processing ---    
    unprocessed_tweet_ids = []
    unprocessed_media_identifiers = [] 
    with get_db_session() as session:
        logger.info("Fetching unprocessed tweet IDs...")
        session.execute("SELECT id FROM tweets WHERE embedding IS NULL")
        unprocessed_tweet_ids = [row['id'] for row in session.fetchall()]
        logger.info(f"Found {len(unprocessed_tweet_ids)} tweets needing embeddings.")

        logger.info("Fetching unprocessed media identifiers...")
        session.execute("""
            SELECT tweet_id, media_url 
            FROM media 
            WHERE type = 'photo' 
            AND embedding IS NULL 
            AND image_desc IS NOT NULL 
            AND image_desc != ''
        """)
        unprocessed_media_identifiers = [(row['tweet_id'], row['media_url']) for row in session.fetchall()]
        logger.info(f"Found {len(unprocessed_media_identifiers)} media items needing embeddings.")

    if not unprocessed_tweet_ids and not unprocessed_media_identifiers:
        logger.info("No items require embedding processing.")
        return

    # --- Process Tweets in Batches ---    
    logger.info(f"Processing {len(unprocessed_tweet_ids)} tweets in batches of {batch_size}...")
    for i in range(0, len(unprocessed_tweet_ids), batch_size):
        batch_ids = unprocessed_tweet_ids[i:i + batch_size]
        logger.info(f"Fetching data for tweet batch {i // batch_size + 1}...")
        tweet_batch_data = fetch_item_data(tweet_ids=batch_ids)
        
        if tweet_batch_data:
            logger.info(f"Processing tweet batch {i // batch_size + 1} with {len(tweet_batch_data)} items...")
            successful_tweets, failed_tweets = process_items_with_retries(
                tweet_batch_data, 
                'tweet',
                lambda item: get_text_embedding(item['text'])
            )
            # Update DB
            processed_count = update_tweet_embeddings(successful_tweets)
            total_processed_tweets += processed_count
            total_failed_tweets += len(failed_tweets)
            if failed_tweets:
                 logger.error(f"Failed to process {len(failed_tweets)} tweets in batch {i // batch_size + 1} after retries.")
        else:
            logger.warning(f"No data fetched for tweet IDs: {batch_ids}")

    # --- Process Media in Batches ---    
    logger.info(f"Processing {len(unprocessed_media_identifiers)} media items in batches of {batch_size}...")
    for i in range(0, len(unprocessed_media_identifiers), batch_size):
        batch_identifiers = unprocessed_media_identifiers[i:i + batch_size]
        logger.info(f"Fetching data for media batch {i // batch_size + 1}...")
        media_batch_data = fetch_item_data(media_identifiers=batch_identifiers)

        if media_batch_data:
            logger.info(f"Processing media batch {i // batch_size + 1} with {len(media_batch_data)} items...")
            successful_media, failed_media = process_items_with_retries(
                media_batch_data, 
                'media',
                lambda item: get_image_embedding(item['media_url'], item['image_desc'])
            )
            # Update DB
            processed_count = update_media_embeddings(successful_media)
            total_processed_media += processed_count
            total_failed_media += len(failed_media)
            if failed_media:
                 logger.error(f"Failed to process {len(failed_media)} media items in batch {i // batch_size + 1} after retries.")
        else:
            logger.warning(f"No data fetched for media identifiers: {batch_identifiers}")

    # --- Final Summary ---    
    logger.info("--- Embedding Generation Summary ---")
    logger.info(f"Successfully processed: {total_processed_tweets} tweets, {total_processed_media} media items.")
    logger.info(f"Failed after retries: {total_failed_tweets} tweets, {total_failed_media} media items.")
    logger.info("--- Embedding Generation Complete ---")

if __name__ == "__main__":
    main() 