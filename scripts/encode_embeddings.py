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
from psycopg2.extras import execute_values # Import for bulk updates

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
            # Determine item ID based on type
            item_id = item.get('id') if item_type == 'tweet' else (item.get('tweet_id'), item.get('media_url'))
            if item_id is None:
                 logger.error(f"Skipping item due to missing ID fields: {item}")
                 permanently_failed_items.append(item)
                 continue

            try:
                # Run the provided processing function (e.g., get_text_embedding or process_media_item)
                processing_result = process_func(item)
                
                # Create a copy to store results without modifying the original list item
                item_copy = item.copy()
                
                # Merge the results into the item copy
                if isinstance(processing_result, dict): # For media returning multiple embeddings
                    item_copy.update(processing_result)
                else: # For tweets returning a single tensor
                    # Maintain the key 'embedding' for tweet processing path
                    item_copy['embedding'] = processing_result 
                
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

    # Handle items that failed the last attempt but didn't reach max retries
    if items_to_process:
         for item in items_to_process:
            item_id = item.get('id') if item_type == 'tweet' else (item.get('tweet_id'), item.get('media_url'))
            # Check item_id again in case it was missing in the loop
            if item_id:
                logger.error(f"{item_type.capitalize()} {item_id} failed after {retry_counts.get(item_id, 'unknown')} attempts (max was {max_retries}).")
            else:
                logger.error(f"An item of type {item_type} failed final attempt (missing ID): {item}")
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

def get_unprocessed_items(limit=100):
    """Get tweets and media items that need embeddings.

    Args:
        limit (int): Max number of items to fetch.
    
    Returns:
        tuple: (list of tweet IDs, list of media identifiers (tweet_id, media_url))
        Media identifiers are returned if either the joint_embedding or image_embedding is missing.
    """
    unprocessed_tweet_ids = []
    unprocessed_media_identifiers = []
    
    with get_db_session() as session:
        # Get tweets without embeddings
        session.execute("""
            SELECT id 
            FROM tweets 
            WHERE embedding IS NULL
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        unprocessed_tweet_ids = [r['id'] for r in session.fetchall()]
        
        # Get media items without joint or image embeddings
        session.execute("""
            SELECT tweet_id, media_url
            FROM media
            WHERE (joint_embedding IS NULL OR image_embedding IS NULL)
            AND type = 'photo'
            LIMIT %s
        """, (limit,))
        unprocessed_media_identifiers = [(r['tweet_id'], r['media_url']) for r in session.fetchall()]

    logger.info(f"Found {len(unprocessed_tweet_ids)} tweets needing embeddings.")
    logger.info(f"Found {len(unprocessed_media_identifiers)} media items needing embeddings.")
    
    return unprocessed_tweet_ids, unprocessed_media_identifiers

def format_vector_for_postgres(vector: torch.Tensor) -> str:
    """Format a torch tensor vector into a string suitable for PostgreSQL vector type."""
    return str(vector.cpu().numpy().tolist()) # Convert to list of floats

def update_tweet_embeddings(successful_tweet_items):
    """Update tweet embeddings in database from successfully processed items."""
    if not successful_tweet_items:
        return 0

    update_values = []
    for item in successful_tweet_items:
        if 'embedding' in item and item['embedding'] is not None:
            vector_str = format_vector_for_postgres(item['embedding'])
            update_values.append((vector_str, item['id']))
        else:
            logger.warning(f"Skipping tweet {item['id']} due to missing or None embedding.")

    if not update_values:
        return 0

    with get_db_session() as session:
        # Use psycopg2 execute_values for efficient bulk update
        execute_values(
            session.cursor,
            """UPDATE tweets SET embedding = v.embedding::vector
            FROM (VALUES %s) AS v(embedding, id)
            WHERE tweets.id = v.id::bigint;""",
            update_values
        )
        count = len(update_values)
        logger.info(f"Bulk updated {count} tweet embeddings.")
        return count

def update_media_embeddings(successful_media_items):
    """Update media joint and image embeddings in database from processed items."""
    if not successful_media_items:
        return 0

    update_values = []
    skipped_count = 0
    for item in successful_media_items:
        # Check for required identifiers first
        tweet_id = item.get('tweet_id')
        media_url = item.get('media_url')
        if not tweet_id or not media_url:
            logger.warning(f"Skipping media item due to missing identifiers: {item}")
            skipped_count += 1
            continue
            
        # Check for both embeddings
        joint_emb = item.get('joint_embedding')
        image_emb = item.get('image_embedding')
        
        if joint_emb is not None and image_emb is not None:
            try:
                joint_vector_str = format_vector_for_postgres(joint_emb)
                image_vector_str = format_vector_for_postgres(image_emb)
                update_values.append((joint_vector_str, image_vector_str, tweet_id, media_url))
            except Exception as e:
                 logger.error(f"Error formatting vectors for media {tweet_id}/{media_url}: {e}")
                 skipped_count += 1
        else:
            missing = []
            if joint_emb is None: missing.append('joint_embedding')
            if image_emb is None: missing.append('image_embedding')
            logger.warning(f"Skipping media {tweet_id}/{media_url} due to missing: {', '.join(missing)}.")
            skipped_count += 1

    if not update_values:
        logger.info(f"No valid media embeddings to update (skipped {skipped_count} items).")
        return 0

    with get_db_session() as session:
        # Use psycopg2 execute_values for efficient bulk update
        execute_values(
            session.cursor,
            """UPDATE media SET 
                   joint_embedding = v.joint_embedding::vector,
                   image_embedding = v.image_embedding::vector
               FROM (VALUES %s) AS v(joint_embedding, image_embedding, tweet_id, media_url)
               WHERE media.tweet_id::bigint = v.tweet_id::bigint 
               AND media.media_url = v.media_url;""",
            update_values
        )
        count = len(update_values)
        logger.info(f"Bulk updated {count} media embeddings (joint and image).")
        return count

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
    unprocessed_tweet_ids, unprocessed_media_identifiers = get_unprocessed_items()

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
            
            # Define the processing function to get both embeddings
            def process_media_item(item):
                try:
                    # This function needs to be modified to return both embeddings
                    # We'll adjust utils.embedding_utils.get_image_embedding next
                    joint_embedding, image_embedding = get_image_embedding(item['media_url'], item['image_desc'])
                    return {'joint_embedding': joint_embedding, 'image_embedding': image_embedding}
                except Exception as e:
                    logger.error(f"Error in get_image_embedding for {item['media_url']}: {e}")
                    raise e # Re-raise to be caught by process_items_with_retries

            successful_media_results, failed_media = process_items_with_retries(
                media_batch_data, 
                'media',
                process_media_item # Use the new processing function
            )

            # Adapt successful_media_results structure for update_media_embeddings
            processed_items_for_update = []
            # Determine which original items were successful. This is approximate.
            # A robust implementation might involve process_items_with_retries returning identifiers.
            original_items_dict = {(item['tweet_id'], item['media_url']): item for item in media_batch_data}
            failed_ids = set((item['tweet_id'], item['media_url']) for item in failed_media)
            successful_original_items = [item for item in media_batch_data if (item['tweet_id'], item['media_url']) not in failed_ids]

            if len(successful_original_items) == len(successful_media_results):
                for original_item, result_embeddings in zip(successful_original_items, successful_media_results):
                    processed_item = {
                        'tweet_id': original_item['tweet_id'],
                        'media_url': original_item['media_url'],
                        **result_embeddings # Add joint_embedding and image_embedding
                    }
                    processed_items_for_update.append(processed_item)
            else:
                # This case indicates a mismatch, log an error or handle appropriately
                logger.error("Mismatch between successful items and results in media batch processing. Cannot reliably map results to items.")
                # Decide how to proceed: skip update for this batch, or try a different mapping?
                # For now, we'll proceed with an empty list, effectively skipping updates for this batch.
                processed_items_for_update = [] 

            # Update DB
            processed_count = update_media_embeddings(processed_items_for_update)
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