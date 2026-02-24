from typing import List, Dict, Optional
import os
import requests

from tqdm import tqdm
import argparse
import logging
from collections import defaultdict
import time
from psycopg2.extras import execute_values  # Import for bulk updates

from db.session import get_db_session
from utils.voyage import (
    voyage_contextualized_embeddings,
    voyage_multimodal_embeddings,
)

# Throttle configuration (env-overridable)
TWEET_SLEEP_SEC = float(os.getenv("VOYAGE_TWEET_SLEEP_SEC", "0.5"))
TWEET_COOLDOWN_SEC = float(os.getenv("VOYAGE_TWEET_COOLDOWN_SEC", "30"))
MEDIA_SLEEP_SEC = float(os.getenv("VOYAGE_MEDIA_SLEEP_SEC", "0.5"))
MEDIA_COOLDOWN_SEC = float(os.getenv("VOYAGE_MEDIA_COOLDOWN_SEC", "30"))

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

                # For deterministic data issues (e.g., missing text), don't bother retrying
                is_retryable = not isinstance(e, ValueError)

                if is_retryable and retry_counts[item_id] < max_retries:
                    next_retry_items.append(item)
                else:
                    logger.error(f"{item_type.capitalize()} {item_id} failed after {retry_counts[item_id]} attempts.")
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
            WHERE (joint_embedding IS NULL OR image_embedding IS NULL)
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
        
        # Get media items without joint or image embeddings (only those with descriptions)
        session.execute("""
            SELECT tweet_id, media_url
            FROM media
            WHERE (joint_embedding IS NULL OR image_embedding IS NULL)
            AND type = 'photo'
            AND image_desc IS NOT NULL
            AND image_desc != ''
            LIMIT %s
        """, (limit,))
        unprocessed_media_identifiers = [(r['tweet_id'], r['media_url']) for r in session.fetchall()]

    logger.info(f"Found {len(unprocessed_tweet_ids)} tweets needing embeddings.")
    logger.info(f"Found {len(unprocessed_media_identifiers)} media items needing embeddings.")
    
    return unprocessed_tweet_ids, unprocessed_media_identifiers

def format_vector_for_postgres(vector: List[float]) -> str:
    """Format a Python list vector into a string suitable for PostgreSQL vector type."""
    return str(list(vector))

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
            WHERE tweets.id = v.id;""",
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
               WHERE media.tweet_id = v.tweet_id
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
            SELECT m.tweet_id, t.text, m.media_url, m.image_desc, m.extr_text
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
                    'text': item['text'],
                    'media_url': item['media_url'],
                    'image_desc': item['image_desc'],
                    'extr_text': item['extr_text'],
                    'type': 'media'
                })
    return items


def _chunked(items: List[Dict], size: int):
    for index in range(0, len(items), size):
        yield items[index:index + size]


def _embed_single_tweet(item: Dict) -> Optional[Dict]:
    text = (item.get('text') or '').strip() or ' '
    outer_attempts = 0
    while True:
        try:
            vectors = voyage_contextualized_embeddings(inputs=[[text]], input_type='document')
            if not vectors:
                logger.warning("Empty embedding for tweet %s", item.get('id'))
                return None
            out = dict(item)
            out['embedding'] = vectors[0]
            return out
        except Exception as e:
            # Cooldown on 429 beyond internal retries
            if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 429 and outer_attempts < 3:
                outer_attempts += 1
                logger.warning(
                    "429 for tweet %s; cooling down %.1fs (outer attempt %d)",
                    item.get('id'),
                    TWEET_COOLDOWN_SEC,
                    outer_attempts,
                )
                time.sleep(TWEET_COOLDOWN_SEC)
                continue
            logger.error(f"Voyage error for tweet {item.get('id')}: {e}")
            return None


def embed_tweets_with_voyage(tweet_items: List[Dict], batch_size: int = 50) -> List[Dict]:
    """Compute tweet embeddings in batches using Voyage contextualized embeddings."""
    if not tweet_items:
        return []

    safe_batch_size = max(1, min(int(batch_size), 1000))
    total_batches = (len(tweet_items) + safe_batch_size - 1) // safe_batch_size
    results: List[Dict] = []

    for batch in tqdm(_chunked(tweet_items, safe_batch_size), total=total_batches, desc="Tweet Batches", unit="batch"):
        inputs = [[(item.get('text') or '').strip() or ' '] for item in batch]
        outer_attempts = 0
        batch_vectors = None

        while True:
            try:
                batch_vectors = voyage_contextualized_embeddings(inputs=inputs, input_type='document')
                break
            except Exception as e:
                # Cooldown on 429 beyond internal retries
                if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 429 and outer_attempts < 3:
                    outer_attempts += 1
                    logger.warning(
                        "429 for tweet batch starting at %s; cooling down %.1fs (outer attempt %d)",
                        batch[0].get('id'),
                        TWEET_COOLDOWN_SEC,
                        outer_attempts,
                    )
                    time.sleep(TWEET_COOLDOWN_SEC)
                    continue
                logger.error(
                    "Voyage error for tweet batch starting at %s: %s",
                    batch[0].get('id'),
                    e,
                )
                break

        if batch_vectors is None:
            logger.info(
                "Falling back to single-item calls for tweet batch starting at %s",
                batch[0].get('id'),
            )
            for item in batch:
                embedded = _embed_single_tweet(item)
                if embedded is not None:
                    results.append(embedded)
                time.sleep(TWEET_SLEEP_SEC)
            continue

        if len(batch_vectors) != len(batch):
            logger.warning(
                "Voyage returned %d vectors for tweet batch of %d items. Missing entries will be retried individually.",
                len(batch_vectors),
                len(batch),
            )

        for i, item in enumerate(batch):
            if i < len(batch_vectors) and batch_vectors[i]:
                out = dict(item)
                out['embedding'] = batch_vectors[i]
                results.append(out)
                continue

            logger.warning("Empty/missing embedding for tweet %s; retrying individually", item.get('id'))
            embedded = _embed_single_tweet(item)
            if embedded is not None:
                results.append(embedded)
            time.sleep(TWEET_SLEEP_SEC)

        # Pacing between API calls
        time.sleep(TWEET_SLEEP_SEC)

    return results


def embed_media_with_voyage(media_items: List[Dict]) -> List[Dict]:
    """Compute embeddings per media item with no cross-item batching, with a progress bar.

    For each media item, send a single request to Voyage MM embeddings with two entries:
      - image-only: content = [{type: image_url, image_url: ...}]
      - joint: [{type: text, text: image_desc?}, {type: image_url, image_url: ...}]

    Includes internal pacing/retry at the client layer; returns dicts with
    'joint_embedding' and 'image_embedding' when successful.
    """
    results: List[Dict] = []
    for it in tqdm(media_items, desc="Media", unit="it"):
        url = it['media_url']
        desc = (it.get('image_desc') or '').strip()
        extr = (it.get('extr_text') or '').strip()
        # Build combined text from description + extracted text for richer joint embedding
        text_parts = [p for p in [desc, extr] if p]
        combined_text = "\n\n".join(text_parts)
        # Build per-item request
        item_inputs = [
            {"content": [{"type": "image_url", "image_url": url}]},
        ]
        joint_content = [{"type": "image_url", "image_url": url}]
        if combined_text:
            joint_content.insert(0, {"type": "text", "text": combined_text})
        item_inputs.append({"content": joint_content})

        outer_attempts = 0
        while True:
            try:
                vecs = voyage_multimodal_embeddings(inputs=item_inputs, input_type='document')
                if len(vecs) >= 2:
                    results.append({
                        **it,
                        'image_embedding': vecs[0],
                        'joint_embedding': vecs[1],
                    })
                else:
                    logger.warning(f"Voyage returned {len(vecs)} vectors for media {url}; skipping")
                break
            except Exception as e:
                if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code == 429 and outer_attempts < 3:
                    outer_attempts += 1
                    logger.warning("429 for media %s; cooling down %.1fs (outer attempt %d)", url, MEDIA_COOLDOWN_SEC, outer_attempts)
                    time.sleep(MEDIA_COOLDOWN_SEC)
                    continue
                logger.error(f"Voyage error for media {url}: {e}")
                break

        # Pacing between items
        time.sleep(MEDIA_SLEEP_SEC)

    return results

def main():
    parser = argparse.ArgumentParser(description='Process tweets and media with Voyage embeddings')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=int(os.getenv("VOYAGE_TWEET_BATCH_SIZE", "50")),
        help='Tweet embedding batch size (max Voyage inputs: 1000). Default: 50',
    )
    # GPU/CPU is irrelevant for hosted APIs; keep option for compatibility no-op.
    parser.add_argument('--force-cpu', action='store_true',
                        help='No-op (hosted embeddings).')

    args = parser.parse_args()
    
    if args.force_cpu:
        logger.info("force-cpu flag ignored for hosted embeddings")

    total_processed_tweets = 0
    total_failed_tweets = 0
    total_processed_media = 0
    total_failed_media = 0

    # --- Fetch all IDs/Identifiers needing processing ---    
    # Fetch a large window to avoid doing client-side batches
    unprocessed_tweet_ids, unprocessed_media_identifiers = get_unprocessed_items(limit=1_000_000)

    if not unprocessed_tweet_ids and not unprocessed_media_identifiers:
        logger.info("No items require embedding processing.")
        return

    # --- Process Tweets (Batched) ---
    if unprocessed_tweet_ids:
        logger.info(f"Fetching data for {len(unprocessed_tweet_ids)} tweets...")
        tweet_data = fetch_item_data(tweet_ids=unprocessed_tweet_ids)
        if tweet_data:
            logger.info(
                "Processing %d tweets in batches of up to %d...",
                len(tweet_data),
                max(1, args.batch_size),
            )
            try:
                successful_tweets = embed_tweets_with_voyage(tweet_data, batch_size=args.batch_size)
                processed_count = update_tweet_embeddings(successful_tweets)
                total_processed_tweets += processed_count
            except Exception as e:
                logger.error(f"Voyage tweet embedding error: {e}")
                total_failed_tweets += len(tweet_data)
        else:
            logger.warning("No tweet data fetched to embed.")

    # --- Process Media Sequentially ---
    if unprocessed_media_identifiers:
        logger.info(f"Fetching data for {len(unprocessed_media_identifiers)} media items...")
        media_data = fetch_item_data(media_identifiers=unprocessed_media_identifiers)
        if media_data:
            logger.info(f"Processing {len(media_data)} media items sequentially (no batching)...")
            try:
                embedded_media = embed_media_with_voyage(media_data)
                processed_count = update_media_embeddings([
                    {
                        'tweet_id': it['tweet_id'],
                        'media_url': it['media_url'],
                        'joint_embedding': it.get('joint_embedding'),
                        'image_embedding': it.get('image_embedding'),
                    }
                    for it in embedded_media
                    if it.get('joint_embedding') is not None and it.get('image_embedding') is not None
                ])
                total_processed_media += processed_count
            except Exception as e:
                logger.error(f"Voyage media embedding error: {e}")
                total_failed_media += len(media_data)
        else:
            logger.warning("No media data fetched to embed.")

    # --- Final Summary ---    
    logger.info("--- Embedding Generation Summary ---")
    logger.info(f"Successfully processed: {total_processed_tweets} tweets, {total_processed_media} media items.")
    logger.info(f"Failed after retries: {total_failed_tweets} tweets, {total_failed_media} media items.")
    logger.info("--- Embedding Generation Complete ---")

if __name__ == "__main__":
    main() 
