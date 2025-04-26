import argparse
from pathlib import Path
import tempfile
import os
import requests
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image
from tqdm import tqdm
import logging
from collections import defaultdict

from utils.vl_utils import setup_vl_model, process_vl_conversation
from db.session import get_db_session
from utils.process_images import resize_image

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# TODO: Implement a more robust retry mechanism, potentially persisting retry state across runs.
MAX_RETRIES = 5
IMAGE_UNAVAILABLE_MARKER = "IMAGE_UNAVAILABLE"
IMAGE_ACCESS_DENIED_MARKER = "IMAGE_ACCESS_DENIED"

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
            # Use tweet_id and media_url as a composite key for media items
            item_id = (item['id'], item['image_url'])
            
            try:
                # process_func is expected to return the full result dictionary
                result = process_func(item)
                # The result already contains necessary info (post_id, media_url, image_description)
                
                # Check for specific non-retryable failure markers
                if result.get('image_description') in [IMAGE_UNAVAILABLE_MARKER, IMAGE_ACCESS_DENIED_MARKER]:
                    logger.warning(f"{item_type.capitalize()} {item_id} marked as {result['image_description']}. Not retrying.")
                    # Treat as success for retry logic, but preserve marker
                    successful_items.append(result) 
                else:
                    # Genuine success
                    successful_items.append(result)
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
            item_id = (item['id'], item['image_url'])
            logger.error(f"{item_type.capitalize()} {item_id} failed after {retry_counts[item_id]} attempts (max was {max_retries}).")
            permanently_failed_items.append(item)

    return successful_items, permanently_failed_items

def get_total_count(overwrite=False):
    """Get total count of media items without descriptions"""
    with get_db_session() as session:
        if overwrite:
            # Count all photos (except quoted tweets)
            session.execute("""
                SELECT COUNT(*) as count
                FROM media m
                JOIN tweets t ON t.id = m.tweet_id
                WHERE m.type = 'photo'
                AND t.quoted_tweet_id IS NULL
            """)
        else:
            # Count only photos without descriptions
            session.execute("""
                SELECT COUNT(*) as count
                FROM media m
                JOIN tweets t ON t.id = m.tweet_id
                WHERE m.type = 'photo'
                AND (m.image_desc IS NULL OR m.image_desc = '')
                AND t.quoted_tweet_id IS NULL
            """)
        result = session.fetchone()
        return result['count'] if result else 0

def get_unprocessed_media(offset=0, batch_size=100, overwrite=False):
    """Fetch media items without descriptions"""
    with get_db_session() as session:
        if overwrite:
            # Get all photos (except quoted tweets)
            session.execute("""
                SELECT t.id, m.media_url, t.text as tweet_text
                FROM tweets t
                JOIN media m ON t.id = m.tweet_id
                WHERE m.type = 'photo'
                AND t.quoted_tweet_id IS NULL
                ORDER BY t.id
                LIMIT %s OFFSET %s
            """, (batch_size, offset))
        else:
            # Get only photos without descriptions
            session.execute("""
                SELECT t.id, m.media_url, t.text as tweet_text
                FROM tweets t
                JOIN media m ON t.id = m.tweet_id
                WHERE m.type = 'photo'
                AND (m.image_desc IS NULL OR m.image_desc = '')
                AND t.quoted_tweet_id IS NULL
                ORDER BY t.id
                LIMIT %s OFFSET %s
            """, (batch_size, offset))
        
        posts = session.fetchall()
        return [{"id": p['id'], "image_url": p['media_url'], "tweet_text": p['tweet_text']} for p in posts]

def process_post(post: Dict[str, Any], vl_chat_processor, vl_gpt, device, dtype) -> Dict[str, Any]:
    """Process a single post with the VL model."""
    # Ensure post_id is correctly identified (it's the tweet ID here)
    tweet_id = post.get('id') or post.get('tweet_id')
    if not tweet_id:
        raise ValueError("Missing tweet ID ('id' or 'tweet_id') in post data")
    
    media_url = post.get('image_url') or post.get('media_url')
    if not media_url:
        raise ValueError("Missing image URL ('image_url' or 'media_url') in post data")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download image from URL
            response = requests.get(media_url, timeout=30) # Added timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
            
            # Convert to PIL Image
            image = Image.open(BytesIO(response.content))
            
            # Resize image
            resized_image = resize_image(image, output_size=(512, 512))
            
            # Convert to RGB mode before saving as JPEG
            resized_image = resized_image.convert('RGB')
            
            # Save to temporary file
            image_path = os.path.join(temp_dir, f"{tweet_id}.jpg")
            resized_image.save(image_path, 'JPEG')
            
            # Create conversation focused on objective description
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder> Provide a concise, high-level semantic summary capturing the core meaning or implication of this image as if summarizing its relevance in casual conversation. Avoid minor details, names, or labels unless essential. Limit your response strictly within two succinct sentences.",
                    "images": [image_path]
                },
                {
                    "role": "Assistant",
                    "content": ""
                }
            ]
            
            # Process with VL model
            answer, sft_format = process_vl_conversation(conversation, vl_chat_processor, vl_gpt, device, dtype)
            
            return {
                "post_id": tweet_id,
                "tweet_text": post['tweet_text'],
                "image_description": answer,
                "media_url": media_url
            }
    except requests.exceptions.RequestException as e:
        status_code = e.response.status_code if e.response is not None else None
        if status_code == 404:
            logger.warning(f"Image not found (404) for {media_url}. Marking as unavailable.")
            return {
                "post_id": tweet_id,
                "media_url": media_url,
                "image_description": IMAGE_UNAVAILABLE_MARKER,
                "tweet_text": post.get('tweet_text', '') # Include for consistency
            }
        elif status_code == 403:
            logger.warning(f"Access denied (403) for {media_url}. Marking as access denied.")
            return {
                "post_id": tweet_id,
                "media_url": media_url,
                "image_description": IMAGE_ACCESS_DENIED_MARKER,
                "tweet_text": post.get('tweet_text', '')
            }
        else:
            # Re-raise other request exceptions for the retry logic to catch
            raise Exception(f"Failed to download image {media_url}: {e}") from e
    except Exception as e:
        # Catch other potential errors (PIL, VL model, etc.) and re-raise
        raise Exception(f"Error processing post {tweet_id} with media {media_url}: {e}") from e

def process_batch(posts, vl_chat_processor, vl_gpt, device, dtype, batch_num, total_batches):
    """Process a batch of media items"""
    if not posts:
        return 0, 0
    
    print(f"\nProcessing batch {batch_num}/{total_batches} ({len(posts)} items)...")
    processed_count = 0
    failed_count = 0
    successful_updates = []
    
    # Process each post
    for post in tqdm(posts, desc=f"Batch {batch_num}/{total_batches}"):
        try:
            result = process_post(post, vl_chat_processor, vl_gpt, device, dtype)
            processed_count += 1
            successful_updates.append(result)
            print(f"\nPost ID: {result['post_id']}")
            print(f"Tweet: {result['tweet_text']}")
            print(f"Model Response: {result['image_description']}")
            print("-" * 80)
        except Exception as e:
            failed_count += 1
            print(f"Error processing post {post['id']}: {str(e)}")
    
    # Update database in a single transaction for all successful results
    if successful_updates:
        with get_db_session() as session:
            for result in successful_updates:
                try:
                    session.execute("""
                        UPDATE media 
                        SET image_desc = %s
                        WHERE tweet_id = %s AND media_url = %s
                    """, (result['image_description'], result['post_id'], result['media_url']))
                except Exception as e:
                    print(f"Error updating post {result['post_id']}: {str(e)}")
                    failed_count += 1
                    processed_count -= 1
            try:
                session.commit()
            except Exception as e:
                print(f"Error committing transaction: {str(e)}")
                session.rollback()
                # If the entire transaction fails, count all updates as failed
                failed_count += len(successful_updates)
                processed_count -= len(successful_updates)
    
    return processed_count, failed_count

def update_media_descriptions(successful_media_items):
    """Update media descriptions in database from successfully processed items."""
    if not successful_media_items:
        return 0

    with get_db_session() as session:
        update_values = []
        # successful_media_items contains dictionaries like the output of process_post
        for item in successful_media_items:
            update_values.append((
                item['image_description'],
                item['post_id'], 
                item['media_url']
            ))

        if update_values:
            from psycopg2.extras import execute_values
            try:
                execute_values(
                    session.cursor, # Pass cursor directly
                    """UPDATE media SET image_desc = v.image_desc
                    FROM (VALUES %s) AS v(image_desc, tweet_id, media_url)
                    WHERE media.tweet_id = v.tweet_id AND media.media_url = v.media_url""",
                    update_values,
                    page_size=len(update_values) # Ensure all values are processed
                )
                session.commit()
                logger.info(f"Bulk updated {len(update_values)} media descriptions.")
                return len(update_values)
            except Exception as e:
                logger.error(f"Database error during bulk update: {e}")
                session.rollback()
                return 0 # Failed update
    return 0

def get_description_stats():
    """Get statistics about image description coverage"""
    with get_db_session() as session:
        session.execute("""
            SELECT 
                COUNT(*) as total_photos,
                COUNT(CASE WHEN image_desc IS NOT NULL AND image_desc != '' THEN 1 END) as with_descriptions,
                COUNT(CASE WHEN image_desc IS NULL OR image_desc = '' THEN 1 END) as without_descriptions
            FROM media m
            JOIN tweets t ON t.id = m.tweet_id
            WHERE m.type = 'photo'
            AND t.quoted_tweet_id IS NULL
        """)
        result = session.fetchone()
        return result

def main():
    parser = argparse.ArgumentParser(description="Process images with DeepSeek VL model")
    parser.add_argument("--batch-size", type=int, default=10, # Reduced default batch size for VL
                        help="Number of items to process per batch (default: 10)")
    parser.add_argument("--small-batch", action="store_true",
                        help="Process just 5 items (for testing)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing descriptions")
    args = parser.parse_args()
    
    # Get and display current description statistics
    stats = get_description_stats()
    total = stats['total_photos']
    with_desc = stats['with_descriptions']
    without_desc = stats['without_descriptions']
    coverage = (with_desc / total * 100) if total > 0 else 0
    
    print("\nCurrent Image Description Coverage:")
    print(f"Total photos: {total:,}")
    print(f"With descriptions: {with_desc:,} ({coverage:.1f}%)")
    print(f"Without descriptions: {without_desc:,}\n")
    
    batch_size = 5 if args.small_batch else args.batch_size
    if args.small_batch:
        print(f"Processing small test batch of {batch_size} items")
    if args.overwrite:
        print("Overwriting existing descriptions")
    
    # Get total count of items to process
    # total_count = get_total_count(overwrite=args.overwrite) # This is informational now
    # print(f"Total media items to process (estimate): {total_count}")
    
    # Setup VL model once at the start
    logger.info("Loading VL model...")
    model_path = "deepseek-ai/deepseek-vl-7b-chat"
    vl_chat_processor, tokenizer, vl_gpt, device, dtype = setup_vl_model(model_path)
    logger.info("VL model loaded successfully")

    total_processed_media = 0
    total_failed_media = 0

    # --- Fetch all Media Identifiers needing processing ---    
    unprocessed_media_identifiers = []
    with get_db_session() as session:
        logger.info("Fetching unprocessed media identifiers...")
        query = """
            SELECT t.id, m.media_url, t.text as tweet_text
            FROM tweets t
            JOIN media m ON t.id = m.tweet_id
            WHERE m.type = 'photo'
            AND t.quoted_tweet_id IS NULL 
        """
        if not args.overwrite:
            query += " AND (m.image_desc IS NULL OR m.image_desc = '')"
        
        session.execute(query)
        # Fetch as dicts for easier use later
        unprocessed_media_identifiers = [
            {'id': row['id'], 'image_url': row['media_url'], 'tweet_text': row['tweet_text']}
            for row in session.fetchall()
        ]
        logger.info(f"Found {len(unprocessed_media_identifiers)} media items needing descriptions.")

    if not unprocessed_media_identifiers:
        logger.info("No media items require description processing.")
        return

    # --- Process Media in Batches ---    
    logger.info(f"Processing {len(unprocessed_media_identifiers)} media items in batches of {batch_size}...")
    for i in range(0, len(unprocessed_media_identifiers), batch_size):
        batch_identifiers_data = unprocessed_media_identifiers[i:i + batch_size]
        # No need to fetch extra data, identifiers_data already has id and image_url
        
        logger.info(f"Processing media batch {i // batch_size + 1} with {len(batch_identifiers_data)} items...")
        
        # Define the processing function for this batch, capturing necessary VL variables
        def batch_process_func(item):
            return process_post(item, vl_chat_processor, vl_gpt, device, dtype)

        successful_media, failed_media = process_items_with_retries(
            batch_identifiers_data, 
            'media',
            batch_process_func # Pass the wrapper function
        )
        
        # Update DB
        processed_count = update_media_descriptions(successful_media)
        total_processed_media += processed_count
        # Note: failed_media contains the original items, not the count of failures during update
        total_failed_media += len(failed_media) 

        if failed_media:
             logger.error(f"Failed to process {len(failed_media)} media items in batch {i // batch_size + 1} after retries.")

        # Optional: Display intermediate progress stats
        # stats = get_description_stats()
        # ... display stats ...

    # --- Final Summary ---    
    logger.info("--- Image Description Generation Summary ---")
    logger.info(f"Successfully processed and updated: {total_processed_media} media items.")
    logger.info(f"Failed after retries: {total_failed_media} media items.")
    logger.info("--- Image Description Generation Complete ---")

    # Final stats display
    stats = get_description_stats()
    total = stats['total_photos']
    with_desc = stats['with_descriptions']
    without_desc = stats['without_descriptions']
    coverage = (with_desc / total * 100) if total > 0 else 0
    
    print("\nFinal Image Description Coverage:")
    print(f"Total photos: {total:,}")
    print(f"With descriptions: {with_desc:,} ({coverage:.1f}%)")
    print(f"Without descriptions: {without_desc:,}\n")

if __name__ == "__main__":
    main()