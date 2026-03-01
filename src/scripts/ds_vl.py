import argparse
from pathlib import Path
import tempfile
import os
import requests
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image
from tqdm import tqdm

from src.utils.vl_utils import setup_vl_model, process_vl_conversation
from src.db.session import get_db_session
from src.utils.process_images import resize_image

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
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download image from URL
        response = requests.get(post['image_url'])
        if response.status_code != 200:
            raise Exception(f"Failed to download image: {response.status_code}")
        
        # Convert to PIL Image
        image = Image.open(BytesIO(response.content))
        
        # Resize image
        resized_image = resize_image(image, output_size=(512, 512))
        
        # Convert to RGB mode before saving as JPEG
        resized_image = resized_image.convert('RGB')
        
        # Save to temporary file
        image_path = os.path.join(temp_dir, f"{post['id']}.jpg")
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
            "post_id": post['id'],
            "tweet_text": post['tweet_text'],
            "image_description": answer,
            "media_url": post['image_url']
        }

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
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Number of items to process per batch (default: 50)")
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
    total_count = get_total_count(overwrite=args.overwrite)
    print(f"Total media items to process: {total_count}")
    
    if total_count == 0:
        print("No items to process!")
        return
    
    # Calculate total number of batches
    total_batches = (total_count + batch_size - 1) // batch_size
    print(f"Will process in {total_batches} batches of {batch_size} items")
    
    # Setup VL model once at the start
    print("Loading VL model...")
    model_path = "deepseek-ai/deepseek-vl-7b-chat"
    vl_chat_processor, tokenizer, vl_gpt, device, dtype = setup_vl_model(model_path)
    print("VL model loaded successfully")
    
    offset = 0
    total_processed = 0
    total_failed = 0
    batch_num = 1
    
    while True:
        # Get next batch of posts
        posts = get_unprocessed_media(offset=offset, batch_size=batch_size, overwrite=args.overwrite)
        if not posts:
            break
            
        # Process the batch
        processed, failed = process_batch(posts, vl_chat_processor, vl_gpt, device, dtype, batch_num, total_batches)
        if processed == 0 and failed == len(posts):
            print("All items in batch failed, stopping...")
            break
            
        total_processed += processed
        total_failed += failed
        offset += batch_size
        
        # Check remaining items
        remaining = get_total_count(overwrite=args.overwrite)
        if remaining == 0:
            print("\nAll items have been processed!")
            print(f"Total processed: {total_processed}")
            print(f"Total failed: {total_failed}")
            print(f"Success rate: {(total_processed / (total_processed + total_failed) * 100):.1f}%")
            break
        else:
            # Get updated statistics
            stats = get_description_stats()
            total = stats['total_photos']
            with_desc = stats['with_descriptions']
            without_desc = stats['without_descriptions']
            coverage = (with_desc / total * 100) if total > 0 else 0
            
            print("\nCurrent Progress:")
            print(f"Overall Progress: {((total_count - remaining) / total_count * 100):.1f}%")
            print(f"Batch Progress: {batch_num}/{total_batches}")
            print(f"This Run - Processed: {total_processed}, Failed: {total_failed}")
            print(f"\nOverall Coverage:")
            print(f"Total photos: {total:,}")
            print(f"With descriptions: {with_desc:,} ({coverage:.1f}%)")
            print(f"Without descriptions: {without_desc:,}")
            print(f"Remaining in queue: {remaining}")
        
        batch_num += 1

if __name__ == "__main__":
    main()