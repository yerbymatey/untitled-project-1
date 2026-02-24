import argparse
import logging
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import requests
import json
import os

# --- Database Imports ---
from db.session import get_db_session
from psycopg2.extras import execute_values # For bulk updates

# --- Try multiple VLM providers to extract text from images ---
# Option 1: OpenAI Vision API (if available)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
has_openai = False
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        has_openai = True
    except ImportError:
        print("OpenAI Python package not installed. Install with: uv pip install openai")
        has_openai = False

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_image(url):
    """
    Download an image from a URL and return it as a PIL Image.
    Args:
        url: URL of the image to download
    Returns:
        PIL.Image: The downloaded image, or None if download failed
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Convert the content to a PIL Image
        image = Image.open(BytesIO(response.content))
        
        # Convert to RGB if it's RGBA or other mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return None

# --- Database Functions ---
def get_unprocessed_media(limit=100):
    """Fetch media items that need text extraction (extr_text is NULL)."""
    with get_db_session() as session:
        session.execute("""
            SELECT tweet_id, media_url
            FROM media
            WHERE type = 'photo' AND extr_text IS NULL
            ORDER BY tweet_id DESC -- Or some other order if needed
            LIMIT %s
        """, (limit,))
        unprocessed_media = session.fetchall()
        logger.info(f"Found {len(unprocessed_media)} media items needing text extraction.")
        return unprocessed_media # List of dicts: [{'tweet_id': ..., 'media_url': ...}]

def update_media_text(successful_items):
    """Update media extr_text in the database for successfully processed items."""
    if not successful_items:
        return 0

    update_values = []
    skipped_count = 0
    for item in successful_items:
        tweet_id = item.get('tweet_id')
        media_url = item.get('media_url')
        extracted_text = item.get('extracted_text') # Key where we store the result
        
        if tweet_id and media_url and extracted_text is not None: # Allow empty string
            update_values.append((extracted_text, tweet_id, media_url))
        else:
            logger.warning(f"Skipping update for media {tweet_id}/{media_url} due to missing data or failed extraction.")
            skipped_count += 1
            
    if not update_values:
        logger.info(f"No valid extracted text to update (skipped {skipped_count} items).")
        return 0

    with get_db_session() as session:
        try:
            execute_values(
                session.cursor,
                """UPDATE media SET extr_text = v.extr_text
                   FROM (VALUES %s) AS v(extr_text, tweet_id, media_url)
                   WHERE media.tweet_id::bigint = v.tweet_id::bigint 
                   AND media.media_url = v.media_url;""",
                update_values
            )
            count = len(update_values)
            logger.info(f"Bulk updated {count} media items with extracted text.")
            return count
        except Exception as e:
            logger.error(f"Database error during bulk update: {e}", exc_info=True)
            session.rollback()
            return 0

# --- Text Extraction using OpenAI ---
def extract_text_with_openai(image_path):
    """Extract text from an image using OpenAI's Vision API."""
    if not has_openai:
        return "OpenAI API not available"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all visible text from this image. Format it in a readable way preserving original layout. Do not describe the image, just extract the text."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_path}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error using OpenAI Vision API: {e}")
        return None

# --- Main Processing Function ---
def process_media_item_for_text(item):
    """
    Processes a single media item dict, extracts text using OpenAI Vision or another available service.
    Expected input item format: {'tweet_id': ..., 'media_url': ...}
    Returns: The extracted text string, or None if failed.
    """
    media_url = item['media_url']
    
    try:
        pil_image = download_image(media_url)
        if pil_image is None:
            logger.warning(f"Failed to download image {media_url}")
            return None
            
        # Option 1: Try with OpenAI if available
        if has_openai:
            # Convert PIL image to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_base64 = buffered.getvalue()
            import base64
            img_base64_str = base64.b64encode(img_base64).decode()
            
            extracted_text = extract_text_with_openai(img_base64_str)
            if extracted_text:
                logger.info(f"Successfully extracted text with OpenAI for {media_url}")
                return extracted_text

        # If all methods fail or aren't available
        logger.warning(f"No text extraction method available for {media_url}")
        return "No text extraction method available"
        
    except Exception as e:
        logger.error(f"Error processing media item {media_url}: {e}", exc_info=True)
        return None

# --- Process Items with Retries ---
def process_items_with_retries(items, item_type, process_func, max_retries=3):
    """Processes items with retries on failure."""
    if not items:
        return [], []

    items_to_process = list(items)
    successful_items = []
    permanently_failed_items = []
    retry_counts = {}
    current_pass = 0

    while items_to_process and current_pass <= max_retries:
        next_retry_items = []
        logger.info(f"Processing pass {current_pass + 1}/{max_retries + 1} for {len(items_to_process)} {item_type} items...")
        
        for item in tqdm(items_to_process, desc=f"Pass {current_pass + 1} - {item_type.capitalize()}"):
            # Create identifier
            item_id = (item.get('tweet_id', ''), item.get('media_url', ''))
            if not all(item_id):
                logger.error(f"Skipping item due to missing ID fields: {item}")
                permanently_failed_items.append(item)
                continue

            try:
                # Run the provided processing function
                processing_result = process_func(item)
                
                # Create a copy to store results
                item_copy = item.copy()
                
                # Add the results
                if isinstance(processing_result, dict):
                    item_copy.update(processing_result)
                else:
                    item_copy['extracted_text'] = processing_result
                
                successful_items.append(item_copy)

            except Exception as e:
                retry_counts[item_id] = retry_counts.get(item_id, 0) + 1
                logger.warning(f"Attempt {retry_counts[item_id]}/{max_retries} failed for {item_type} {item_id}: {e}")
                
                if retry_counts[item_id] < max_retries:
                    next_retry_items.append(item)
                else:
                    logger.error(f"{item_type.capitalize()} {item_id} failed after {max_retries} retries.")
                    permanently_failed_items.append(item)

        items_to_process = next_retry_items
        current_pass += 1

    # Handle any remaining failed items
    for item in items_to_process:
        item_id = (item.get('tweet_id', ''), item.get('media_url', ''))
        if all(item_id):
            logger.error(f"{item_type.capitalize()} {item_id} failed after {retry_counts.get(item_id, 0)} attempts.")
        permanently_failed_items.append(item)

    return successful_items, permanently_failed_items

# --- Main Execution Logic ---
def main():
    parser = argparse.ArgumentParser(description='Extract text from media images using OpenAI Vision')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Number of items to process per batch (default: 5)')
    parser.add_argument('--limit', type=int, default=10,
                        help='Total number of items to process in this run (default: 10)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retries for processing a single item (default: 3)')
    
    args = parser.parse_args()
    
    # Check if we have any text extraction methods available
    if not has_openai:
        logger.warning("No text extraction methods available. Please install OpenAI or other vision API.")
    
    batch_size = args.batch_size
    total_limit = args.limit
    max_retries = args.max_retries
    
    total_processed_successfully = 0
    total_failed_permanently = 0
    items_processed_count = 0

    logger.info(f"Starting text extraction process. Processing up to {total_limit} items.")

    media_to_process = get_unprocessed_media(limit=total_limit)

    if not media_to_process:
        logger.info("No media items found requiring text extraction.")
        return

    for i in range(0, len(media_to_process), batch_size):
        batch_items = media_to_process[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} with {len(batch_items)} items...")

        successful_results, failed_items = process_items_with_retries(
            batch_items, 
            'media_text_extraction', 
            lambda item: process_media_item_for_text(item), 
            max_retries=max_retries
        )
        
        processed_count = update_media_text(successful_results)
        total_processed_successfully += processed_count
        total_failed_permanently += len(failed_items)
        items_processed_count += len(batch_items)

        if failed_items:
             logger.error(f"Failed to process {len(failed_items)} media items in batch {i // batch_size + 1} after retries.")

    logger.info("--- Text Extraction Summary ---")
    logger.info(f"Successfully processed and updated: {total_processed_successfully} media items.")
    logger.info(f"Failed permanently after retries: {total_failed_permanently} media items.")
    logger.info("--- Text Extraction Complete ---")

if __name__ == "__main__":
    main()