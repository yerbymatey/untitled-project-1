import torch # Keep for potential type hints, though MLX handles tensors
import argparse
import logging
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from io import BytesIO # Needed for Docling

# --- Database Imports ---
from db.session import get_db_session
from psycopg2.extras import execute_values # For bulk updates

# --- Model & Processing Imports ---
# Using SmolDocling with MLX VLM
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config, stream_generate

# Docling imports for parsing the output
from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.document import DocTagsDocument, DoclingDocument

# --- Utility Imports ---
# No longer using DEVICE directly as MLX handles it
from utils.colnomic_utils import download_image # Using download_image from colnomic_utils
from scripts.encode_embeddings import process_items_with_retries # Reusing retry logic

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Using SmolDocling MLX model
GENERATIVE_VLM_NAME = "ds4sd/SmolDocling-256M-preview-mlx-bf16"
EXTRACTION_PROMPT = "Convert this page to docling." # Specific prompt for SmolDocling

# --- Model Loading (MLX VLM) ---
generative_model = None
generative_processor = None
model_config = None # Store model config if needed
model_loaded = False

def load_generative_vlm():
    global generative_model, generative_processor, model_config, model_loaded
    if model_loaded:
        return
    
    logger.info(f"Loading GENERATIVE VLM: {GENERATIVE_VLM_NAME} using MLX")
    try:
        # MLX load handles device placement automatically (CPU or Apple Silicon GPU)
        generative_model, generative_processor = load(GENERATIVE_VLM_NAME)
        model_config = load_config(GENERATIVE_VLM_NAME) # Load config for chat template
        
        model_loaded = True
        logger.info("Generative VLM (MLX) and processor loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Generative VLM (MLX): {e}", exc_info=True)
        raise

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
                   WHERE media.tweet_id = v.tweet_id
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

# --- Processing Function (Updated for SmolDocling/MLX) ---
def process_media_item_for_text(item):
    """
    Processes a single media item dict, extracts text using SmolDocling (MLX).
    Expected input item format: {'tweet_id': ..., 'media_url': ...}
    Returns: The extracted text string (as Markdown), or None if failed.
    """
    if not model_loaded:
        load_generative_vlm()
        if not model_loaded:
            logger.error("Generative VLM (MLX) not loaded. Cannot process item.")
            raise RuntimeError("Generative VLM (MLX) failed to load")

    media_url = item['media_url']
    
    try:
        pil_image = download_image(media_url) # Assuming download_image returns PIL image
        if pil_image is None:
            logger.warning(f"Failed to download image {media_url}")
            return None # Signal failure

        # Prepare input using MLX VLM utils
        formatted_prompt = apply_chat_template(generative_processor, model_config, EXTRACTION_PROMPT, num_images=1)
        
        # Generate DocTags output using stream_generate
        logger.debug(f"Generating DocTags for {media_url}...")
        output_doctags = ""
        # Use stream_generate as in the example
        for token in stream_generate(
            generative_model, generative_processor, formatted_prompt, [pil_image], 
            max_tokens=4096, # SmolDocling might need more tokens for full page
            verbose=False # Keep logs clean
        ):
            output_doctags += token.text
            # Stop condition from example
            if "</doctag>" in token.text: 
                break 
        
        if not output_doctags or not output_doctags.strip().endswith("</doctag>"):
             logger.warning(f"Incomplete or missing DocTags generated for {media_url}. Output: {output_doctags[:200]}...")
             # Decide how to handle incomplete tags - return None or try to parse anyway?
             # Returning None for now to be safe.
             return None

        logger.debug(f"Generated DocTags (first 200 chars): {output_doctags[:200]}...")

        # Parse DocTags and convert to Markdown using docling-core
        doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([output_doctags], [pil_image])
        doc = DoclingDocument(name=f"media_{item['tweet_id']}_{media_url.split('/')[-1]}") # Create name
        doc.load_from_doctags(doctags_doc)
        
        extracted_markdown = doc.export_to_markdown()
        logger.debug(f"Successfully extracted text (Markdown) for {media_url}")
        return extracted_markdown.strip()

    except Exception as e:
        logger.error(f"Error processing media item {media_url} with MLX/Docling: {e}", exc_info=True)
        raise e # Re-raise for process_items_with_retries

# --- Main Execution Logic (Largely unchanged, uses updated process_media_item_for_text) ---
def main():
    parser = argparse.ArgumentParser(description='Extract text from media images using SmolDocling (MLX)')
    parser.add_argument('--batch-size', type=int, default=5, # MLX models might still be intensive, adjust batch
                        help='Number of items to process per batch (default: 5)')
    parser.add_argument('--limit', type=int, default=100,
                        help='Total number of items to process in this run (default: 100)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retries for processing a single item (default: 3)')
    # No --force-cpu needed for MLX
    
    args = parser.parse_args()
    
    # MLX handles device selection automatically
    logger.info(f"Using MLX (device selected automatically)")

    # Load the generative VLM (and processor) early
    try:
        load_generative_vlm()
    except Exception:
        logger.critical("Failed to load the generative VLM (MLX). Exiting.")
        return
        
    batch_size = args.batch_size
    total_limit = args.limit
    max_retries = args.max_retries
    
    total_processed_successfully = 0
    total_failed_permanently = 0
    items_processed_count = 0

    logger.info(f"Starting text extraction process using SmolDocling. Processing up to {total_limit} items.")

    media_to_process = get_unprocessed_media(limit=total_limit)

    if not media_to_process:
        logger.info("No media items found requiring text extraction.")
        return

    for i in range(0, len(media_to_process), batch_size):
        batch_items = media_to_process[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} with {len(batch_items)} items...")

        successful_results, failed_items = process_items_with_retries(
            batch_items, 
            'media_text_extraction_smoldocling', # Changed type name slightly
            lambda item: {'extracted_text': process_media_item_for_text(item)}, 
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
