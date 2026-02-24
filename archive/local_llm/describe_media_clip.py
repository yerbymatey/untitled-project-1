import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import argparse
import os
import sys
import requests
from PIL import Image
from io import BytesIO
import tempfile
from huggingface_hub import snapshot_download

from db.session import get_db_session
from utils.vl_utils import setup_vl_model, process_vl_conversation
from utils.process_images import resize_image

# Using OpenAI's CLIP model with 768 dimensions
VISION_MODEL_NAME = "openai/clip-vit-large-patch14-336"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Using device: {DEVICE}")

def ensure_model_files(model_name):
    """Ensure model files are downloaded"""
    try:
        snapshot_download(repo_id=model_name, local_files_only=False)
        return True
    except Exception as e:
        print(f"Error downloading model {model_name}: {e}", file=sys.stderr)
        return False

def get_unprocessed_media(limit=100):
    """Get media items that don't have descriptions yet"""
    with get_db_session() as session:
        query = """
        SELECT tweet_id, media_url, type
        FROM media
        WHERE (image_desc IS NULL OR image_desc = '')
        AND type = 'photo'
        LIMIT %s
        """
        session.execute(query, (limit,))
        results = session.fetchall()
        return [(row['tweet_id'], row['media_url'], row['type']) for row in results]

def update_media_descriptions(tweet_ids, media_urls, descriptions):
    """Update image descriptions in the media table"""
    with get_db_session() as session:
        for tweet_id, media_url, description in zip(tweet_ids, media_urls, descriptions):
            query = """
            UPDATE media 
            SET image_desc = %s
            WHERE tweet_id = %s AND media_url = %s
            """
            session.execute(query, (description, tweet_id, media_url))
        session.commit()

def process_media_batch(batch_size=100):
    """Process a batch of media items for image descriptions"""
    print(f"Loading VL model...")
    
    # Setup VL model
    model_path = "deepseek-ai/deepseek-vl-7b-chat"
    vl_chat_processor, tokenizer, vl_gpt, device, dtype = setup_vl_model(model_path)
    
    # Process media without descriptions
    media_data = get_unprocessed_media(batch_size)
    if not media_data:
        print("No media items found without descriptions")
        return
    
    print(f"Processing {len(media_data)} media items for descriptions...")
    tweet_ids = []
    media_urls = []
    all_descriptions = []
    
    for tweet_id, media_url, media_type in tqdm(media_data, desc="Processing media"):
        try:
            # Create a temporary directory for the image
            with tempfile.TemporaryDirectory() as temp_dir:
                # Download image from URL
                response = requests.get(media_url)
                if response.status_code != 200:
                    print(f"Failed to download image from {media_url}")
                    continue
                
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
                description, sft_format = process_vl_conversation(conversation, vl_chat_processor, vl_gpt, device, dtype)
                
                all_descriptions.append(description)
                tweet_ids.append(tweet_id)
                media_urls.append(media_url)
                
        except Exception as e:
            print(f"Error processing image for tweet {tweet_id}: {e}")
            continue
    
    # Update media descriptions
    if all_descriptions:
        update_media_descriptions(tweet_ids, media_urls, all_descriptions)
        print(f"Updated {len(all_descriptions)} media descriptions")
    else:
        print("No descriptions were generated")

def main():
    parser = argparse.ArgumentParser(description='Process media with DeepSeek VL for descriptions')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of items to process (default: 100)')
    parser.add_argument('--small-batch', action='store_true',
                        help='Process just 5 items (for testing)')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force using CPU even if GPU is available')
    
    args = parser.parse_args()
    
    # Force CPU if requested
    global DEVICE
    if args.force_cpu:
        DEVICE = "cpu"
        print("Forcing CPU usage")
    
    print(f"Using device: {DEVICE}")
    
    # Use a small batch size for testing if requested
    batch_size = 5 if args.small_batch else args.batch_size
    if args.small_batch:
        print(f"Processing small test batch of {batch_size} items")
    
    while True:
        process_media_batch(batch_size)
        
        # Check if there are any remaining media items without descriptions
        with get_db_session() as session:
            session.execute("""
                SELECT COUNT(*) as count 
                FROM media 
                WHERE (image_desc IS NULL OR image_desc = '')
                AND type = 'photo'
            """)
            result = session.fetchone()
            remaining = result['count'] if result else 0
            
        if remaining == 0:
            print("All media items have been processed!")
            break
        else:
            print(f"Remaining media items to process: {remaining}")

if __name__ == "__main__":
    main() 