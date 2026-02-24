import sys
import os
import logging
from PIL import Image
from io import BytesIO
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_image(url):
    """
    Download an image from a URL and return it as a PIL Image.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
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
def get_unprocessed_media_from_db(limit=5):
    """Connect to the database and fetch a few unprocessed media URLs."""
    try:
        # This will use the existing DB session code
        from db.session import get_db_session
        
        with get_db_session() as session:
            session.execute("""
                SELECT tweet_id, media_url
                FROM media
                WHERE type = 'photo' AND extr_text IS NULL
                ORDER BY tweet_id DESC
                LIMIT %s
            """, (limit,))
            unprocessed_media = session.fetchall()
            logger.info(f"Found {len(unprocessed_media)} media items needing text extraction.")
            return unprocessed_media
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        # Return some example URLs for testing - using real URLs that should work
        return [
            {"tweet_id": "00000", "media_url": "https://pbs.twimg.com/profile_images/1727929124410044416/1-Ysa3vz_400x400.jpg"},
            {"tweet_id": "00000", "media_url": "https://pbs.twimg.com/profile_images/1771908017999544320/L-6pHBAb_400x400.jpg"}
        ]

def main():
    # Get a few images from the DB
    media_items = get_unprocessed_media_from_db(limit=5)
    
    if not media_items:
        logger.info("No media items found requiring text extraction.")
        return
    
    logger.info(f"Testing image download for {len(media_items)} items")
    
    for item in media_items:
        media_url = item['media_url']
        tweet_id = item['tweet_id']
        
        logger.info(f"Downloading image from {media_url} (tweet {tweet_id})")
        image = download_image(media_url)
        
        if image:
            # Print basic info about the image
            width, height = image.size
            mode = image.mode
            format = image.format
            logger.info(f"Successfully downloaded image: {width}x{height}, mode={mode}, format={format}")
            
            # You could save the image temporarily to inspect it
            # output_dir = "test_output"
            # os.makedirs(output_dir, exist_ok=True)
            # filename = os.path.join(output_dir, f"{tweet_id}_{os.path.basename(media_url)}")
            # image.save(filename)
            # logger.info(f"Saved image to {filename}")
        else:
            logger.error(f"Failed to download image from {media_url}")

if __name__ == "__main__":
    main()