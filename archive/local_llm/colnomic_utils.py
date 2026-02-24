import torch
from PIL import Image
from io import BytesIO
import requests
import logging
from transformers.utils.import_utils import is_flash_attn_2_available

# Note: ColQwen2_5 and ColQwen2_5_Processor are for EMBEDDING generation (retrieval),
# not direct text generation from images.
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# For device placement
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

# This model is designed for RETRIEVAL EMBEDDINGS, not generation.
COLNOMIC_EMBED_MODEL_NAME = "nomic-ai/colnomic-embed-multimodal-7b"
# EXTRACTION_PROMPT = "Extract all text visible in this image." # Prompt will be used with a GENERATIVE VLM, not this model.

# --- Model Loading (for Embeddings, if needed) ---
# This section loads the Colnomic model intended for generating embeddings,
# similar to the original example. It is NOT used for the text generation task.
colnomic_embed_model = None
colnomic_embed_processor = None
embed_model_loaded = False

def load_colnomic_embedding_model():
    global colnomic_embed_model, colnomic_embed_processor, embed_model_loaded
    if embed_model_loaded:
        return
    
    logger.info(f"Loading Colnomic EMBEDDING model: {COLNOMIC_EMBED_MODEL_NAME} onto device: {DEVICE}")
    try:
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() and DEVICE != 'cpu' else "sdpa"
        torch_dtype = torch.bfloat16 if DEVICE == 'cuda' else torch.float32

        colnomic_embed_model = ColQwen2_5.from_pretrained(
            COLNOMIC_EMBED_MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map=DEVICE,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        ).eval()

        colnomic_embed_processor = ColQwen2_5_Processor.from_pretrained(COLNOMIC_EMBED_MODEL_NAME)
        
        embed_model_loaded = True
        logger.info("Colnomic EMBEDDING model and processor loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading Colnomic EMBEDDING model: {e}", exc_info=True)
        raise

# --- Helper for downloading images ---
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

# Optional: Add a main block for testing if needed
# if __name__ == '__main__':
#     test_url = "..." # Add a test image URL
#     extracted = get_extracted_text_from_image(test_url)
#     if extracted:
#         print("Extracted Text:")
#         print(extracted)
#     else:
#         print("Failed to extract text.")