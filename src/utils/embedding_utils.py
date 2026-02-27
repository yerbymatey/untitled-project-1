import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import requests
from io import BytesIO
from typing import Union, List

# Using Nomic models
TEXT_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
VISION_MODEL_NAME = "nomic-ai/nomic-embed-vision-v1.5"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EMBEDDING_DIM = 768

# Initialize models once
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True).to(DEVICE)
text_model.eval()

def mean_pooling(model_output, attention_mask):
    """Mean pooling of token embeddings"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def download_image(url):
    """Download image from URL and convert to PIL Image"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def get_text_embedding(texts: Union[str, List[str]], task_prefix: str = "search_document") -> torch.Tensor:
    """Generate text embedding using Nomic's text model"""
    # Handle single text input
    if isinstance(texts, str):
        texts = [texts]
    
    # Add task prefix to texts
    prefixed_texts = [f"{task_prefix}: {text}" for text in texts]
    
    # Tokenize and encode
    encoded_input = tokenizer(prefixed_texts, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}
    
    # Generate embeddings
    with torch.no_grad():
        model_output = text_model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings

def get_image_embedding(image_url: str, description: str = None) -> torch.Tensor:
    """Generate image embedding using Nomic's vision model"""
    processor = AutoImageProcessor.from_pretrained(VISION_MODEL_NAME)
    vision_model = AutoModel.from_pretrained(VISION_MODEL_NAME, trust_remote_code=True).to(DEVICE)
    vision_model.eval()
    
    # Download and process image
    image = download_image(image_url)
    if image is None:
        raise ValueError(f"Failed to download image from {image_url}")
    
    with torch.no_grad():
        # Process image
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        img_emb = vision_model(**inputs).last_hidden_state
        image_features = F.normalize(img_emb[:, 0], p=2, dim=1)
        
        # If description is provided, process it with text model
        if description:
            text_features = get_text_embedding(description)[0]  # Get first (only) embedding
            
            # Combine image and text features
            joint_features = (image_features + text_features.unsqueeze(0)) / 2
            return F.normalize(joint_features, p=2, dim=1)
        
        return image_features

def get_embeddings_for_tweet(tweet_text: Union[str, List[str]]) -> torch.Tensor:
    """Generate embedding for tweet text(s)"""
    return get_text_embedding(tweet_text, task_prefix="search_document")

def get_query_embedding(query: Union[str, List[str]]) -> torch.Tensor:
    """Generate embedding for search query/queries"""
    return get_text_embedding(query, task_prefix="search_query")

def get_embeddings_for_media(image_url: str, description: str) -> torch.Tensor:
    """Generate joint embedding for media (image + description)"""
    # Add type prefix to description
    prefixed_description = f"image: {description}"
    return get_image_embedding(image_url, prefixed_description) 