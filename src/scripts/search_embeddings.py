import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
import os
import sys
from huggingface_hub import snapshot_download
import numpy as np
from typing import List, Dict, Tuple

from src.db.session import get_db_session
from src.utils.embedding_utils import (
    get_text_embedding,
    mean_pooling,
    DEVICE,
    EMBEDDING_DIM,
    TEXT_MODEL_NAME
)

# Initialize models once
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME, trust_remote_code=True).to(DEVICE)
text_model.eval()

def ensure_model_files(model_name):
    """Ensure model files are downloaded"""
    try:
        snapshot_download(repo_id=model_name, local_files_only=False)
        return True
    except Exception as e:
        print(f"Error downloading model {model_name}: {e}", file=sys.stderr)
        return False

def get_query_embedding(query: str) -> torch.Tensor:
    """Generate embedding for the search query"""
    sentences = [f'search_query: {query}']
    
    # Tokenize and encode
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}
    
    # Generate embeddings
    with torch.no_grad():
        model_output = text_model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.squeeze(0)

def compute_similarities(query_embedding: torch.Tensor, stored_embeddings: List[List[float]]) -> torch.Tensor:
    """Compute cosine similarities between query and stored embeddings"""
    if not stored_embeddings:
        return torch.tensor([])
    
    # Convert stored embeddings to tensor
    stored_tensor = torch.tensor(stored_embeddings).to(DEVICE)
    
    # Compute cosine similarities
    similarities = torch.matmul(stored_tensor, query_embedding)
    return similarities

def debug_vector_stats(vector_str: str):
    """Print statistics about a vector stored in the database"""
    try:
        # Convert string representation to numpy array
        vector = np.array(eval(vector_str))
        print(f"\nStored vector shape: {vector.shape}")
        print(f"Stored vector norm: {np.linalg.norm(vector):.4f}")
        print(f"Stored vector mean: {np.mean(vector):.4f}")
        print(f"Stored vector std: {np.std(vector):.4f}")
    except Exception as e:
        print(f"Error analyzing vector: {e}")

def search_tweets(query: str, limit: int = 10) -> List[Dict]:
    """
    Search tweets using tensor operations for similarity computation
    
    Args:
        query (str): Search query text
        limit (int): Maximum number of results to return
        
    Returns:
        list: List of dictionaries containing tweet data and similarity scores
    """
    try:
        # Generate query embedding
        query_embedding = get_query_embedding(query)
        
        with get_db_session() as session:
            # Get all tweet embeddings - cast vector to text then parse
            session.execute("""
                SELECT t.id, t.text, t.created_at, t.embedding::text as embedding_array
                FROM tweets t
                WHERE t.embedding IS NOT NULL
            """)
            results = session.fetchall()
            
            if not results:
                return []
            
            # Parse text representation back into list
            stored_embeddings = []
            tweet_data = []
            for r in results:
                try:
                    # Remove brackets and split into floats
                    vector_str = r['embedding_array'].strip('[]')
                    vector = [float(x) for x in vector_str.split(',')]
                    stored_embeddings.append(vector)
                    tweet_data.append({
                        'id': r['id'],
                        'text': r['text'],
                        'created_at': r['created_at']
                    })
                except Exception as e:
                    print(f"Error parsing vector for tweet {r['id']}: {e}")
                    continue
            
            if not stored_embeddings:
                return []
            
            # Compute similarities
            similarities = compute_similarities(query_embedding, stored_embeddings)
            
            # Sort by similarity and get top results
            if len(similarities) > 0:
                top_indices = torch.argsort(similarities, descending=True)[:limit]
                
                # Build final results
                search_results = []
                for idx in top_indices:
                    i = idx.item()
                    search_results.append({
                        **tweet_data[i],
                        'similarity': similarities[i].item()
                    })
                
                return search_results
            
            return []
            
    except Exception as e:
        print(f"Error searching tweets: {e}")
        return []

def search_media(query: str, limit: int = 10) -> List[Dict]:
    """
    Search media using tensor operations for similarity computation
    
    Args:
        query (str): Search query text
        limit (int): Maximum number of results to return
        
    Returns:
        list: List of dictionaries containing media data and similarity scores
    """
    try:
        # Generate query embedding
        query_embedding = get_query_embedding(query)
        
        with get_db_session() as session:
            # Get all media embeddings - cast vector to text then parse
            session.execute("""
                SELECT m.tweet_id, t.text, m.media_url, m.image_desc, t.created_at, 
                       m.embedding::text as embedding_array
                FROM media m
                JOIN tweets t ON t.id = m.tweet_id
                WHERE m.embedding IS NOT NULL
                AND m.type = 'photo'
            """)
            results = session.fetchall()
            
            if not results:
                return []
            
            # Parse text representation back into list
            stored_embeddings = []
            media_data = []
            for r in results:
                try:
                    # Remove brackets and split into floats
                    vector_str = r['embedding_array'].strip('[]')
                    vector = [float(x) for x in vector_str.split(',')]
                    stored_embeddings.append(vector)
                    media_data.append({
                        'tweet_id': r['tweet_id'],
                        'text': r['text'],
                        'media_url': r['media_url'],
                        'image_desc': r['image_desc'],
                        'created_at': r['created_at']
                    })
                except Exception as e:
                    print(f"Error parsing vector for media {r['tweet_id']}: {e}")
                    continue
            
            if not stored_embeddings:
                return []
            
            # Compute similarities
            similarities = compute_similarities(query_embedding, stored_embeddings)
            
            # Sort by similarity and get top results
            if len(similarities) > 0:
                top_indices = torch.argsort(similarities, descending=True)[:limit]
                
                # Build final results
                search_results = []
                for idx in top_indices:
                    i = idx.item()
                    search_results.append({
                        **media_data[i],
                        'similarity': similarities[i].item()
                    })
                
                return search_results
            
            return []
            
    except Exception as e:
        print(f"Error searching media: {e}")
        return []

def search_all(query: str, limit: int = 10) -> Tuple[List[Dict], List[Dict]]:
    """
    Search both tweets and media with a single query
    
    Args:
        query (str): Search query text
        limit (int): Maximum number of results to return for each type
        
    Returns:
        tuple: (tweet_results, media_results)
    """
    return search_tweets(query, limit), search_media(query, limit) 