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

from db.session import get_db_session
from utils.embedding_utils import (
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
            # Also LEFT JOIN to get associated media info (picking one media item per tweet)
            # Using ROW_NUMBER() for compatibility and clarity
            query = """
            WITH RankedTweets AS (
                SELECT
                    t.id,
                    t.text,
                    t.created_at,
                    t.embedding::text as embedding_array,
                    m.media_url,
                    m.image_desc,
                    ROW_NUMBER() OVER(PARTITION BY t.id ORDER BY m.media_url NULLS LAST) as rn -- Pick one media per tweet, handle NULLs
                FROM tweets t
                LEFT JOIN media m ON t.id = m.tweet_id AND m.type = 'photo'
                WHERE t.embedding IS NOT NULL
            )
            SELECT
                id,
                text,
                created_at,
                embedding_array,
                media_url,
                image_desc
            FROM RankedTweets
            WHERE rn = 1;
            """
            session.execute(query)
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
                        'created_at': r['created_at'],
                        'media_url': r['media_url'],
                        'image_desc': r['image_desc']
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

def search_media(query: str, limit: int = 10, weight_joint: float = 0.5, weight_image: float = 0.5) -> List[Dict]:
    """
    Search media using tensor operations for similarity computation.
    Compares the text query against both the joint (text+image) embedding 
    and the image-only embedding, then combines the scores.
    
    Args:
        query (str): Search query text
        limit (int): Maximum number of results to return
        weight_joint (float): Weight for the joint embedding similarity score.
        weight_image (float): Weight for the image embedding similarity score.
        
    Returns:
        list: List of dictionaries containing media data and combined similarity scores
    """
    # Ensure weights sum approximately to 1 if needed, or handle as desired.
    # For simplicity, we assume they are provided appropriately.
    
    try:
        # Generate query embedding
        query_embedding = get_query_embedding(query)
        
        with get_db_session() as session:
            # Get all media embeddings (joint and image) - cast vector to text then parse
            session.execute("""
                SELECT 
                    m.tweet_id, 
                    t.text, 
                    m.media_url, 
                    m.image_desc, 
                    t.created_at, 
                    m.joint_embedding::text as joint_embedding_array,
                    m.image_embedding::text as image_embedding_array
                FROM media m
                JOIN tweets t ON t.id = m.tweet_id
                WHERE m.joint_embedding IS NOT NULL
                  AND m.image_embedding IS NOT NULL 
                  AND m.type = 'photo'
            """)
            results = session.fetchall()
            
            if not results:
                return []
            
            # Parse text representations back into lists
            stored_joint_embeddings = []
            stored_image_embeddings = []
            media_data = []
            valid_indices = [] # Keep track of indices with valid embeddings
            for idx, r in enumerate(results):
                try:
                    # Parse joint embedding
                    joint_vector_str = r['joint_embedding_array'].strip('[]')
                    joint_vector = [float(x) for x in joint_vector_str.split(',')]
                    
                    # Parse image embedding
                    image_vector_str = r['image_embedding_array'].strip('[]')
                    image_vector = [float(x) for x in image_vector_str.split(',')]
                    
                    # Check embedding dimensions match expected (optional but good practice)
                    if len(joint_vector) != EMBEDDING_DIM or len(image_vector) != EMBEDDING_DIM:
                        logger.warning(f"Skipping media {r['tweet_id']}/{r['media_url']} due to unexpected embedding dimension.")
                        continue

                    stored_joint_embeddings.append(joint_vector)
                    stored_image_embeddings.append(image_vector)
                    media_data.append({
                        'tweet_id': r['tweet_id'],
                        'text': r['text'],
                        'media_url': r['media_url'],
                        'image_desc': r['image_desc'],
                        'created_at': r['created_at']
                    })
                    valid_indices.append(idx) # Store original index if parsing succeeds

                except Exception as e:
                    print(f"Error parsing vectors for media {r['tweet_id']}/{r['media_url']}: {e}")
                    continue # Skip this item if parsing fails
            
            if not media_data:
                return []
            
            # Compute similarities for both embedding types
            joint_similarities = compute_similarities(query_embedding, stored_joint_embeddings)
            image_similarities = compute_similarities(query_embedding, stored_image_embeddings)
            
            # Combine similarities using weights
            # Ensure tensors have the same shape
            if joint_similarities.shape != image_similarities.shape:
                 raise ValueError("Shape mismatch between joint and image similarities")

            combined_scores = (weight_joint * joint_similarities) + (weight_image * image_similarities)
            
            # Sort by combined score and get top results
            if len(combined_scores) > 0:
                # argsort returns indices into the *filtered* lists (stored_embeddings, media_data)
                top_indices_in_filtered = torch.argsort(combined_scores, descending=True)[:limit]
                
                # Build final results using the filtered media_data
                search_results = []
                for idx_in_filtered in top_indices_in_filtered:
                    original_data_index = idx_in_filtered.item() # Index within media_data list
                    search_results.append({
                        **media_data[original_data_index],
                        'similarity': combined_scores[original_data_index].item(),
                        'joint_similarity': joint_similarities[original_data_index].item(), # Optional: include individual scores
                        'image_similarity': image_similarities[original_data_index].item()  # Optional: include individual scores
                    })
                
                return search_results
            
            return []
            
    except Exception as e:
        print(f"Error searching media: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return []

def search_media_image_only(query: str, limit: int = 10) -> List[Dict]:
    """
    Search media based *only* on the similarity between the text query 
    and the image-only embedding.
    
    Args:
        query (str): Search query text
        limit (int): Maximum number of results to return
        
    Returns:
        list: List of dictionaries containing media data and image similarity scores
    """
    try:
        # Generate query embedding
        query_embedding = get_query_embedding(query)
        
        with get_db_session() as session:
            # Get only the image embeddings - cast vector to text then parse
            session.execute("""
                SELECT 
                    m.tweet_id, 
                    t.text, 
                    m.media_url, 
                    m.image_desc, 
                    t.created_at, 
                    m.image_embedding::text as image_embedding_array
                FROM media m
                JOIN tweets t ON t.id = m.tweet_id
                WHERE m.image_embedding IS NOT NULL 
                  AND m.type = 'photo'
            """)
            results = session.fetchall()
            
            if not results:
                return []
            
            # Parse text representation back into list
            stored_image_embeddings = []
            media_data = []
            for r in results:
                try:
                    # Parse image embedding
                    image_vector_str = r['image_embedding_array'].strip('[]')
                    image_vector = [float(x) for x in image_vector_str.split(',')]
                    
                    if len(image_vector) != EMBEDDING_DIM:
                        logger.warning(f"Skipping media {r['tweet_id']}/{r['media_url']} due to unexpected embedding dimension.")
                        continue
                        
                    stored_image_embeddings.append(image_vector)
                    media_data.append({
                        'tweet_id': r['tweet_id'],
                        'text': r['text'],
                        'media_url': r['media_url'],
                        'image_desc': r['image_desc'],
                        'created_at': r['created_at']
                    })
                except Exception as e:
                    print(f"Error parsing image vector for media {r['tweet_id']}/{r['media_url']}: {e}")
                    continue # Skip this item if parsing fails
            
            if not media_data:
                return []
            
            # Compute similarities against image embeddings
            image_similarities = compute_similarities(query_embedding, stored_image_embeddings)
            
            # Sort by image similarity and get top results
            if len(image_similarities) > 0:
                top_indices = torch.argsort(image_similarities, descending=True)[:limit]
                
                # Build final results
                search_results = []
                for idx in top_indices:
                    i = idx.item()
                    search_results.append({
                        **media_data[i],
                        'similarity': image_similarities[i].item()
                    })
                
                return search_results
            
            return []
            
    except Exception as e:
        print(f"Error searching media (image only): {e}")
        import traceback
        traceback.print_exc()
        return []

def search_traditional_combined(query: str, limit: int = 10) -> List[Dict]:
    """
    Performs a combined search across tweets and media, ranking all items together.

    Compares the text query individually against:
    1. Tweet text embeddings
    2. Media joint embeddings
    3. Media image embeddings

    All results are pooled and ranked purely by their calculated cosine similarity
    to the query embedding. A single media item might appear twice if both its
    joint and image embeddings rank highly.

    Args:
        query (str): Search query text
        limit (int): Maximum number of results to return from the combined list.

    Returns:
        list: A single list of dictionaries, containing tweet or media data,
              ranked by similarity score. Each item includes a 'type' field
              ('tweet', 'media_joint', 'media_image') and 'similarity'.
    """
    all_results = []
    try:
        query_embedding = get_query_embedding(query)

        with get_db_session() as session:
            # 1. Fetch Tweet Embeddings
            session.execute("""
                SELECT
                    t.id,
                    t.text,
                    t.created_at,
                    t.embedding::text as embedding_array,
                    m.media_url, -- Include for context, might be null
                    m.image_desc -- Include for context, might be null
                FROM tweets t
                LEFT JOIN (
                    SELECT DISTINCT ON (tweet_id) tweet_id, media_url, image_desc
                    FROM media WHERE type = 'photo' ORDER BY tweet_id, media_url
                ) m ON t.id = m.tweet_id
                WHERE t.embedding IS NOT NULL
            """)
            tweet_db_results = session.fetchall()

            if tweet_db_results:
                tweet_embeddings = []
                tweet_data_map = {} # Store data by index
                for i, r in enumerate(tweet_db_results):
                    try:
                        vector_str = r['embedding_array'].strip('[]')
                        vector = [float(x) for x in vector_str.split(',')]
                        if len(vector) == EMBEDDING_DIM:
                           tweet_embeddings.append(vector)
                           tweet_data_map[len(tweet_embeddings) - 1] = { # Map index in embedding list to data
                               'id': r['id'],
                               'text': r['text'],
                               'created_at': r['created_at'],
                               'media_url': r['media_url'],
                               'image_desc': r['image_desc']
                           }
                        else:
                            logger.warning(f"Skipping tweet {r['id']} due to unexpected embedding dimension.")
                    except Exception as e:
                        logger.error(f"Error parsing vector for tweet {r['id']}: {e}")

                if tweet_embeddings:
                    tweet_similarities = compute_similarities(query_embedding, tweet_embeddings)
                    for i, score in enumerate(tweet_similarities):
                        if i in tweet_data_map: # Check if data was stored for this index
                            all_results.append({
                                'type': 'tweet',
                                **tweet_data_map[i],
                                'similarity': score.item()
                            })


            # 2. Fetch Media Embeddings (Joint and Image)
            session.execute("""
                SELECT
                    m.tweet_id,
                    t.text,
                    m.media_url,
                    m.image_desc,
                    t.created_at,
                    m.joint_embedding::text as joint_embedding_array,
                    m.image_embedding::text as image_embedding_array
                FROM media m
                JOIN tweets t ON t.id = m.tweet_id
                WHERE m.joint_embedding IS NOT NULL
                  AND m.image_embedding IS NOT NULL
                  AND m.type = 'photo'
            """)
            media_db_results = session.fetchall()

            if media_db_results:
                media_joint_embeddings = []
                media_image_embeddings = []
                media_data_map = {} # Store data by index (applies to both lists)

                for i, r in enumerate(media_db_results):
                    try:
                        joint_vector_str = r['joint_embedding_array'].strip('[]')
                        joint_vector = [float(x) for x in joint_vector_str.split(',')]

                        image_vector_str = r['image_embedding_array'].strip('[]')
                        image_vector = [float(x) for x in image_vector_str.split(',')]

                        if len(joint_vector) == EMBEDDING_DIM and len(image_vector) == EMBEDDING_DIM:
                           media_joint_embeddings.append(joint_vector)
                           media_image_embeddings.append(image_vector)
                           idx = len(media_joint_embeddings) - 1 # Current index
                           media_data_map[idx] = {
                               'tweet_id': r['tweet_id'],
                               'text': r['text'], # Tweet text associated with media
                               'media_url': r['media_url'],
                               'image_desc': r['image_desc'],
                               'created_at': r['created_at']
                           }
                        else:
                             logger.warning(f"Skipping media {r['tweet_id']}/{r['media_url']} due to unexpected embedding dimension.")

                    except Exception as e:
                        logger.error(f"Error parsing vectors for media {r['tweet_id']}/{r['media_url']}: {e}")

                # Compute similarities for joint embeddings
                if media_joint_embeddings:
                    joint_similarities = compute_similarities(query_embedding, media_joint_embeddings)
                    for i, score in enumerate(joint_similarities):
                         if i in media_data_map:
                            all_results.append({
                                'type': 'media_joint',
                                **media_data_map[i],
                                'similarity': score.item()
                            })

                # Compute similarities for image embeddings
                if media_image_embeddings:
                    image_similarities = compute_similarities(query_embedding, media_image_embeddings)
                    for i, score in enumerate(image_similarities):
                        if i in media_data_map:
                           all_results.append({
                               'type': 'media_image',
                               **media_data_map[i],
                               'similarity': score.item()
                           })

        # 3. Sort all collected results by similarity
        all_results.sort(key=lambda x: x['similarity'], reverse=True)

        # 4. Return top N
        return all_results[:limit]

    except Exception as e:
        print(f"Error during traditional combined search: {e}")
        import traceback
        traceback.print_exc()
        return []

def search_all(query: str, limit: int = 10) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Search tweets (text vs text), media (text vs combined), and media (text vs image only) 
    with a single query.
    
    Args:
        query (str): Search query text
        limit (int): Maximum number of results to return for each type
        
    Returns:
        tuple: (tweet_results, media_combined_results, media_image_only_results)
    """
    tweet_results = search_tweets(query, limit)
    media_combined_results = search_media(query, limit) # Uses default 0.5/0.5 weights
    media_image_only_results = search_media_image_only(query, limit)
    
    return tweet_results, media_combined_results, media_image_only_results 